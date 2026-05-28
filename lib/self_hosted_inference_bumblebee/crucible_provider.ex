defmodule SelfHostedInferenceBumblebee.CrucibleProvider do
  @moduledoc """
  Bumblebee implementation of the Crucible runtime provider contract.

  This module owns live model loading and telemetry extraction for
  `SelfHostedInferenceCore.CrucibleRuntime`. The provider-neutral runtime kernel
  never calls Bumblebee directly.
  """

  @behaviour SelfHostedInferenceCore.CrucibleRuntime.Provider

  alias CrucibleBumblebee.{
    Live,
    ManualGeneration,
    ModelBundle,
    ModelLoader,
    Preflight,
    TraceWriter
  }

  alias CrucibleBumblebee.ModelLoader.Options

  defstruct [
    :bundle,
    :loader_options,
    :tap_plan,
    :capability_report,
    :surface
  ]

  @impl true
  def init(opts) when is_list(opts) do
    options = loader_options(opts)

    with {:ok, bundle} <- ModelLoader.load(options) do
      tap_plan = Live.forward_tap_plan()
      preflight = Preflight.run!(bundle, tap_plan)

      {:ok,
       %__MODULE__{
         bundle: bundle,
         loader_options: options,
         tap_plan: tap_plan,
         capability_report: preflight.capability_report,
         surface: preflight.surface
       }}
    end
  rescue
    error -> {:error, {:bumblebee_provider_start_failed, Exception.message(error)}}
  end

  @impl true
  def forward(%__MODULE__{bundle: %ModelBundle{} = bundle} = state, input, opts) do
    prompt = prompt_from_input(input)
    name = Keyword.get(opts, :trace_name, "hosted_runtime_#{safe_name(bundle.model_id)}")
    trace_id = Keyword.get_lazy(opts, :trace_id, &trace_id/0)
    run_id = Keyword.get_lazy(opts, :run_id, &run_id/0)
    root = state.loader_options.artifact_root
    trace_path = TraceWriter.output_path("hosted_#{name}", "trace.jsonl", root: root)
    report_path = TraceWriter.output_path("hosted_#{name}", "capability_report.json", root: root)
    started = System.monotonic_time(:millisecond)
    forward_started = System.monotonic_time(:millisecond)

    TraceWriter.reset!(trace_path)
    TraceWriter.write_capability_report!(report_path, state.capability_report)

    TraceWriter.write!(trace_path, :trace_start,
      trace_id: trace_id,
      run_id: run_id,
      provider_kind: provider_kind(state),
      model_id: bundle.model_id,
      model_family: bundle.model_family,
      backend: bundle.backend,
      hosted_runtime_id: Keyword.get(opts, :runtime_id)
    )

    TraceWriter.write!(trace_path, :provider_capability_report,
      trace_id: trace_id,
      capability_report: state.capability_report,
      capability_report_digest: Crucible.CanonicalJSON.digest(state.capability_report)
    )

    TraceWriter.write!(trace_path, :forward_start,
      trace_id: trace_id,
      prompt_digest: CrucibleSignalTrace.Digest.prefixed_text(prompt)
    )

    logits = Live.run_logits(bundle, prompt)

    signal =
      TraceWriter.signal_from_logits(logits, %{
        signal_id: "sig_final_logits",
        trace_id: trace_id,
        run_id: run_id,
        model_id: bundle.model_id,
        model_family: bundle.model_family,
        model_revision: bundle.revision,
        backend: bundle.backend,
        capture_method: :hosted_loaded_bundle
      })

    TraceWriter.write!(trace_path, :signal_record, trace_id: trace_id, signal: signal)

    TraceWriter.write!(trace_path, :forward_end,
      trace_id: trace_id,
      forward_time_ms: elapsed_ms(forward_started)
    )

    TraceWriter.write!(trace_path, :trace_end,
      trace_id: trace_id,
      status: :ok,
      duration_ms: elapsed_ms(started)
    )

    trace =
      trace_path
      |> CrucibleSignalTrace.Ingest.from_jsonl!([])
      |> ensure_final_logits()

    {:ok, trace}
  rescue
    error -> {:error, {:bumblebee_forward_failed, Exception.message(error)}}
  end

  @impl true
  def generate(%__MODULE__{bundle: %{model_family: family}} = state, input, opts)
      when family in [:gpt2, :qwen3] do
    prompt = prompt_from_input(input)

    case ManualGeneration.run(state.bundle, prompt,
           max_new_tokens:
             Keyword.get(opts, :max_new_tokens, state.loader_options.max_new_tokens || 1),
           strategy: Keyword.get(opts, :generation_strategy, :greedy),
           seed: Keyword.get(opts, :seed, state.loader_options.seed),
           stop_token_ids: Keyword.get(opts, :stop_token_ids, [])
         ) do
      {:ok, generation} ->
        trace_path = write_generation_trace!(state, generation, prompt, opts)

        {:ok,
         %{
           model_id: state.bundle.model_id,
           backend: state.bundle.backend,
           generated_token_ids: generation.generated_token_ids,
           decoded_text: generation.decoded_text,
           step_count: length(generation.steps),
           success_level: :generation_step_logits,
           trace_path: trace_path
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def generate(%__MODULE__{} = state, _input, _opts) do
    {:error, {:generation_unavailable_for_model_family, state.bundle.model_family}}
  end

  @impl true
  def capabilities(%__MODULE__{} = state), do: state.capability_report

  @impl true
  def provider_kind(%__MODULE__{}), do: :elixir_bumblebee

  @impl true
  def model_id(%__MODULE__{} = state), do: state.bundle.model_id

  @impl true
  def backend(%__MODULE__{} = state), do: state.bundle.backend

  @impl true
  def surface_id(%__MODULE__{} = state), do: state.surface.id

  @impl true
  def ready?(%__MODULE__{}), do: true

  @impl true
  def tokenizer_loaded?(%__MODULE__{}), do: true

  @impl true
  def model_loaded?(%__MODULE__{}), do: true

  @impl true
  def state_machine(%__MODULE__{}) do
    [
      :init,
      :select_provider,
      :select_backend,
      :load_tokenizer,
      :load_model,
      :preflight_surface,
      :compile_tap_plan,
      :ready
    ]
  end

  defp loader_options(opts) do
    opts
    |> Keyword.take([
      :model_id,
      :tokenizer_id,
      :revision,
      :backend,
      :offline?,
      :cache_dir,
      :prompt,
      :max_new_tokens,
      :seed,
      :artifact_root,
      :architecture,
      :module,
      :diagnostic_path
    ])
    |> Options.new()
  end

  defp write_generation_trace!(state, generation, prompt, opts) do
    name =
      Keyword.get(
        opts,
        :trace_name,
        "hosted_generation_#{safe_name(state.bundle.model_id)}"
      )

    trace_id = trace_id()
    run_id = run_id()
    root = state.loader_options.artifact_root
    trace_path = TraceWriter.output_path("hosted_#{name}", "trace.jsonl", root: root)
    report_path = TraceWriter.output_path("hosted_#{name}", "capability_report.json", root: root)

    TraceWriter.reset!(trace_path)
    TraceWriter.write_capability_report!(report_path, state.capability_report)

    TraceWriter.write!(trace_path, :trace_start,
      trace_id: trace_id,
      run_id: run_id,
      provider_kind: provider_kind(state),
      model_id: state.bundle.model_id,
      model_family: state.bundle.model_family,
      backend: state.bundle.backend
    )

    TraceWriter.write!(trace_path, :provider_capability_report,
      trace_id: trace_id,
      capability_report: state.capability_report,
      capability_report_digest: Crucible.CanonicalJSON.digest(state.capability_report)
    )

    TraceWriter.write!(trace_path, :generation_start,
      trace_id: trace_id,
      prompt_digest: CrucibleSignalTrace.Digest.prefixed_text(prompt),
      max_new_tokens: length(generation.steps),
      decode_mode: Keyword.get(opts, :generation_strategy, :greedy)
    )

    Enum.each(generation.steps, fn step ->
      signal =
        TraceWriter.signal_from_tensor(step.logits, %{
          signal_id: "sig_generation_step_logits_#{step.step_index}",
          trace_id: trace_id,
          run_id: run_id,
          signal_type: :generation_step_logits,
          model_id: state.bundle.model_id,
          model_family: state.bundle.model_family,
          model_revision: state.bundle.revision,
          backend: state.bundle.backend,
          token_index: step.step_index,
          node_name: "generation_step_logits",
          capture_method: :manual_autoregressive_loop,
          capability_status: :captured
        })

      TraceWriter.write!(trace_path, :signal_record, trace_id: trace_id, signal: signal)

      TraceWriter.write!(trace_path, :generation_step,
        trace_id: trace_id,
        step_index: step.step_index,
        generated_token_id: step.token_id,
        generated_token_text: step.token_text,
        logits_signal_id: signal.signal_id,
        entropy: step.entropy,
        margin: step.margin,
        top_k: step.top_k
      )
    end)

    TraceWriter.write!(trace_path, :generation_end,
      trace_id: trace_id,
      status: :ok,
      generated_token_ids: generation.generated_token_ids,
      generated_text: generation.decoded_text
    )

    TraceWriter.write!(trace_path, :trace_end, trace_id: trace_id, status: :ok)
    trace_path
  end

  defp prompt_from_input(%{prompt: prompt}) when is_binary(prompt), do: prompt
  defp prompt_from_input(%{"prompt" => prompt}) when is_binary(prompt), do: prompt
  defp prompt_from_input(prompt) when is_binary(prompt), do: prompt
  defp prompt_from_input(_input), do: "Hi"

  defp ensure_final_logits(%Crucible.ForwardTrace{} = trace) do
    %{trace | final_logits: Enum.find(trace.signals, &(&1.signal_type == :final_logits))}
  end

  defp trace_id, do: "tr_#{System.unique_integer([:positive])}"
  defp run_id, do: "run_#{System.unique_integer([:positive])}"
  defp elapsed_ms(start_ms), do: System.monotonic_time(:millisecond) - start_ms

  defp safe_name(name) do
    name
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9_.-]+/, "_")
    |> String.trim("_")
  end
end
