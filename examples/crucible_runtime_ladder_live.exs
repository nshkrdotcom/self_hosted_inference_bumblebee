defmodule SelfHostedInferenceBumblebee.Examples.CrucibleRuntimeLadderLive do
  @moduledoc false

  alias CrucibleBumblebee.{Artifacts, LiveMatrix}
  alias SelfHostedInferenceBumblebee.CrucibleProvider
  alias SelfHostedInferenceCore.CrucibleRuntime

  def main(argv) do
    argv = normalize_argv(argv)
    gate? = "--gate" in argv

    {opts, _rest, _invalid} =
      OptionParser.parse(argv,
        strict: [
          backend: :string,
          rungs: :string,
          artifact_root: :string,
          forward_timeout_ms: :integer,
          generation_timeout_ms: :integer,
          gate: :boolean
        ]
      )

    run_live(opts, gate?)
  end

  defp run_live(opts, gate?) do
    root = Keyword.get(opts, :artifact_root)
    backend = Keyword.get(opts, :backend, "binary")
    forward_timeout_ms = Keyword.get(opts, :forward_timeout_ms, 240_000)
    generation_timeout_ms = Keyword.get(opts, :generation_timeout_ms, 240_000)

    Artifacts.ensure_layout!(root: root)

    rows =
      opts
      |> models()
      |> Enum.map(fn model ->
        run_model(model, backend, root, forward_timeout_ms, generation_timeout_ms)
      end)

    report_path = write_report!(rows, root)
    result = %{ok: Enum.all?(rows, &(&1.result == "passed")), rows: rows, report_path: report_path}
    IO.inspect(result)

    if gate? and not result.ok, do: System.halt(7)
  end

  defp models(opts) do
    LiveMatrix.model_ladder()
    |> Enum.reject(&Map.has_key?(&1, :expected_blocker))
    |> then(fn models ->
      case Keyword.get(opts, :rungs) do
        nil ->
          models

        rungs ->
          allowed = rungs |> String.split(",", trim: true) |> MapSet.new()
          Enum.filter(models, &MapSet.member?(allowed, &1.rung))
      end
    end)
  end

  defp run_model(model, backend, root, forward_timeout_ms, generation_timeout_ms) do
    id = :"crucible-runtime-ladder-#{model.rung}-#{System.unique_integer([:positive])}"
    started = System.monotonic_time(:millisecond)

    with {:ok, pid} <- start_runtime(model, id, backend, root),
         true <- CrucibleRuntime.ready?(pid),
         {:ok, lease} <- CrucibleRuntime.lease(pid, owner_ref: "hosted-ladder"),
         {:ok, trace} <- forward(pid, model, backend, forward_timeout_ms) do
      generation = maybe_generate(pid, model, generation_timeout_ms)
      :ok = CrucibleRuntime.release(lease)

      row = %{
        rung: model.rung,
        model_id: model.model_id,
        family: model.family,
        backend: backend,
        hosted_runtime: true,
        ready: true,
        lease: true,
        forward: true,
        generation: generation.ok,
        generation_result: generation.result,
        generation_step_count: generation.step_count,
        trace_id: trace.trace_id,
        signal_count: length(trace.signals),
        result: "passed",
        duration_ms: elapsed_ms(started)
      }

      write_row!(row, root)
      row
    else
      false ->
        row = failed_row(model, backend, :not_ready, started)
        write_row!(row, root)
        row

      {:error, reason} ->
        row = failed_row(model, backend, reason, started)
        write_row!(row, root)
        row
    end
  end

  defp start_runtime(model, id, backend, root) do
    CrucibleRuntime.start_child(
      id: id,
      provider_module: CrucibleProvider,
      provider_opts: [
        model_id: model.model_id,
        tokenizer_id: Map.get(model, :tokenizer_id, model.model_id),
        backend: backend,
        architecture: model.architecture,
        module: Map.get(model, :module),
        artifact_root: root,
        max_new_tokens: 1
      ]
    )
  end

  defp forward(pid, model, backend, timeout_ms) do
    CrucibleRuntime.forward(pid, nil, %{prompt: "Hi"},
      trace_name: "#{model.rung}_#{Artifacts.safe_name(model.model_id)}_#{backend}",
      timeout: timeout_ms
    )
  end

  defp maybe_generate(pid, %{family: family}, timeout_ms) when family in [:gpt2, :qwen3] do
    case CrucibleRuntime.generate(pid, nil, "Hi", max_new_tokens: 1, timeout: timeout_ms) do
      {:ok, result} ->
        %{ok: true, result: "passed", step_count: result.step_count}

      {:error, reason} ->
        %{ok: false, result: inspect(reason), step_count: 0}
    end
  end

  defp maybe_generate(_pid, _model, _timeout_ms) do
    %{ok: false, result: "non_causal_generation", step_count: 0}
  end

  defp failed_row(model, backend, reason, started) do
    %{
      rung: model.rung,
      model_id: model.model_id,
      family: model.family,
      backend: backend,
      hosted_runtime: true,
      ready: false,
      lease: false,
      forward: false,
      generation: false,
      result: "failed",
      error: inspect(reason),
      duration_ms: elapsed_ms(started)
    }
  end

  defp write_row!(row, root) do
    Artifacts.append_jsonl!(:model_matrix, "hosted_runtime_ladder.jsonl", row, root: root)
  end

  defp write_report!(rows, root) do
    path = Artifacts.path!(:reports, "hosted_runtime_matrix.md", root: root)

    columns = [
      {:rung, "rung"},
      {:model_id, "model_id"},
      {:family, "family"},
      {:backend, "backend"},
      {:result, "result"},
      {:ready, "ready"},
      {:lease, "lease"},
      {:forward, "forward"},
      {:generation, "generation"},
      {:generation_result, "generation_result"},
      {:generation_step_count, "generation_step_count"},
      {:duration_ms, "duration_ms"},
      {:error, "error"},
      {:trace_id, "trace_id"},
      {:signal_count, "signal_count"}
    ]

    body =
      [
        "# Hosted Runtime Matrix\n\n",
        "| ",
        columns |> Enum.map(&elem(&1, 1)) |> Enum.join(" | "),
        " |\n| ",
        columns |> Enum.map(fn _ -> "---" end) |> Enum.join(" | "),
        " |\n",
        Enum.map(rows, fn row ->
          [
            "| ",
            columns |> Enum.map(fn {key, _label} -> cell(Map.get(row, key)) end) |> Enum.join(" | "),
            " |\n"
          ]
        end)
      ]
      |> IO.iodata_to_binary()

    File.write!(path, body)
    path
  end

  defp cell(nil), do: ""
  defp cell(value), do: value |> to_string() |> String.replace("|", "\\|")

  defp normalize_argv(["--" | rest]), do: rest
  defp normalize_argv(args), do: args

  defp elapsed_ms(start_ms), do: System.monotonic_time(:millisecond) - start_ms
end

SelfHostedInferenceBumblebee.Examples.CrucibleRuntimeLadderLive.main(System.argv())
