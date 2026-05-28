defmodule SelfHostedInferenceBumblebee.QwenSakanaLoader do
  @moduledoc """
  High-level loader for the artifact-driven TRINITY coordinator.

  It returns a single struct-like map containing:

    * the Qwen model_info/tokenizer used for hidden-state extraction,
    * the standalone Axon routing-head model and params,
    * artifact manifest metadata,
    * inferred `num_agents`, `num_roles`, and hidden size.

  Provider LLM calls are not performed here.
  """

  alias SelfHostedInferenceBumblebee.{
    AdaptedQwenPatch,
    Extractor,
    HeadLoader,
    RoutingHead,
    SLMProfile
  }

  alias SelfHostedInferenceBumblebee.Runtime.{Preflight, Profile}
  alias SelfHostedInferenceCore.RouteLogits

  @type t :: %{
          required(:model_info) => map(),
          required(:tokenizer) => map(),
          required(:routing_model) => Axon.t(),
          required(:routing_params) => struct(),
          required(:manifest) => map(),
          required(:artifact_dir) => String.t(),
          required(:num_agents) => pos_integer(),
          required(:num_roles) => pos_integer(),
          required(:hidden_size) => pos_integer(),
          required(:backend) => term(),
          required(:runtime_profile) => SelfHostedInferenceBumblebee.Runtime.Profile.t()
        }

  @doc """
  Loads the Sakana-adapted Qwen backbone and routing head.

  Options:

    * `:artifact_dir` - defaults to `AdaptedQwenPatch.default_output_dir/0`.
    * `:num_roles` - defaults to `3`.
    * `:runtime_profile` - a `SelfHostedInferenceBumblebee.Runtime.Profile` name or struct.
      Defaults to `:cuda_exla` (canonical production lane). The profile
      determines the Nx backend tuple and whether CUDA must be present.
    * `:backend` - explicit override. When supplied, overrides the
      backend derived from `:runtime_profile`.
    * `:require_cuda` - explicit override. When supplied, overrides the
      profile's `require_cuda?` flag.

  The defaults preserve the canonical CUDA-shaped production lane unless a
  caller selects a different runtime profile or explicit override.
  """
  @spec load(keyword()) :: {:ok, t()} | {:error, term()}
  def load(opts \\ []) when is_list(opts) do
    opts =
      Keyword.validate!(opts,
        artifact_dir: AdaptedQwenPatch.default_output_dir(),
        num_roles: 3,
        runtime_profile: :cuda_exla,
        backend: nil,
        require_cuda: nil
      )

    profile = Profile.resolve(opts[:runtime_profile])
    backend = opts[:backend] || profile.nx_backend

    require_cuda? =
      case opts[:require_cuda] do
        nil -> profile.require_cuda?
        b when is_boolean(b) -> b
      end

    if require_cuda? do
      Preflight.put_cuda_backend!()
    end

    slm_profile =
      SLMProfile.qwen_coordinator()
      |> Map.put(:adapted_artifact_dir, opts[:artifact_dir])
      |> Map.put(:artifact_patch_options,
        patch_router_head: false,
        allow_incomplete: false,
        cast_tensors: true
      )
      |> Map.update!(:load_options, fn lo -> Keyword.put(lo, :backend, backend) end)

    with {:ok, {model_info, tokenizer}} <- SLMProfile.load_profile(slm_profile),
         {:ok, manifest} <- AdaptedQwenPatch.load_manifest(opts[:artifact_dir]),
         head_weights <-
           AdaptedQwenPatch.load_router_head!(opts[:artifact_dir], manifest: manifest),
         {:ok, head_state} <-
           HeadLoader.build_routing_state(head_weights,
             num_roles: opts[:num_roles],
             backend: backend
           ),
         :ok <-
           HeadLoader.assert_shape_invariants!(head_state, manifest) do
      {:ok,
       %{
         model_info: model_info,
         tokenizer: tokenizer,
         routing_model: head_state.model,
         routing_params: head_state.params,
         manifest: manifest,
         artifact_dir: opts[:artifact_dir],
         num_agents: head_state.num_agents,
         num_roles: head_state.num_roles,
         hidden_size: head_state.hidden_size,
         backend: backend,
         runtime_profile: profile
       }}
    else
      {:error, reason} -> {:error, reason}
    end
  rescue
    e ->
      {:error, {:coordinator_load_error, Exception.message(e)}}
  end

  @doc """
  Runs the atomic extract+project operation for a loaded Qwen/Sakana adapter.
  """
  def route(%{} = coordinator, messages, _opts \\ []) when is_list(messages) do
    with {:ok, routed} <-
           Extractor.extract_and_route(
             coordinator.model_info,
             coordinator.tokenizer,
             messages,
             &route_hidden_vector(&1, coordinator)
           ) do
      {:ok, route_logits(routed.route, routed, messages, coordinator.runtime_profile)}
    end
  end

  defp route_hidden_vector(hidden_vector, coordinator) do
    # EXLA may donate the route input during the head forward pass. Keep a host
    # snapshot for router trace diagnostics before passing the CUDA tensor into
    # Axon.
    route_input =
      hidden_vector
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> transfer_route_input(coordinator.backend)

    RoutingHead.route(
      coordinator.routing_model,
      coordinator.routing_params,
      route_input,
      coordinator.num_agents,
      coordinator.num_roles
    )
  end

  defp transfer_route_input(vector_snapshot, nil), do: vector_snapshot

  defp transfer_route_input(vector_snapshot, backend),
    do: Nx.backend_transfer(vector_snapshot, backend)

  defp route_logits(route, routed, messages, runtime_profile) do
    agent_logits = Nx.to_flat_list(route.agent_logits)
    role_logits = Nx.to_flat_list(route.role_logits)
    transcript_hash = transcript_hash(messages)

    %RouteLogits{
      role_logits: role_logits,
      agent_logits: agent_logits,
      selected_role_id: route.role_id,
      selected_agent_id: route.agent_id,
      token_count: routed.token_count,
      transcript_hash: transcript_hash,
      route_hash_inputs:
        route_hash_inputs(route.agent_id, route.role_id, agent_logits, role_logits),
      backend_label: Preflight.tensor_backend(route.logits),
      runtime_profile: runtime_profile.name,
      margins: %{agent: top_margin(agent_logits), role: top_margin(role_logits)}
    }
  end

  defp route_hash_inputs(agent_id, role_id, agent_logits, role_logits) do
    %{
      "schema" => "self_hosted_inference_bumblebee.route_hash_inputs.v1",
      "agent_id" => agent_id,
      "role_id" => role_id,
      "logits_rounded" => Enum.map(agent_logits ++ role_logits, &Float.round(&1, 6))
    }
  end

  defp transcript_hash(messages) when is_list(messages) do
    messages
    |> Enum.map(fn message ->
      %{
        role: Map.get(message, :role, Map.get(message, "role")),
        content: Map.get(message, :content, Map.get(message, "content"))
      }
    end)
    |> :erlang.term_to_binary([:compressed])
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp top_margin(logits) do
    logits
    |> Enum.sort(:desc)
    |> case do
      [first, second | _] -> first - second
      [_] -> 0.0
      [] -> 0.0
    end
  end
end
