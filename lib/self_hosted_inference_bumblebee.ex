defmodule SelfHostedInferenceBumblebee do
  @moduledoc """
  Bumblebee-backed self-hosted inference adapters.
  """

  alias SelfHostedInferenceBumblebee.{Adapter, QwenSakanaLoader}
  alias SelfHostedInferenceCore.RouteLogits

  @doc "Returns package metadata."
  @spec metadata() :: %{app: atom(), version: String.t()}
  def metadata do
    %{
      app: :self_hosted_inference_bumblebee,
      version: to_string(Application.spec(:self_hosted_inference_bumblebee, :vsn))
    }
  end

  @doc "Returns the concrete `SelfHostedInferenceCore.Backend` module."
  @spec backend_module() :: module()
  def backend_module, do: SelfHostedInferenceBumblebee.Backend

  @doc "Builds the canonical Qwen/Sakana adapter descriptor."
  @spec qwen_sakana_adapter(keyword() | map()) :: Adapter.t()
  def qwen_sakana_adapter(opts \\ []), do: Adapter.qwen_sakana(opts)

  @doc "Builds a deterministic mock adapter for default CI."
  @spec mock_tiny_adapter(keyword() | map()) :: Adapter.t()
  def mock_tiny_adapter(opts \\ []), do: Adapter.mock_tiny(opts)

  @doc "Loads a concrete adapter runtime."
  @spec load(Adapter.t() | keyword() | map()) :: {:ok, map() | Adapter.t()} | {:error, term()}
  def load(%Adapter{runtime_profile: :mock_tiny} = adapter), do: {:ok, adapter}

  def load(%Adapter{} = adapter) do
    opts =
      adapter.runtime_options
      |> Keyword.put(:runtime_profile, adapter.runtime_profile)
      |> maybe_put(:artifact_dir, adapter.artifact_dir)
      |> Keyword.put(:num_roles, adapter.route_head.num_roles)

    QwenSakanaLoader.load(opts)
  end

  def load(attrs) when is_list(attrs) or is_map(attrs) do
    attrs
    |> Adapter.new()
    |> case do
      {:ok, adapter} -> load(adapter)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Runs the atomic extract+project operation and returns route logits.
  """
  @spec route(Adapter.t() | map(), [map()], keyword()) ::
          {:ok, RouteLogits.t()} | {:error, term()}
  def route(adapter, messages, opts \\ [])

  def route(%Adapter{runtime_profile: :mock_tiny} = adapter, messages, opts)
      when is_list(messages) and is_list(opts) do
    {:ok, mock_route(adapter, messages, opts)}
  end

  def route(%Adapter{} = adapter, messages, opts) when is_list(messages) and is_list(opts) do
    with {:ok, loaded} <- load(adapter) do
      route(loaded, messages, opts)
    end
  end

  def route(%{} = loaded, messages, opts) when is_list(messages) and is_list(opts) do
    QwenSakanaLoader.route(loaded, messages, opts)
  end

  def route(_adapter, _messages, _opts), do: {:error, :invalid_route_request}

  defp mock_route(%Adapter{} = adapter, messages, _opts) do
    transcript_hash = transcript_hash(messages)

    <<agent_seed, role_seed, margin_seed, _rest::binary>> =
      Base.decode16!(transcript_hash, case: :lower)

    agent_id = rem(agent_seed, adapter.route_head.num_agents)
    role_id = rem(role_seed, adapter.route_head.num_roles)
    agent_logits = one_hot_logits(adapter.route_head.num_agents, agent_id, margin_seed / 255)
    role_logits = one_hot_logits(adapter.route_head.num_roles, role_id, margin_seed / 255)

    %RouteLogits{
      role_logits: role_logits,
      agent_logits: agent_logits,
      selected_role_id: role_id,
      selected_agent_id: agent_id,
      token_count: token_count(messages),
      transcript_hash: transcript_hash,
      route_hash_inputs: route_hash_inputs(agent_id, role_id, agent_logits, role_logits),
      backend_label: :mock_tiny,
      runtime_profile: :mock_tiny,
      margins: %{agent: top_margin(agent_logits), role: top_margin(role_logits)}
    }
  end

  defp one_hot_logits(count, selected, margin) do
    Enum.map(0..(count - 1), fn
      ^selected -> 1.0 + margin
      _ -> 0.0
    end)
  end

  defp route_hash_inputs(agent_id, role_id, agent_logits, role_logits) do
    %{
      "schema" => "self_hosted_inference_bumblebee.route_hash_inputs.v1",
      "agent_id" => agent_id,
      "role_id" => role_id,
      "logits_rounded" => Enum.map(agent_logits ++ role_logits, &Float.round(&1, 6))
    }
  end

  defp transcript_hash(messages) do
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

  defp token_count(messages) do
    messages
    |> Enum.map_join(" ", fn message ->
      to_string(Map.get(message, :content, Map.get(message, "content", "")))
    end)
    |> String.split()
    |> length()
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

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, _key, ""), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
