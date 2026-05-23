defmodule SelfHostedInferenceBumblebee.Adapter do
  @moduledoc """
  Runtime adapter descriptor for Bumblebee-backed route heads.
  """

  alias SelfHostedInferenceBumblebee.RouteHeadSpec
  alias SelfHostedInferenceCore.AdapterRef

  @default_model_id "Qwen/Qwen3-0.6B"

  @enforce_keys [:adapter_ref, :model_id, :route_head, :runtime_profile]
  defstruct [
    :adapter_ref,
    :model_id,
    :artifact_dir,
    :route_head,
    :hidden_layer,
    :runtime_profile,
    runtime_options: []
  ]

  @type t :: %__MODULE__{
          adapter_ref: AdapterRef.t(),
          model_id: String.t(),
          artifact_dir: String.t() | nil,
          route_head: RouteHeadSpec.t(),
          hidden_layer: integer() | nil,
          runtime_profile: atom() | SelfHostedInferenceBumblebee.Runtime.Profile.t(),
          runtime_options: keyword()
        }

  @spec qwen_sakana(keyword() | map()) :: t()
  def qwen_sakana(attrs \\ []) do
    attrs = Map.new(attrs)

    new!(
      adapter_ref:
        get_value(attrs, :adapter_ref) ||
          AdapterRef.new!(
            id: :trinity_qwen3_0_6b_sakana,
            version: "0.1.0",
            contract: :route_logits_v1
          ),
      model_id: get_value(attrs, :model_id, @default_model_id),
      artifact_dir: get_value(attrs, :artifact_dir),
      route_head:
        get_value(attrs, :route_head) ||
          RouteHeadSpec.new!(input_dim: 1024, num_agents: 7, num_roles: 3, head_variant: :linear),
      hidden_layer: get_value(attrs, :hidden_layer, 26),
      runtime_profile: get_value(attrs, :runtime_profile, :cuda_exla),
      runtime_options: get_value(attrs, :runtime_options, [])
    )
  end

  @spec mock_tiny(keyword() | map()) :: t()
  def mock_tiny(attrs \\ []) do
    attrs = Map.new(attrs)

    new!(
      adapter_ref:
        get_value(attrs, :adapter_ref) ||
          AdapterRef.new!(id: :mock_tiny, version: "0.1.0", contract: :route_logits_v1),
      model_id: get_value(attrs, :model_id, "mock-tiny"),
      artifact_dir: get_value(attrs, :artifact_dir),
      route_head:
        get_value(attrs, :route_head) ||
          RouteHeadSpec.new!(input_dim: 8, num_agents: 7, num_roles: 3, head_variant: :linear),
      hidden_layer: get_value(attrs, :hidden_layer),
      runtime_profile: get_value(attrs, :runtime_profile, :mock_tiny),
      runtime_options: get_value(attrs, :runtime_options, [])
    )
  end

  @spec new(keyword() | map() | t()) :: {:ok, t()} | {:error, term()}
  def new(%__MODULE__{} = adapter), do: {:ok, adapter}
  def new(attrs) when is_list(attrs), do: attrs |> Map.new() |> new()

  def new(attrs) when is_map(attrs) do
    with {:ok, adapter_ref} <- AdapterRef.new(get_value(attrs, :adapter_ref)),
         {:ok, route_head} <-
           RouteHeadSpec.new(get_value(attrs, :route_head, default_route_head())) do
      {:ok,
       %__MODULE__{
         adapter_ref: adapter_ref,
         model_id: get_value(attrs, :model_id, @default_model_id),
         artifact_dir: get_value(attrs, :artifact_dir),
         route_head: route_head,
         hidden_layer: get_value(attrs, :hidden_layer),
         runtime_profile: get_value(attrs, :runtime_profile, :cuda_exla),
         runtime_options: get_value(attrs, :runtime_options, [])
       }}
    end
  end

  def new(attrs), do: {:error, {:invalid_adapter, attrs}}

  @spec new!(keyword() | map() | t()) :: t()
  def new!(attrs) do
    case new(attrs) do
      {:ok, adapter} -> adapter
      {:error, reason} -> raise ArgumentError, "invalid adapter: #{inspect(reason)}"
    end
  end

  defp get_value(attrs, key, default \\ nil),
    do: Map.get(attrs, key, Map.get(attrs, Atom.to_string(key), default))

  defp default_route_head do
    RouteHeadSpec.new!(input_dim: 1024, num_agents: 7, num_roles: 3, head_variant: :linear)
  end
end
