defmodule SelfHostedInferenceBumblebee.RouteHeadSpec do
  @moduledoc """
  Shape contract for an adapter routing head.
  """

  @enforce_keys [:input_dim, :num_agents, :num_roles, :head_variant]
  defstruct [:input_dim, :num_agents, :num_roles, :head_variant]

  @type t :: %__MODULE__{
          input_dim: pos_integer(),
          num_agents: pos_integer(),
          num_roles: pos_integer(),
          head_variant: atom()
        }

  @spec new(keyword() | map() | t()) :: {:ok, t()} | {:error, term()}
  def new(%__MODULE__{} = spec), do: {:ok, spec}
  def new(attrs) when is_list(attrs), do: attrs |> Map.new() |> new()

  def new(attrs) when is_map(attrs) do
    spec = %__MODULE__{
      input_dim: get_value(attrs, :input_dim, 1024),
      num_agents: get_value(attrs, :num_agents, 7),
      num_roles: get_value(attrs, :num_roles, 3),
      head_variant: get_value(attrs, :head_variant, :linear)
    }

    with :ok <- positive_integer(spec.input_dim, :input_dim),
         :ok <- positive_integer(spec.num_agents, :num_agents),
         :ok <- positive_integer(spec.num_roles, :num_roles),
         true <- spec.head_variant in [:linear, :block_diagonal, :sparse] do
      {:ok, spec}
    else
      false -> {:error, {:invalid_head_variant, spec.head_variant}}
      {:error, reason} -> {:error, reason}
    end
  end

  def new(attrs), do: {:error, {:invalid_route_head_spec, attrs}}

  @spec new!(keyword() | map() | t()) :: t()
  def new!(attrs) do
    case new(attrs) do
      {:ok, spec} -> spec
      {:error, reason} -> raise ArgumentError, "invalid route head spec: #{inspect(reason)}"
    end
  end

  defp positive_integer(value, _field) when is_integer(value) and value > 0, do: :ok
  defp positive_integer(value, field), do: {:error, {:invalid_positive_integer, field, value}}

  defp get_value(attrs, key, default),
    do: Map.get(attrs, key, Map.get(attrs, Atom.to_string(key), default))
end
