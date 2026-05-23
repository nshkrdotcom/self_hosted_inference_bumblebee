defmodule SelfHostedInferenceBumblebee.HeadLoader do
  @moduledoc """
  Utilities for building a standalone Axon routing head from Sakana head weights.

  The Qwen/Bumblebee causal-LM params do not contain a `routing_head`; the routing
  head is a separate Axon model.  This module makes that separation explicit.
  """

  alias CrucibleTensorPatch.BackendLabel
  alias SelfHostedInferenceBumblebee.RoutingHead
  alias SelfHostedInferenceBumblebee.Runtime.Preflight

  @routing_head_layer "routing_head"

  @type build_result :: %{
          required(:model) => Axon.t(),
          required(:params) => struct(),
          required(:hidden_size) => pos_integer(),
          required(:output_count) => pos_integer(),
          required(:num_agents) => pos_integer(),
          required(:num_roles) => pos_integer()
        }

  @doc """
  Builds a routing head model and initialized params from `{output_count, hidden_size}` weights.

  Options:

    * `:num_roles` - default `3`.
    * `:backend` - optional backend for initialized params, e.g. `{EXLA.Backend, client: :cuda}`.
    * `:model_opts` - forwarded to `RoutingHead.build_model/4`.
  """
  @spec build_routing_state(term(), keyword()) :: {:ok, build_result()} | {:error, term()}
  def build_routing_state(head_weights, opts \\ [])

  def build_routing_state(%Nx.Tensor{} = head_weights, opts) when is_list(opts) do
    opts = Keyword.validate!(opts, num_roles: 3, backend: nil, model_opts: [])

    with {:ok, output_count, hidden_size, num_agents, num_roles} <-
           infer_head_dimensions(head_weights, opts[:num_roles]) do
      model = RoutingHead.build_model(hidden_size, num_agents, num_roles, opts[:model_opts])
      {init_fn, _predict_fn} = Axon.build(model)

      input =
        Nx.broadcast(0.0, {1, hidden_size})
        |> Nx.as_type(:f32)
        |> maybe_transfer(opts[:backend])

      params = init_fn.(input, Axon.ModelState.empty())

      {:ok,
       %{
         model: model,
         params: put_head_weights!(params, head_weights, backend: opts[:backend]),
         hidden_size: hidden_size,
         output_count: output_count,
         num_agents: num_agents,
         num_roles: num_roles
       }}
    end
  end

  def build_routing_state(_head_weights, _opts), do: {:error, :invalid_head_weights}

  @doc """
  Replaces `routing_head.kernel` and `routing_head.bias` in an Axon state.

  `head_weights` must be shaped `{output_count, hidden_size}` and is transposed
  to Axon's dense-kernel layout `{hidden_size, output_count}`.
  """
  def put_head_weights!(%Axon.ModelState{} = params, %Nx.Tensor{} = head_weights, opts \\ []) do
    opts = Keyword.validate!(opts, backend: nil, cast: true)

    {output_count, hidden_size} = Nx.shape(head_weights)
    data = params.data

    layer_key = resolve_map_key!(data, @routing_head_layer)
    layer = Map.fetch!(data, layer_key)

    kernel_key = resolve_map_key!(layer, "kernel")
    bias_key = resolve_map_key!(layer, "bias")

    existing_kernel = Map.fetch!(layer, kernel_key)
    existing_bias = Map.fetch!(layer, bias_key)

    expected_kernel_shape = {hidden_size, output_count}

    unless Nx.shape(existing_kernel) == expected_kernel_shape do
      raise ArgumentError,
            "routing head kernel shape mismatch: expected #{inspect(expected_kernel_shape)}, got #{inspect(Nx.shape(existing_kernel))}"
    end

    backend = opts[:backend] || backend_from_tensor(existing_kernel)
    target_type = Nx.type(existing_kernel)

    kernel =
      head_weights
      |> Nx.transpose()
      |> maybe_cast(target_type, opts[:cast])
      |> maybe_transfer(backend)

    bias =
      Nx.broadcast(0.0, Nx.shape(existing_bias))
      |> maybe_cast(Nx.type(existing_bias), true)
      |> maybe_transfer(backend)

    patched_layer =
      layer
      |> Map.put(kernel_key, kernel)
      |> Map.put(bias_key, bias)

    %{params | data: Map.put(data, layer_key, patched_layer)}
  end

  defp infer_head_dimensions(head_weights, num_roles)
       when is_integer(num_roles) and num_roles > 0 do
    case Nx.shape(head_weights) do
      {output_count, hidden_size}
      when output_count > num_roles and hidden_size > 0 ->
        {:ok, output_count, hidden_size, output_count - num_roles, num_roles}

      shape ->
        {:error, {:invalid_head_shape, shape}}
    end
  end

  defp infer_head_dimensions(_head_weights, num_roles),
    do: {:error, {:invalid_num_roles, num_roles}}

  defp resolve_map_key!(container, key) when is_map(container) and is_binary(key) do
    if Map.has_key?(container, key) do
      key
    else
      existing_atom_key(container, key) || raise_missing_map_key!(key)
    end
  end

  defp resolve_map_key!(container, key) when is_map(container) do
    if Map.has_key?(container, key) do
      key
    else
      raise_missing_map_key!(key)
    end
  end

  defp raise_missing_map_key!(key), do: raise(ArgumentError, "missing map key #{inspect(key)}")

  defp existing_atom_key(container, key) do
    Enum.find(Map.keys(container), fn
      atom when is_atom(atom) -> Atom.to_string(atom) == key
      _ -> false
    end)
  end

  defp maybe_cast(tensor, target_type, true), do: Nx.as_type(tensor, target_type)

  defp maybe_cast(tensor, target_type, false) do
    if Nx.type(tensor) == target_type do
      tensor
    else
      raise ArgumentError,
            "routing head type mismatch: expected #{inspect(target_type)}, got #{inspect(Nx.type(tensor))}"
    end
  end

  defp maybe_transfer(tensor, nil), do: tensor
  defp maybe_transfer(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp backend_from_tensor(tensor) do
    case BackendLabel.from_label(Preflight.tensor_backend(tensor)) do
      {:ok, backend_spec} -> backend_spec
      {:error, _} -> nil
    end
  end

  @doc """
  Validates structural invariants between a built routing-head state and the
  manifest the head was loaded from.

  Checks two load-bearing invariants for the Sakana-adapted head:

    1. `num_agents + num_roles == manifest["router_head_shape"][0]` (today 10).
    2. `hidden_size == manifest["router_head_shape"][1]` (today 1024).

  This is the only thing standing between a refreshed Sakana checkpoint with a
  different agent count and silently mis-sliced agent vs role logits. The
  manifest's `router_head_shape` is the authoritative size declaration; the
  built head's dimensions come from `build_routing_state/2` parsing the
  weights, so when these disagree the artifact has drifted out from under
  the loader.

  Returns `:ok` on success; raises `ArgumentError` otherwise.
  """
  @spec assert_shape_invariants!(map(), map()) :: :ok | no_return()
  def assert_shape_invariants!(head_state, manifest)
      when is_map(head_state) and is_map(manifest) do
    head_shape = manifest_router_head_shape(manifest)
    %{num_agents: num_agents, num_roles: num_roles, hidden_size: hidden_size} = head_state

    case head_shape do
      [out_dim, hidden_dim] ->
        unless num_agents + num_roles == out_dim do
          raise ArgumentError,
                "router head shape mismatch: manifest declares output_count=#{out_dim} " <>
                  "but built head has num_agents=#{num_agents} + num_roles=#{num_roles} = " <>
                  "#{num_agents + num_roles}"
        end

        unless hidden_size == hidden_dim do
          raise ArgumentError,
                "router head hidden-size mismatch: manifest declares hidden=#{hidden_dim} " <>
                  "but built head has hidden_size=#{hidden_size}"
        end

        :ok

      other ->
        raise ArgumentError,
              "router head shape in manifest is malformed; expected [out_dim, hidden_dim], got: " <>
                inspect(other)
    end
  end

  defp manifest_router_head_shape(manifest) do
    shape = Map.get(manifest, "router_head_shape") || semantic_head_shape(manifest)

    case shape do
      [_, _] = pair -> pair
      [_ | _] = list -> list
      _ -> nil
    end
  end

  defp semantic_head_shape(manifest) do
    routing =
      manifest
      |> Map.get("python_semantic_manifest", %{})
      |> Map.get("routing", %{})

    Map.get(routing, "head_shape")
  end
end
