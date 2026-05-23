defmodule SelfHostedInferenceBumblebee.RoutingHead do
  @moduledoc """
  A routing head that maps SLM hidden states to agent/role logits.

  The default is a single dense projection (`:linear`). Optional variants include
  `:block_diagonal` and `:sparse` for ablation work.
  """

  @head_variant_by_name %{
    "linear" => :linear,
    "block_diagonal" => :block_diagonal,
    "sparse" => :sparse
  }
  @known_head_variants Map.values(@head_variant_by_name)
  @selection_mode_by_name %{
    "argmax" => :argmax,
    "softmax" => :softmax,
    "softmax_argmax" => :softmax,
    "sample" => :sample
  }

  @doc """
  Builds the Axon model structure.

  Supported options:

    * `:head` - one of `:linear`, `:block_diagonal`, or `:sparse` (default `:linear`).
    * `:blocks` - number of blocks for `:block_diagonal` (default `1`).
    * `:sparse_k` - fixed feature width for `:sparse`.
  """
  def build_model(input_dim \\ 1024, num_agents \\ 7, num_roles \\ 3, opts \\ [])
      when is_integer(input_dim) and input_dim > 0 and
             is_integer(num_agents) and num_agents > 0 and
             is_integer(num_roles) and num_roles > 0 and
             is_list(opts) do
    head_opts = parse_head_options!(opts)
    total_outputs = num_agents + num_roles

    validate_head_dimensions!(input_dim, total_outputs, head_opts)

    build_model_for_head_variant(input_dim, total_outputs, head_opts)
  end

  defp build_model_for_head_variant(input_dim, total_outputs, head_opts) do
    case head_opts[:head] do
      :linear ->
        Axon.input("hidden_state", shape: {nil, input_dim})
        |> Axon.dense(total_outputs, name: "routing_head")

      :block_diagonal ->
        build_block_diagonal_model(input_dim, total_outputs, head_opts[:blocks])

      :sparse ->
        build_sparse_model(input_dim, total_outputs, head_opts[:sparse_k])
    end
  end

  @doc """
  Returns variant metadata and partition layout used for diagnostics and tests.
  """
  def variant_metadata(input_dim, num_agents, num_roles, opts \\ [])
      when is_integer(input_dim) and is_integer(num_agents) and is_integer(num_roles) and
             is_list(opts) do
    head_opts = parse_head_options!(opts)
    total_outputs = num_agents + num_roles

    validate_head_dimensions!(input_dim, total_outputs, head_opts)

    base = %{
      input_dim: input_dim,
      num_agents: num_agents,
      num_roles: num_roles,
      output_dim: total_outputs,
      head: head_opts[:head]
    }

    case head_opts[:head] do
      :linear ->
        base
        |> Map.put(:blocks, 1)
        |> Map.put(:effective_sparse_k, input_dim)
        |> Map.put(:input_partitions, [{0, input_dim}])
        |> Map.put(:output_partitions, [{0, total_outputs}])
        |> Map.put(:parameter_count, dense_param_count(input_dim, total_outputs))

      :block_diagonal ->
        in_counts = partition_counts(input_dim, head_opts[:blocks])
        out_counts = partition_counts(total_outputs, head_opts[:blocks])

        base
        |> Map.put(:blocks, head_opts[:blocks])
        |> Map.put(:input_partitions, partitions_with_offsets(in_counts))
        |> Map.put(:output_partitions, partitions_with_offsets(out_counts))
        |> Map.put(:parameter_count, block_diagonal_param_count(in_counts, out_counts))
        |> Map.put(:effective_sparse_k, nil)

      :sparse ->
        sparse_k = effective_sparse_k(head_opts[:sparse_k], input_dim)

        base
        |> Map.put(:blocks, 1)
        |> Map.put(:effective_sparse_k, sparse_k)
        |> Map.put(:input_partitions, [{0, sparse_k}])
        |> Map.put(:output_partitions, [{0, total_outputs}])
        |> Map.put(:parameter_count, dense_param_count(sparse_k, total_outputs))
    end
  end

  @doc """
  Returns trainable parameter count for the selected variant.
  """
  def parameter_count(input_dim, num_agents, num_roles, opts \\ []) when is_list(opts) do
    variant_metadata(input_dim, num_agents, num_roles, opts).parameter_count
  end

  @doc "Returns raw logits as a rank-2 tensor with shape {batch, num_agents+num_roles}."
  def output_logits(model, params, penultimate_tensor) do
    Axon.predict(model, params, %{"hidden_state" => penultimate_tensor})
  end

  @doc """
  Runs the real Axon forward pass and returns route details.

  The five-argument form preserves the imported Sakana runtime behavior: hard
  argmax over agent logits and role logits.
  """
  def route(model, params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    route(model, params, penultimate_tensor, num_agents, num_roles, [])
  end

  @doc """
  Runs the forward pass with explicit selection options.

  Options:

    * `:agent_selection` - `:argmax`, `:softmax`, `:softmax_argmax`, or `:sample`.
    * `:role_selection` - `:argmax`, `:softmax`, `:softmax_argmax`, or `:sample`.
    * `:return_probabilities` - include softmax probability tensors.
    * `:temperature` - positive softmax temperature.
    * `:seed` - deterministic sampling seed, either integer or `{a, b, c}`.

  Defaults remain deterministic argmax for both splits.
  """
  def route(model, params, penultimate_tensor, num_agents, num_roles, opts) when is_list(opts) do
    opts = normalize_route_opts!(opts)
    logits = output_logits(model, params, penultimate_tensor)
    validate_logits!(logits, num_agents, num_roles)

    logits_1d = Nx.squeeze(logits, axes: [0])

    agent_logits = Nx.slice(logits_1d, [0], [num_agents])
    role_logits = Nx.slice(logits_1d, [num_agents], [num_roles])

    agent = select_from_logits(agent_logits, opts.agent_selection, opts, :agent)
    role = select_from_logits(role_logits, opts.role_selection, opts, :role)

    if not is_integer(agent.id) or not is_integer(role.id) do
      raise ArgumentError, "invalid selection output from coordination head"
    end

    %{
      agent_id: agent.id,
      role_id: role.id,
      logits: logits,
      agent_logits: agent_logits,
      role_logits: role_logits,
      agent_selection_mode: opts.agent_selection,
      role_selection_mode: opts.role_selection,
      selection_temperature: opts.temperature,
      selection_seed: opts.seed,
      agent_probabilities: Map.get(agent, :probabilities),
      role_probabilities: Map.get(role, :probabilities)
    }
  end

  @doc "Alias for `route/6` kept for call sites that prefer an explicit name."
  def route_with_options(model, params, penultimate_tensor, num_agents, num_roles, opts) do
    route(model, params, penultimate_tensor, num_agents, num_roles, opts)
  end

  @doc "Runs the forward pass and returns `{agent_id, role_id}`."
  def forward(model, params, penultimate_tensor, num_agents \\ 7, num_roles \\ 3) do
    route = route(model, params, penultimate_tensor, num_agents, num_roles)
    {route.agent_id, route.role_id}
  end

  defp normalize_route_opts!(opts) do
    opts =
      Keyword.validate!(opts,
        agent_selection: :argmax,
        role_selection: :argmax,
        return_probabilities: false,
        temperature: 1.0,
        seed: nil
      )

    temperature = opts[:temperature]

    unless is_number(temperature) and temperature > 0 do
      raise ArgumentError, "temperature must be a positive number"
    end

    agent_selection = normalize_selection_mode!(opts[:agent_selection], :agent_selection)
    role_selection = normalize_selection_mode!(opts[:role_selection], :role_selection)

    %{
      agent_selection: agent_selection,
      role_selection: role_selection,
      return_probabilities: opts[:return_probabilities],
      temperature: temperature / 1,
      seed: opts[:seed]
    }
  end

  defp normalize_selection_mode!(mode, _key) when mode in [:argmax, :softmax, :sample],
    do: mode

  defp normalize_selection_mode!(:softmax_argmax, _key), do: :softmax

  defp normalize_selection_mode!(value, key) when is_binary(value) do
    case Map.fetch(@selection_mode_by_name, normalized_option_name(value)) do
      {:ok, mode} ->
        mode

      :error ->
        raise ArgumentError,
              "#{key} must be :argmax, :softmax, :softmax_argmax, or :sample, got #{inspect(value)}"
    end
  end

  defp normalize_selection_mode!(value, key) do
    raise ArgumentError,
          "#{key} must be :argmax, :softmax, :softmax_argmax, or :sample, got #{inspect(value)}"
  end

  defp normalized_option_name(value) do
    value
    |> String.trim()
    |> String.downcase()
    |> String.replace("-", "_")
  end

  defp normalize_head_variant!(head) when head in @known_head_variants, do: head

  defp normalize_head_variant!(head) when is_binary(head) do
    case Map.fetch(@head_variant_by_name, normalized_option_name(head)) do
      {:ok, variant} -> variant
      :error -> raise_invalid_head_variant!(head)
    end
  end

  defp normalize_head_variant!(head), do: raise_invalid_head_variant!(head)

  defp raise_invalid_head_variant!(head) do
    raise ArgumentError,
          "invalid head variant #{inspect(head)}"
  end

  defp parse_head_options!(opts) do
    head =
      opts
      |> Keyword.get(:head, :linear)
      |> normalize_head_variant!()

    blocks = Keyword.get(opts, :blocks, 1)
    sparse_k = Keyword.get(opts, :sparse_k, nil)

    blocks = normalize_blocks(head, blocks)
    sparse_k = normalize_sparse_k(sparse_k)

    %{head: head, blocks: blocks, sparse_k: sparse_k}
  end

  defp normalize_blocks(:block_diagonal, value) do
    if is_integer(value) and value > 0 do
      value
    else
      raise ArgumentError, "blocks must be a positive integer"
    end
  end

  defp normalize_blocks(_head, _value), do: 1

  defp normalize_sparse_k(value) when is_integer(value) and value > 0, do: value
  defp normalize_sparse_k(nil), do: nil

  defp normalize_sparse_k(_value),
    do: raise(ArgumentError, "sparse_k must be nil or positive integer")

  defp validate_head_dimensions!(input_dim, output_dim, head_opts) do
    if head_opts[:head] == :block_diagonal &&
         (head_opts[:blocks] > input_dim || head_opts[:blocks] > output_dim) do
      raise ArgumentError,
            "block_diagonal requires blocks <= input_dim and <= num_agents+num_roles, got blocks=#{head_opts[:blocks]}, input_dim=#{input_dim}, output_dim=#{output_dim}"
    end

    if head_opts[:head] == :sparse do
      sparse_k = head_opts[:sparse_k] || input_dim

      unless is_integer(sparse_k) and sparse_k >= 1 and sparse_k <= input_dim do
        raise ArgumentError,
              "sparse_k must be between 1 and input_dim (#{input_dim}), got #{sparse_k}"
      end
    end

    :ok
  end

  defp build_block_diagonal_model(input_dim, output_dim, blocks) do
    input_node = Axon.input("hidden_state", shape: {nil, input_dim})
    in_counts = partition_counts(input_dim, blocks)
    out_counts = partition_counts(output_dim, blocks)

    in_partitions = partitions_with_offsets(in_counts)
    out_partitions = partitions_with_offsets(out_counts)

    block_layers =
      Enum.zip(in_partitions, out_partitions)
      |> Enum.with_index()
      |> Enum.map(fn {{{in_start, in_count}, {_, out_count}}, idx} ->
        input_slice =
          Axon.nx(input_node, fn x ->
            Nx.slice(x, [0, in_start], [Nx.axis_size(x, 0), in_count])
          end)

        Axon.dense(input_slice, out_count, name: "routing_head_block_#{idx}")
      end)

    Axon.concatenate(block_layers, axis: 1, name: "routing_head")
  end

  defp build_sparse_model(input_dim, output_dim, sparse_k) do
    input_node = Axon.input("hidden_state", shape: {nil, input_dim})
    keep = effective_sparse_k(sparse_k, input_dim)

    sliced_input =
      Axon.nx(input_node, fn x ->
        Nx.slice(x, [0, 0], [Nx.axis_size(x, 0), keep])
      end)

    Axon.dense(sliced_input, output_dim, name: "routing_head")
  end

  defp effective_sparse_k(nil, input_dim), do: input_dim
  defp effective_sparse_k(sparse_k, _input_dim), do: sparse_k

  defp partition_counts(total, parts) do
    base = div(total, parts)
    remainder = rem(total, parts)

    0..(parts - 1)
    |> Enum.map(fn index ->
      if index < remainder do
        base + 1
      else
        base
      end
    end)
  end

  defp partitions_with_offsets(counts) do
    {_size, partitions} =
      Enum.reduce(counts, {0, []}, fn count, {offset, acc} ->
        {offset + count, [{offset, count} | acc]}
      end)

    Enum.reverse(partitions)
  end

  defp block_diagonal_param_count(in_counts, out_counts) do
    Enum.zip(in_counts, out_counts)
    |> Enum.reduce(0, fn {in_count, out_count}, acc ->
      acc + in_count * out_count + out_count
    end)
  end

  defp dense_param_count(input_dim, output_dim), do: input_dim * output_dim + output_dim

  defp select_from_logits(logits, :argmax, opts, _split) do
    result = %{id: Nx.to_number(Nx.argmax(logits))}

    if opts.return_probabilities do
      Map.put(result, :probabilities, softmax_1d(logits, opts.temperature))
    else
      result
    end
  end

  defp select_from_logits(logits, :softmax, opts, _split) do
    probabilities = softmax_1d(logits, opts.temperature)
    %{id: Nx.to_number(Nx.argmax(probabilities)), probabilities: probabilities}
  end

  defp select_from_logits(logits, :sample, opts, split) do
    probabilities = softmax_1d(logits, opts.temperature)
    id = sample_probability_index(probabilities, opts.seed, split)
    %{id: id, probabilities: probabilities}
  end

  defp softmax_1d(logits, temperature) do
    scaled = Nx.divide(logits, temperature)
    shifted = Nx.subtract(scaled, Nx.reduce_max(scaled))
    exp = Nx.exp(shifted)
    Nx.divide(exp, Nx.sum(exp))
  end

  defp sample_probability_index(probabilities, seed, split) do
    values = Nx.to_flat_list(probabilities)

    rng_seed = normalize_sample_seed(seed, split)
    :rand.seed(:exsss, rng_seed)
    draw = :rand.uniform()

    values
    |> Enum.with_index()
    |> Enum.reduce_while(0.0, fn {probability, index}, acc ->
      next = acc + probability

      if draw <= next do
        {:halt, index}
      else
        {:cont, next}
      end
    end)
    |> then(fn
      value when is_integer(value) -> value
      _ -> max(length(values) - 1, 0)
    end)
  end

  defp normalize_sample_seed({a, b, c}, :agent)
       when is_integer(a) and is_integer(b) and is_integer(c),
       do: {a, b, c}

  defp normalize_sample_seed({a, b, c}, :role)
       when is_integer(a) and is_integer(b) and is_integer(c),
       do: {a + 17, b + 31, c + 43}

  defp normalize_sample_seed(nil, split),
    do: normalize_sample_seed(System.unique_integer([:positive]), split)

  defp normalize_sample_seed(seed, :agent) when is_integer(seed), do: {seed, seed + 1, seed + 2}

  defp normalize_sample_seed(seed, :role) when is_integer(seed),
    do: {seed + 17, seed + 31, seed + 43}

  defp normalize_sample_seed(seed, _split) do
    raise ArgumentError,
          "seed must be nil, integer, or {integer, integer, integer}, got #{inspect(seed)}"
  end

  defp validate_logits!(logits, num_agents, num_roles) do
    output_dim = num_agents + num_roles
    logits_shape = Nx.shape(logits)

    case logits_shape do
      {1, ^output_dim} ->
        :ok

      {_batch, ^output_dim} ->
        raise ArgumentError,
              "coordination routing expects a single example, got shape #{inspect(logits_shape)}"

      {_batch, _dim} ->
        raise ArgumentError,
              "coordination head must output #{output_dim} logits, got shape #{inspect(logits_shape)}"

      _ ->
        raise ArgumentError, "invalid coordination head output shape #{inspect(logits_shape)}"
    end
  end
end
