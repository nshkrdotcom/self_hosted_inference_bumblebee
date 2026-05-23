defmodule SelfHostedInferenceBumblebee.AdaptedQwenPatch do
  @moduledoc """
  Runtime loader and patcher for persisted Sakana artifacts.
  """

  alias Axon.ModelState
  alias CrucibleSafetensors.Reader
  alias CrucibleTensorPatch.BackendLabel
  alias SelfHostedInferenceBumblebee.Runtime.Preflight

  @manifest_version 1
  @default_output_dir "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"
  @manifest_file "manifest.json"
  @router_head_file "router_head.safetensors"
  @router_head_tensor_key "trinity_router_head"
  @adapted_tensors_file "adapted_tensors.safetensors"
  @export_log_file "export.log.jsonl"
  @checkpoint_dir_name "checkpoints"
  @artifact_layout_single_file "single_file"
  @artifact_layout_checkpoint_directory "checkpoint_directory"

  @status_complete "complete"

  @required_manifest_keys [
    "artifact_version",
    "status",
    "selected_tensors",
    "adapted_tensors_artifact",
    "router_head_artifact",
    "router_head_shape",
    "artifact_layout",
    "selected_tensor_count",
    "selected_singular_value_count",
    "source_vector_shape",
    "source_vector_sha256",
    "scale_offset_count",
    "router_head_tensor_key",
    "base_model_repo",
    "bumblebee_module",
    "architecture",
    "xla_target",
    "export_backend",
    "source_vector_path",
    "source_vector_tensor",
    "export_complete",
    "source_split",
    "split"
  ]

  @identity_keys [
    "base_model_repo",
    "bumblebee_module",
    "architecture",
    "xla_target",
    "export_backend",
    "source_vector_path",
    "source_vector_tensor",
    "source_vector_sha256",
    "selected_tensors",
    "selected_singular_value_count",
    "scale_offset_count",
    "router_head_shape",
    "selected_tensor_count",
    "source_vector_shape"
  ]

  @identity_selected_tensor_keys [
    "path",
    "artifact_key",
    "shape",
    "singular_values",
    "type",
    "segments"
  ]

  @doc "Returns canonical manifest version."
  def manifest_version, do: @manifest_version

  @doc "Returns canonical output directory."
  def default_output_dir, do: @default_output_dir

  @doc "Returns canonical manifest file name."
  def manifest_file, do: @manifest_file

  @doc "Returns canonical router-head file name."
  def router_head_file, do: @router_head_file

  @doc "Returns canonical router-head tensor key."
  def router_head_tensor_key, do: @router_head_tensor_key

  @doc "Returns canonical adapted-tensors file name."
  def adapted_tensors_file, do: @adapted_tensors_file

  @doc "Returns canonical checkpoint directory name."
  def checkpoint_directory_name, do: @checkpoint_dir_name

  @doc "Returns manifest path in an output directory."
  def manifest_path(out_dir), do: Path.join(out_dir, @manifest_file)

  @doc "Returns checkpoints path in an output directory."
  def checkpoint_path(out_dir), do: Path.join(out_dir, @checkpoint_dir_name)

  @doc "Returns single-file artifact layout key."
  def artifact_layout_single_file, do: @artifact_layout_single_file

  @doc "Returns checkpoint-directory artifact layout key."
  def artifact_layout_checkpoint_directory, do: @artifact_layout_checkpoint_directory

  @doc "Returns canonical export event log file name."
  def export_log_file, do: @export_log_file

  @doc "Returns export log path in an output directory."
  def export_log_path(out_dir), do: Path.join(out_dir, @export_log_file)

  @doc """
  Loads and validates manifest JSON from disk.
  """
  def load_manifest(out_dir) when is_binary(out_dir) do
    with {:ok, body} <- File.read(manifest_path(out_dir)),
         {:ok, decoded} <- Jason.decode(body),
         normalized <- normalize_string_keys(decoded),
         {:ok, manifest} <- validate_manifest(normalized) do
      {:ok, manifest}
    else
      {:error, :enoent} ->
        {:error, :missing_manifest}

      {:error, %Jason.DecodeError{} = reason} ->
        {:error, {:invalid_manifest, reason}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def load_manifest(_), do: {:error, :invalid_output_dir}

  @doc """
  Loads and validates manifest JSON from disk and raises on failure.
  """
  def load_manifest!(out_dir) when is_binary(out_dir) do
    case load_manifest(out_dir) do
      {:ok, manifest} ->
        manifest

      {:error, reason} ->
        raise ArgumentError,
              "unable to load manifest from #{inspect(out_dir)}: #{inspect(reason)}"
    end
  end

  @doc """
  Writes manifest JSON atomically using `.tmp`.
  """
  def write_manifest!(out_dir, manifest) when is_binary(out_dir) and is_map(manifest) do
    File.mkdir_p!(out_dir)

    path = manifest_path(out_dir)
    tmp = path <> ".tmp"

    encoded = Jason.encode!(normalize_string_keys(manifest))
    File.write!(tmp, encoded)
    File.rename!(tmp, path)
    :ok
  end

  @doc """
  Loads adapted tensors from either single-file artifact or checkpoint directory.
  """
  def load_adapted_tensors!(out_dir, opts \\ []) when is_binary(out_dir) do
    opts = Keyword.validate!(opts, manifest: nil, allow_incomplete: false)
    manifest = opts[:manifest] || load_manifest!(out_dir)
    ensure_manifest_complete!(manifest, opts[:allow_incomplete])

    entries = selected_tensors(manifest)

    if entries == [] do
      raise ArgumentError, "manifest has no selected_tensors"
    end

    case field(manifest, "artifact_layout", @artifact_layout_checkpoint_directory) do
      @artifact_layout_single_file ->
        load_adapted_tensors_single_file(out_dir, manifest, entries)

      @artifact_layout_checkpoint_directory ->
        load_adapted_tensors_from_checkpoints(out_dir, manifest, entries)

      layout ->
        raise ArgumentError, "unknown artifact_layout #{inspect(layout)}"
    end
  end

  defp load_adapted_tensors_single_file(out_dir, manifest, entries) do
    path = Path.join(out_dir, field(manifest, "adapted_tensors_artifact", @adapted_tensors_file))
    validate_file_sha256!(path, field(manifest, "adapted_tensors_sha256"), :adapted_tensors)
    _header = Reader.open!(path)

    Enum.reduce(entries, %{}, fn entry, acc ->
      path_key = field(entry, "path")
      artifact_key = field(entry, "artifact_key", path_key)
      entry_tensor = load_safetensor_tensor!(path, artifact_key)
      ensure_tensor_shape_and_type!(entry_tensor, path_key, entry)

      Map.put(acc, artifact_key, entry_tensor)
    end)
  end

  defp load_adapted_tensors_from_checkpoints(out_dir, _manifest, entries) do
    Enum.reduce(entries, %{}, fn entry, acc ->
      if field(entry, "status", "complete") != "complete" do
        raise ArgumentError,
              "cannot load incomplete adapted tensor #{inspect(field(entry, "path"))}"
      end

      path_key = field(entry, "path")
      artifact_key = field(entry, "artifact_key", path_key)
      checkpoint_path = Path.join(out_dir, field(entry, "checkpoint_path", ""))
      validate_file_sha256!(checkpoint_path, field(entry, "checkpoint_sha256"), path_key)

      checkpoint_tensor = load_safetensor_tensor!(checkpoint_path, artifact_key)
      ensure_tensor_shape_and_type!(checkpoint_tensor, path_key, entry)

      Map.put(acc, artifact_key, checkpoint_tensor)
    end)
  end

  @doc """
  Loads router head tensor from safetensors artifact with optional shape check.
  """
  def load_router_head!(out_dir, opts \\ []) do
    opts = Keyword.validate!(opts, manifest: nil, allow_incomplete: false, expected_shape: nil)
    manifest = opts[:manifest] || load_manifest!(out_dir)
    ensure_manifest_complete!(manifest, opts[:allow_incomplete])

    path = Path.join(out_dir, field(manifest, "router_head_artifact", @router_head_file))
    validate_file_sha256!(path, field(manifest, "router_head_sha256"), :router_head)
    key = field(manifest, "router_head_tensor_key", @router_head_tensor_key)
    tensor = load_safetensor_tensor!(path, key)

    expected_shape =
      case opts[:expected_shape] do
        nil -> field(manifest, "router_head_shape")
        custom -> custom
      end

    validate_shape!(tensor, expected_shape, :router_head)
    tensor
  end

  @doc """
  Patches params with adapted tensors using manifest-selected paths.
  """
  def patch_params!(params, manifest, tensors), do: patch_params!(params, manifest, tensors, [])

  def patch_params!(%ModelState{} = params, manifest, tensors, opts)
      when is_map(manifest) and is_map(tensors) do
    %{params | data: patch_params!(params.data, manifest, tensors, opts)}
  end

  def patch_params!(params, manifest, tensors, opts)
      when is_map(params) and is_map(manifest) and is_map(tensors) do
    cast_tensors = Keyword.get(opts, :cast_tensors, true)
    allow_incomplete = Keyword.get(opts, :allow_incomplete, false)
    selected = selected_tensors(manifest)
    ensure_manifest_complete!(manifest, allow_incomplete)

    Enum.reduce(selected, params, fn entry, acc ->
      path = field(entry, "path")
      key = field(entry, "artifact_key", path)
      tensor = Map.fetch!(tensors, key)
      segments = normalize_segments(field(entry, "segments"), path)
      target = fetch_nested_tensor(acc, segments, path)
      patched = align_tensor_for_target!(tensor, target, cast_tensors, path)
      patch_param!(acc, segments, patched, path)
    end)
  end

  def patch_params!(_params, _manifest, _tensors, _opts) do
    raise ArgumentError, "unsupported params container for patching"
  end

  @doc """
  Patches model params and routing head from a prepared artifact directory.
  """
  def patch_model_info!(model_info, out_dir, opts \\ []) when is_map(model_info) do
    opts =
      Keyword.validate!(
        opts,
        manifest: nil,
        allow_incomplete: false,
        cast_tensors: true,
        cast_head: true,
        patch_router_head: true,
        head_transfer: true
      )

    manifest = opts[:manifest] || load_manifest!(out_dir)
    ensure_manifest_complete!(manifest, opts[:allow_incomplete])

    params = field(model_info, :params, field(model_info, "params"))

    if not is_map(params) and not is_struct(params, ModelState) do
      raise ArgumentError, "invalid model_info: missing params"
    end

    tensors =
      load_adapted_tensors!(out_dir,
        manifest: manifest,
        allow_incomplete: opts[:allow_incomplete]
      )

    patched_params =
      patch_params!(params, manifest, tensors,
        cast_tensors: opts[:cast_tensors],
        cast_head: opts[:cast_head],
        allow_incomplete: opts[:allow_incomplete]
      )

    head_weights =
      load_router_head!(out_dir, manifest: manifest, allow_incomplete: opts[:allow_incomplete])

    final_params =
      if opts[:patch_router_head] do
        patched_params
        |> patch_router_head!(head_weights,
          cast: opts[:cast_head],
          transfer: opts[:head_transfer]
        )
      else
        patched_params
      end

    manifest_file_path = manifest_path(out_dir)
    manifest_hash = file_sha256!(manifest_file_path)

    model_info
    |> Map.put(:params, final_params)
    |> Map.put("params", final_params)
    |> Map.put(:trinity_artifact_manifest, manifest)
    |> Map.put(:trinity_artifact_manifest_hash, manifest_hash)
    |> Map.put(:trinity_artifact_manifest_path, manifest_file_path)
    |> Map.put(
      :trinity_artifact_trace_metadata,
      trace_metadata(manifest, manifest_hash, manifest_file_path)
    )
  end

  @doc "Returns a compact trace metadata map for orchestrator events."
  def trace_metadata(model_info_or_manifest, manifest_hash \\ nil, manifest_path \\ nil)
      when is_map(model_info_or_manifest) do
    manifest =
      case Map.get(model_info_or_manifest, "trinity_artifact_manifest") do
        nil -> Map.get(model_info_or_manifest, :trinity_artifact_manifest)
        other -> other
      end

    if is_map(manifest) do
      resolved_manifest_path =
        manifest_path ||
          Map.get(model_info_or_manifest, :trinity_artifact_manifest_path) ||
          Map.get(model_info_or_manifest, "trinity_artifact_manifest_path")

      resolved_manifest_hash =
        manifest_hash ||
          Map.get(model_info_or_manifest, :trinity_artifact_manifest_hash) ||
          Map.get(model_info_or_manifest, "trinity_artifact_manifest_hash")

      %{
        "trinity_artifact_manifest_hash" => resolved_manifest_hash,
        "trinity_artifact_manifest_path" => resolved_manifest_path,
        "trinity_artifact_base_model_repo" => field(manifest, "base_model_repo"),
        "trinity_artifact_bumblebee_module" => field(manifest, "bumblebee_module"),
        "trinity_artifact_architecture" => field(manifest, "architecture"),
        "trinity_artifact_source_vector_sha256" => field(manifest, "source_vector_sha256")
      }
    else
      %{}
    end
  end

  @doc "Computes file SHA-256 in lowercase hex."
  def file_sha256!(path) when is_binary(path) do
    File.open!(path, [:read, :binary], fn file ->
      stream = IO.binstream(file, 1_048_576)

      Enum.reduce(stream, :crypto.hash_init(:sha256), fn chunk, acc ->
        :crypto.hash_update(acc, chunk)
      end)
      |> :crypto.hash_final()
      |> Base.encode16(case: :lower)
    end)
  end

  @doc "Computes tensor SHA-256 in lowercase hex after host transfer."
  def tensor_sha256(%Nx.Tensor{} = tensor) do
    binary =
      tensor
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_binary()

    :crypto.hash(:sha256, binary)
    |> Base.encode16(case: :lower)
  end

  @doc """
  Returns selected tensor entries list.
  """
  def selected_tensors(manifest) when is_map(manifest) do
    field(manifest, "selected_tensors", [])
  end

  @doc """
  Checks whether the identity fields match for resume workflows.
  """
  def identity_matches?(expected, observed) when is_map(expected) and is_map(observed) do
    key_values_match? =
      Enum.all?(@identity_keys, fn key ->
        if key == "selected_tensors" do
          identity_selected_tensors(field(expected, key)) ==
            identity_selected_tensors(field(observed, key))
        else
          normalize_identity_value(field(expected, key)) ==
            normalize_identity_value(field(observed, key))
        end
      end)

    key_values_match?
  end

  defp identity_selected_tensors(entries) when is_list(entries) do
    entries
    |> Enum.map(fn entry ->
      entry
      |> Map.take(@identity_selected_tensor_keys)
      |> Map.new(fn {key, value} -> {key, normalize_identity_value(value)} end)
    end)
  end

  defp identity_selected_tensors(_entries), do: []

  @doc "Returns required identity keys."
  def required_identity_keys, do: @identity_keys

  defp patch_router_head!(%ModelState{} = params, head_weights, opts) do
    %{params | data: patch_router_head!(params.data, head_weights, opts)}
  end

  defp patch_router_head!(params, head_weights, opts) do
    layer_key = resolve_map_key(params, "routing_head")

    if is_nil(layer_key) do
      raise ArgumentError, "missing routing_head layer in params"
    end

    layer = Map.fetch!(params, layer_key)
    kernel_key = resolve_map_key(layer, "kernel")
    bias_key = resolve_map_key(layer, "bias")

    if is_nil(kernel_key) do
      raise ArgumentError, "missing routing_head.kernel in params"
    end

    if is_nil(bias_key) do
      raise ArgumentError, "missing routing_head.bias in params"
    end

    kernel = Map.fetch!(layer, kernel_key)
    existing_shape = Nx.shape(kernel)

    unless tuple_size(existing_shape) == 2 do
      raise ArgumentError, "routing_head kernel shape #{inspect(existing_shape)} is invalid"
    end

    {hidden_size, output_count} = existing_shape
    expected_head_shape = {output_count, hidden_size}

    unless Nx.shape(head_weights) == expected_head_shape do
      raise ArgumentError,
            "router head shape mismatch: expected #{inspect(expected_head_shape)} got #{inspect(Nx.shape(head_weights))}"
    end

    target_type = Nx.type(kernel)
    target_backend = backend_from_label(Preflight.tensor_backend(kernel))
    cast? = Keyword.get(opts, :cast, true)
    transfer? = Keyword.get(opts, :transfer, false)

    kernel_tensor =
      head_weights
      |> Nx.transpose()
      |> align_tensor!(target_type, target_backend, cast?, transfer?)

    bias_tensor =
      Nx.broadcast(0.0, {output_count})
      |> align_tensor!(target_type, target_backend, cast?, transfer?)

    layer = Map.put(layer, kernel_key, kernel_tensor)
    layer = Map.put(layer, bias_key, bias_tensor)

    put_in(params, [layer_key], layer)
  end

  defp align_tensor!(tensor, target_type, target_backend, true, transfer?) do
    tensor
    |> Nx.as_type(target_type)
    |> maybe_transfer(transfer?, target_backend)
  end

  defp align_tensor!(%Nx.Tensor{} = tensor, target_type, target_backend, false, transfer?) do
    if Nx.type(tensor) != target_type do
      raise ArgumentError,
            "tensor type mismatch: expected #{inspect(target_type)}, got #{inspect(Nx.type(tensor))}"
    end

    maybe_transfer(tensor, transfer?, target_backend)
  end

  defp align_tensor_for_target!(%Nx.Tensor{} = tensor, %Nx.Tensor{} = target, cast?, path) do
    target_type = Nx.type(target)
    target_shape = Nx.shape(target)
    target_backend = backend_from_label(Preflight.tensor_backend(target))

    if Nx.shape(tensor) != target_shape do
      raise ArgumentError,
            "adapted tensor #{inspect(path)} shape mismatch: expected #{inspect(target_shape)}, got #{inspect(Nx.shape(tensor))}"
    end

    align_tensor!(tensor, target_type, target_backend, cast?, true)
  end

  defp maybe_transfer(%Nx.Tensor{} = tensor, true, backend) do
    Nx.backend_transfer(tensor, backend)
  end

  defp maybe_transfer(%Nx.Tensor{} = tensor, false, _backend), do: tensor

  defp backend_from_label(label), do: BackendLabel.from_label!(label)

  defp ensure_manifest_complete!(_manifest, true), do: :ok

  defp ensure_manifest_complete!(manifest, false) do
    if field(manifest, "partial_debug_only", false) do
      raise ArgumentError, "artifact is partial_debug_only"
    end

    if field(manifest, "artifact_version") != @manifest_version do
      raise ArgumentError, "unsupported artifact version"
    end

    if field(manifest, "status") != @status_complete do
      raise ArgumentError, "manifest status #{inspect(field(manifest, "status"))} is incomplete"
    end

    if field(manifest, "export_complete") != true do
      raise ArgumentError, "manifest export_complete=false"
    end

    :ok
  end

  defp ensure_tensor_shape_and_type!(tensor, path, entry) do
    expected_shape = normalize_shape(field(entry, "shape"))

    if expected_shape && Nx.shape(tensor) != expected_shape do
      raise ArgumentError,
            "adapted tensor #{inspect(path)} shape mismatch: expected #{inspect(expected_shape)}, got #{inspect(Nx.shape(tensor))}"
    end

    expected_type = normalize_type(field(entry, "type"))
    actual_type = normalize_type(Nx.type(tensor))

    if expected_type && actual_type != expected_type do
      raise ArgumentError,
            "adapted tensor #{inspect(path)} type mismatch: expected #{inspect(expected_type)}, got #{inspect(actual_type)}"
    end
  end

  defp validate_shape!(_tensor, nil, _label), do: :ok

  defp validate_shape!(tensor, expected, label) do
    expected_shape = normalize_shape(expected)

    unless is_tuple(expected_shape) and Nx.shape(tensor) == expected_shape do
      raise ArgumentError,
            "#{label} shape mismatch: expected #{inspect(expected_shape)} got #{inspect(Nx.shape(tensor))}"
    end
  end

  defp validate_file_sha256!(_path, nil, _label), do: :ok
  defp validate_file_sha256!(_path, "", _label), do: :ok

  defp validate_file_sha256!(path, expected, label) when is_binary(expected) do
    actual = file_sha256!(path)

    if actual != expected do
      raise ArgumentError,
            "#{label} sha256 mismatch: expected #{expected}, got #{actual}"
    end
  end

  defp validate_manifest(manifest) do
    with :ok <- validate_required_keys(manifest),
         :ok <- validate_selected_tensors(manifest) do
      {:ok, manifest}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp validate_required_keys(manifest) do
    case Enum.find(@required_manifest_keys, &(field(manifest, &1) == nil)) do
      nil -> :ok
      missing -> {:error, {:missing_manifest_key, missing}}
    end
  end

  defp validate_selected_tensors(manifest) do
    case field(manifest, "selected_tensors") do
      entries when is_list(entries) -> :ok
      _ -> {:error, :invalid_selected_tensors}
    end
  end

  defp load_safetensor_tensor!(path, key) when is_binary(path) do
    header = Reader.open!(path)
    key = normalize_string_key(key)
    tensor_info = Map.fetch!(header.tensors, key)

    case Reader.read_tensor(header, tensor_info) do
      {:ok, slice} ->
        slice.data
        |> Nx.from_binary(tensor_info.dtype)
        |> Nx.reshape(List.to_tuple(tensor_info.shape))

      {:error, exception} ->
        raise exception
    end
  end

  defp patch_param!(_container, [], _tensor, _path),
    do: raise(ArgumentError, "cannot patch empty segment path")

  defp patch_param!(container, [segment], tensor, path) do
    put_child!(container, segment, tensor, path)
  end

  defp patch_param!(container, [segment | rest], tensor, path) do
    container
    |> fetch_child!(segment, path)
    |> patch_param!(rest, tensor, path)
    |> then(&put_child!(container, segment, &1, path))
  end

  defp fetch_nested_tensor(_container, [], path) do
    raise ArgumentError, "invalid segment path for #{inspect(path)}"
  end

  defp fetch_nested_tensor(container, segments, path) when is_list(segments) do
    Enum.reduce(segments, container, fn segment, acc -> fetch_child!(acc, segment, path) end)
  end

  defp fetch_child!(container, segment, path) when is_map(container) do
    case resolve_map_key(container, segment) do
      nil -> raise ArgumentError, "missing path segment #{inspect(segment)} for #{inspect(path)}"
      resolved -> Map.fetch!(container, resolved)
    end
  end

  defp fetch_child!(container, segment, _path) when is_list(container) and is_integer(segment) do
    if valid_index?(segment, length(container)) do
      Enum.at(container, segment)
    else
      raise ArgumentError, "missing list index #{inspect(segment)}"
    end
  end

  defp fetch_child!(container, segment, _path) when is_tuple(container) and is_integer(segment) do
    if valid_index?(segment, tuple_size(container)) do
      elem(container, segment)
    else
      raise ArgumentError, "missing tuple index #{inspect(segment)}"
    end
  end

  defp fetch_child!(container, segment, path) do
    raise ArgumentError,
          "cannot descend into #{inspect(container)} at #{inspect(segment)} for #{inspect(path)}"
  end

  defp put_child!(container, segment, value, path) when is_map(container) do
    case resolve_map_key(container, segment) do
      nil -> raise ArgumentError, "missing map key #{inspect(segment)} for #{inspect(path)}"
      resolved -> Map.put(container, resolved, value)
    end
  end

  defp put_child!(container, segment, value, _path)
       when is_list(container) and is_integer(segment) do
    if valid_index?(segment, length(container)) do
      List.update_at(container, segment, fn _ -> value end)
    else
      raise ArgumentError, "missing list index #{inspect(segment)}"
    end
  end

  defp put_child!(container, segment, value, _path)
       when is_tuple(container) and is_integer(segment) do
    if valid_index?(segment, tuple_size(container)) do
      put_elem(container, segment, value)
    else
      raise ArgumentError, "missing tuple index #{inspect(segment)}"
    end
  end

  defp put_child!(container, segment, _value, path) do
    raise ArgumentError,
          "cannot patch #{inspect(path)} at #{inspect(segment)} in #{inspect(container)}"
  end

  defp valid_index?(segment, size), do: segment >= 0 and segment < size

  defp resolve_map_key(container, key) when is_map(container) do
    cond do
      Map.has_key?(container, key) ->
        key

      is_binary(key) ->
        existing_atom_map_key(container, key)

      is_atom(key) ->
        string_map_key(container, key)

      true ->
        nil
    end
  end

  defp existing_atom_map_key(container, key) do
    Enum.find(Map.keys(container), fn
      atom_key when is_atom(atom_key) -> Atom.to_string(atom_key) == key
      _ -> false
    end)
  end

  defp string_map_key(container, key) do
    string_key = Atom.to_string(key)
    if Map.has_key?(container, string_key), do: string_key
  end

  defp normalize_segments(nil, path), do: String.split(path, ".")

  defp normalize_segments(segments, _path) when is_list(segments) do
    Enum.map(segments, &normalize_segment/1)
  end

  defp normalize_segments(other, path) do
    raise ArgumentError, "invalid segments #{inspect(other)} for #{inspect(path)}"
  end

  defp normalize_segment(value) when is_integer(value), do: value
  defp normalize_segment(value) when is_binary(value), do: parse_integer(value) || value
  defp normalize_segment(value), do: value

  defp parse_integer(value) when is_binary(value) do
    case Integer.parse(value) do
      {int, ""} -> int
      _ -> nil
    end
  end

  defp normalize_shape(value) when is_tuple(value), do: value
  defp normalize_shape(value) when is_list(value), do: List.to_tuple(value)
  defp normalize_shape(_), do: nil

  defp normalize_type(nil), do: nil
  defp normalize_type(value) when is_atom(value), do: Atom.to_string(value)
  defp normalize_type(value) when is_binary(value), do: value
  defp normalize_type(value), do: inspect(value)

  defp normalize_identity_value(nil), do: nil

  defp normalize_identity_value(value) when is_list(value),
    do: Enum.map(value, &normalize_identity_value/1)

  defp normalize_identity_value(value) when is_tuple(value), do: Tuple.to_list(value)
  defp normalize_identity_value(value) when is_map(value), do: normalize_identity_value_map(value)
  defp normalize_identity_value(value) when is_atom(value), do: Atom.to_string(value)
  defp normalize_identity_value(value), do: value

  defp normalize_identity_value_map(value) when is_map(value) do
    Map.new(value, fn {key, v} -> {normalize_identity_value(key), normalize_identity_value(v)} end)
  end

  defp normalize_string_keys(value) when is_map(value) do
    Map.new(value, fn {key, item} ->
      {normalize_string_key(key), normalize_string_keys(item)}
    end)
  end

  defp normalize_string_keys(value) when is_list(value) do
    Enum.map(value, &normalize_string_keys/1)
  end

  defp normalize_string_keys(value), do: value

  defp normalize_string_key(nil), do: nil
  defp normalize_string_key(key) when is_atom(key), do: Atom.to_string(key)
  defp normalize_string_key(key), do: to_string(key)

  defp field(map, key) when is_map(map) do
    string_key = normalize_string_key(key)

    case Map.fetch(map, string_key) do
      {:ok, value} ->
        value

      :error ->
        Map.get(map, key)
    end
  end

  defp field(map, key, default) when is_map(map) do
    string_key = normalize_string_key(key)

    case Map.fetch(map, string_key) do
      {:ok, value} ->
        value

      :error ->
        Map.get(map, key, default)
    end
  end
end
