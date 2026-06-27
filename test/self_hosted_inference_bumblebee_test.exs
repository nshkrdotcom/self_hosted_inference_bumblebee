defmodule SelfHostedInferenceBumblebeeTest do
  use ExUnit.Case, async: false

  alias SelfHostedInferenceBumblebee.{Adapter, Backend}
  alias SelfHostedInferenceBumblebee.Runtime.Profile
  alias SelfHostedInferenceCore.{AdapterRef, InstanceSpec, RouteLogits, RuntimeRegistry}

  setup do
    _ = SelfHostedInferenceCore.unregister_backend(:bumblebee)
    :ok = SelfHostedInferenceCore.register_backend(Backend)

    on_exit(fn ->
      _ = SelfHostedInferenceCore.stop_all_instances()
      _ = SelfHostedInferenceCore.unregister_backend(:bumblebee)
    end)

    :ok
  end

  test "backend manifest exposes adapter refs" do
    manifest = Backend.manifest()

    assert manifest.backend == :bumblebee
    assert manifest.startup_kind == :attach_existing_service
    assert manifest.capabilities.route_logits? == true
    assert manifest.capabilities.crucible_runtime_provider? == true
    assert Enum.any?(manifest.metadata.adapter_refs, &(&1.id == :trinity_qwen3_0_6b_sakana))
    assert Enum.any?(manifest.metadata.adapter_refs, &(&1.id == :mock_tiny))
  end

  test "instance spec with adapter_ref starts backend under self_hosted_inference_core" do
    adapter_ref = Adapter.mock_tiny().adapter_ref

    spec =
      InstanceSpec.new!(
        backend: :bumblebee,
        adapter_ref: adapter_ref,
        backend_options: %{model_identity: "mock"}
      )

    assert {:ok, %{instance: snapshot, reused?: false}} =
             SelfHostedInferenceCore.ensure_instance(spec)

    assert snapshot.adapter_ref == adapter_ref

    assert RuntimeRegistry.whereis({:bumblebee, adapter_ref}) ==
             RuntimeRegistry.whereis(snapshot.instance_id)
  end

  test "mock adapter returns deterministic route logits without tensor fields" do
    adapter = Adapter.mock_tiny()
    messages = [%{role: "user", content: "Route this deterministic prompt."}]

    assert {:ok, %RouteLogits{} = first} = SelfHostedInferenceBumblebee.route(adapter, messages)
    assert {:ok, %RouteLogits{} = second} = SelfHostedInferenceBumblebee.route(adapter, messages)

    assert first == second

    assert first.transcript_hash ==
             "6f7d00cd47d137afbaf4bf479a305cd2e11b5ceb4138230648eb21a57c9e5106"

    assert is_integer(first.selected_agent_id)
    assert is_integer(first.selected_role_id)
    assert is_list(first.agent_logits)
    assert is_list(first.role_logits)
    refute contains_tensor?(first)
  end

  test "hidden-state extraction is not a public tensor boundary" do
    public_functions =
      SelfHostedInferenceBumblebee.Extractor.__info__(:functions)
      |> Keyword.keys()
      |> Enum.map(&Atom.to_string/1)

    refute Enum.any?(public_functions, &String.contains?(&1, "hidden"))
    refute Enum.any?(public_functions, &String.contains?(&1, "vector"))
  end

  test "route hash inputs use the runtime-stable decision and rounded-logit payload" do
    adapter = Adapter.mock_tiny()
    messages = [%{role: "user", content: "What is 17 + 25? Answer briefly."}]

    assert {:ok, %RouteLogits{} = logits} = SelfHostedInferenceBumblebee.route(adapter, messages)

    assert logits.route_hash_inputs["schema"] ==
             "self_hosted_inference_bumblebee.route_hash_inputs.v1"

    assert logits.route_hash_inputs["agent_id"] == logits.selected_agent_id
    assert logits.route_hash_inputs["role_id"] == logits.selected_role_id
    assert length(logits.route_hash_inputs["logits_rounded"]) == 10
  end

  test "runtime profile resolution table matches the coordinator backend lanes" do
    assert Profile.resolve(:cuda_exla).nx_backend == {EXLA.Backend, client: :cuda}
    assert Profile.resolve(:host_exla).nx_backend == {EXLA.Backend, client: :host}
    assert Profile.resolve(:binary).nx_backend == Nx.BinaryBackend
    assert Profile.resolve(:mock_tiny).nx_backend == Nx.BinaryBackend
  end

  test "optional Apple backends are absent-safe" do
    for profile <- [:emlx, :emily] do
      resolved = Profile.resolve(profile)
      assert resolved.name == profile
      assert_raise RuntimeError, fn -> Profile.put_default_backend!(resolved) end
    end
  end

  test "XLA target preflight rejects unsupported CUDA targets" do
    original = System.get_env("XLA_TARGET")

    try do
      System.put_env("XLA_TARGET", "cuda14")
      assert_raise Mix.Error, fn -> XlaTargetValidator.validate!() end
    after
      if original,
        do: System.put_env("XLA_TARGET", original),
        else: System.delete_env("XLA_TARGET")
    end
  end

  test "adapter refs are typed, stable core contracts" do
    adapter_ref =
      AdapterRef.new!(
        id: :trinity_qwen3_0_6b_sakana,
        version: "0.1.0",
        contract: :route_logits_v1
      )

    adapter = Adapter.qwen_sakana(adapter_ref: adapter_ref, runtime_profile: :mock_tiny)

    assert adapter.adapter_ref == adapter_ref

    assert AdapterRef.key(adapter.adapter_ref) ==
             {:trinity_qwen3_0_6b_sakana, "0.1.0", :route_logits_v1}
  end

  test "startup plan exposes the Crucible provider module" do
    adapter_ref = Adapter.mock_tiny().adapter_ref

    spec =
      InstanceSpec.new!(
        backend: :bumblebee,
        adapter_ref: adapter_ref,
        backend_options: %{model_identity: "mock"}
      )

    assert {:ok, plan} = Backend.startup_plan(spec)
    assert plan.backend_state.crucible_provider == SelfHostedInferenceBumblebee.CrucibleProvider

    assert plan.backend_state.formal_crucible_provider ==
             SelfHostedInferenceBumblebee.FormalCrucibleProvider

    assert plan.metadata.crucible_provider == SelfHostedInferenceBumblebee.CrucibleProvider

    assert plan.metadata.formal_crucible_provider ==
             SelfHostedInferenceBumblebee.FormalCrucibleProvider
  end

  defp contains_tensor?(%Nx.Tensor{}), do: true

  defp contains_tensor?(%_{} = value) do
    value
    |> Map.from_struct()
    |> Map.values()
    |> Enum.any?(&contains_tensor?/1)
  end

  defp contains_tensor?(value) when is_map(value) do
    value
    |> Map.values()
    |> Enum.any?(&contains_tensor?/1)
  end

  defp contains_tensor?(value) when is_list(value), do: Enum.any?(value, &contains_tensor?/1)
  defp contains_tensor?(_value), do: false
end
