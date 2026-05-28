defmodule SelfHostedInferenceBumblebee.CrucibleProviderLiveTest do
  use ExUnit.Case, async: false

  alias SelfHostedInferenceBumblebee.CrucibleProvider
  alias SelfHostedInferenceCore.CrucibleRuntime

  @moduletag :live_cpu_heavy
  @moduletag timeout: 300_000

  @models [
    "hf-internal-testing/tiny-random-gpt2",
    "gpt2"
  ]

  setup do
    configured_root = System.get_env("CRUCIBLE_ARTIFACT_ROOT")

    root =
      configured_root ||
        Path.join(
          System.tmp_dir!(),
          "self_hosted_inference_bumblebee_live_#{System.unique_integer([:positive])}"
        )

    on_exit(fn ->
      stop_crucible_runtimes()

      if is_nil(configured_root) do
        File.rm_rf!(root)
      end
    end)

    {:ok, root: root}
  end

  for model_id <- @models do
    @tag model_id: model_id
    test "loads #{model_id}, runs hosted forward, and captures generation logits", %{
      root: root,
      model_id: model_id
    } do
      id = :"crucible-provider-live-#{System.unique_integer([:positive])}"

      assert {:ok, pid} =
               CrucibleRuntime.start_child(
                 id: id,
                 provider_module: CrucibleProvider,
                 provider_opts: [
                   model_id: model_id,
                   tokenizer_id: model_id,
                   backend: :binary,
                   artifact_root: root,
                   max_new_tokens: 1
                 ]
               )

      assert CrucibleRuntime.ready?(pid)
      assert {:ok, lease} = CrucibleRuntime.lease(pid, owner_ref: "live-test")

      assert {:ok, %Crucible.ForwardTrace{} = trace} =
               CrucibleRuntime.forward(pid, nil, %{prompt: "Hi"},
                 trace_name: "live_#{safe_name(model_id)}",
                 timeout: 240_000
               )

      assert trace.model_id == model_id
      assert trace.final_logits.signal_type == :final_logits
      assert trace.signals != []

      assert {:ok, generation} =
               CrucibleRuntime.generate(pid, nil, "Hi", max_new_tokens: 1, timeout: 240_000)

      assert generation.model_id == model_id
      assert generation.success_level == :generation_step_logits
      assert generation.step_count == 1
      assert File.exists?(generation.trace_path)

      assert :ok = CrucibleRuntime.release(lease)
    end
  end

  defp stop_crucible_runtimes do
    for snapshot <- CrucibleRuntime.list_snapshots() do
      if pid = CrucibleRuntime.whereis(snapshot.id) do
        try do
          DynamicSupervisor.terminate_child(
            SelfHostedInferenceCore.CrucibleRuntimeSupervisor,
            pid
          )
        catch
          :exit, _reason -> :ok
        end
      end
    end
  end

  defp safe_name(name) do
    name
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9_.-]+/, "_")
    |> String.trim("_")
  end
end
