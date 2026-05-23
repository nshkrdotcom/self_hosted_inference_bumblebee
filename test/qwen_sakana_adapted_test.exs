defmodule SelfHostedInferenceBumblebee.QwenSakanaAdaptedTest do
  use ExUnit.Case, async: false

  @moduletag :qwen_sakana_adapted
  @moduletag timeout: 300_000

  alias SelfHostedInferenceBumblebee.Adapter
  alias SelfHostedInferenceCore.RouteLogits

  @cases_path "/home/home/p/g/n/trinity_coordinator/examples/fixtures/qwen_router_prompt_eval_cases.json"
  @snapshot_path "/home/home/p/g/n/trinity_coordinator/examples/fixtures/qwen_router_prompt_eval_logits.json"

  test "routes all coordinator prompt-eval fixtures through the adapted Qwen/Sakana artifact" do
    artifact_dir =
      System.get_env("TRINITY_ARTIFACT_DIR") ||
        "/home/home/p/g/n/trinity_coordinator/priv/sakana_trinity/adapted_qwen3_0_6b_layer26"

    adapter = Adapter.qwen_sakana(artifact_dir: artifact_dir, runtime_profile: :cuda_exla)
    assert {:ok, loaded} = SelfHostedInferenceBumblebee.load(adapter)

    cases = Jason.decode!(File.read!(@cases_path))["cases"]
    snapshots = @snapshot_path |> File.read!() |> Jason.decode!() |> Map.fetch!("cases")
    snapshots_by_id = Map.new(snapshots, &{&1["id"], &1})

    results =
      Enum.map(cases, fn case_spec ->
        assert {:ok, %RouteLogits{} = logits} =
                 SelfHostedInferenceBumblebee.route(loaded, case_spec["messages"])

        expected = case_spec["expected"]
        snapshot = Map.fetch!(snapshots_by_id, case_spec["id"])

        assert logits.selected_agent_id == expected["agent_id"]
        assert logits.selected_role_id == expected["role_id"]
        assert logits.selected_agent_id == snapshot["agent_id"]
        assert logits.selected_role_id == snapshot["role_id"]
        assert logits.token_count == snapshot["token_count"]
        assert logits.transcript_hash == snapshot["transcript_hash"]

        case_spec["id"]
      end)

    assert length(results) == 37
  end
end
