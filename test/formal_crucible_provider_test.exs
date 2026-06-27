defmodule SelfHostedInferenceBumblebee.FormalCrucibleProviderTest do
  use ExUnit.Case, async: true

  alias CrucibleBumblebee.{ModelBundle, ModelSurface, Preflight}
  alias CrucibleTap.TapPlan
  alias SelfHostedInferenceBumblebee.{CrucibleProvider, FormalCrucibleProvider}

  test "declares the formal provider behaviour" do
    behaviours = FormalCrucibleProvider.__info__(:attributes)[:behaviour] || []

    assert Crucible.Provider in behaviours
  end

  test "exposes formal provider metadata without loading a model" do
    state = fixture_state()

    assert FormalCrucibleProvider.provider_kind(state) == :model
    assert FormalCrucibleProvider.model_ref(state) == "model:bumblebee-fixture"
    assert FormalCrucibleProvider.backend(state) == :binary
    assert FormalCrucibleProvider.ready?(state)
    assert :ok = FormalCrucibleProvider.shutdown(state, :normal)

    assert %Crucible.Provider.ProviderHealth{} = FormalCrucibleProvider.health(state)
  end

  test "returns the canonical tap surface and capability report" do
    state = fixture_state()

    assert {:ok, %CrucibleTap.Surface{} = surface} =
             FormalCrucibleProvider.surface(state, "model:bumblebee-fixture", [])

    assert surface.metadata.surface_id == :bumblebee_fixture_surface

    assert {:ok, %Crucible.CapabilityReport{} = report} =
             FormalCrucibleProvider.capabilities(state)

    assert report.model_id == "model:bumblebee-fixture"
    assert report.required_missing == []
  end

  test "compiles supported tap plans and fails closed for unsupported required taps" do
    state = fixture_state()
    assert {:ok, surface} = FormalCrucibleProvider.surface(state, "model:bumblebee-fixture", [])

    supported_plan =
      TapPlan.new!(
        [[id: "logits", signal_type: :final_logits, layers: [:final], required?: true]],
        plan_id: "formal-bumblebee-supported"
      )

    assert {:ok, compiled} =
             FormalCrucibleProvider.compile(state, supported_plan, surface, [])

    assert compiled.plan_id == "formal-bumblebee-supported"
    assert [%{tap_id: "logits"}] = compiled.matched

    unsupported_plan =
      TapPlan.new!(
        [[id: "hidden", signal_type: :hidden_state, layers: [0], required?: true]],
        plan_id: "formal-bumblebee-unsupported"
      )

    assert {:error, {:tap_compile_failed, report}} =
             FormalCrucibleProvider.compile(state, unsupported_plan, surface, [])

    assert report.required_missing == ["hidden"]
  end

  defp fixture_state do
    surface =
      ModelSurface.new!(
        :gpt2,
        [
          [
            id: "final_logits",
            signal_type: :final_logits,
            layer_name: "final_logits",
            layer_index: :final,
            operations: [:read, :route_on],
            capture_modes: [:summary]
          ]
        ],
        %{
          surface_id: :bumblebee_fixture_surface,
          capabilities: %{final_logits: true, hidden_state: false}
        }
      )

    %CrucibleProvider{
      bundle: %ModelBundle{
        model_id: "model:bumblebee-fixture",
        tokenizer_id: "model:bumblebee-fixture",
        backend: :binary,
        model_family: :gpt2,
        revision: "test"
      },
      loader_options: nil,
      tap_plan: nil,
      capability_report:
        Crucible.CapabilityReport.new(
          provider_kind: :elixir_bumblebee,
          model_id: "model:bumblebee-fixture",
          model_family: :gpt2,
          backend: :binary,
          supported: ["logits"],
          resource_budget: Preflight.resource_budget()
        ),
      surface: surface
    }
  end
end
