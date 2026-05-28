defmodule SelfHostedInferenceBumblebee.Backend do
  @moduledoc """
  `SelfHostedInferenceCore.Backend` implementation for Bumblebee adapters.
  """

  alias SelfHostedInferenceBumblebee.Adapter
  alias SelfHostedInferenceCore.{Backend, BackendManifest, InstanceSpec}
  alias SelfHostedInferenceCore.Backend.StartupPlan

  @behaviour Backend

  @impl Backend
  def backend_id, do: :bumblebee

  @impl Backend
  def manifest do
    BackendManifest.new!(
      backend: backend_id(),
      runtime_kind: :service,
      management_modes: [:externally_managed],
      startup_kind: :attach_existing_service,
      protocols: [:route_logits],
      capabilities: %{
        route_logits?: true,
        qwen_sakana_adapted?: true,
        hidden_state_public?: false,
        crucible_runtime_provider?: true
      },
      supported_surfaces: [:local_subprocess, :lower_simulation],
      resource_profile: %{gpu_recommended?: true, canonical_runtime_profile: :cuda_exla},
      metadata: %{
        adapter_refs: [Adapter.qwen_sakana().adapter_ref, Adapter.mock_tiny().adapter_ref],
        optional_backends: [:emlx, :emily]
      }
    )
  end

  @impl Backend
  def startup_plan(%InstanceSpec{} = spec) do
    adapter_ref = InstanceSpec.adapter_ref(spec) || Adapter.qwen_sakana().adapter_ref
    model_identity = Map.get(spec.backend_options, :model_identity, "qwen3-0.6b-sakana")

    {:ok,
     %StartupPlan{
       backend: backend_id(),
       instance_key:
         "bumblebee:#{inspect(SelfHostedInferenceCore.AdapterRef.key(adapter_ref))}:#{model_identity}",
       startup_kind: :attach_existing_service,
       management_mode: :externally_managed,
       transport: nil,
       health_interval_ms: nil,
       endpoint_template: %{
         protocol: :route_logits,
         base_url: "memory://self-hosted-inference-bumblebee",
         provider_identity: :bumblebee,
         model_identity: model_identity,
         source_runtime: __MODULE__,
         source_runtime_ref:
           "adapter:#{inspect(SelfHostedInferenceCore.AdapterRef.key(adapter_ref))}",
         capabilities: %{route_logits?: true},
         metadata: %{
           adapter_ref: adapter_ref,
           crucible_provider: SelfHostedInferenceBumblebee.CrucibleProvider
         }
       },
       backend_state: %{
         adapter_ref: adapter_ref,
         model_identity: model_identity,
         crucible_provider: SelfHostedInferenceBumblebee.CrucibleProvider
       },
       metadata: %{
         adapter_ref: adapter_ref,
         crucible_provider: SelfHostedInferenceBumblebee.CrucibleProvider
       }
     }}
  end

  @impl Backend
  def probe_readiness(state) do
    {:ready, %{base_url: "memory://self-hosted-inference-bumblebee"}, state}
  end

  @impl Backend
  def health_check(state), do: {:ok, :healthy, %{}, state}

  @impl Backend
  def handle_transport_event(_event, state), do: {:pending, state}

  @impl Backend
  def shutdown(_state, _transport_pid), do: :ok
end
