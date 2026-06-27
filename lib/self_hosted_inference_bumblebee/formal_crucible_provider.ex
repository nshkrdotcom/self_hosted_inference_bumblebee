defmodule SelfHostedInferenceBumblebee.FormalCrucibleProvider do
  @moduledoc """
  Formal `Crucible.Provider` adapter for the Bumblebee runtime provider.

  `SelfHostedInferenceBumblebee.CrucibleProvider` remains the SHIC-native
  provider. This module exposes the shared provider ABI while delegating model
  loading, forward execution, generation, and trace writing to the SHIC-native
  implementation.
  """

  @behaviour Crucible.Provider

  alias Crucible.CapabilityReport
  alias Crucible.Provider.ProviderHealth
  alias CrucibleBumblebee.Preflight
  alias CrucibleTap.{Surface, TapPlan}
  alias SelfHostedInferenceBumblebee.CrucibleProvider

  @formal_provider_kind :model

  @impl true
  def init(opts), do: CrucibleProvider.init(opts)

  @impl true
  def surface(%CrucibleProvider{} = state, _model_ref, _opts), do: provider_surface(state)

  @impl true
  def capabilities(%CrucibleProvider{} = state), do: {:ok, state.capability_report}

  @impl true
  def compile(%CrucibleProvider{} = state, %TapPlan{} = tap_plan, %Surface{} = surface, opts) do
    case CapabilityReport.negotiate(tap_plan, surface,
           provider_kind: provider_kind(state),
           model_id: model_ref(state),
           model_family: state.bundle.model_family,
           backend: backend(state),
           resource_budget: Keyword.get(opts, :resource_budget, Preflight.resource_budget())
         ) do
      {:ok, compiled_plan, _report} -> {:ok, compiled_plan}
      {:error, reason} -> {:error, reason}
    end
  end

  @impl true
  def forward(%CrucibleProvider{} = state, input, compiled_plan, opts) do
    state
    |> maybe_put_capability_report(opts)
    |> CrucibleProvider.forward(input, Keyword.put(opts, :compiled_plan, compiled_plan))
  end

  @impl true
  def generate(%CrucibleProvider{} = state, input, compiled_plan, opts) do
    state
    |> maybe_put_capability_report(opts)
    |> CrucibleProvider.generate(input, Keyword.put(opts, :compiled_plan, compiled_plan))
  end

  @impl true
  def ready?(%CrucibleProvider{} = state), do: CrucibleProvider.ready?(state)

  @impl true
  def health(%CrucibleProvider{} = state) do
    ProviderHealth.new!(
      status: :ok,
      uptime_seconds: 0,
      last_latency_ms: nil,
      error_count: 0,
      memory_bytes: nil,
      details: %{
        provider: :bumblebee,
        model_id: model_ref(state),
        model_family: state.bundle.model_family,
        backend: backend(state)
      }
    )
  end

  @impl true
  def provider_kind(%CrucibleProvider{}), do: @formal_provider_kind

  @impl true
  def model_ref(%CrucibleProvider{} = state), do: CrucibleProvider.model_id(state)

  @impl true
  def backend(%CrucibleProvider{} = state), do: CrucibleProvider.backend(state)

  @impl true
  def shutdown(%CrucibleProvider{}, _reason), do: :ok

  defp provider_surface(%CrucibleProvider{surface: %Surface{} = surface}), do: {:ok, surface}

  defp provider_surface(%CrucibleProvider{surface: %{surface: %Surface{} = surface}}),
    do: {:ok, surface}

  defp provider_surface(%CrucibleProvider{surface: surface}),
    do: {:error, {:invalid_provider_surface, surface}}

  defp maybe_put_capability_report(%CrucibleProvider{} = state, opts) do
    case Keyword.get(opts, :capability_report) do
      %CapabilityReport{} = report -> %{state | capability_report: report}
      _other -> state
    end
  end
end
