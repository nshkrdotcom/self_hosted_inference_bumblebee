if bootstrap = System.get_env("MIX_WORKSPACE_OPS_BOOTSTRAP"), do: Code.require_file(bootstrap)

unless Code.ensure_loaded?(XlaTargetValidator) do
  Code.require_file("build_support/xla_target_validator.exs", __DIR__)
end

unless Code.ensure_loaded?(Mix.Tasks.Compile.XlaEnvPreflight) do
  Code.require_file("build_support/mix_tasks_compile_xla_env_preflight.exs", __DIR__)
end

XlaTargetValidator.validate_root_project!(__DIR__)

defmodule SelfHostedInferenceBumblebee.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/nshkrdotcom/self_hosted_inference_bumblebee"
  @bumblebee_ref "cbe271afafcacff04d298046f4b11711712b4123"

  def project do
    [
      app: :self_hosted_inference_bumblebee,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      compilers: [:xla_env_preflight] ++ Mix.compilers(),
      deps: deps(),
      aliases: aliases(),
      dialyzer: [plt_add_deps: :apps_direct],
      name: "SelfHostedInferenceBumblebee",
      description: "Bumblebee/Nx runtime backend for self-hosted inference adapters",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs(),
      package: package()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  def cli do
    [
      preferred_envs: [
        ci: :test,
        credo: :test,
        dialyzer: :test,
        docs: :dev
      ]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.12.1", override: true},
      {:exla, "~> 0.12.0", override: true},
      {:axon, "~> 0.8.1"},
      {:bumblebee, github: "North-Shore-AI/bumblebee", ref: @bumblebee_ref, override: true},
      workspace_dep({:self_hosted_inference_core, "~> 0.1.0", override: true}),
      workspace_dep({:execution_plane, "~> 0.1.0", override: true}),
      workspace_dep({:execution_plane_process, "~> 0.1.0", override: true}),
      workspace_dep({:crucible_safetensors, "~> 0.1.0", override: true}),
      workspace_dep({:crucible_factorization, "~> 0.1.0", override: true}),
      workspace_dep({:crucible_tensor_patch, "~> 0.1.0", override: true}),
      workspace_dep({:crucible_model_registry, "~> 0.3.1", override: true}),
      workspace_dep({:crucible_bumblebee, "~> 0.1.0"}),
      workspace_dep({:crucible_provider_contracts, "~> 0.1.0"}),
      workspace_dep({:crucible_signal, "~> 0.1.0"}),
      workspace_dep({:crucible_signal_trace, "~> 0.1.0"}),
      workspace_dep({:crucible_tap, "~> 0.1.0"}),
      {:jason, "~> 1.4.5"},
      {:credo, "~> 1.7.19", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40.3", only: [:dev, :test], runtime: false}
    ]
  end

  defp aliases do
    [
      ci: [
        "deps.get",
        "format --check-formatted",
        "compile --warnings-as-errors",
        "test",
        "credo --strict",
        "dialyzer --format short",
        "docs"
      ]
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md"],
      source_ref: "main",
      source_url: @source_url,
      homepage_url: @source_url
    ]
  end

  defp workspace_dep(committed) do
    if function_exported?(MixWorkspaceOpsBootstrap, :dep, 2),
      do: apply(MixWorkspaceOpsBootstrap, :dep, [committed, __DIR__]),
      else: committed
  end

  defp package do
    [
      name: "self_hosted_inference_bumblebee",
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib mix.exs README.md LICENSE)
    ]
  end
end
