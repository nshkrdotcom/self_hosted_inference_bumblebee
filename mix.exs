unless Code.ensure_loaded?(DependencySources) do
  Code.require_file("build_support/dependency_sources.exs", __DIR__)
end

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
  @bumblebee_ref "d0774e8ab8c4d5ac60ade95ec8dc9e1f0efd7306"

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
      {:nx, "~> 0.12", override: true},
      {:exla, "~> 0.12", override: true},
      {:axon, "~> 0.7"},
      {:bumblebee, github: "elixir-nx/bumblebee", ref: @bumblebee_ref, override: true},
      DependencySources.dep(:self_hosted_inference_core, __DIR__, override: true),
      DependencySources.dep(:execution_plane, __DIR__, override: true),
      DependencySources.dep(:execution_plane_process, __DIR__, override: true),
      DependencySources.dep(:crucible_safetensors, __DIR__, override: true),
      DependencySources.dep(:crucible_factorization, __DIR__, override: true),
      DependencySources.dep(:crucible_tensor_patch, __DIR__, override: true),
      DependencySources.dep(:crucible_model_registry, __DIR__, override: true),
      {:jason, "~> 1.4"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40.1", only: [:dev, :test], runtime: false}
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

  defp package do
    [
      name: "self_hosted_inference_bumblebee",
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib build_support mix.exs README.md LICENSE)
    ]
  end
end
