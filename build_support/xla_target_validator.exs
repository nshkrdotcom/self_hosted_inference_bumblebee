defmodule XlaTargetValidator do
  @moduledoc """
  Validates the `XLA_TARGET` OS environment variable against the values
  accepted by the bundled `xla` dependency.

  Lives under `build_support/` (not `lib/`) so it can be `Code.require_file`d
  from `mix.exs` at top level. That makes the same validator usable from:

    * `mix.exs` (eager top-level validation, catches `mix test`,
      `mix deps.compile`, `mix deps.update`, etc., before EXLA tries to
      compile and before the project's own `:compilers` get a chance to
      run);
    * `Mix.Tasks.Compile.XlaEnvPreflight` (the project's preflight
      compiler, surfaced as a normal `mix compile` step);
    * `Mix.Tasks.Trinity.Env.Check` (operator-invoked
      `mix trinity.env.check`).

  The recognised target list is intentionally kept in lock-step with the
  bundled `xla` version. As of `xla 0.10.x` (used by EXLA 0.12+), the
  supported set is `cpu`, `cuda`, `cuda12`, `cuda13`, `rocm`, `tpu`.
  `cuda12` remains the recommended default for CUDA hosts; `cuda13` is
  newly accepted (the previous bundled `xla 0.9.x` rejected it).
  """

  @supported_xla_targets ["cpu", "cuda", "cuda12", "cuda13", "rocm", "tpu"]
  @recommended "cuda12"

  @doc "Validates `XLA_TARGET`. Returns `:ok` or raises a `Mix.Error`."
  @spec validate!() :: :ok
  def validate! do
    case raw_xla_target() do
      nil -> :ok
      "" -> :ok
      value when value in @supported_xla_targets -> :ok
      value when is_binary(value) -> raise_invalid!(value)
    end
  end

  @doc """
  Validates unless `repo_root` is itself inside a Mix-managed `deps/` tree.

  This lets `trinity_coordinator` enforce its own bundled-XLA target list when
  it is the root project, while avoiding surprising parent projects that consume
  it as a transitive dependency and may manage XLA differently.
  """
  @spec validate_root_project!(Path.t()) :: :ok
  def validate_root_project!(repo_root) when is_binary(repo_root) do
    if repo_under_mix_deps?(repo_root) do
      :ok
    else
      validate!()
    end
  end

  @doc "Returns true when `repo_root` has a path component named `deps`."
  @spec repo_under_mix_deps?(Path.t()) :: boolean()
  def repo_under_mix_deps?(repo_root) when is_binary(repo_root) do
    repo_root
    |> Path.expand()
    |> Path.split()
    |> Enum.member?("deps")
  end

  @doc "Returns the list of XLA_TARGET values accepted by the bundled xla."
  @spec supported_xla_targets() :: [String.t()]
  def supported_xla_targets, do: @supported_xla_targets

  @doc "Returns the recommended XLA_TARGET for CUDA-capable hosts."
  @spec recommended_xla_target() :: String.t()
  def recommended_xla_target, do: @recommended

  @doc "Reads `XLA_TARGET` from the OS environment. Exposed for testing."
  @spec raw_xla_target() :: String.t() | nil
  def raw_xla_target, do: System.get_env("XLA_TARGET")

  defp raise_invalid!(value) do
    accepted = Enum.map_join(@supported_xla_targets, ", ", &inspect/1)

    Mix.raise(
      "XLA_TARGET=#{inspect(value)} is not accepted by the bundled xla 0.10.x. " <>
        "Accepted values: #{accepted}. " <>
        "Recommended for CUDA hosts: export XLA_TARGET=#{@recommended}. " <>
        "Recommended for CPU hosts: unset XLA_TARGET (or use cpu). " <>
        "The bundled xla rejects unrecognised targets at compile time, so EXLA " <>
        "cannot compile until XLA_TARGET is corrected. See " <>
        "guides/troubleshooting.md (\"XLA_TARGET Rejected At Compile Time\") " <>
        "for the canonical recipe."
    )
  end
end
