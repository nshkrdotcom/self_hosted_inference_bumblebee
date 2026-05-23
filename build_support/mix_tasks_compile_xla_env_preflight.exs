defmodule Mix.Tasks.Compile.XlaEnvPreflight do
  @shortdoc "Validates XLA_TARGET against the bundled xla before project compilation"
  @moduledoc """
  Mix compiler that validates the `XLA_TARGET` OS environment variable
  against the targets accepted by the bundled `xla` dependency before
  the project's own source files compile.

  ## Why this exists

  Without this preflight, an unrecognised `XLA_TARGET` (for example
  `cuda14` against the bundled `xla 0.10.x`) surfaces as a `RuntimeError`
  stacktrace from `deps/exla/mix.exs` during dependency compilation:

      ** (RuntimeError) expected XLA_TARGET to be one of
         "cpu", "cuda", "rocm", "tpu", "cuda12", "cuda13", but got: "cuda14"
          (xla 0.10.0) lib/xla.ex:82: XLA.xla_target/0
          ...

  With this preflight, the project surfaces the failure with a single
  readable line and a concrete remediation (`export XLA_TARGET=cuda12`).

  ## How this fits

  This compiler runs as part of the *project's* compile step, not during
  dependency compilation. Mix compiles dependencies before invoking the
  project's `:compilers` list, so this compiler alone would not catch
  the dependency-compile failure mode. To close that gap, the same
  validation is also invoked eagerly from `mix.exs` at top level, so
  `mix test`, `mix deps.compile`, `mix deps.update`, and other tasks
  that evaluate `mix.exs` before touching deps all benefit.

  The compiler module exists in addition to (not instead of) the
  top-level eager check, because it also produces an explicit
  `==> xla_env_preflight` step in normal `mix compile` output, which
  is the conventional place an Elixir developer looks for build-graph
  preflight checks.

  ## Behaviour

  The compiler delegates entirely to `XlaTargetValidator.validate!/0`.
  When the validator returns `:ok`, the compiler returns `{:noop, []}`
  (no source files were produced, no diagnostics emitted). When the
  validator raises, the raise propagates and Mix surfaces it as a
  build error.

  This compiler is intentionally a no-op in terms of source artefacts.
  It exists for its side-effect (the validation) and its diagnostic
  surface only.
  """

  use Mix.Task.Compiler

  @repo_root Path.expand("..", __DIR__)

  @impl Mix.Task.Compiler
  def run(_argv) do
    XlaTargetValidator.validate_root_project!(@repo_root)
    {:noop, []}
  end
end
