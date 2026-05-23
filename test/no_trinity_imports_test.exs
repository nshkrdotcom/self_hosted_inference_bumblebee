defmodule SelfHostedInferenceBumblebee.NoTrinityImportsTest do
  use ExUnit.Case, async: true

  @repo_root Path.expand("..", __DIR__)
  @forbidden ["Trinity.", "TrinityCoordinator.", "TrinityFramework."]

  test "lib has no TRINITY framework or coordinator imports" do
    violations =
      @repo_root
      |> Path.join("lib/**/*.ex")
      |> Path.wildcard()
      |> Enum.flat_map(fn path ->
        contents = File.read!(path)

        @forbidden
        |> Enum.filter(&String.contains?(contents, &1))
        |> Enum.map(&"#{Path.relative_to(path, @repo_root)} contains #{&1}")
      end)

    assert violations == []
  end
end
