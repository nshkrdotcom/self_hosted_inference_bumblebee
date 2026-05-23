defmodule SelfHostedInferenceBumblebee.NoLibOsEnvTest do
  use ExUnit.Case, async: true

  @repo_root Path.expand("..", __DIR__)

  test "lib does not read OS environment directly" do
    violations =
      @repo_root
      |> Path.join("lib/**/*.ex")
      |> Path.wildcard()
      |> Enum.filter(&(File.read!(&1) =~ "System.get_env"))

    assert violations == []
  end
end
