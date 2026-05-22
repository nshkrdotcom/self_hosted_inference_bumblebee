defmodule SelfHostedInferenceBumblebeeTest do
  use ExUnit.Case
  doctest SelfHostedInferenceBumblebee

  test "greets the world" do
    assert SelfHostedInferenceBumblebee.hello() == :world
  end
end
