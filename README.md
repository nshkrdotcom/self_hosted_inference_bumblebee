# SelfHostedInferenceBumblebee

Bumblebee/Nx runtime backend for `self_hosted_inference_core`.

This package owns the concrete model runtime for self-hosted Bumblebee
adapters. For TRINITY it loads the adapted Qwen/Sakana artifact, keeps
hidden-state extraction and route-head execution inside the backend, and
returns typed `SelfHostedInferenceCore.RouteLogits` results.

## Gates

Default CI:

```sh
mix ci
```

Opt-in CUDA artifact parity:

```sh
XLA_TARGET=cuda12 TRINITY_ARTIFACT_DIR=/path/to/adapted_qwen3_0_6b_layer26 mix test --only qwen_sakana_adapted --timeout 300000
```

## Installation

Until published to Hex, depend on the GitHub repo:

```elixir
def deps do
  [
    {:self_hosted_inference_bumblebee,
     github: "nshkrdotcom/self_hosted_inference_bumblebee"}
  ]
end
```
