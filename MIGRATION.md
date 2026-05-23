# Migration Notes

This repo was created during TRINITY decomposition Phase 7.

Source material for the initial implementation:

- `nshkrdotcom/trinity_coordinator` tag `v0.1.0-monolith`
- source commit `64144a2983950e5fc9f2db2d26323a576c7379a1`
- `build_support/xla_target_validator.exs`
- `build_support/mix_tasks_compile_xla_env_preflight.exs`
- `lib/trinity_coordinator/extractor.ex`
- runtime portions of `lib/trinity_coordinator/coordination_head.ex`
- runtime portions of `lib/trinity_coordinator/sakana/coordinator.ex`
- runtime portions of `lib/trinity_coordinator/sakana/head.ex`
- runtime/patching portions of `lib/trinity_coordinator/sakana/artifact.ex`
- `lib/trinity_coordinator/runtime.ex`
- `lib/trinity_coordinator/runtime_profile.ex`
- runtime portions of `lib/trinity_coordinator/slm_profile.ex`

The destination package owns hidden-state extraction and route-head execution as
one backend operation. The hidden vector is not exposed as a public bridge
boundary; public routing returns `SelfHostedInferenceCore.RouteLogits`.
