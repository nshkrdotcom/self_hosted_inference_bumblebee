%{
  deps: %{
    self_hosted_inference_core: %{
      path: "../self_hosted_inference_core",
      github: %{repo: "nshkrdotcom/self_hosted_inference_core", branch: "main"},
      hex: "~> 0.1.0",
      default_order: [:path, :github, :hex],
      publish_order: [:hex]
    },
    execution_plane: %{
      path: "../execution_plane/core/execution_plane",
      github: %{
        repo: "nshkrdotcom/execution_plane",
        branch: "main",
        subdir: "core/execution_plane"
      },
      hex: "~> 0.1.0",
      default_order: [:path, :github, :hex],
      publish_order: [:hex]
    },
    execution_plane_process: %{
      path: "../execution_plane/runtimes/execution_plane_process",
      github: %{
        repo: "nshkrdotcom/execution_plane",
        branch: "main",
        subdir: "runtimes/execution_plane_process"
      },
      hex: "~> 0.1.0",
      default_order: [:path, :github, :hex],
      publish_order: [:hex]
    },
    crucible_safetensors: %{
      path: "../../North-Shore-AI/crucible_safetensors",
      github: %{repo: "North-Shore-AI/crucible_safetensors", branch: "main"},
      hex: "~> 0.1.0",
      default_order: [:path, :github, :hex],
      publish_order: [:hex]
    },
    crucible_factorization: %{
      path: "../../North-Shore-AI/crucible_factorization",
      github: %{repo: "North-Shore-AI/crucible_factorization", branch: "main"},
      hex: "~> 0.1.0",
      default_order: [:path, :github, :hex],
      publish_order: [:hex]
    },
    crucible_tensor_patch: %{
      path: "../../North-Shore-AI/crucible_tensor_patch",
      github: %{repo: "North-Shore-AI/crucible_tensor_patch", branch: "main"},
      hex: "~> 0.1.0",
      default_order: [:path, :github, :hex],
      publish_order: [:hex]
    },
    crucible_model_registry: %{
      path: "../../North-Shore-AI/crucible_model_registry",
      github: %{repo: "North-Shore-AI/crucible_model_registry", branch: "main"},
      hex: "~> 0.3.1",
      default_order: [:path, :github, :hex],
      publish_order: [:hex]
    }
  }
}
