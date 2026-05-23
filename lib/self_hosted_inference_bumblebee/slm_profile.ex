defmodule SelfHostedInferenceBumblebee.SLMProfile do
  @moduledoc """
  Profile descriptors for coordinator SLM backends.

  Profiles define model loading metadata without coupling the router to concrete
  repository names throughout application code.
  """
  alias SelfHostedInferenceBumblebee.AdaptedQwenPatch

  @type profile_status :: :ready | :pending | :unsupported

  @type profile :: %{
          required(:name) => atom(),
          required(:repo) => term(),
          required(:module) => module() | nil,
          required(:architecture) => atom(),
          required(:expected_hidden_size) => pos_integer() | nil,
          required(:xla_target) => String.t(),
          required(:status) => profile_status,
          optional(:load_options) => keyword(),
          optional(:notes) => String.t()
        }

  @type compatibility_report :: %{
          required(:name) => atom(),
          required(:repo) => term(),
          required(:module) => module() | nil,
          required(:architecture) => atom(),
          required(:supported_text_modules) => [atom()],
          required(:status) => :compatible | {:incompatible, term()},
          required(:xla_target) => String.t()
        }

  @doc "Small fast verification profile used by test and local smoke checks."
  def tiny_gpt2 do
    %{
      name: :tiny_gpt2,
      repo: {:hf, "hf-internal-testing/tiny-random-gpt2"},
      module: Bumblebee.Text.Gpt2,
      architecture: :base,
      expected_hidden_size: 32,
      xla_target: "cuda12",
      status: :ready,
      notes: "Fast local profile for deterministic verification"
    }
  end

  @doc "Production intent profile for a Qwen-class coordinator model."
  def qwen_coordinator do
    %{
      name: :qwen_coordinator,
      repo: {:hf, "Qwen/Qwen3-0.6B"},
      module: Bumblebee.Text.Qwen3,
      architecture: :for_causal_language_modeling,
      expected_hidden_size: 1024,
      xla_target: "cuda12",
      status: :ready,
      load_options: [
        type: :bf16
      ],
      notes:
        "Qwen3-0.6B causal-LM coordinator profile for hidden-state extraction and Sakana SVF tensor selection"
    }
  end

  @doc """
  Production artifact-driven Sakana-adapted Qwen coordinator profile.

  This profile patches the Qwen backbone tensors only.  The routing head is a
  separate Axon model and must be initialized from the artifact router-head
  tensor via `SelfHostedInferenceBumblebee.HeadLoader` or
  `SelfHostedInferenceBumblebee.QwenSakanaLoader`.

  Keeping the SLM params and routing-head params separate avoids trying to add
  `routing_head` into Bumblebee's causal-LM param tree.
  """
  def qwen_sakana_adapted do
    %{
      name: :qwen_sakana_adapted,
      repo: {:hf, "Qwen/Qwen3-0.6B"},
      module: Bumblebee.Text.Qwen3,
      architecture: :for_causal_language_modeling,
      expected_hidden_size: 1024,
      xla_target: "cuda12",
      status: :ready,
      adapted_artifact_dir: AdaptedQwenPatch.default_output_dir(),
      artifact_patch_options: [
        patch_router_head: false,
        allow_incomplete: false,
        cast_tensors: true
      ],
      load_options: [
        type: :bf16
      ],
      notes:
        "Qwen3-0.6B causal-LM coordinator profile with runtime Sakana SVF backbone artifacts; routing head is loaded separately"
    }
  end

  @doc "Returns a profile by known name."
  def profile(:tiny_gpt2), do: {:ok, tiny_gpt2()}
  def profile(:qwen_coordinator), do: {:ok, qwen_coordinator()}
  def profile(:qwen_sakana_adapted), do: {:ok, qwen_sakana_adapted()}
  def profile(other), do: {:error, {:unknown_profile, other}}

  @doc "Load by known profile name or full profile map after compatibility validation."
  def load_profile(name) when is_atom(name) do
    with {:ok, profile} <- resolve_profile(name) do
      load_profile(profile)
    end
  end

  def load_profile(profile) when is_map(profile) do
    with {:ok, ready_profile} <- ensure_ready_profile(profile),
         {:ok, {model_info, tokenizer}} <-
           SelfHostedInferenceBumblebee.Extractor.load_slm_model(
             ready_profile.repo,
             ready_profile.module,
             ready_profile.architecture,
             Map.get(ready_profile, :load_options, [])
           ),
         {:ok, model_info} <- apply_profile_artifact_patch(ready_profile, model_info) do
      {:ok, {model_info, tokenizer}}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  def load_profile(_), do: {:error, :invalid_profile}

  @doc """
  Runs compatibility checks without attempting a full model fetch.

  The probe returns structured status so callers can present actionable blockers
  before attempting provider or model-load paths.
  """
  @spec compatibility_probe(atom() | map()) :: {:ok, compatibility_report()} | {:error, term()}
  def compatibility_probe(name_or_profile)

  def compatibility_probe(name) when is_atom(name) do
    case resolve_profile(name) do
      {:ok, profile} -> compatibility_probe(profile)
      {:error, reason} -> {:error, reason}
    end
  end

  def compatibility_probe(profile) when is_map(profile) do
    if validate_profile_signature(profile) == :ok do
      report = %{
        name: profile.name,
        repo: profile.repo,
        module: profile.module,
        architecture: profile.architecture,
        supported_text_modules: supported_text_modules(),
        status: compatibility_status(profile),
        xla_target: profile.xla_target
      }

      {:ok, report}
    else
      {:error, :invalid_profile}
    end
  end

  def compatibility_probe(_), do: {:error, :invalid_profile}

  defp compatibility_status(%{status: :ready} = profile) do
    case probe_module(profile.module) do
      :ok -> :compatible
      {:incompatible, reason} -> {:incompatible, reason}
    end
  end

  defp compatibility_status(%{status: status}),
    do: {:incompatible, {:unsupported_profile_status, status}}

  defp resolve_profile(name) when is_atom(name) do
    case profile(name) do
      {:ok, profile} -> {:ok, profile}
      {:error, reason} -> {:error, reason}
    end
  end

  defp ensure_ready_profile(profile) when is_map(profile) do
    with :ok <- validate_profile_signature(profile),
         {:ok, report} <- compatibility_probe(profile),
         :ok <- validate_status_compatibility(profile.name, report.status) do
      {:ok, profile}
    end
  end

  defp validate_status_compatibility(_name, :compatible), do: :ok

  defp validate_status_compatibility(name, {:incompatible, reason}) do
    {:error, {:unsupported_profile, name, reason}}
  end

  defp validate_profile_signature(profile) do
    with :ok <-
           ensure_keys(profile, [
             :name,
             :repo,
             :module,
             :architecture,
             :status,
             :expected_hidden_size,
             :xla_target
           ]),
         :ok <- ensure_key_types(profile) do
      :ok
    else
      _ ->
        {:error, :invalid_profile}
    end
  end

  defp ensure_keys(profile, keys) do
    missing = Enum.find(keys, fn key -> not Map.has_key?(profile, key) end)

    if missing == nil, do: :ok, else: {:error, {:missing_field, missing}}
  end

  defp ensure_key_types(profile) do
    with :ok <- validate_profile_name(profile.name),
         :ok <- validate_repo(profile.repo),
         :ok <- validate_architecture(profile.architecture),
         :ok <- validate_status(profile.status),
         :ok <- validate_expected_hidden_size(profile.expected_hidden_size),
         :ok <- validate_xla_target(profile.xla_target) do
      :ok
    else
      _ -> {:error, :invalid_profile}
    end
  end

  defp validate_profile_name(name) when is_atom(name), do: :ok
  defp validate_profile_name(_), do: {:error, :invalid_profile}

  defp validate_repo(repo) when is_tuple(repo), do: :ok
  defp validate_repo(_), do: {:error, :invalid_profile}

  defp validate_architecture(arch) when is_atom(arch), do: :ok
  defp validate_architecture(_), do: {:error, :invalid_profile}

  defp validate_status(status) when status in [:ready, :pending, :unsupported], do: :ok
  defp validate_status(_), do: {:error, :invalid_profile}

  defp validate_expected_hidden_size(size) when is_integer(size) and size > 0, do: :ok
  defp validate_expected_hidden_size(_), do: {:error, :invalid_profile}

  defp validate_xla_target("cuda12"), do: :ok
  defp validate_xla_target("cuda13"), do: :ok
  defp validate_xla_target(_), do: {:error, :invalid_profile}

  defp probe_module(nil), do: {:incompatible, :missing_module}

  defp probe_module(module) when is_atom(module) do
    if Code.ensure_loaded?(module) do
      :ok
    else
      {:incompatible, {:missing_bumblebee_module, module}}
    end
  end

  defp probe_module(_), do: {:incompatible, :invalid_module}

  defp apply_profile_artifact_patch(profile, model_info) when is_map(profile) do
    case Map.get(profile, :adapted_artifact_dir) do
      nil ->
        {:ok, model_info}

      "" ->
        {:ok, model_info}

      artifact_dir ->
        opts =
          profile
          |> Map.get(:artifact_patch_options, [])
          |> Keyword.put_new(:patch_router_head, false)

        {:ok, AdaptedQwenPatch.patch_model_info!(model_info, artifact_dir, opts)}
    end
  rescue
    e ->
      {:error, {:artifact_patch_error, Exception.message(e)}}
  end

  defp supported_text_modules do
    Application.spec(:bumblebee, :modules)
    |> List.wrap()
    |> Enum.filter(fn module ->
      String.starts_with?(Atom.to_string(module), "Elixir.Bumblebee.Text.")
    end)
    |> Enum.sort()
    |> Enum.uniq()
  end
end
