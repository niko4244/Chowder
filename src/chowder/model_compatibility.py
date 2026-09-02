from __future__ import annotations

from .hf_resilience import with_hub_retries


class IncompatibleModelArchitectureError(RuntimeError):
    """base_model's architecture is not usable with AutoModelForCausalLM,
    which Chowder's transformers-peft training and evaluation workers both
    assume. Raised during preflight, before any GPU-hours are reserved or a
    worker subprocess is spawned -- discovering an architecture mismatch
    there is a config-time mistake, not a training failure, and should look
    like one rather than surfacing as a confusing error deep inside a
    spawned subprocess after paying for the reservation and process
    startup.
    """


def check_causal_lm_architecture(
    *, base_model: str, revision: str | None, offline: bool, label: str
) -> None:
    """Resolve base_model's config (config.json only -- no weight download)
    and verify its architecture is registered under AutoModelForCausalLM.

    Uses AutoModelForCausalLM's own model-mapping registry rather than a
    hand-maintained allowlist, so newly supported architectures need no
    change here. If that registry is ever unavailable (a future
    transformers version changing a private attribute this relies on), the
    check is skipped rather than blocking a run: it is a cheap early
    warning, not the only line of defense -- a genuinely incompatible model
    still fails later, inside the worker subprocess, just more expensively.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = with_hub_retries(
        lambda: AutoConfig.from_pretrained(
            base_model,
            revision=revision,
            trust_remote_code=False,
            local_files_only=offline,
        ),
        label=f"config resolution for {base_model}",
    )
    try:
        mapping = AutoModelForCausalLM._model_mapping
        compatible = type(config) in mapping
    except AttributeError:
        return
    if not compatible:
        model_type = getattr(config, "model_type", type(config).__name__)
        raise IncompatibleModelArchitectureError(
            f"{label}: {base_model!r} has architecture {model_type!r}, which is not "
            "registered under AutoModelForCausalLM; Chowder's transformers-peft "
            "backend requires a causal-LM-compatible architecture"
        )
