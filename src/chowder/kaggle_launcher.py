"""Kaggle-adapted parent-evaluation launcher (Qwen3.8 program, Phase 2).

Reuses `parent_tournament.evaluate_parent` / `.tokenizer_evidence` /
`.verify_parent_integrity` / `.LocalParent` and `parent_eval.ParentEvalSpec`
completely unchanged -- this module contains no re-implementation of
evaluation, scoring, tokenizer measurement, or fingerprinting logic. It
only adapts what is unavoidably platform-specific:

1. **Environment fingerprinting** (`capture_environment_fingerprint`) --
   records the exact dependency/hardware facts the mission requires,
   reusing `kaggle_equivalence.BackendFingerprint`'s shape so a Kaggle
   run's fingerprint is directly comparable to a local run's.
2. **A VRAM preflight** (`preflight_vram_headroom`) -- a T4 has 16 GiB,
   not the ~35+ GiB headroom `parent_tournament._enforce_commit_headroom`
   already gates locally. This mirrors that function's fail-fast
   philosophy (refuse before the load with actionable numbers, rather
   than a silent OOM crash mid-generation) for the one resource that
   differs on Kaggle. It is a conservative *estimate*, not a promise --
   if the model genuinely does not fit, this module raises rather than
   attempting a workaround (no quantization change, no offloading trick)
   that would itself become an undeclared, unaccounted-for divergence
   from the local protocol.
3. **The one spec field known to be hardware-incompatible on T4**
   (`build_kaggle_eval_spec`). Confirmed by reading
   `evaluators.base_text_worker._dtype` directly: it raises when
   `precision="bf16"` is requested on a device where
   `torch.cuda.is_bf16_supported()` is False, which is the case for
   every T4 (Turing, compute capability 7.5 -- no bf16 tensor cores).
   `precision` is part of `ParentEvalSpec.to_dict()` and therefore part
   of the protocol digest, so this module never silently swaps it in a
   spec it hands to the reused worker without also returning the exact,
   citable justification string `kaggle_equivalence.qualify_backend`
   requires as a `declared_digest_divergence_reasons` entry. No other
   field is ever changed for Kaggle's sake.

What this module does NOT do
-------------------------------
- It does not talk to the Kaggle API, Kaggle Secrets, or the Hugging
  Face Hub -- those live in the `kaggle/` directory's notebook-facing
  scripts, which import this module rather than the reverse.
- It never runs a real model load or generation itself in this
  repository's test suite; `run_kaggle_parent_evaluation`'s real work is
  delegated to `parent_tournament.evaluate_parent` (injectable as
  `evaluate_parent_fn` so tests can verify this module's own plumbing
  -- spec adaptation, preflight, fingerprint capture -- without any GPU).
- It never targets more than one GPU. `base_text_worker.evaluate` pins
  the whole quantized model onto a single CUDA device index
  (`device_map={"": index}`); there is no multi-GPU device-map path to
  reuse without modifying that off-limits file. A Kaggle "T4 x2"
  allocation is therefore used as one T4 per evaluation job, honestly
  recorded as `gpu_count=1` in the environment fingerprint -- never
  assumed to behave like a single 32 GiB card.
"""
from __future__ import annotations

import importlib.metadata
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .kaggle_equivalence import BackendFingerprint
from .parent_eval import ParentEvalSpec, ParentTokenizerEvidence
from .parent_tournament import LocalParent, ParentRunResult, evaluate_parent

#: T4 total capacity commonly reported by `nvidia-smi` on Kaggle's GPU
#: notebook image is close to, but slightly under, the vendor-nominal
#: 16 GiB (driver/ECC reservation). This is a cross-check ceiling only --
#: the real gate is `free_vram_gib_fn`'s *live* measurement at preflight
#: time, not this constant.
KAGGLE_T4_USABLE_VRAM_GIB = 14.8

#: Real, measured evidence beats a formula: parent A's actual retry7 run
#: on the local RTX 5060 Ti (16 GiB card, 4-bit + bf16, protocol v2, the
#: same 256-token budget) peaked at `peak_gpu_mib_sampled = 16004` MiB --
#: read directly from that run's persisted evidence, not estimated. That
#: is roughly 15.63 GiB: this workload occupies essentially the *entire*
#: nominal capacity of a 16 GiB-class card, with very little margin, on
#: hardware this program already knows works. A single T4 (also a
#: 16 GiB-class card) is therefore genuinely marginal, not a comfortable
#: fit -- this module surfaces that risk with a preflight check rather
#: than assuming it away. `estimate_required_vram_gib` uses this measured
#: figure (scaled for the fp16-vs-bf16 activation-dtype difference, which
#: changes activation/KV-cache footprint but not the 4-bit weight size)
#: as its primary basis whenever a caller supplies a reference
#: measurement; parents A/B/C/D share an identical measured tensor census
#: (docs/QWEN38_SPARSE_PROGRAM.md), so parent A's real number is a
#: defensible proxy for C/D even though they have not been measured
#: directly yet.
REFERENCE_PEAK_GPU_MIB_PARENT_A = 16004

#: A caller-suppliable safety margin over the reference measurement, to
#: absorb cross-architecture kernel/driver variance (T4 vs the reference
#: card) -- not a claim that the real number will differ by exactly this
#: much.
_VRAM_ESTIMATE_MARGIN = 1.10

#: Fallback estimate when no real reference measurement is available at
#: all (used only if `reference_peak_gpu_mib_sampled` is omitted). nf4
#: double-quantized 4-bit weights run roughly 4x smaller than the
#: bf16/fp16 shard bytes a manifest records (safetensors stores the
#: *unquantized* checkpoint at 2 bytes/parameter; bitsandbytes quantizes
#: on load), so weights alone are ~0.25x the manifest's total_weight_bytes.
#: The 1.35x pad covers activations, the KV cache for a 256-token
#: generation budget, and quantization-constant overhead -- markedly less
#: reliable than a real measurement, and this module prefers the
#: measured path whenever one is available.
_QUANTIZED_WEIGHT_FRACTION = 0.25
_VRAM_ESTIMATE_PAD = 1.35

#: The single spec field this module ever overrides for Kaggle, and why.
#: Cited verbatim into `kaggle_equivalence.build_equivalence_report`'s
#: `declared_digest_divergence_reasons` by callers wiring a real
#: qualification run -- kept as one named constant so the local and
#: Kaggle sides of a qualification can never accidentally use worded
#: differently.
PRECISION_DIVERGENCE_REASON = (
    "precision: bf16 (local) vs fp16 (kaggle) -- T4 GPUs (compute capability 7.5) "
    "do not support bf16 tensor cores; torch.cuda.is_bf16_supported() is False "
    "there, and evaluators.base_text_worker._dtype raises RuntimeError on a bf16 "
    "request in that case. fp16 is the closest technically possible substitute; "
    "no other spec field is changed for Kaggle."
)


class KagglePreflightError(RuntimeError):
    """A Kaggle-side resource or configuration preflight failed; refuse to launch."""


def estimate_required_vram_gib(
    total_weight_bytes: int,
    *,
    reference_peak_gpu_mib_sampled: int | None = None,
    margin: float = _VRAM_ESTIMATE_MARGIN,
    pad: float = _VRAM_ESTIMATE_PAD,
) -> float:
    """VRAM estimate for a 4-bit load on Kaggle, preferring real measurement.

    When `reference_peak_gpu_mib_sampled` is supplied (e.g.
    `REFERENCE_PEAK_GPU_MIB_PARENT_A`, or a fresher real measurement once
    one exists), this is `reference_peak_gpu_mib_sampled / 2**10 * margin`
    -- real evidence from an actual completed run, scaled by a modest
    cross-architecture margin, never re-derived from a formula. Without a
    reference, falls back to a formula-based estimate from
    `total_weight_bytes` (the acquired parent's own
    `LocalModelManifest.total_weight_bytes`) -- clearly the weaker
    evidence path, used only when no real measurement exists yet.
    """
    if total_weight_bytes <= 0:
        raise ValueError("total_weight_bytes must be positive")
    if reference_peak_gpu_mib_sampled is not None:
        if reference_peak_gpu_mib_sampled <= 0:
            raise ValueError("reference_peak_gpu_mib_sampled must be positive")
        return (reference_peak_gpu_mib_sampled / 2**10) * margin
    quantized_bytes = total_weight_bytes * _QUANTIZED_WEIGHT_FRACTION
    return (quantized_bytes / 2**30) * pad


def preflight_vram_headroom(
    required_gib: float,
    *,
    free_vram_gib_fn: Callable[[], float],
    usable_ceiling_gib: float = KAGGLE_T4_USABLE_VRAM_GIB,
) -> float:
    """Fail fast, with actionable numbers, before a load Kaggle's T4 cannot
    hold -- the exact philosophy `parent_tournament._enforce_commit_headroom`
    already applies locally, for the resource that differs here.

    `free_vram_gib_fn` is real GPU state on Kaggle (e.g. wrapping
    `torch.cuda.mem_get_info`), injected so this can be exercised in
    tests without any GPU.
    """
    free_gib = free_vram_gib_fn()
    ceiling = min(free_gib, usable_ceiling_gib)
    if required_gib > ceiling:
        raise KagglePreflightError(
            f"estimated requirement {required_gib:.1f} GiB exceeds the usable T4 "
            f"ceiling {ceiling:.1f} GiB (free reported: {free_gib:.1f} GiB, usable "
            f"cap: {usable_ceiling_gib:.1f} GiB); refusing to launch rather than "
            "risk a mid-generation OOM. This module does not change the "
            "quantization recipe or offload weights to work around it -- if the "
            "model genuinely does not fit, that is a real blocker to report, not "
            "a reason to diverge from the local protocol silently."
        )
    return ceiling


def build_kaggle_eval_spec(reference_spec: ParentEvalSpec, *, precision: str = "fp16") -> ParentEvalSpec:
    """The reference (local) protocol spec, unchanged except `precision`.

    Returns a new `ParentEvalSpec` -- every suite, `quantization`,
    `max_model_len`, and `require_thinking_efficiency_telemetry` value is
    copied from `reference_spec` verbatim. Only `precision` differs, and
    only because T4 hardware cannot run the reference's bf16 (see
    `PRECISION_DIVERGENCE_REASON`). The resulting digest will therefore
    differ from `reference_spec.digest()` by exactly that one field --
    expected, and the reason is exported as a named constant so a caller
    never has to reconstruct it.
    """
    return replace(reference_spec, precision=precision)


def local_parent_for_kaggle(*, label: str, revision: str, kaggle_dataset_root: str | Path) -> LocalParent:
    """Build the same `LocalParent` shape `parent_tournament.py` uses,
    pointed at a Kaggle input-dataset mount instead of a local drive
    letter. Raises via `LocalParent.__post_init__` (unchanged) if the
    directory or manifest is not actually present -- exactly the same
    fail-closed check a local run gets."""
    root = Path(kaggle_dataset_root)
    return LocalParent(
        label=label,
        revision=revision,
        local_path=str(root),
        manifest_path=str(root.parent / f"{root.name}.manifest.json"),
    )


def capture_environment_fingerprint(
    *,
    chowder_commit_sha: str,
    tokenizer_identity_sha256: str | None,
    quantization: str,
    dtype: str,
    device_map_summary: str,
    torch_module: Any | None = None,
) -> BackendFingerprint:
    """Record every environment field the mission requires, live.

    `torch_module` is injectable so this can run (and be tested) without
    a real CUDA device: pass a fake object exposing `cuda.is_available()`,
    `cuda.device_count()`, `cuda.get_device_name(i)`, and `version.cuda`
    for a synthetic GPU fingerprint. In production, omit it and the real
    `torch` is imported lazily -- this module does not import torch at
    module load time, so a CPU-only environment can still import and use
    every non-GPU-dependent function above.
    """
    if torch_module is None:
        import torch as torch_module  # local import: torch is optional for the pure-python helpers above

    gpu_models: tuple[str, ...] = ()
    gpu_count = 0
    cuda_runtime_version: str | None = None
    if torch_module.cuda.is_available():
        gpu_count = torch_module.cuda.device_count()
        gpu_models = tuple(torch_module.cuda.get_device_name(i) for i in range(gpu_count))
        cuda_runtime_version = getattr(torch_module.version, "cuda", None)

    def _version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return BackendFingerprint(
        python_version=_python_version(),
        torch_version=_version("torch"),
        transformers_version=_version("transformers"),
        bitsandbytes_version=_version("bitsandbytes"),
        accelerate_version=_version("accelerate"),
        cuda_runtime_version=cuda_runtime_version,
        gpu_models=gpu_models,
        gpu_count=gpu_count,
        device_map_summary=device_map_summary,
        quantization=quantization,
        dtype=dtype,
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        chowder_commit_sha=chowder_commit_sha,
    )


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def run_kaggle_parent_evaluation(
    registry: Any,
    parent: LocalParent,
    reference_spec: ParentEvalSpec,
    *,
    tokenizer: ParentTokenizerEvidence,
    total_weight_bytes: int,
    free_vram_gib_fn: Callable[[], float],
    output_root: str | Path,
    seed: int,
    reference_peak_gpu_mib_sampled: int | None = REFERENCE_PEAK_GPU_MIB_PARENT_A,
    timeout_seconds: float | None = None,
    evaluate_parent_fn: Callable[..., ParentRunResult] = evaluate_parent,
) -> tuple[ParentRunResult, ParentEvalSpec]:
    """The single Kaggle notebook entry point: preflight, adapt, evaluate.

    Delegates the actual verify/load/generate/aggregate/persist sequence
    to `evaluate_parent_fn` (defaults to the real, unmodified
    `parent_tournament.evaluate_parent`) unchanged -- this function adds
    only the VRAM preflight and the one-field spec adaptation described
    in this module's docstring. `tokenizer` must still come from the
    caller having already run the real, unmodified
    `parent_tournament.tokenizer_evidence` + `ensure_parent_tokenizer_compatible`
    pairwise gate against the reference parent, exactly as
    `parent_tournament.run_tournament` does locally -- this module does
    not re-derive or bypass that gate.

    Returns `(result, kaggle_spec)` so the caller has the exact adapted
    spec (and therefore its digest) to record alongside the result for
    the equivalence report.
    """
    preflight_vram_headroom(
        estimate_required_vram_gib(
            total_weight_bytes, reference_peak_gpu_mib_sampled=reference_peak_gpu_mib_sampled
        ),
        free_vram_gib_fn=free_vram_gib_fn,
    )
    kaggle_spec = build_kaggle_eval_spec(reference_spec)
    result = evaluate_parent_fn(
        registry,
        parent,
        kaggle_spec,
        tokenizer=tokenizer,
        output_root=output_root,
        device="cuda:0",
        quantization=kaggle_spec.quantization,
        precision=kaggle_spec.precision,
        seed=seed,
        timeout_seconds=timeout_seconds,
    )
    return result, kaggle_spec
