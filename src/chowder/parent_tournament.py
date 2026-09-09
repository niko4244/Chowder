"""The real parent tournament: protected nine-dimension evaluation of local parents.

This module executes what `parent_eval.py` specified and
`parent_suite_content.py` authored: the first real A/B comparison of
native-Qwen3.8 parent checkpoints under the frozen protected suite v1.

Design constraints (each closes a specific honesty hazard):

- **One evaluation interpretation.** Model execution is delegated to
  `evaluators.base_text_worker` in a subprocess — the same worker the
  rest of Chowder uses — so the tournament cannot drift from the
  platform's evaluation semantics. This module orchestrates; it does
  not reimplement generation, scoring, or fingerprinting.
- **Integrity before inference.** Every parent's local manifest is
  re-verified against its directory before any GPU work; a divergence
  stops that parent's evaluation with the exact divergence record
  instead of scoring a model whose bytes changed behind the manifest.
- **Tokenizer identity is the comparability gate.** Cross-parent score
  comparison presupposes shared tokenization; `ensure_parent_tokenizer_compatible`
  is applied to every pairing before any model loads, from measured
  tokenizer evidence (class, vocab size, serialized-asset identity
  hash), never from names.
- **Capability and behavior stay separate.** Aggregation goes through
  `aggregate_parent_result`, whose split is structural. This module
  adds no aggregate of its own.
- **Six items per dimension is a real limit.** The comparison table
  classifies deltas in item units: a difference smaller than one item
  is a tie, one item is a weak signal, two or more is a clear
  difference — and the table says so instead of ranking on noise.
- **Evidence or it did not happen.** Each parent run records the suite
  manifest digest, protocol digest, model manifest digest and pin,
  tokenizer evidence, worker runtime/versions, wall-clock, sampled
  peak VRAM, and the per-item prediction files' digests, then persists
  an FK-anchored `evaluation_runs` row via
  `record_parent_tournament_result`.
- **Commit headroom before the load, not a segfault after.** The 4-bit
  load path (transformers 5.16.1 + bitsandbytes 0.50.2 staging a
  ~52 GiB checkpoint) drives Windows commit charge up by ~68-71 GiB
  above its launch baseline, and when that requirement crosses the
  system commit limit the process dies as a silent native access
  violation (exit 3221225477) with no Python traceback -- measured,
  not inferred (2026-09-07 differential, see docs/HANDOFF.md
  "4-bit load crash" and the evidence files under
  `C:/Users/nikma/Chowder-Protected/diagnostics/`): 5/5 crashes
  under insufficient headroom (44 GiB -> early crash; ~55-60 GiB ->
  mid-load crash; 49 GiB forced in a controlled stress test ->
  crash at 414 s), 5/5 completions at ~90 GiB headroom. Before every
  worker launch this module measures commit headroom and refuses to
  start the load below `CHOWDER_MIN_COMMIT_HEADROOM_GIB` (default 80),
  with the measured numbers in the error, because a fail-fast with
  actionable numbers beats a seven-minute silent segfault. A worker
  that still dies natively (e.g. headroom consumed concurrently) is
  retried up to `_NATIVE_CRASH_RETRIES` times after re-checking the
  gate -- the crash is environment-transient, not deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .local_model_manifest import (
    LocalModelManifest,
    verify_local_model_manifest,
)
from .models import Experiment, ExperimentStatus, Hypothesis
from .parent_eval import (
    ParentEvalSpec,
    ParentTokenizerEvidence,
    aggregate_parent_result,
    ensure_parent_tokenizer_compatible,
    record_parent_tournament_result,
    ParentTokenizerMismatch,)
from .parent_suite_content import build_tournament_spec


class ParentTournamentError(RuntimeError):
    """The tournament cannot be executed honestly."""


@dataclass(frozen=True)
class LocalParent:
    """One tournament participant: a verified local checkpoint at a pin."""

    label: str
    revision: str
    local_path: str
    manifest_path: str

    def __post_init__(self) -> None:
        for label_, value in (
            ("label", self.label),
            ("revision", self.revision),
            ("local_path", self.local_path),
            ("manifest_path", self.manifest_path),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"local parent {label_} must be a non-empty string")
        if not Path(self.local_path).is_dir():
            raise ParentTournamentError(
                f"parent {self.label!r} local dir missing: {self.local_path}"
            )
        if not Path(self.manifest_path).is_file():
            raise ParentTournamentError(
                f"parent {self.label!r} manifest missing: {self.manifest_path}"
            )

    def load_manifest(self) -> LocalModelManifest:
        return LocalModelManifest.from_dict(
            json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
        )


#: The program's documented parents (docs/QWEN38_SPARSE_PROGRAM.md pins),
#: as lazy factories: the paths are machine-specific, so they are validated
#: only when a caller on this machine actually asks for a parent — never at
#: import time, which would break collection on any machine without the
#: local-models volumes (CI). The fail-closed check still happens, exactly
#: where execution begins.


def parent_a() -> LocalParent:
    """Parent A (official control) on this machine; raises if absent."""
    return LocalParent(
        label="parent-a-qwen38-27b-official",
        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        local_path=r"F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B",
        manifest_path=r"F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B.manifest.json",
    )


def parent_b() -> LocalParent:
    """Parent B (OrcaRouter prior) on this machine; raises if absent."""
    return LocalParent(
        label="parent-b-orcarouter-uncensored",
        revision="404ea47aaa5d8a8b00049c9e9750089aca011ab2",
        local_path=r"G:\Local Models\HuggingFace\orcarouter\Qwen3.8-27B-Uncensored",
        manifest_path=r"G:\Local Models\HuggingFace\orcarouter\Qwen3.8-27B-Uncensored.manifest.json",
    )

#: Tokenizer asset files whose content defines tokenizer identity.
_V3_PROBE_CORPUS_SHA256 = (
    "a05451e901d819a5b81b2fedb3fe991761eec28c3b0ef730d5c39166bcf909e4"
)
_V3_PROBE_CORPUS_PATH = (
    r"C:/Users/nikma/Chowder-Protected/calibration/phase4-parent-a/corpus.jsonl"
)
_V3_PROBE_PASSAGES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 370, 371, 740, 741, 1160, 1161)


def tokenizer_behavior_evidence(parent: LocalParent) -> dict[str, Any]:
    """Behavioral tokenizer evidence: token IDs on the pinned public-domain probe.

    Protocol v3's comparability question is not "are the serialization bytes
    equal" (v2's question) but "does this parent tokenize the tournament's
    inputs identically to the reference". The probe is a fixed passage
    selection from the hashed Phase 4 calibration corpus (public-domain
    Gutenberg text, `not_protected_suite_content: true` in its manifest --
    never tournament content), so the evidence is reproducible by anyone
    holding the corpus.
    """
    # Corpus integrity FIRST: the pinned probe must be the exact hashed
    # bytes before any tokenizer is loaded (and before the heavy
    # transformers import, which CI environments do not carry).
    root = Path(parent.local_path)
    corpus = Path(_V3_PROBE_CORPUS_PATH)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    if digest != _V3_PROBE_CORPUS_SHA256:
        raise ParentTournamentError(
            "v3 tokenizer probe corpus hash mismatch: expected "
            f"{_V3_PROBE_CORPUS_SHA256}, got {digest}"
        )
    from transformers import AutoTokenizer

    rows = corpus.read_text(encoding="utf-8").splitlines()
    texts = []
    for i in _V3_PROBE_PASSAGES:
        texts.append(str(json.loads(rows[i])["text"]))
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), trust_remote_code=False, local_files_only=True
    )
    token_ids: list[list[int]] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.append([int(x) for x in encoded])
    token_count = sum(len(ids) for ids in token_ids)
    if token_count < 1000:
        raise ParentTournamentError(
            f"v3 tokenizer probe produced only {token_count} tokens for "
            f"parent {parent.label!r}; probe coverage is too small to be evidence"
        )
    return {"token_ids": token_ids, "token_count": token_count}


def ensure_parent_tokenizer_behavior_compatible(
    reference: LocalParent, candidate: LocalParent
) -> dict[str, Any]:
    """v3 gate: fail closed unless the parents tokenize the probe identically.

    Replaces v2's asset-identity equality (which refused C and D despite
    behaviorally identical tokenization -- serialization hygiene, not
    substance). Class and vocab size are still recorded via
    `tokenizer_evidence`; the returned evidence goes into the run record.
    """
    ref = tokenizer_behavior_evidence(reference)
    cand = tokenizer_behavior_evidence(candidate)
    if ref["token_ids"] != cand["token_ids"]:
        first = None
        for i, (a, b) in enumerate(zip(ref["token_ids"], cand["token_ids"])):
            for j in range(min(len(a), len(b))):
                if a[j] != b[j]:
                    first = (i, j)
                    break
            if first is not None:
                break
        raise ParentTokenizerMismatch(
            "v3 tokenizer behavior mismatch: parent "
            f"{candidate.label!r} does not tokenize the pinned public-domain "
            f"probe identically to {reference.label!r} "
            f"(first divergence at passage/position {first}); cross-parent "
            "score comparability requires provably shared tokenization."
        )
    return {
        "mode": "behavior",
        "probe_corpus_sha256": _V3_PROBE_CORPUS_SHA256,
        "passages": list(_V3_PROBE_PASSAGES),
        "token_count": ref["token_count"],
        "reference_label": reference.label,
        "candidate_label": candidate.label,
        "identical": True,
    }


_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def tokenizer_evidence(parent: LocalParent, *, offline: bool = True) -> ParentTokenizerEvidence:
    """Measure one parent's tokenizer identity from its serialized assets.

    The identity hash covers every tokenizer asset file present on disk
    (sorted name -> sha256), so a changed vocabulary, template, or
    special-token map changes the identity. Loading uses the local path
    with `local_files_only` — the tournament never re-downloads.
    """
    from transformers import AutoTokenizer

    root = Path(parent.local_path)
    material: dict[str, str] = {}
    for name in _TOKENIZER_ASSETS:
        path = root / name
        if path.is_file():
            material[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not material:
        raise ParentTournamentError(
            f"parent {parent.label!r} has no tokenizer assets to fingerprint"
        )
    identity = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), trust_remote_code=False, local_files_only=offline
    )
    return ParentTokenizerEvidence(
        tokenizer_class=type(tokenizer).__name__,
        vocab_size=len(tokenizer),
        identity_sha256=identity,
    )


def verify_parent_integrity(parent: LocalParent) -> dict[str, Any]:
    """Re-verify a parent's full manifest against its directory.

    Full mode re-hashes every weight shard; the returned record is the
    evidence that the evaluated bytes are the manifested bytes. Any
    divergence raises — a changed checkpoint must never be silently
    scored.
    """
    manifest = parent.load_manifest()
    if manifest.mode != "full":
        raise ParentTournamentError(
            f"parent {parent.label!r} manifest is mode={manifest.mode!r}; the "
            "tournament requires a full-mode manifest (every shard hashed)"
        )
    # rehash_weights=True forces every weight shard to be re-hashed even if
    # the manifest had been fast-mode: the tournament's integrity evidence
    # is byte-level, not size-level.
    verification = verify_local_model_manifest(
        manifest, parent.local_path, rehash_weights=True
    )
    if not verification.clean:
        raise ParentTournamentError(
            f"parent {parent.label!r} integrity FAILED: {verification.divergences}"
        )
    return {
        "manifest_sha256": manifest.manifest_sha256,
        "checked_files": verification.checked_files,
        "weight_shards": len(manifest.weight_files),
        "total_weight_bytes": manifest.total_weight_bytes,
    }


#: `EvalSuiteSpec`'s exact field set: `ParentSuiteSpec.to_dict()` carries
#: the extra `dimension` label, which the *worker's* suite dataclass does
#: not accept. This mirror is the single serialization boundary between
#: the tournament's suite view and the worker's — same semantic content,
#: worker-serializable shape, one field list to keep in sync.
_EVAL_SUITE_FIELDS = (
    "name",
    "dataset",
    "prompt_field",
    "expected_field",
    "scoring",
    "max_new_tokens",
    "use_chat_template",
    # v3: the worker renders through the canonical template when set.
    "canonical_rendering",
)


def _worker_spec_payload(
    suites: tuple,
    parent: LocalParent,
    run_dir: Path,
    *,
    device: str,
    quantization: str,
    precision: str,
    seed: int,
    timeout_seconds: float | None,
    canonical_rendering: bool = False,
) -> dict[str, Any]:
    """The flat `BaseTextEvalSpec` payload `base_text_worker.main` parses.

    `canonical_rendering` is spec-level on `ParentEvalSpec` (not a per-suite
    field), so it is injected into every suite dict explicitly — passing it
    through the per-suite dicts alone would silently drop it and the worker
    would fall back to each parent's own template (the exact v2 behavior v3
    exists to remove).
    """
    return {
        "base_model": parent.local_path,
        "output_dir": str(run_dir),
        "suites": [
            {
                **{key: value for key, value in suite.to_dict().items() if key in _EVAL_SUITE_FIELDS},
                "canonical_rendering": bool(canonical_rendering),
            }
            for suite in suites
        ],
        "revision": None,
        "precision": precision,
        "quantization": quantization,
        "device": device,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "trust_remote_code": False,
        "offline": True,
    }


def _peak_vram_sampler(stop: threading.Event, result: dict[str, Any]) -> None:
    """Sample total GPU memory used across all GPUs until stopped."""
    peak = 0
    while not stop.is_set():
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in out.stdout.splitlines():
                peak = max(peak, int(line.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        result["peak_mib"] = peak
        stop.wait(5.0)


_NATIVE_CRASH_RETRIES = 2
_NATIVE_CRASH_EXIT_CODES = {3221225477, 139}  # Windows access violation; POSIX SIGSEGV
_MIN_COMMIT_HEADROOM_GIB = float(
    os.environ.get("CHOWDER_MIN_COMMIT_HEADROOM_GIB", "80")
)


def _commit_state() -> tuple[float, float] | None:
    """Return (used_gib, limit_gib) of Windows commit charge, or None when
    unmeasurable (non-Windows). Read from the OS, never estimated."""
    if sys.platform != "win32":
        return None
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return None
    used = (stat.ullTotalPageFile - stat.ullAvailPageFile) / 2**30
    return used, stat.ullTotalPageFile / 2**30


def _enforce_commit_headroom() -> float:
    """Fail fast when commit headroom cannot safely hold a 4-bit load.

    Returns the measured headroom in GiB. On non-Windows platforms the
    measurement is unavailable and the gate passes (the crash class it
    guards against was only observed on Windows/WDDM).
    """
    state = _commit_state()
    if state is None:
        return float("nan")
    used, limit = state
    headroom = limit - used
    if headroom < _MIN_COMMIT_HEADROOM_GIB:
        raise ParentTournamentError(
            f"refusing to start a 4-bit load with {headroom:.1f} GiB of commit "
            f"headroom ({used:.1f}/{limit:.1f} GiB used); the 4-bit staging path "
            f"needs ~68-71 GiB above its baseline and dies as a silent native "
            f"access violation when it hits the commit limit. Close memory-heavy "
            f"applications or raise the pagefile, or override with "
            f"CHOWDER_MIN_COMMIT_HEADROOM_GIB."
        )
    return headroom


def _run_worker(spec_payload: dict[str, Any], run_dir: Path, *, timeout_seconds: float | None) -> dict[str, Any]:
    """Launch `base_text_worker` on a serialized spec and return its result."""
    spec_path = run_dir / "eval-spec.json"
    result_path = run_dir / "eval-result.json"
    spec_path.write_text(
        json.dumps(spec_payload, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    stdout_path = run_dir / "worker-stdout.log"
    stderr_path = run_dir / "worker-stderr.log"
    command = [
        sys.executable,
        "-m",
        "chowder.evaluators.base_text_worker",
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
    ]
    stop = threading.Event()
    sampler: dict[str, Any] = {"peak_mib": 0}
    thread = threading.Thread(target=_peak_vram_sampler, args=(stop, sampler), daemon=True)
    thread.start()
    # Unbuffered stderr so a native crash cannot swallow the tqdm position
    # (the pre-fix runs appeared to crash at "0%" only because buffered
    # stderr was lost with the process).
    worker_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    commit_headroom = _enforce_commit_headroom()
    attempts = 0
    started = time.perf_counter()
    try:
        proc = None
        while True:
            attempts += 1
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                proc = subprocess.run(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout_seconds,
                    env=worker_env,
                )
            if proc.returncode == 0 and result_path.is_file():
                break
            if proc.returncode in _NATIVE_CRASH_EXIT_CODES and attempts <= _NATIVE_CRASH_RETRIES:
                # Native crashes of this class are environment-transient
                # (commit pressure), not deterministic: re-check the gate
                # so a retry under worse pressure fails loudly instead of
                # segfaulting again, then relaunch from scratch.
                _enforce_commit_headroom()
                continue
            tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise ParentTournamentError(
                f"evaluation worker failed (exit {proc.returncode}, "
                f"attempt {attempts}); stderr tail:\n{tail}"
            )
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        thread.join(timeout=15)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["wall_seconds"] = round(elapsed, 1)
    result["peak_gpu_mib_sampled"] = sampler.get("peak_mib", 0)
    result["commit_headroom_gib_at_launch"] = (
        round(commit_headroom, 1) if commit_headroom == commit_headroom else None
    )
    result["worker_attempts"] = attempts
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ParentRunResult:
    """One parent's completed protected evaluation."""

    parent: LocalParent
    report: Any  # ParentEvalReport
    worker_result: Mapping[str, Any]
    integrity: Mapping[str, Any]
    tokenizer: ParentTokenizerEvidence
    run_dir: str
    evaluation_run_id: str


def evaluate_parent(
    registry: Any,
    parent: LocalParent,
    spec: ParentEvalSpec,
    *,
    tokenizer: ParentTokenizerEvidence,
    output_root: str | Path,
    device: str = "cuda:0",
    quantization: str = "4bit",
    precision: str = "bf16",
    seed: int = 20260907,
    timeout_seconds: float | None = None,
) -> ParentRunResult:
    """Verify, evaluate, aggregate, and persist one parent.

    `tokenizer` must come from `tokenizer_evidence` after the pairwise
    identity gate has passed — it is recorded, never re-derived here.
    The seed is a parameter, not a constant, so the orchestrator can
    guarantee every parent in one tournament runs under the *same* seed.
    """
    integrity = verify_parent_integrity(parent)
    run_dir = Path(output_root) / parent.label
    run_dir.mkdir(parents=True, exist_ok=False)

    worker_result = _run_worker(
        _worker_spec_payload(
            spec.suites,
            parent,
            run_dir,
            device=device,
            quantization=quantization,
            precision=precision,
            seed=seed,
            timeout_seconds=timeout_seconds,
            canonical_rendering=spec.canonical_rendering,
        ),
        run_dir,
        timeout_seconds=timeout_seconds,
    )
    metrics = {k: float(v) for k, v in worker_result["metrics"].items()}
    report = aggregate_parent_result(
        spec=spec,
        base_model=parent.label,
        revision=parent.revision,
        metrics=metrics,
        evidence={
            "local_path": parent.local_path,
            "model_manifest_sha256": integrity["manifest_sha256"],
            "integrity_checked_files": integrity["checked_files"],
            "tokenizer_assets_note": "identity hashed from serialized assets on disk",
            "suite_manifest_note": "frozen protected root; dataset sha256s in suite manifest",
            "quantization": quantization,
            "precision": precision,
            "wall_seconds": worker_result["wall_seconds"],
            "peak_gpu_mib_sampled": worker_result["peak_gpu_mib_sampled"],
            "worker_runtime": worker_result.get("runtime", {}),
            "worker_versions": worker_result.get("versions", {}),
            "suite_evidence": worker_result.get("suites", {}),
        },
    )
    # Digest every prediction file into the evidence: the per-item record
    # must be traceable from the persisted row without copying raw text.
    prediction_digests = {
        path.name: _file_sha256(path)
        for path in sorted(run_dir.glob("predictions-*.jsonl"))
    }
    report.evidence["prediction_file_sha256"] = prediction_digests

    gpu_hours = worker_result["wall_seconds"] / 3600.0
    experiment_id = f"exp-parent-baseline-{parent.label}"
    # evaluation_runs is foreign-keyed to experiments: record the anchoring
    # experiment row first (the "experiment" here *is* the baseline
    # evaluation), then the outcome. Real registries get a real row;
    # test doubles without record_experiment are recorded as outcomes only.
    if hasattr(registry, "record_experiment"):
        registry.record_experiment(
            Experiment(
                experiment_id=experiment_id,
                parent_id=None,
                hypothesis=Hypothesis(
                    observation=(
                        f"parent {parent.label} has no protected-suite baseline"
                    ),
                    suspected_cause="no parent-selection evidence exists yet",
                    intervention=(
                        "evaluate the untouched parent under frozen protected "
                        "suite v1 with the platform's standard text worker"
                    ),
                    expected_deltas={},
                ),
                config_patch={
                    "parent_baseline": True,
                    "local_path": parent.local_path,
                    "revision": parent.revision,
                    "quantization": quantization,
                    "precision": precision,
                    "seed": seed,
                },
                estimated_gpu_hours=max(round(gpu_hours, 6), 0.001),
                status=ExperimentStatus.PASSED,
                tags=("parent-tournament", "protected-suite-v1"),
            )
        )
    outcome = record_parent_tournament_result(
        registry,
        report=report,
        run_id=f"tournament-{parent.label}-{int(time.time())}",
        experiment_id=experiment_id,
        artifact_ref=str(run_dir),
        gpu_hours=round(gpu_hours, 6),
    )
    return ParentRunResult(
        parent=parent,
        report=report,
        worker_result=worker_result,
        integrity=integrity,
        tokenizer=tokenizer,
        run_dir=str(run_dir),
        evaluation_run_id=outcome.run_id,
    )


def run_tournament(
    registry: Any,
    parents: tuple[LocalParent, ...],
    frozen_root: str | Path,
    *,
    output_root: str | Path,
    device: str = "cuda:0",
    quantization: str = "4bit",
    precision: str = "bf16",
    seed: int = 20260907,
    timeout_seconds: float | None = None,
    protocol_version: str = "v2",
) -> dict[str, Any]:
    """Execute the protected tournament over `parents` in order.

    Raises before any GPU work if integrity or tokenizer comparability
    fails for any participant. `protocol_version` selects the protocol
    generation: v2 (own-template rendering, asset-identity tokenizer
    gate) or v3 (canonical-template rendering, behavioral tokenizer
    gate). Returns the full evidence bundle including the honest
    comparison table.
    """
    if len(parents) < 2:
        raise ParentTournamentError("a tournament needs at least two parents")
    spec = build_tournament_spec(frozen_root, protocol_version=protocol_version)

    # Gates before any model load: integrity for everyone, then tokenizer
    # identity pairwise against the first parent (the comparability gate).
    integrity_by_parent: dict[str, dict[str, Any]] = {}
    for parent in parents:
        integrity_by_parent[parent.label] = verify_parent_integrity(parent)
    tokenizer_by_parent: dict[str, ParentTokenizerEvidence] = {}
    behavior_evidence: dict[str, Any] = {}
    reference = parents[0]
    tokenizer_by_parent[reference.label] = tokenizer_evidence(reference)
    if spec.protocol_version == "v3":
        # v3 comparability: identical tokenization BEHAVIOR on the pinned
        # public-domain probe (serialization may differ; behavior may not).
        behavior_evidence[reference.label] = {"mode": "reference"}
        for parent in parents[1:]:
            behavior_evidence[parent.label] = ensure_parent_tokenizer_behavior_compatible(
                reference, parent
            )
            tokenizer_by_parent[parent.label] = tokenizer_evidence(parent)
    else:
        # v2 comparability: byte-identical serialized tokenizer assets.
        for parent in parents[1:]:
            evidence = tokenizer_evidence(parent)
            ensure_parent_tokenizer_compatible(tokenizer_by_parent[reference.label], evidence)
            tokenizer_by_parent[parent.label] = evidence

    results: dict[str, ParentRunResult] = {}
    for parent in parents:
        results[parent.label] = evaluate_parent(
            registry,
            parent,
            spec,
            tokenizer=tokenizer_by_parent[parent.label],
            output_root=output_root,
            device=device,
            quantization=quantization,
            precision=precision,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )

    return {
        "protocol_sha256": spec.digest(),
        "suite_count": len(spec.suites),
        "seed": seed,
        "tokenizer_behavior_evidence": behavior_evidence,
        "tokenizer_evidence": {
            label: tok.to_dict()
            if hasattr(tok, "to_dict")
            else {
                "tokenizer_class": tok.tokenizer_class,
                "vocab_size": tok.vocab_size,
                "identity_sha256": tok.identity_sha256,
            }
            for label, tok in tokenizer_by_parent.items()
        },
        "integrity": integrity_by_parent,
        "runs": {
            label: {
                "evaluation_run_id": result.evaluation_run_id,
                "report": result.report.to_dict(),
                "wall_seconds": result.worker_result["wall_seconds"],
                "peak_gpu_mib_sampled": result.worker_result["peak_gpu_mib_sampled"],
            }
            for label, result in results.items()
        },
        "comparison": compare_reports({label: result.report for label, result in results.items()}),
    }


# ---- Honest comparison ---------------------------------------------------------
#: With six items per suite, one item is 1/6 of a dimension's mean.
_ITEMS_PER_SUITE = 6
_ONE_ITEM = 1.0 / _ITEMS_PER_SUITE


def compare_reports(reports: Mapping[str, Any]) -> dict[str, Any]:
    """Compare tournament rows without manufacturing certainty.

    Every dimension delta is classified in item units: |delta| < one
    item is a "tie" (six-item noise), exactly one item is a "weak
    signal", more is a "clear difference". Capability and behavior are
    compared separately; no overall score is produced.
    """
    labels = sorted(reports)
    if len(labels) != 2:
        raise ParentTournamentError("the A/B comparison needs exactly two parents")
    left, right = reports[labels[0]], reports[labels[1]]
    dimensions: dict[str, Any] = {}
    for dimension in sorted(set(left.dimensions) | set(right.dimensions)):
        l_mean = left.dimensions.get(dimension).mean if dimension in left.dimensions else None
        r_mean = right.dimensions.get(dimension).mean if dimension in right.dimensions else None
        if l_mean is None or r_mean is None:
            classification = "unmeasured"
            delta = None
        else:
            delta = round(r_mean - l_mean, 6)
            magnitude = abs(delta) * _ITEMS_PER_SUITE
            if magnitude < 0.5:
                classification = "tie"
            elif magnitude < 1.5:
                classification = "weak-signal"
            else:
                classification = "clear-difference"
        dimensions[dimension] = {
            "left": l_mean,
            "right": r_mean,
            "delta_right_minus_left": delta,
            "classification": classification,
        }
    return {
        "left": labels[0],
        "right": labels[1],
        "dimensions": dimensions,
        "capability_mean": {"left": left.capability_mean, "right": right.capability_mean},
        "behavior_mean": {"left": left.behavior_mean, "right": right.behavior_mean},
        "note": (
            "six items per suite: ties are preserved and one-item differences are "
            "weak signals, not rankings"
        ),
    }
