"""Fail-closed acquisition/preflight tooling for Qwen3.8 parents C and D.

Parents A (control) and B (primary) are already cached and verified on
this machine; C (`OBLITERATUS/Qwen3.8-27B-OBLITERATED`) and D
(`DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-
NM-DAU`) are not. This module prepares that acquisition so it can happen
with minimal manual work once A/B's tournament finishes -- it does not
perform it. Nothing in this module is invoked against the network or a
real filesystem destination during this task; every network- and
disk-touching step is an injected callable so it can be exercised with
synthetic fixtures in tests and only wired to the real Hugging Face Hub /
local filesystem by a caller that has actually decided to acquire.

Hard boundary this module respects
-----------------------------------
- Never imports `torch`. Never loads model weights into memory, CPU or
  GPU. Tokenizer measurement (`parent_tournament.tokenizer_evidence`) is
  the only "model-adjacent" step this module ever triggers, and it is
  strictly a text-only, CPU tokenizer load -- the same call the real
  tournament already makes before any model load, reused unchanged.
- Never touches `Chowder-Protected` in any way.
- Every pin (`PARENT_C_PIN`, `PARENT_D_PIN`) is the exact revision recorded
  in `docs/QWEN38_SPARSE_PROGRAM.md`'s pinned-revision table and
  `qwen38_campaign.default_qwen38_campaign_manifest`'s comparison
  parents -- reusing `qwen38_campaign.ParentPin`'s own validation (a full
  40-character commit sha; a branch name or `main`/`latest` raises at
  construction) rather than re-implementing pin validation here.

Reused, not reinvented
------------------------
- `local_model_manifest.py` for the manifest/hash/verify machinery --
  identical provenance standard to parents A and B.
- `parameter_accounting.account_parameters` for shard/parameter/config
  metadata, from real safetensors headers, stdlib-only.
- `parent_tournament.tokenizer_evidence` /
  `parent_eval.ensure_parent_tokenizer_compatible` for tokenizer identity
  and the tokenizer-comparability gate against the protected evaluation
  protocol -- the exact functions the real tournament runs before any
  model load.
- `hf_resilience.with_hub_retries` for transient-vs-permanent Hub error
  classification around the (injected) download call.

Fail-closed conditions
------------------------
- a pin naming a branch/`main`/`latest` (raised by `ParentPin` itself,
  not by this module -- reuse, not a parallel check);
- a remote listing containing no `.safetensors` files (GGUF-only
  artifact -- "the GGUF is not the training parent", the exact rule
  `docs/QWEN38_SPARSE_PROGRAM.md` already resolved for Comparison C);
- no candidate destination volume has the conservatively-padded required
  free space;
- any manifest-listed file missing or size-drifted after acquisition
  (`local_model_manifest.verify_local_model_manifest`);
- a destination whose existing manifest does not match what a fresh full
  verification finds (manifest divergence) -- re-acquisition is refused,
  not silently accepted;
- a partially-populated destination directory with no valid manifest
  (a prior download that never completed) is detected and reported, not
  treated as already-acquired.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hf_resilience import with_hub_retries
from .local_model_manifest import (
    LocalModelManifest,
    LocalModelManifestError,
    build_local_model_manifest,
    verify_local_model_manifest,
    write_manifest_file,
)
from .parameter_accounting import ParameterAccountingError, account_parameters
from .parent_eval import ParentTokenizerEvidence, ParentTokenizerMismatch, ensure_parent_tokenizer_compatible
from .qwen38_campaign import ParentPin

#: The program's two not-yet-acquired comparison parents
#: (docs/QWEN38_SPARSE_PROGRAM.md's pinned-revision table; identical
#: values to qwen38_campaign.default_qwen38_campaign_manifest's
#: comparison_parents -- restated here as the acquisition module's own
#: named constants rather than importing a full campaign manifest, which
#: additionally requires a non-empty repair policy this module has no
#: reason to know about).
PARENT_C_PIN = ParentPin(
    repo="OBLITERATUS/Qwen3.8-27B-OBLITERATED",
    revision="a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8",
    role="comparison",
)
PARENT_D_PIN = ParentPin(
    repo=(
        "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-"
        "Heretic-Uncensored-NM-DAU"
    ),
    revision="81c73940f94023f7d64e3ae6abcc653fc837d415",
    role="comparison",
)

#: Parents A and B, already cached locally, restated here (same pins as
#: qwen38_campaign.default_qwen38_campaign_manifest's native_control/
#: primary_parent) so a Kaggle acquisition of *any* of the four parents --
#: including re-acquiring A for the Phase 3 equivalence qualification
#: run -- can use this module's one acquisition path instead of a second,
#: Kaggle-specific pin table.
PARENT_A_PIN = ParentPin(
    repo="Qwen/Qwen3.8-27B",
    revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    role="control",
)
PARENT_B_PIN = ParentPin(
    repo="orcarouter/Qwen3.8-27B-Uncensored",
    revision="404ea47aaa5d8a8b00049c9e9750089aca011ab2",
    role="primary",
)

#: Every pinned parent, keyed by the program's A/B/C/D role letters.
ALL_PARENT_PINS: dict[str, ParentPin] = {
    "A": PARENT_A_PIN,
    "B": PARENT_B_PIN,
    "C": PARENT_C_PIN,
    "D": PARENT_D_PIN,
}

#: Tournament labels (== `parent_tournament.LocalParent.label` /
#: `ParentEvalReport.base_model`) for every role. A and B match the exact
#: strings `parent_tournament.parent_a()`/`parent_b()` already use --
#: C and D have no equivalent factory yet in that off-limits file (only A
#: and B are defined there today), so this module names them here as its
#: own explicit, citable convention rather than leaving every caller to
#: invent one independently. If `parent_tournament.py` later grows
#: `parent_c()`/`parent_d()` factories, whoever adds them should match
#: these exact strings so existing evidence keyed on them stays joinable.
PARENT_LABELS: dict[str, str] = {
    "A": "parent-a-qwen38-27b-official",
    "B": "parent-b-orcarouter-uncensored",
    "C": "parent-c-obliteratus",
    "D": "parent-d-davidau-turbo-fable",
}

#: Conservative padding over the raw remote file total: local staging
#: files during a Hub download, filesystem block-size rounding, and a
#: safety margin against an under-reported remote size.
DISK_HEADROOM_FACTOR = 1.15

_WEIGHT_SUFFIX = ".safetensors"
_GGUF_SUFFIX = ".gguf"


class AcquisitionError(RuntimeError):
    """The C/D acquisition cannot proceed honestly; fail closed."""


@dataclass(frozen=True)
class RemoteFileInfo:
    """One file the Hub reports for a pinned revision."""

    path: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("RemoteFileInfo.path must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("RemoteFileInfo.size_bytes must be a non-negative int")


@dataclass(frozen=True)
class RemoteRepoListing:
    """A pinned revision's remote file listing -- metadata only, no bytes fetched."""

    pin: ParentPin
    files: tuple[RemoteFileInfo, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise AcquisitionError(
                f"{self.pin.repo}@{self.pin.revision} reports zero files; refusing to "
                "plan an acquisition against an empty or unreadable revision"
            )

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @property
    def safetensors_files(self) -> tuple[RemoteFileInfo, ...]:
        return tuple(f for f in self.files if f.path.endswith(_WEIGHT_SUFFIX))

    @property
    def gguf_files(self) -> tuple[RemoteFileInfo, ...]:
        return tuple(f for f in self.files if f.path.endswith(_GGUF_SUFFIX))


ListRemoteFilesFn = Callable[[str, str], Sequence[RemoteFileInfo]]
SnapshotDownloadFn = Callable[..., str]


def fetch_remote_listing(pin: ParentPin, *, list_files_fn: ListRemoteFilesFn) -> RemoteRepoListing:
    """Metadata-only Hub listing for `pin`. Performs no network access
    itself -- `list_files_fn(repo, revision)` is the caller's real
    (or, in tests, synthetic) Hub call. Wrapped in `with_hub_retries` so
    a transient Hub failure while merely *listing* files does not need a
    second retry layer built on top."""
    files = with_hub_retries(
        lambda: list_files_fn(pin.repo, pin.revision),
        label=f"list remote files for {pin.repo}@{pin.revision[:12]}",
    )
    return RemoteRepoListing(pin=pin, files=tuple(files))


def refuse_gguf_only(listing: RemoteRepoListing) -> None:
    """Fail closed on a revision that carries no real Safetensors/Transformers
    checkpoint -- the exact "GGUF is not the training parent" rule
    `docs/QWEN38_SPARSE_PROGRAM.md` already resolved once for Comparison C."""
    if listing.safetensors_files:
        return
    if listing.gguf_files:
        raise AcquisitionError(
            f"{listing.pin.repo}@{listing.pin.revision} has GGUF files but no "
            ".safetensors files; the GGUF is not the training parent -- resolve "
            "the GGUF card's `base_model` metadata to the real Transformers/"
            "Safetensors repo and pin that instead (see docs/QWEN38_SPARSE_PROGRAM.md, "
            "'Comparison C resolution record')"
        )
    raise AcquisitionError(
        f"{listing.pin.repo}@{listing.pin.revision} has no .safetensors files and no "
        "GGUF files either; this is not a checkpoint this program's lineage policy "
        "can train from"
    )


def preflight_disk_capacity(
    listing: RemoteRepoListing, candidate_roots: Sequence[tuple[Path, int]]
) -> Path:
    """Pick the first candidate root with enough conservatively-padded
    free space, or raise. `candidate_roots` is `(path, free_bytes)` pairs
    the caller has already measured (e.g. via `shutil.disk_usage`) -- kept
    a pure function so tests never touch real disk state."""
    required = int(listing.total_bytes * DISK_HEADROOM_FACTOR)
    for root, free_bytes in candidate_roots:
        if free_bytes >= required:
            return root
    raise AcquisitionError(
        f"no candidate destination volume has >= {required:,} bytes "
        f"({required / 2**30:.1f} GiB, {DISK_HEADROOM_FACTOR}x the remote total) free; "
        f"checked: {[(str(root), free) for root, free in candidate_roots]}"
    )


def _manifest_path_for(destination: Path) -> Path:
    """The same `<model_dir>.manifest.json` convention parents A/B use
    (`LOCAL_MODELS.md`; e.g. `Qwen3.8-27B.manifest.json` beside
    `Qwen3.8-27B/`), so tooling that reads A/B's manifests works
    unchanged for C/D."""
    return destination.parent / f"{destination.name}.manifest.json"


def _load_existing_manifest(destination: Path) -> LocalModelManifest | None:
    manifest_path = _manifest_path_for(destination)
    if not manifest_path.is_file():
        return None
    try:
        return LocalModelManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (LocalModelManifestError, json.JSONDecodeError, KeyError, TypeError):
        return None


def check_already_acquired(destination: Path) -> LocalModelManifest | None:
    """Restart-safety / idempotency check.

    Returns the verified manifest when `destination` already holds a
    complete, byte-clean acquisition matching its own recorded manifest.
    Returns `None` for: no manifest yet, a manifest that no longer
    verifies clean (partial download, truncation, or content drift), or
    a missing directory -- every one of those means "acquire (again)",
    never "assume it's fine".
    """
    if not destination.is_dir():
        return None
    existing = _load_existing_manifest(destination)
    if existing is None:
        return None
    verification = verify_local_model_manifest(existing, destination, rehash_weights=True)
    if not verification.clean:
        return None
    return existing


def verify_expected_files_present(destination: Path, expected_relative_paths: Sequence[str]) -> None:
    """Fail closed when acquisition left the destination incomplete.

    `expected_relative_paths` should be every *non-GGUF* file the remote
    listing reported (weights, config, tokenizer assets) -- this module
    never expects a GGUF sibling to have been downloaded.
    """
    missing = [path for path in expected_relative_paths if not (destination / path).is_file()]
    if missing:
        raise AcquisitionError(
            f"acquisition incomplete at {destination}: {len(missing)} expected file(s) "
            f"missing (first few: {missing[:5]})"
        )


def inspect_architecture_metadata(destination: Path) -> dict[str, Any]:
    """Stdlib-only `config.json` read -- no torch, no safetensors import.

    Mirrors the architecture table `docs/QWEN38_SPARSE_PROGRAM.md` already
    records for A/B/C/D from real `config.json` reads (architecture,
    model_type, nested text_config, MTP/vision tensor presence is left to
    `parameter_accounting.account_parameters`, which reads tensor names
    directly from safetensors headers rather than guessing from config).
    """
    config_path = destination / "config.json"
    if not config_path.is_file():
        raise AcquisitionError(f"no config.json at {destination}; cannot inspect architecture metadata")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "has_nested_text_config": isinstance(text_config, Mapping),
        "text_config_model_type": text_config.get("model_type") if isinstance(text_config, Mapping) else None,
        "num_hidden_layers": (text_config or config).get("num_hidden_layers"),
        "hidden_size": (text_config or config).get("hidden_size"),
        "mtp_num_hidden_layers": config.get("mtp_num_hidden_layers"),
    }


@dataclass(frozen=True)
class TokenizerGateResult:
    """Outcome of gating a newly acquired parent's tokenizer against a
    reference (A's, by convention). A mismatch is recorded, not raised --
    D's tokenizer class is already known to differ
    (`docs/QWEN38_SPARSE_PROGRAM.md`), and that is expected, informative
    evidence for later cross-parent score comparability, not a reason to
    fail the acquisition of the checkpoint itself."""

    compatible: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"compatible": self.compatible, "detail": self.detail}


def gate_tokenizer_compatibility(
    reference: ParentTokenizerEvidence, candidate: ParentTokenizerEvidence
) -> TokenizerGateResult:
    """Reuses `parent_eval.ensure_parent_tokenizer_compatible` unchanged --
    this module adds no fuzzy-matching logic of its own."""
    try:
        ensure_parent_tokenizer_compatible(reference, candidate)
    except ParentTokenizerMismatch as exc:
        return TokenizerGateResult(compatible=False, detail=str(exc))
    return TokenizerGateResult(compatible=True, detail="tokenizer identity matches the reference")


@dataclass(frozen=True)
class AcquisitionResult:
    pin: ParentPin
    destination: str
    manifest: LocalModelManifest
    already_present: bool
    architecture: Mapping[str, Any]
    parameter_accounting: Mapping[str, Any] | None
    tokenizer: ParentTokenizerEvidence | None
    tokenizer_gate: TokenizerGateResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pin": self.pin.to_dict(),
            "destination": self.destination,
            "manifest": self.manifest.to_dict(),
            "already_present": self.already_present,
            "architecture": dict(self.architecture),
            "parameter_accounting": dict(self.parameter_accounting) if self.parameter_accounting else None,
            "tokenizer": (
                {
                    "tokenizer_class": self.tokenizer.tokenizer_class,
                    "vocab_size": self.tokenizer.vocab_size,
                    "identity_sha256": self.tokenizer.identity_sha256,
                }
                if self.tokenizer is not None
                else None
            ),
            "tokenizer_gate": self.tokenizer_gate.to_dict() if self.tokenizer_gate is not None else None,
        }


def acquire_parent(
    pin: ParentPin,
    destination: Path,
    *,
    list_files_fn: ListRemoteFilesFn,
    snapshot_download_fn: SnapshotDownloadFn,
    candidate_roots: Sequence[tuple[Path, int]] | None = None,
    measure_tokenizer_fn: Callable[[Path], ParentTokenizerEvidence] | None = None,
    reference_tokenizer: ParentTokenizerEvidence | None = None,
) -> AcquisitionResult:
    """Acquire (or confirm already-acquired) one pinned parent.

    Restart-safe: if `destination` already verifies clean against its own
    manifest, `snapshot_download_fn` is never called and
    `already_present=True` is returned. Otherwise: lists the remote
    revision (metadata only), refuses a GGUF-only revision, optionally
    checks disk headroom against `candidate_roots`, downloads via the
    injected `snapshot_download_fn`, verifies every expected file is
    present, builds and writes a full-mode manifest (identical provenance
    standard to parents A/B), inspects config-level architecture metadata
    and (when safetensors headers are readable) Phase 11 parameter
    accounting, and -- when `measure_tokenizer_fn`/`reference_tokenizer`
    are supplied -- measures and gates tokenizer identity.

    Never imports torch. Never loads model weights.
    """
    existing = check_already_acquired(destination)
    if existing is not None:
        return _finish_result(pin, destination, existing, already_present=True,
                               measure_tokenizer_fn=measure_tokenizer_fn,
                               reference_tokenizer=reference_tokenizer)

    listing = fetch_remote_listing(pin, list_files_fn=list_files_fn)
    refuse_gguf_only(listing)
    if candidate_roots is not None:
        preflight_disk_capacity(listing, candidate_roots)

    with_hub_retries(
        lambda: snapshot_download_fn(repo_id=pin.repo, revision=pin.revision, local_dir=str(destination)),
        label=f"acquire {pin.repo}@{pin.revision[:12]}",
    )

    expected_files = [f.path for f in listing.files if not f.path.endswith(_GGUF_SUFFIX)]
    verify_expected_files_present(destination, expected_files)

    manifest = build_local_model_manifest(destination, mode="full")
    write_manifest_file(manifest, _manifest_path_for(destination))

    return _finish_result(pin, destination, manifest, already_present=False,
                           measure_tokenizer_fn=measure_tokenizer_fn,
                           reference_tokenizer=reference_tokenizer)


def _finish_result(
    pin: ParentPin,
    destination: Path,
    manifest: LocalModelManifest,
    *,
    already_present: bool,
    measure_tokenizer_fn: Callable[[Path], ParentTokenizerEvidence] | None,
    reference_tokenizer: ParentTokenizerEvidence | None,
) -> AcquisitionResult:
    architecture = inspect_architecture_metadata(destination)
    try:
        accounting = account_parameters(destination).to_dict()
    except ParameterAccountingError:
        accounting = None

    tokenizer: ParentTokenizerEvidence | None = None
    tokenizer_gate: TokenizerGateResult | None = None
    if measure_tokenizer_fn is not None:
        tokenizer = measure_tokenizer_fn(destination)
        if reference_tokenizer is not None:
            tokenizer_gate = gate_tokenizer_compatibility(reference_tokenizer, tokenizer)

    return AcquisitionResult(
        pin=pin,
        destination=str(destination),
        manifest=manifest,
        already_present=already_present,
        architecture=architecture,
        parameter_accounting=accounting,
        tokenizer=tokenizer,
        tokenizer_gate=tokenizer_gate,
    )
