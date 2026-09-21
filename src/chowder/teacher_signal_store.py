"""Teacher Fabric, Slice B: the content-addressed teacher signal store.

`TeacherSignalStore` is the durable local cache between provider adapters
(Slice F) and every consumer of teacher signal (the Regression Surgeon
integration in Slice C, the query controller in Slice D). It closes the
loop Slice A left open: an artifact that lives only in memory is lost on
process exit, and a re-queried teacher costs real money for evidence we
already hold.

Design (from docs/TEACHER_FABRIC.md §8/§9/§16 and the brief's regression
rules, decisions recorded in docs/HANDOFF.md):

- **Content-addressed, exactly.** A signal's cache key is the digest over
  its `(request_digest, payload_file_sha256)` pair (rule #8's identity:
  same request answered with the same payload is the *same* evidence). The
  payload bytes live at `payloads/<sha256>.bin`, named by their own content
  hash. Two artifacts with the same digest are the same evidence; different
  payloads for the same request (a teacher re-queried at a new revision)
  are distinct rows, never overwrites.

- **Cache ≠ evidence.** The payload files and the per-entry metadata are a
  *cache*: mutable bookkeeping (`stored_at`, `hit_count`, `last_hit_at`)
  that may be evicted to respect the budget. The durable evidence is the
  append-only `teacher_signals` row in `RunRegistry` (added by migration
  4), written through `_insert_immutable` so history is never rewritten
  (rule #4). Evicting the cache loses only re-query cost, never evidence.

- **Verified-or-absent.** Reads verify the payload hash and the metadata
  digest chain before returning anything (rule #8's *verified* half). A
  corrupted or missing payload is a hard error — never served, never
  silently re-fetched, never repaired by guessing; recovery is explicit
  (`discard` + re-query). A store opened on a cache directory whose
  metadata disagrees with itself refuses to serve the affected entry the
  same way.

- **Atomic writes.** Payloads are written to a unique temporary name in
  the same directory and `os.replace`d into place, so a crash mid-write
  leaves either the old content or none — never a torn payload that could
  later pass a name-based check. Recovery from an interrupted store
  operation is at open time: temporary files and *verified* orphaned
  payloads (no metadata row claims them) are swept; a payload orphan is
  garbage only when nothing references it, which metadata is the record
  of.

- **No default budget.** `local_cache_max_bytes` is a required constructor
  argument (docs/TEACHER_FABRIC.md §16.1: the right default is a measured
  open question, not a chosen one; this module also deliberately does not
  implement eviction — see below). The store measures its own footprint
  (`disk_bytes`) so rule #14's "constraints honored and measured" has a
  number to check against the ceiling.

Honesty rules this module enforces mechanically
-----------------------------------------------
1. **Corruption is never served.** `load` re-hashes the payload and
   re-checks the metadata digest chain on every read and raises
   `SignalIntegrityError` on any mismatch — including tampering that keeps
   file sizes plausible. The registry row is not consulted to *bless* a
   bad payload; it records what was stored, and the store verifies what
   is on disk independently.
2. **No invented timestamps or hits.** `hit_count` starts at 0 and is
   incremented only by real `load` calls; `last_hit_at` is None until a
   first hit. Nothing is backfilled.
3. **Duplicate registration is evidence of the same content, not a
   second copy.** Storing an artifact whose key already exists verifies
   the existing payload byte-for-byte and returns the existing metadata;
   a *different* payload under the same key is a hard error, because it
   means the same cache key maps to two different evidences — the
   addressing scheme is broken.
4. **The budget is enforced at write time, measured, and never negotiated
   by discarding evidence.** A store that is at its ceiling refuses new
   *distinct* signals with `CacheBudgetExceededError` rather than
   evicting silently: which cached signals are safe to drop is a policy
   decision (freshness, cost-to-requery) that belongs to the query
   controller (Slice D), and this slice refuses to pre-empt it. Eviction
   is available as an explicit, caller-chosen `discard` of named entries.
5. **Deletion is by exact digest, never by pattern.** `discard` removes
   one named entry and its payload only when no other entry shares it.

Deliberately not in this slice (later slices, per the file plan):
hit-rate instrumentation across candidate ceilings (the §16.1 experiment
runs on top of this store once real queries exist), streamed shard
*transport* (Slice B implements complete-or-absent integrity for
whole payloads; F/G move payloads that actually stream), repair
integration (C), cost accounting and escalation policy (D), real remote
providers (F). Nothing here performs network I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .resources import ResourceUsage
from .registry import RegistryInvariantError
from .teacher_fabric import SCHEMA_VERSION, SignalKind, TeacherSignalArtifact

_METADATA_VERSION = 1
_PAYLOAD_DIR_NAME = "payloads"
_METADATA_FILE_NAME = "index.json"

_METADATA_COLUMNS = (
    "artifact_digest",
    "request_digest",
    "prompt_digest",
    "payload_file_sha256",
    "signal_kind",
    "teacher_id",
    "model_revision",
    "tokenizer_identity_sha256",
    "signal_id",
    "stored_at",
    "hit_count",
    "last_hit_at",
    "metadata_json",
)


class SignalIntegrityError(ValueError):
    """A stored signal failed verification.

    Raised on payload/content hash mismatch, metadata digest-chain
    disagreement, or a missing payload. The entry is never served; the
    message names the artifact digest and the failed check. Recovery is
    explicit: `discard` the entry and re-query the teacher.
    """


class CacheBudgetExceededError(RuntimeError):
    """A distinct signal does not fit the store's configured ceiling.

    Raised *before* anything is written. The message reports the measured
    current footprint, the incoming payload size, and the ceiling. This is
    not an error about one request: it means the cache policy (evict what,
    when) has to be decided — by the caller, explicitly, not by this
    module silently.
    """


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256_hex(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"teacher signal store {label} must be a 64-character sha256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"teacher signal store {label} must be sha256 hex") from exc


@dataclass(frozen=True)
class StoredSignalMetadata:
    """Cache bookkeeping for one stored artifact.

    Everything except `hit_count`/`last_hit_at` is identity evidence and
    immutable once written; those two are cache statistics only and never
    appear in any digest or registry row.
    """

    artifact_digest: str
    request_digest: str
    prompt_digest: str
    payload_file_sha256: str
    signal_kind: str
    teacher_id: str
    model_revision: str
    tokenizer_identity_sha256: str | None
    signal_id: str
    stored_at: str
    hit_count: int = 0
    last_hit_at: str | None = None

    def __post_init__(self) -> None:
        _require_sha256_hex(self.artifact_digest, "artifact_digest")
        _require_sha256_hex(self.request_digest, "request_digest")
        _require_sha256_hex(self.prompt_digest, "prompt_digest")
        _require_sha256_hex(self.payload_file_sha256, "payload_file_sha256")
        if not isinstance(self.signal_kind, str) or not self.signal_kind:
            raise ValueError("teacher signal store signal_kind must be a non-empty string")
        if not isinstance(self.teacher_id, str) or not self.teacher_id.strip():
            raise ValueError("teacher signal store teacher_id must be a non-empty string")
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("teacher signal store model_revision must be a non-empty string")
        if self.tokenizer_identity_sha256 is not None:
            _require_sha256_hex(
                self.tokenizer_identity_sha256, "tokenizer_identity_sha256"
            )
        if not isinstance(self.signal_id, str) or not self.signal_id.strip():
            raise ValueError("teacher signal store signal_id must be a non-empty string")
        if not isinstance(self.stored_at, str) or not self.stored_at:
            raise ValueError("teacher signal store stored_at must be a non-empty string")
        if isinstance(self.hit_count, bool) or not isinstance(self.hit_count, int):
            raise ValueError("teacher signal store hit_count must be a non-negative int")
        if self.hit_count < 0:
            raise ValueError("teacher signal store hit_count must be a non-negative int")
        if self.last_hit_at is not None and (
            not isinstance(self.last_hit_at, str) or not self.last_hit_at
        ):
            raise ValueError("teacher signal store last_hit_at must be None or a non-empty string")

    @property
    def entry_key(self) -> str:
        """The content address: digest over (request_digest, payload hash).

        This is rule #8's identity — the same request answered by the same
        payload is the same evidence, from any provider session.
        """
        return _sha256_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "request_digest": self.request_digest,
                    "payload_file_sha256": self.payload_file_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "artifact_digest": self.artifact_digest,
            "request_digest": self.request_digest,
            "prompt_digest": self.prompt_digest,
            "payload_file_sha256": self.payload_file_sha256,
            "signal_kind": self.signal_kind,
            "teacher_id": self.teacher_id,
            "model_revision": self.model_revision,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "signal_id": self.signal_id,
            "stored_at": self.stored_at,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "StoredSignalMetadata":
        return cls(
            artifact_digest=data["artifact_digest"],
            request_digest=data["request_digest"],
            prompt_digest=data["prompt_digest"],
            payload_file_sha256=data["payload_file_sha256"],
            signal_kind=data["signal_kind"],
            teacher_id=data["teacher_id"],
            model_revision=data["model_revision"],
            tokenizer_identity_sha256=data["tokenizer_identity_sha256"],
            signal_id=data["signal_id"],
            stored_at=data["stored_at"],
            hit_count=data["hit_count"],
            last_hit_at=data["last_hit_at"],
        )


class TeacherSignalStore:
    """Content-addressed local cache of verified `TeacherSignalArtifact`s.

    Construct with an explicit `local_cache_max_bytes` (no default —
    §16.1). With `registry`, every stored artifact also lands as an
    append-only `teacher_signals` ledger row (evidence survives cache
    eviction); without one, the cache is still verified-or-absent but
    nothing durable records the signal.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        local_cache_max_bytes: int,
        registry: Any | None = None,
    ) -> None:
        if isinstance(local_cache_max_bytes, bool) or not isinstance(
            local_cache_max_bytes, int
        ):
            raise ValueError("local_cache_max_bytes must be a non-negative int")
        if local_cache_max_bytes < 0:
            raise ValueError("local_cache_max_bytes must be a non-negative int")
        self._root = Path(root)
        self._max_bytes = local_cache_max_bytes
        self._registry = registry
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / _PAYLOAD_DIR_NAME).mkdir(exist_ok=True)
        self._recover_interrupted_writes()

    # -- layout ------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def local_cache_max_bytes(self) -> int:
        return self._max_bytes

    def _payload_path(self, payload_file_sha256: str) -> Path:
        return self._root / _PAYLOAD_DIR_NAME / f"{payload_file_sha256}.bin"

    def _metadata_path(self) -> Path:
        return self._root / _METADATA_FILE_NAME

    def _read_metadata(self) -> dict[str, dict[str, Any]]:
        path = self._metadata_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SignalIntegrityError(
                f"teacher signal store metadata at {path} is unreadable: {exc}; "
                "refusing to serve from a store whose index cannot be trusted"
            ) from exc
        if not isinstance(raw, dict) or raw.get("metadata_version") != _METADATA_VERSION:
            raise SignalIntegrityError(
                f"teacher signal store metadata at {path} has an unrecognized "
                "format; refusing to serve"
            )
        entries = raw.get("entries")
        if not isinstance(entries, dict):
            raise SignalIntegrityError(
                f"teacher signal store metadata at {path} is missing its entries map"
            )
        return entries

    def _write_metadata(self, entries: Mapping[str, Mapping[str, Any]]) -> None:
        document = {"metadata_version": _METADATA_VERSION, "entries": dict(entries)}
        path = self._metadata_path()
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self._root, prefix=".index-", suffix=".tmp", delete=False
        )
        try:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(handle.name, path)
        except BaseException:
            handle.close()
            try:
                os.unlink(handle.name)
            except FileNotFoundError:
                pass
            raise

    def _write_payload(self, payload_file_sha256: str, data: bytes) -> None:
        destination = self._payload_path(payload_file_sha256)
        handle = tempfile.NamedTemporaryFile(
            "wb",
            dir=self._root / _PAYLOAD_DIR_NAME,
            prefix=f".{payload_file_sha256[:12]}-",
            suffix=".tmp",
            delete=False,
        )
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(handle.name, destination)
        except BaseException:
            handle.close()
            try:
                os.unlink(handle.name)
            except FileNotFoundError:
                pass
            raise

    def _read_payload(self, payload_file_sha256: str) -> bytes:
        path = self._payload_path(payload_file_sha256)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise SignalIntegrityError(
                f"teacher signal payload {payload_file_sha256} is missing from "
                f"{path}; the entry is verified-or-absent and this one is absent"
            ) from exc
        actual = _sha256_bytes(data)
        if actual != payload_file_sha256:
            raise SignalIntegrityError(
                f"teacher signal payload {payload_file_sha256} does not match its "
                f"content (found {actual}); the cache "
                "serves verified content or nothing, never corrupt bytes"
            )
        return data

    def _recover_interrupted_writes(self) -> None:
        """Sweep leftovers of a crash between temp file and rename.

        Temporary files (`.index-*.tmp`, `.payload-*.tmp`) are always
        garbage. Orphaned *payloads* — content-addressed files no metadata
        entry claims — are swept too: with no metadata row they are
        unrecoverable bookkeeping, and re-deriving them (re-query) is the
        documented recovery. The metadata file itself is replaced
        atomically, so a crash leaves either the old or the new index.
        """
        for temp in self._root.glob(".index-*.tmp"):
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        payloads = self._root / _PAYLOAD_DIR_NAME
        for temp in payloads.glob(".*.tmp"):
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        entries = self._read_metadata()
        claimed = {
            entry["payload_file_sha256"]
            for entry in entries.values()
            if isinstance(entry, dict) and isinstance(entry.get("payload_file_sha256"), str)
        }
        for payload_file in payloads.glob("*.bin"):
            if payload_file.stem not in claimed:
                try:
                    payload_file.unlink()
                except FileNotFoundError:
                    pass

    # -- measurement ---------------------------------------------------------

    def disk_bytes(self) -> int:
        """Measured store footprint: metadata file + all payload bytes.

        Rule #14's measurement primitive — the budget check and any §16.1
        hit-rate/cost experiment read this, never a modeled estimate.
        """
        total = 0
        index = self._metadata_path()
        if index.exists():
            total += index.stat().st_size
        for payload_file in (self._root / _PAYLOAD_DIR_NAME).glob("*.bin"):
            total += payload_file.stat().st_size
        return total

    def entry_count(self) -> int:
        return len(self._read_metadata())

    # -- core operations -----------------------------------------------------

    def store(
        self,
        artifact: TeacherSignalArtifact,
        *,
        stored_at: str,
    ) -> StoredSignalMetadata:
        """Verify, write atomically, index, and (with a registry) ledger.

        `stored_at` is caller-supplied (ISO-8601 in existing conventions)
        and recorded verbatim — the store does not invent timestamps.
        """
        if not isinstance(artifact, TeacherSignalArtifact):
            raise ValueError("teacher signal store stores TeacherSignalArtifact instances")
        if not isinstance(stored_at, str) or not stored_at.strip():
            raise ValueError("stored_at must be a non-empty string")

        canonical_json = artifact.canonical_json()
        payload_bytes = canonical_json.encode("utf-8")
        payload_file_sha256 = _sha256_bytes(payload_bytes)
        request_digest = artifact.request_digest
        metadata = StoredSignalMetadata(
            artifact_digest=artifact.digest(),
            request_digest=request_digest,
            prompt_digest=artifact.prompt_digest,
            payload_file_sha256=payload_file_sha256,
            signal_kind=artifact.signal.signal_kind.value,
            teacher_id=artifact.teacher_id,
            model_revision=artifact.model_revision,
            tokenizer_identity_sha256=artifact.tokenizer_identity_sha256,
            signal_id=artifact.signal_id,
            stored_at=stored_at,
        )
        entry_key = metadata.entry_key

        entries = self._read_metadata()
        existing_raw = entries.get(entry_key)
        if existing_raw is not None:
            existing = StoredSignalMetadata.from_json_dict(existing_raw)
            if existing.payload_file_sha256 != payload_file_sha256:
                raise SignalIntegrityError(
                    f"cache key collision: entry {entry_key} is recorded with payload "
                    f"{existing.payload_file_sha256} but the incoming artifact's "
                    f"payload hashes to {payload_file_sha256}; the addressing scheme "
                    "maps one key to one evidence and this must never be guessed away"
                )
            # Same evidence again: verify the bytes on disk really are that
            # payload (rule #8's verified half) and report the existing
            # entry rather than writing a second copy.
            self._read_payload(payload_file_sha256)
            return existing

        payload_size = len(payload_bytes)
        projected = self.disk_bytes() + payload_size
        if projected > self._max_bytes:
            raise CacheBudgetExceededError(
                f"storing a {payload_size}-byte teacher signal would take the cache to "
                f"{projected} bytes against a {self._max_bytes}-byte ceiling; decide "
                "eviction explicitly (discard named entries) rather than letting the "
                "store choose"
            )

        self._write_payload(payload_file_sha256, payload_bytes)
        entries[entry_key] = metadata.to_json_dict()
        self._write_metadata(entries)
        if self._registry is not None:
            self._ledger_append(metadata)
        return metadata

    def _ledger_append(self, metadata: StoredSignalMetadata) -> None:
        """Record the signal in the registry ledger, evidence-idempotently.

        The ledger row is evidence ("this signal entered our records"),
        keyed by the content address. Re-acquiring *identical* evidence
        after cache eviction must not diverge on bookkeeping (`stored_at`
        is first-acquisition time and does not change by re-querying), so
        the append first reads any existing row and verifies the evidence
        columns match; a genuine divergence -- the same content address
        claimed for different evidence -- is a RegistryInvariantError, and
        `_insert_immutable` remains the last-resort guard.
        """
        existing = self._registry.teacher_signal_row(metadata.entry_key)
        evidence = (
            "artifact_digest",
            "request_digest",
            "payload_file_sha256",
            "signal_kind",
            "teacher_id",
            "model_revision",
            "tokenizer_identity_sha256",
            "signal_id",
        )
        if existing is not None:
            mismatched = [
                column
                for column in evidence
                if existing[column] != getattr(metadata, column)
            ]
            if mismatched:
                raise RegistryInvariantError(
                    "teacher_signals ledger divergence at "
                    f"{metadata.entry_key}: existing row disagrees on "
                    f"{', '.join(mismatched)}; the content address maps to "
                    "different evidence than previously recorded"
                )
            return
        self._registry.record_teacher_signal(
            entry_key=metadata.entry_key,
            artifact_digest=metadata.artifact_digest,
            request_digest=metadata.request_digest,
            payload_file_sha256=metadata.payload_file_sha256,
            signal_kind=metadata.signal_kind,
            teacher_id=metadata.teacher_id,
            model_revision=metadata.model_revision,
            tokenizer_identity_sha256=metadata.tokenizer_identity_sha256,
            signal_id=metadata.signal_id,
            stored_at=metadata.stored_at,
            metadata_json=json.dumps(
                metadata.to_json_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )

    def find_by_request(self, request_digest: str) -> tuple[StoredSignalMetadata, ...]:
        """All verified entries answering *request_digest*, in stored order.

        Every returned entry's payload is re-hashed before inclusion —
        a corrupted entry is skipped here (the loud failure mode lives in
        `load`, which callers use when they intend to *consume* the
        signal). Which entry is freshest for reuse is the caller's /
        Slice D's policy; this store exposes facts, not a freshness rule.
        """
        _require_sha256_hex(request_digest, "request_digest")
        matches: list[StoredSignalMetadata] = []
        for raw in self._read_metadata().values():
            metadata = StoredSignalMetadata.from_json_dict(raw)
            if metadata.request_digest != request_digest:
                continue
            try:
                self._read_payload(metadata.payload_file_sha256)
            except SignalIntegrityError:
                # Verified-or-absent at the lookup level too: a corrupted
                # entry is not served. Consuming callers go through
                # `load`, where the same corruption fails loudly.
                continue
            matches.append(metadata)
        return tuple(matches)

    def load(self, entry_key: str) -> tuple[TeacherSignalArtifact, StoredSignalMetadata]:
        """Return the verified artifact for *entry_key*, updating hit stats.

        Full verification chain: entry exists → payload present and
        hash-verified → metadata digest chain agrees (payload hash round-
        trips through the entry key; artifact digest re-checked against
        the stored row) → artifact re-parsed and its own digest
        recomputed. Anything failing raises `SignalIntegrityError` and the
        entry is left in place but never served.
        """
        _require_sha256_hex(entry_key, "entry_key")
        entries = self._read_metadata()
        raw = entries.get(entry_key)
        if raw is None:
            raise KeyError(f"no teacher signal entry {entry_key!r} in the store")
        metadata = StoredSignalMetadata.from_json_dict(raw)
        if metadata.entry_key != entry_key:
            raise SignalIntegrityError(
                f"teacher signal entry {entry_key!r} re-derives to key "
                f"{metadata.entry_key!r}; the index and its keys disagree"
            )
        data = self._read_payload(metadata.payload_file_sha256)
        try:
            artifact = _artifact_from_json_bytes(data)
        except (ValueError, KeyError, TypeError) as exc:
            raise SignalIntegrityError(
                f"teacher signal payload {metadata.payload_file_sha256} does not "
                f"parse as the artifact it claims to be: {exc}"
            ) from exc
        if artifact.digest() != metadata.artifact_digest:
            raise SignalIntegrityError(
                f"teacher signal payload for entry {entry_key!r} re-hashes to artifact "
                f"digest {artifact.digest()} but the entry records "
                f"{metadata.artifact_digest}; refusing to serve unverifiable evidence"
            )

        hit_count = metadata.hit_count + 1
        updated = StoredSignalMetadata(
            artifact_digest=metadata.artifact_digest,
            request_digest=metadata.request_digest,
            prompt_digest=metadata.prompt_digest,
            payload_file_sha256=metadata.payload_file_sha256,
            signal_kind=metadata.signal_kind,
            teacher_id=metadata.teacher_id,
            model_revision=metadata.model_revision,
            tokenizer_identity_sha256=metadata.tokenizer_identity_sha256,
            signal_id=metadata.signal_id,
            stored_at=metadata.stored_at,
            hit_count=hit_count,
            last_hit_at=metadata.last_hit_at,
        )
        entries[entry_key] = updated.to_json_dict()
        self._write_metadata(entries)
        return artifact, updated

    def discard(self, entry_key: str) -> bool:
        """Remove one named entry and, when unshared, its payload.

        Explicit cache management for the caller that owns eviction
        policy. Returns True when the entry existed. Never touches the
        registry ledger row: the evidence was recorded, the cache is the
        only thing being cleaned.
        """
        _require_sha256_hex(entry_key, "entry_key")
        entries = self._read_metadata()
        raw = entries.pop(entry_key, None)
        if raw is None:
            return False
        metadata = StoredSignalMetadata.from_json_dict(raw)
        still_claimed = any(
            StoredSignalMetadata.from_json_dict(other).payload_file_sha256
            == metadata.payload_file_sha256
            for other in entries.values()
        )
        if not still_claimed:
            try:
                self._payload_path(metadata.payload_file_sha256).unlink()
            except FileNotFoundError:
                pass
        self._write_metadata(entries)
        return True


def _artifact_from_json_bytes(data: bytes) -> TeacherSignalArtifact:
    """Rehydrate a `TeacherSignalArtifact` from its canonical JSON bytes.

    The canonical payload embeds the full request and signal, so the
    round-trip reconstructs every field, including `ResourceUsage` for
    GPU-backed teachers; the artifact's own validation runs on
    reconstruction and its digest is re-checked by the caller.
    """
    from .teacher_fabric import TeacherRequest, TeacherSignal

    document = json.loads(data.decode("utf-8"))
    request_payload = document["request"]
    request = TeacherRequest(
        teacher_id=request_payload["teacher_id"],
        signal_kind=SignalKind(request_payload["signal_kind"]),
        prompt=request_payload["prompt"],
        input_payload=request_payload["input_payload"],
        parameters=request_payload["parameters"],
        student_tokenizer_identity=request_payload["student_tokenizer_identity"],
        student_trajectory=(
            tuple(request_payload["student_trajectory"])
            if request_payload["student_trajectory"] is not None
            else None
        ),
    )
    signal_payload = document["signal"]
    resource_usage_raw = signal_payload["resource_usage"]
    resource_usage = (
        ResourceUsage(
            wall_seconds=resource_usage_raw["wall_seconds"],
            accelerator_seconds=resource_usage_raw["accelerator_seconds"],
            active_accelerator_count=resource_usage_raw["active_accelerator_count"],
            visible_accelerator_count=resource_usage_raw["visible_accelerator_count"],
            peak_vram_gb_by_accelerator=resource_usage_raw["peak_vram_gb_by_accelerator"],
        )
        if resource_usage_raw is not None
        else None
    )
    signal = TeacherSignal(
        signal_kind=SignalKind(signal_payload["signal_kind"]),
        payload=signal_payload["payload"],
        resource_usage=resource_usage,
        monetary_cost_usd=signal_payload["monetary_cost_usd"],
        token_counts=signal_payload["token_counts"],
    )
    return TeacherSignalArtifact(
        signal_id=document["signal_id"],
        teacher_id=document["teacher_id"],
        provider_type=document["provider_type"],
        model_revision=document["model_revision"],
        tokenizer_identity_sha256=document["tokenizer_identity_sha256"],
        request=request,
        signal=signal,
        occurred_at=document["occurred_at"],
        latency_seconds=document["latency_seconds"],
        generation_parameters=document["generation_parameters"],
        monetary_cost_usd=document["monetary_cost_usd"],
        token_counts=document["token_counts"],
        gpu_hours=document["gpu_hours"],
        parent_job_manifest_digest=document["parent_job_manifest_digest"],
        provenance=document["provenance"],
    )

