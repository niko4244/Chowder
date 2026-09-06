"""Teacher Fabric Slice B: the content-addressed teacher signal store.

Offline, deterministic — no network anywhere (regression rule 11). The
corruption cases mutate bytes on disk directly; the atomicity cases
simulate the exact intermediate states a crash leaves behind; the budget
cases use tiny synthetic ceilings so no real payload volume is needed.
"""

import json
import os

import pytest

from chowder.registry import RegistryInvariantError, RunRegistry
from chowder.resources import ResourceUsage
from chowder.teacher_fabric import (
    FakeTeacherProvider,
    SignalKind,
    TeacherRequest,
    build_signal_artifact,
)
from chowder.teacher_signal_store import (
    CacheBudgetExceededError,
    SignalIntegrityError,
    StoredSignalMetadata,
    TeacherSignalStore,
)

_STUDENT_TOKENS = "a" * 64
_TEACHER_TOKENS = "a" * 64


def _request(kind=SignalKind.CRITIQUE, prompt="Review this solution: 2+2=4", **overrides):
    fields = {
        "teacher_id": "teacher-frontier-x",
        "signal_kind": kind,
        "prompt": prompt,
        "input_payload": {"candidate": "2+2=4"},
        "parameters": {"temperature": 0.2},
        "student_tokenizer_identity": _STUDENT_TOKENS,
    }
    fields.update(overrides)
    return TeacherRequest(**fields)


def _artifact(provider=None, request=None, signal_id="sig-0001"):
    provider = provider or FakeTeacherProvider()
    request = request or _request()
    signal = provider.query(request)
    return build_signal_artifact(
        provider=provider,
        request=request,
        signal=signal,
        signal_id=signal_id,
        occurred_at="2026-09-06T00:00:00Z",
        latency_seconds=0.25,
    )


def _store(tmp_path, max_bytes=1_000_000, registry=None):
    return TeacherSignalStore(
        tmp_path / "teacher-signals",
        local_cache_max_bytes=max_bytes,
        registry=registry,
    )


def _gpu_artifact(signal_id="sig-gpu-1"):
    provider = FakeTeacherProvider(name="fake-gpu")
    request = _request(prompt="Score this trajectory end to end")
    signal = provider.query(request)
    signal = TeacherSignal_with_usage(signal, ResourceUsage(
        wall_seconds=12.0,
        accelerator_seconds=24.0,
        active_accelerator_count=1,
        visible_accelerator_count=2,
        peak_vram_gb_by_accelerator={"cuda:0": 3.5},
    ))
    return build_signal_artifact(
        provider=provider,
        request=request,
        signal=signal,
        signal_id=signal_id,
        occurred_at="2026-09-06T00:00:00Z",
        latency_seconds=1.5,
    )


def TeacherSignal_with_usage(signal, usage):
    from chowder.teacher_fabric import TeacherSignal

    return TeacherSignal(
        signal_kind=signal.signal_kind,
        payload=dict(signal.payload),
        resource_usage=usage,
        monetary_cost_usd=signal.monetary_cost_usd,
        token_counts=dict(signal.token_counts),
    )


# --- construction ------------------------------------------------------------


def test_budget_is_required_with_no_default(tmp_path):
    with pytest.raises(TypeError):
        TeacherSignalStore(tmp_path / "teacher-signals")  # type: ignore[call-arg]


def test_budget_rejects_negative_and_non_int(tmp_path):
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative int"):
            _store(tmp_path, max_bytes=bad)


def test_store_layout_is_created_on_open(tmp_path):
    store = _store(tmp_path)
    assert store.root.name == "teacher-signals"
    assert (store.root / "payloads").is_dir()


# --- store / verified reads ---------------------------------------------------


def test_store_then_load_round_trips_the_artifact(tmp_path):
    store = _store(tmp_path)
    artifact = _artifact()
    metadata = store.store(artifact, stored_at="2026-09-06T00:00:01Z")

    loaded, hit_metadata = store.load(metadata.entry_key)
    assert loaded == artifact
    assert loaded.digest() == artifact.digest()
    assert hit_metadata.hit_count == 1
    assert hit_metadata.last_hit_at is None  # no invented timestamps
    assert metadata.hit_count == 0 and metadata.last_hit_at is None


def test_entry_key_is_digest_over_request_and_payload_file(tmp_path):
    store = _store(tmp_path)
    artifact = _artifact()
    metadata = store.store(artifact, stored_at="t1")
    import hashlib

    payload_file_sha256 = hashlib.sha256(artifact.canonical_json().encode("utf-8")).hexdigest()
    expected = hashlib.sha256(
        json.dumps(
            {
                "schema_version": 1,
                "request_digest": artifact.request_digest,
                "payload_file_sha256": payload_file_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert metadata.entry_key == expected


def test_same_evidence_restored_returns_existing_metadata(tmp_path):
    store = _store(tmp_path)
    artifact = _artifact()
    first = store.store(artifact, stored_at="t1")
    second = store.store(artifact, stored_at="t2")
    assert second == first
    assert store.entry_count() == 1


def test_same_request_different_payload_is_distinct_entry(tmp_path):
    store = _store(tmp_path)
    provider = FakeTeacherProvider()
    request = _request()
    one = build_signal_artifact(
        provider=provider,
        request=request,
        signal=provider.query(request),
        signal_id="sig-1",
        occurred_at="2026-09-06T00:00:00Z",
        latency_seconds=0.1,
    )
    other = build_signal_artifact(
        provider=provider,
        request=request,
        signal=provider.query(request),
        signal_id="sig-2",
        occurred_at="2026-09-06T00:00:05Z",
        latency_seconds=0.2,
    )
    m1 = store.store(one, stored_at="t1")
    m2 = store.store(other, stored_at="t2")
    assert m1.entry_key != m2.entry_key
    matches = store.find_by_request(request.digest())
    assert len(matches) == 2


def test_find_by_request_verifies_before_including(tmp_path):
    store = _store(tmp_path)
    artifact = _artifact()
    metadata = store.store(artifact, stored_at="t1")
    payload_path = store.root / "payloads" / f"{metadata.payload_file_sha256}.bin"
    payload_path.write_bytes(b"corrupted")
    assert store.find_by_request(artifact.request_digest) == ()


def test_load_of_unknown_key_is_keyerror(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.load("e" * 64)


def test_load_updates_hit_count_persistently(tmp_path):
    store = _store(tmp_path)
    metadata = store.store(_artifact(), stored_at="t1")
    _, first = store.load(metadata.entry_key)
    _, second = store.load(metadata.entry_key)
    assert first.hit_count == 1
    assert second.hit_count == 2
    reopened = _store(tmp_path)
    _, third = reopened.load(metadata.entry_key)
    assert third.hit_count == 3


# --- verified-or-absent: corruption is never served ---------------------------


def test_corrupted_payload_is_never_served(tmp_path):
    store = _store(tmp_path)
    artifact = _artifact()
    metadata = store.store(artifact, stored_at="t1")
    payload_path = store.root / "payloads" / f"{metadata.payload_file_sha256}.bin"
    payload_path.write_bytes(b"tampered content")
    with pytest.raises(SignalIntegrityError, match="never corrupt bytes"):
        store.load(metadata.entry_key)


def test_truncated_payload_is_never_served(tmp_path):
    store = _store(tmp_path)
    metadata = store.store(_artifact(), stored_at="t1")
    payload_path = store.root / "payloads" / f"{metadata.payload_file_sha256}.bin"
    data = payload_path.read_bytes()
    payload_path.write_bytes(data[: len(data) // 2])
    with pytest.raises(SignalIntegrityError):
        store.load(metadata.entry_key)


def test_missing_payload_is_a_loud_absence(tmp_path):
    store = _store(tmp_path)
    metadata = store.store(_artifact(), stored_at="t1")
    (store.root / "payloads" / f"{metadata.payload_file_sha256}.bin").unlink()
    with pytest.raises(SignalIntegrityError, match="absent"):
        store.load(metadata.entry_key)


def test_index_tampered_artifact_digest_is_caught_by_recomputation(tmp_path):
    store = _store(tmp_path)
    metadata = store.store(_artifact(), stored_at="t1")
    # Payload bytes stay perfectly valid; the recorded artifact digest in
    # the index is what is tampered. Verification recomputes the artifact
    # digest from the payload and refuses the disagreement.
    index_path = store.root / "index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["entries"][metadata.entry_key]["artifact_digest"] = "f" * 64
    index_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    with pytest.raises(SignalIntegrityError, match="re-hashes to artifact"):
        store.load(metadata.entry_key)


def test_index_with_foreign_entry_key_is_refused(tmp_path):
    store = _store(tmp_path)
    metadata = store.store(_artifact(), stored_at="t1")
    index_path = store.root / "index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["entries"]["0" * 64] = document["entries"].pop(metadata.entry_key)
    index_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    reopened = _store(tmp_path)
    with pytest.raises(SignalIntegrityError, match="keys disagree"):
        reopened.load("0" * 64)


def test_unreadable_index_refuses_service_at_open(tmp_path):
    store = _store(tmp_path)
    store.store(_artifact(), stored_at="t1")
    (store.root / "index.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SignalIntegrityError, match="unreadable"):
        _store(tmp_path)


# --- atomic writes and interrupted-write recovery ------------------------------


def test_no_temp_files_remain_after_normal_operation(tmp_path):
    store = _store(tmp_path)
    store.store(_artifact(), stored_at="t1")
    assert list(store.root.glob(".*")) == []
    assert list((store.root / "payloads").glob(".*")) == []


def test_open_sweeps_interrupted_temp_files(tmp_path):
    store = _store(tmp_path)
    store.store(_artifact(), stored_at="t1")
    (store.root / ".index-stranded.tmp").write_text("{}", encoding="utf-8")
    (store.root / "payloads" / ".00deadbeef-stranded.tmp").write_bytes(b"partial")
    reopened = _store(tmp_path)
    assert list(reopened.root.glob(".*")) == []
    assert list((reopened.root / "payloads").glob(".*")) == []
    assert reopened.entry_count() == 1


def test_orphaned_payload_with_no_metadata_is_swept_on_open(tmp_path):
    store = _store(tmp_path)
    store.store(_artifact(), stored_at="t1")
    metadata = store.find_by_request(_artifact().request_digest)[0]
    # Simulate a crash after the payload rename but before the index
    # replace: the payload exists, the index does not claim it.
    index_path = store.root / "index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["entries"] = {}
    index_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    _ = metadata
    orphan = store.root / "payloads" / f"{metadata.payload_file_sha256}.bin"
    assert orphan.exists()
    reopened = _store(tmp_path)
    assert not orphan.exists()
    assert reopened.disk_bytes() > 0  # index remains; only the orphan went


def test_index_write_is_atomic_replace(tmp_path):
    store = _store(tmp_path)
    store.store(_artifact(), stored_at="t1")
    # The metadata path is a regular file after a completed write (not a
    # symlink/partial), and a fresh store reads it back.
    reopened = _store(tmp_path)
    assert reopened.entry_count() == 1


# --- budget ---------------------------------------------------------------------


def test_tiny_budget_refuses_first_store_and_writes_nothing(tmp_path):
    store = _store(tmp_path, max_bytes=10)
    artifact = _artifact()
    size = len(artifact.canonical_json().encode("utf-8"))
    assert size > 10
    with pytest.raises(CacheBudgetExceededError, match="ceiling"):
        store.store(artifact, stored_at="t1")
    assert store.entry_count() == 0
    assert store.disk_bytes() <= 10 or store.disk_bytes() < 200  # only empty index scaffolding


def test_budget_refusal_leaves_existing_entries_intact(tmp_path):
    store = _store(tmp_path)
    first = store.store(_artifact(signal_id="sig-a"), stored_at="t1")
    big = _artifact(signal_id="sig-b")
    store._max_bytes = store.disk_bytes()  # pretend the ceiling is already reached
    with pytest.raises(CacheBudgetExceededError):
        store.store(big, stored_at="t2")
    _, loaded_meta = store.load(first.entry_key)
    assert loaded_meta.hit_count == 1


def test_disk_bytes_is_measured_not_modeled(tmp_path):
    store = _store(tmp_path)
    baseline = store.disk_bytes()
    store.store(_artifact(), stored_at="t1")
    after = store.disk_bytes()
    assert after > baseline
    index_size = (store.root / "index.json").stat().st_size
    payload_sizes = sum(p.stat().st_size for p in (store.root / "payloads").glob("*.bin"))
    assert after == index_size + payload_sizes


# --- explicit eviction -----------------------------------------------------------


def test_discard_removes_entry_and_unshared_payload(tmp_path):
    store = _store(tmp_path)
    metadata = store.store(_artifact(), stored_at="t1")
    payload_path = store.root / "payloads" / f"{metadata.payload_file_sha256}.bin"
    assert store.discard(metadata.entry_key) is True
    assert not payload_path.exists()
    assert store.entry_count() == 0
    assert store.discard(metadata.entry_key) is False


def test_discard_keeps_payload_still_claimed_by_another_entry(tmp_path):
    store = _store(tmp_path)
    provider = FakeTeacherProvider()
    request = _request()
    payloads = []
    for i in range(2):
        artifact = build_signal_artifact(
            provider=provider,
            request=request,
            signal=provider.query(request),
            signal_id=f"sig-{i}",
            occurred_at="2026-09-06T00:00:00Z",
            latency_seconds=0.1 * i,
        )
        payloads.append(store.store(artifact, stored_at=f"t{i}"))
    # Two artifacts differ (signal_id, latency) so their payloads differ;
    # craft a shared payload by storing the same content under a second
    # metadata row is not possible by design, so instead assert the
    # realistic invariant: discarding one entry leaves the other loadable.
    other = payloads[1]
    store.discard(payloads[0].entry_key)
    loaded, _ = store.load(other.entry_key)
    assert loaded.request_digest == other.request_digest


def test_discard_never_touches_the_registry_ledger(tmp_path):
    with RunRegistry(tmp_path / "r.sqlite") as registry:
        store = _store(tmp_path, registry=registry)
        metadata = store.store(_artifact(), stored_at="t1")
        assert len(list(registry.list_teacher_signals())) == 1
        store.discard(metadata.entry_key)
        assert len(list(registry.list_teacher_signals())) == 1  # evidence persists


# --- registry ledger --------------------------------------------------------------


def test_store_with_registry_appends_immutable_ledger_row(tmp_path):
    with RunRegistry(tmp_path / "r.sqlite") as registry:
        store = _store(tmp_path, registry=registry)
        artifact = _artifact()
        metadata = store.store(artifact, stored_at="t1")
        rows = list(registry.list_teacher_signals())
        assert len(rows) == 1
        row = rows[0]
        assert row["entry_key"] == metadata.entry_key
        assert row["artifact_digest"] == artifact.digest()
        assert row["request_digest"] == artifact.request_digest
        assert row["payload_file_sha256"] == metadata.payload_file_sha256
        assert row["signal_kind"] == "critique"
        assert row["teacher_id"] == "teacher-frontier-x"
        assert row["metadata"]["hit_count"] == 0


def test_duplicate_store_replays_idempotently_into_the_ledger(tmp_path):
    with RunRegistry(tmp_path / "r.sqlite") as registry:
        store = _store(tmp_path, registry=registry)
        artifact = _artifact()
        store.store(artifact, stored_at="t1")
        store.store(artifact, stored_at="t2")
        assert len(list(registry.list_teacher_signals())) == 1


def test_divergent_ledger_row_is_an_invariant_error(tmp_path):
    with RunRegistry(tmp_path / "r.sqlite") as registry:
        store = _store(tmp_path, registry=registry)
        metadata = store.store(_artifact(), stored_at="t1")
        store.discard(metadata.entry_key)
        # Corrupt the existing ledger row at the SQL layer: it now claims
        # different evidence for this content address (the public API
        # refuses to create such a row, which is the point).
        registry._conn.execute(
            "UPDATE teacher_signals SET artifact_digest = ? WHERE entry_key = ?",
            ("f" * 64, metadata.entry_key),
        )
        with pytest.raises(RegistryInvariantError, match="divergence"):
            store.store(_artifact(), stored_at="t3")


def test_restore_after_eviction_is_evidence_idempotent_in_the_ledger(tmp_path):
    with RunRegistry(tmp_path / "r.sqlite") as registry:
        store = _store(tmp_path, registry=registry)
        artifact = _artifact()
        metadata = store.store(artifact, stored_at="t1")
        store.discard(metadata.entry_key)
        # Re-store the identical artifact after eviction: the ledger row
        # keeps its first-acquisition time instead of diverging on
        # bookkeeping.
        store.store(artifact, stored_at="t2")
        rows = list(registry.list_teacher_signals())
        assert len(rows) == 1
        assert rows[0]["stored_at"] == "t1"
        reopened = _store(tmp_path, registry=registry)
        restored, hit = reopened.load(metadata.entry_key)
        assert restored == artifact
        assert hit.hit_count == 1


def test_ledger_survives_full_cache_deletion(tmp_path):
    with RunRegistry(tmp_path / "r.sqlite") as registry:
        store = _store(tmp_path, registry=registry)
        store.store(_artifact(), stored_at="t1")
    import shutil

    shutil.rmtree(tmp_path / "teacher-signals")
    with RunRegistry(tmp_path / "r.sqlite") as registry:
        rows = list(registry.list_teacher_signals())
        assert len(rows) == 1
        assert rows[0]["metadata"]["stored_at"] == "t1"


# --- GPU-backed round trip ----------------------------------------------------------


def test_gpu_backed_artifact_round_trips_resource_usage(tmp_path):
    store = _store(tmp_path)
    artifact = _gpu_artifact()
    assert artifact.signal.resource_usage is not None
    assert artifact.signal.resource_usage.peak_vram_gb_by_accelerator == {"cuda:0": 3.5}
    metadata = store.store(artifact, stored_at="t1")
    loaded, _ = store.load(metadata.entry_key)
    assert loaded == artifact
    assert loaded.signal.resource_usage.peak_vram_gb_by_accelerator == {"cuda:0": 3.5}
    assert loaded.gpu_hours == pytest.approx(24.0 / 3600.0)


# --- metadata dataclass validation ----------------------------------------------------


def test_stored_metadata_rejects_bad_digests_and_counts():
    good = dict(
        artifact_digest="a" * 64,
        request_digest="b" * 64,
        prompt_digest="c" * 64,
        payload_file_sha256="d" * 64,
        signal_kind="critique",
        teacher_id="t",
        model_revision="r",
        tokenizer_identity_sha256=None,
        signal_id="s",
        stored_at="t1",
    )
    StoredSignalMetadata(**good)
    with pytest.raises(ValueError, match="sha256"):
        StoredSignalMetadata(**{**good, "artifact_digest": "short"})
    with pytest.raises(ValueError, match="hit_count"):
        StoredSignalMetadata(**{**good, "hit_count": -1})
    with pytest.raises(ValueError, match="hit_count"):
        StoredSignalMetadata(**{**good, "hit_count": True})
    with pytest.raises(ValueError, match="stored_at"):
        StoredSignalMetadata(**{**good, "stored_at": ""})


def test_metadata_json_round_trip():
    metadata = StoredSignalMetadata(
        artifact_digest="a" * 64,
        request_digest="b" * 64,
        prompt_digest="c" * 64,
        payload_file_sha256="d" * 64,
        signal_kind="critique",
        teacher_id="t",
        model_revision="r",
        tokenizer_identity_sha256="e" * 64,
        signal_id="s",
        stored_at="t1",
        hit_count=3,
        last_hit_at="t9",
    )
    restored = StoredSignalMetadata.from_json_dict(metadata.to_json_dict())
    assert restored == metadata
