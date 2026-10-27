"""Source-bound actual receipt checks and adversarial mutation negatives."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from chowder_batch import reap_v38_independent_receipt_audit_worker13_20261027 as audit
from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as v38


JOURNAL = Path(
    "F:/reap-v2-human164-v38-private-q4-sealed-original-20261027-1325/"
    "q4-actual-independent-per-case.private.jsonl"
)


@pytest.fixture(scope="module")
def authentic_row():
    if not JOURNAL.is_file():
        pytest.skip("original F: native receipt not mounted")
    return json.loads(JOURNAL.read_bytes().splitlines()[0])


def serialize(*rows):
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def test_original_q4_q8_sha_bound_model_program_and_session():
    for arm in ("q4", "q8"):
        manifest = audit._source_manifest(arm)
        assert manifest["source_rows"] == 164
        assert manifest["original_model_sha256"] == v38.ARMS[arm].original_model_sha256
        assert manifest["original_serving_session"] == v38.ARMS[arm].session
        assert manifest["original_rows_raw_sha256"] == v38.ARMS[arm].source_sha256
        assert manifest["canonical_raw_sha256"] == v38.CANONICAL_SHA256


def test_real_first_item_native_receipt_validates(authentic_row):
    result = audit.audit_bytes("q4", serialize(authentic_row))
    assert result["verified_contiguous_rows"] == 1
    assert result["verified_actual_executions"] == 1
    assert result["full_164_independent_correct"] is None
    assert result["gen0_status"] == "BLOCKED"


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(index=2),
    lambda r: r.update(id="HumanEval/37"),
    lambda r: r.update(source_session="forged"),
    lambda r: r.update(original_model_sha256="0" * 64),
    lambda r: r.update(program_sha256="0" * 64),
    lambda r: r.update(independent_research_correct=not r["independent_research_correct"]),
    lambda r: r["live_external_vm_host_witness"].update(source_program_sha256="0" * 64),
    lambda r: r["live_external_vm_host_witness"]["outside_host_exact_staged_source_witness"]
        ["immutable_executable_memfd"].update(seals=0),
    lambda r: r["live_external_vm_host_witness"]["outside_host_exact_staged_source_witness"]
        ["immutable_executable_memfd"].update(sha256="0" * 64),
    lambda r: r["live_external_vm_host_witness"].update(
        external_owned_process_extinction={"cgroup_procs": [123], "same_owned_processes_live": []}),
    lambda r: r["live_external_vm_host_witness"]["outside_host_exact_staged_source_witness"]
        .update(net_interfaces=["lo", "eth0"]),
    lambda r: r["live_external_vm_host_witness"]["outside_host_exact_staged_source_witness"]
        .update(mount_inventory=[{"point": "/etc/passwd", "source": "/dev/sdd",
                                  "fstype": "ext4", "options": ["ro"]}]),
    lambda r: r["live_external_vm_host_witness"]["outside_host_exact_staged_source_witness"]
        .update(process_status=[]),
    lambda r: r.update(frozen_v2_gate2_eligible=True),
])
def test_mutated_authentic_receipt_fails_closed(authentic_row, mutate):
    row = copy.deepcopy(authentic_row)
    mutate(row)
    with pytest.raises(ValueError, match="REFUSED independent"):
        audit.audit_bytes("q4", serialize(row))


def test_duplicate_nonmonotonic_original_row_fails(authentic_row):
    with pytest.raises(ValueError, match="nonmonotonic"):
        audit.audit_bytes("q4", serialize(authentic_row, authentic_row))


def test_truncated_or_unfsynced_tail_and_duplicate_json_key_refused(authentic_row):
    with pytest.raises(ValueError, match="truncated"):
        audit.audit_bytes("q4", serialize(authentic_row)[:-1])
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit.audit_bytes("q4", b'{"index":0,"index":0}\n')


def test_bad_saved_original_manifest_rejected(authentic_row):
    with pytest.raises(ValueError, match="manifest diverges"):
        audit.audit_bytes("q4", serialize(authentic_row), saved_manifest=b"{}")


def test_live_snapshot_requires_fresh_exclusive_f_root(authentic_row, tmp_path):
    with pytest.raises(ValueError, match="exclusive fresh"):
        audit.audit_live("q4", JOURNAL, snapshot_dir=tmp_path / "not-f-root")
