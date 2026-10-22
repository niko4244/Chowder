"""CPU-only content identity and stale/malformed inventory rejection."""

import json
import os
import hashlib

import pytest

from chowder import moe_placement_gates as mpg
from chowder.local_model_manifest import model_content_digest
from chowder.moe_checkpoint_inventory import (
    CheckpointInventoryError,
    enroll,
    verified_identity,
)


def checkpoint(root, weight=b"abcdefgh12345678"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"num_experts": 160}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(weight)
    return root


def test_full_sha_enrollment_then_metadata_only_verification(tmp_path, monkeypatch):
    root = checkpoint(tmp_path / "model")
    first = enroll(root)
    assert len(first["files"]) == 2
    assert first["content_sha256"] == model_content_digest(root).digest
    assert verified_identity(root)["content_sha256"] == first["content_sha256"]

    # Weight bytes are read for SHA exactly once on enrollment. The fast path
    # must not open these multi-gigabyte payloads in an ordinary gate check.
    import chowder.moe_checkpoint_inventory as inventory

    original = inventory.sha256_file
    seen = []

    def count_hash(path):
        seen.append(path.name)
        if path.suffix == ".safetensors":
            raise AssertionError("weight payload was rehashed during a quick gate check")
        return original(path)

    monkeypatch.setattr(inventory, "sha256_file", count_hash)
    assert verified_identity(root)["inventory_sha256"] == first["inventory_sha256"]
    assert "config.json" in seen


def test_missing_inventory_and_config_only_legacy_table_fail_closed(tmp_path):
    root = checkpoint(tmp_path / "model")
    with pytest.raises(CheckpointInventoryError, match="inventory missing"):
        verified_identity(root)
    old = mpg.render_block({"base_model": str(root), "artifact": mpg.artifact_identity(root)})
    with pytest.raises(mpg.PlacementGateError, match="not satisfied|lacks a verified"):
        mpg.assert_declaration_allowed(old.replace("not measured", "pass"), ["campaign"])


def test_same_size_weight_replacement_is_rejected_even_with_identical_config(tmp_path):
    root = checkpoint(tmp_path / "model")
    first = enroll(root)
    old_config_sha = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    weight = root / "model.safetensors"
    weight.write_bytes(b"qrstuvwx12345678")  # exactly the same size
    assert hashlib.sha256((root / "config.json").read_bytes()).hexdigest() == old_config_sha
    with pytest.raises(CheckpointInventoryError, match="payload changed"):
        verified_identity(root)
    assert first["content_sha256"]


def test_inventory_checksum_tampering_and_extra_shards_refused(tmp_path):
    root = checkpoint(tmp_path / "model")
    enroll(root)
    path = root / ".chowder-moe-sha256-inventory.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["files"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CheckpointInventoryError, match="digest mismatches"):
        verified_identity(root)
    # Restoring the inventory then adding another shard is also a mismatch.
    path.unlink()
    enroll(root)
    (root / "extra.safetensors").write_bytes(b"extra shard")
    with pytest.raises(CheckpointInventoryError, match="incomplete or changed"):
        verified_identity(root)


def test_semantic_bytes_checked_even_with_restored_mtime(tmp_path):
    root = checkpoint(tmp_path / "model")
    enroll(root)
    path = root / "config.json"
    before = path.stat()
    path.write_text('{"num_experts": 128}', encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(CheckpointInventoryError, match="changed|SHA256 mismatch"):
        verified_identity(root)


def test_different_verified_weight_sets_cannot_mix_witness_and_gate(tmp_path):
    a = checkpoint(tmp_path / "a", b"a" * 16)
    b = checkpoint(tmp_path / "b", b"b" * 16)
    enroll(a)
    enroll(b)
    identity_a = mpg.artifact_identity(a)
    identity_b = mpg.artifact_identity(b)
    assert identity_a["config_sha256"] == identity_b["config_sha256"]
    assert identity_a["shard_bytes"] == identity_b["shard_bytes"]
    assert identity_a["content_sha256"] != identity_b["content_sha256"]
    record = {"artifact": identity_a, "base_model": str(a)}
    table = mpg.render_block(record)
    with pytest.raises(mpg.PlacementGateError, match="SHA inventory/content pin"):
        mpg.splice_witness_rows(table, {
            "schema": mpg.WITNESS_SCHEMA,
            "placement": mpg.EXPERT_PLACEMENT,
            "artifact": identity_b,
            "gates": {"smoke": {"verdict": "pass", "value": "all good"}},
        })


def test_campaign_gate_refuses_same_size_weight_mutation_after_enrollment(tmp_path):
    root = checkpoint(tmp_path / "model")
    enroll(root)
    identity = mpg.artifact_identity(root)
    record = {
        "base_model": str(root),
        "artifact": identity,
        "arms": {"int8": {
            "peak_vram_gb": 10.0,
            "s_per_step": 1.0,
            "gates": {
                "load_census": "measured", "memory_vram_pass": True,
                "host_rss": "measured", "gradient": "pass", "smoke": "pass",
                "router_balance": "pass", "wddm_spill": "pass",
            },
        }},
    }
    table = mpg.render_block(record)
    assert mpg.assert_declaration_allowed(table, ["campaign.json"], base_model=root)
    (root / "model.safetensors").write_bytes(b"abcdefghijklmnop")  # size unchanged
    with pytest.raises(mpg.PlacementGateError, match="inventory verification failed"):
        mpg.assert_declaration_allowed(table, ["campaign.json"], base_model=root)


def test_reenrollment_never_overwrites_existing_inventory(tmp_path):
    root = checkpoint(tmp_path / "model")
    enroll(root)
    with pytest.raises(CheckpointInventoryError, match="refusing overwrite"):
        enroll(root)
