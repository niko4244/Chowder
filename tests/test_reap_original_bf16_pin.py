"""Hermetic official-original BF16 identity and forged D0 evidence regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from chowder_batch import reap_original_bf16_pin as pin
from chowder_batch import reap_hidden_checkpoint as cp


def _fixture(tmp_path, monkeypatch):
    """Tiny 26-shard/11-asset stand-in; full data checks, no real model loads."""
    model, weights = tmp_path / "model", tmp_path / "weights"
    model.mkdir()
    weights.mkdir()
    shards = {
        f"model-{i:05d}-of-00026.safetensors": f"VALID-BF16-SOURCE-{i:02d}".encode()
        for i in range(1, 27)
    }
    metadata = {
        name: f"OFFICIAL-{name}".encode()
        for name in (
            "config.json", "configuration.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "generation_config.json", "merges.txt", "vocab.json",
            "preprocessor_config.json", "video_preprocessor_config.json",
        )
    }
    metadata["model.safetensors.index.json"] = json.dumps({
        "weight_map": {f"tensor.{i}": name for i, name in enumerate(shards)},
    }).encode()
    for name, content in shards.items():
        (weights / name).write_bytes(content)
    for name, content in metadata.items():
        (model / name).write_bytes(content)
    original = pin.load_original_pin()
    stub = {
        **original,
        "weight_shards": {
            name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in shards.items()
        },
        "model_metadata": {
            name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in metadata.items()
        },
    }
    monkeypatch.setattr(pin, "load_original_pin", lambda: stub)
    return model, weights, stub


def test_real_checked_in_upstream_pin_has_fixed_expected_inventory():
    doc = pin.load_original_pin()
    assert doc["repository"] == "Qwen/Qwen3.6-35B-A3B"
    assert doc["revision"] == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert len(doc["weight_shards"]) == 26
    assert len(doc["model_metadata"]) == 11
    assert doc["frozen_retokens"]["count"] == 704
    assert doc["frozen_retokens"]["sum"] == 1_192_632
    assert doc["frozen_retokens"]["sha256"] == (
        "78b8f482f61cfd4b20bba6982a20c8eeb5a757b20069f2520273d4a315335f92"
    )


def test_upstream_pin_rejects_same_size_mutations_and_self_rehashed_weights(
    tmp_path, monkeypatch,
):
    model, weights, stub = _fixture(tmp_path, monkeypatch)
    expected = pin.validate_registered_source_metadata(model, weights)
    assert expected["revision"] == pin.REVISION
    correct = {
        **{f"weights/{n}": d["sha256"] for n, d in stub["weight_shards"].items()},
        **{f"model/{n}": d["sha256"] for n, d in stub["model_metadata"].items()},
    }
    assert pin.validate_registered_source_artifacts(correct) == expected
    shard = weights / "model-00001-of-00026.safetensors"
    raw = shard.read_bytes()
    shard.write_bytes(b"X" + raw[1:])
    # Header/size-only admission cannot establish payload equality.
    pin.validate_registered_source_metadata(model, weights)
    forged = dict(correct)
    forged["weights/" + shard.name] = hashlib.sha256(shard.read_bytes()).hexdigest()
    with pytest.raises(pin.OriginalBF16PinError, match="independently pinned"):
        pin.validate_registered_source_artifacts(forged)
    shard.write_bytes(raw)
    pin.validate_registered_source_metadata(model, weights)


@pytest.mark.parametrize("change", ["index", "metadata", "missing", "symlink"])
def test_registered_small_metadata_or_shard_inventory_tampering_fails(
    tmp_path, monkeypatch, change,
):
    model, weights, _ = _fixture(tmp_path, monkeypatch)
    if change == "index":
        index = model / "model.safetensors.index.json"
        content = json.loads(index.read_text())
        content["weight_map"]["extra"] = "unexpected.safetensors"
        index.write_text(json.dumps(content))
    elif change == "metadata":
        path = model / "config.json"
        data = path.read_bytes()
        path.write_bytes(data[:-1] + b"X")
    elif change == "missing":
        (weights / "model-00008-of-00026.safetensors").unlink()
    elif change == "symlink":
        path = weights / "model-00008-of-00026.safetensors"
        content = path.read_bytes()
        path.unlink()
        other = tmp_path / "outside.safetensors"
        other.write_bytes(content)
        try:
            path.symlink_to(other)
        except OSError:
            pytest.skip("filesystem does not allow symlinks")
    with pytest.raises(pin.OriginalBF16PinError):
        pin.validate_registered_source_metadata(model, weights)


def test_registered_token_identity_rejects_permutation_same_census():
    original = pin.load_original_pin()
    lengths = original["frozen_retokens"]["lengths"]
    identity = {
        "calibration_sha256": original["frozen_calibration_sha256"],
        "lengths": lengths,
        "token_ids_sha256": original["frozen_retokens"]["sha256"],
    }
    pin.validate_registered_token_identity(identity, n_seqs=704, n_tokens=1_192_632)
    wrong = dict(identity, token_ids_sha256="0" * 64)
    with pytest.raises(pin.OriginalBF16PinError, match="ordered token"):
        pin.validate_registered_token_identity(wrong, n_seqs=704, n_tokens=1_192_632)
    swapped = list(lengths)
    i = next(i for i, v in enumerate(swapped) if v != swapped[0])
    swapped[0], swapped[i] = swapped[i], swapped[0]
    with pytest.raises(pin.OriginalBF16PinError, match="ordered token"):
        pin.validate_registered_token_identity(
            dict(identity, lengths=swapped), n_seqs=704, n_tokens=1_192_632,
        )


def _forged_receipt(out: Path, *, fake_stats: bool) -> dict:
    source = pin.load_original_pin()
    artifacts = {
        **{f"weights/{name}": value["sha256"]
           for name, value in source["weight_shards"].items()},
        **{f"model/{name}": value["sha256"]
           for name, value in source["model_metadata"].items()},
    }
    identity = {
        "output_dir": str(out.resolve()),
        "calibration_sha256": source["frozen_calibration_sha256"],
        "lengths": source["frozen_retokens"]["lengths"],
        "token_ids_sha256": source["frozen_retokens"]["sha256"],
        "num_experts": 256,
        "top_k": 8,
        "layers": list(range(40)),
        "artifacts": artifacts,
        "trusted_upstream_pin": pin.validate_registered_source_artifacts(artifacts),
    }
    chain, timings, prev = [], [], None
    identity_hash = cp._digest_bytes(cp._canonical(identity))
    for generation in range(41):
        layer = generation - 1
        stat_sha = None
        timing = None
        if generation:
            path = out / f"layer-{layer:02d}.pt"
            if fake_stats:
                path.write_bytes(b"NOT A TORCH STATISTICS TENSOR")
            else:
                cnt = torch.zeros(256, dtype=torch.float64)
                cnt[0] = 1_192_632 * 8
                gate = torch.zeros(256, dtype=torch.float64)
                gate[0] = 1_192_632
                stats = {
                    "count": cnt,
                    "gate_sum": gate,
                    "norm_sum": gate,
                    "reap_sum": gate / 2,
                }
                torch.save(stats, path)
            stat_sha = cp.sha256_file(path)
            timing = {"layer": layer, "routed_tokens": 1_192_632 * 8}
            timings.append(timing)
        manifest = {
            "schema": cp.SCHEMA,
            "generation": generation,
            "layer": layer,
            "buffer": generation % 2,
            "state_sha256": "a" * 64,
            "stats_sha256": stat_sha,
            "timing": timing,
            "previous_sha256": prev,
            "identity_sha256": identity_hash,
        }
        prev = cp._digest_bytes(cp._canonical(manifest))
        chain.append({"manifest": manifest, "manifest_sha256": prev})
    return {
        "n_seqs": 704,
        "n_tokens": 1_192_632,
        "calibration_sha256": source["frozen_calibration_sha256"],
        "calibration_ppl": 10,
        "layers": timings,
        "hidden_storage": {
            "type": "dual file-backed CPU BF16 bits, atomic layer checkpoint",
            "last_committed_generation": 40,
        },
        "checkpoint_receipt": {
            "schema": cp.SCHEMA,
            "generation": 40,
            "root_sha256": prev,
            "source_identity": identity,
            "source_identity_sha256": identity_hash,
            "chain": chain,
            "active_state_sha256": "a" * 64,
        },
    }


def test_self_consistent_forged_704_d0_with_non_torch_stats_rejected(tmp_path):
    observation = _forged_receipt(tmp_path, fake_stats=True)
    # This *would* have passed earlier: all self hashes match, and the pinned
    # model hashes can be copied from the public manifest without computation.
    cp.validate_observation_receipt(tmp_path, observation, expected_generation=40)
    with pytest.raises(ValueError, match="statistics cannot be decoded"):
        cp.validate_observation_receipt(
            tmp_path, observation, require_registered_original=True,
        )
    from chowder_batch.reap_observe_supervised import _check_observation_complete
    (tmp_path / "observation.json").write_text(json.dumps(observation), encoding="utf-8")
    with pytest.raises(ValueError, match="statistics cannot be decoded"):
        _check_observation_complete(tmp_path, tmp_path / "registered-calibration.jsonl")


def test_self_rehashed_forged_original_source_fails_against_publisher_pin(tmp_path):
    observed = _forged_receipt(tmp_path, fake_stats=False)
    receipt = observed["checkpoint_receipt"]
    identity = receipt["source_identity"]
    first = sorted(k for k in identity["artifacts"] if k.startswith("weights/"))[0]
    identity["artifacts"][first] = "0" * 64
    # Rewrite EVERY self-attested SHA and linked manifest as an attacker could.
    replacement = cp._digest_bytes(cp._canonical(identity))
    receipt["source_identity_sha256"] = replacement
    previous = None
    for entry in receipt["chain"]:
        manifest = entry["manifest"]
        manifest["identity_sha256"] = replacement
        manifest["previous_sha256"] = previous
        previous = entry["manifest_sha256"] = cp._digest_bytes(cp._canonical(manifest))
    receipt["root_sha256"] = previous
    # All self-consistency hashes still verify in legacy generic mode.
    cp.validate_observation_receipt(tmp_path, observed)
    # Registered original D0 must NOT let the supplied receipt pin itself.
    with pytest.raises(ValueError, match="unregistered BF16 original source"):
        cp.validate_observation_receipt(
            tmp_path, observed, require_registered_original=True,
        )


@pytest.mark.xfail(
    reason="Residual scientific gate: an untrusted local writer can synthesize "
           "plausible stats and self-hashes; producer attestation is not yet independent",
    strict=False,
)
def test_self_consistent_forged_704_d0_with_fake_but_valid_tensors_limit(tmp_path):
    # Track the remaining external execution-attestation gate: local,
    # syntactically realistic files are not proof of BF16 forward execution.
    observation = _forged_receipt(tmp_path, fake_stats=False)
    with pytest.raises(ValueError, match="independently authenticated computation"):
        cp.validate_observation_receipt(
            tmp_path, observation, require_registered_original=True,
        )
