"""Adversarial CPU-only independent 42-chunk/733-tensor provenance checks."""

import json

import pytest

from chowder_batch import reap_gguf_q4_independent_cert as cert


SOURCE_SHA = "8" * 64
STAGE_SHA = "9" * 64


def _fixture(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    names = [f"blk.{layer}.{suffix}" for layer in range(40)
             for suffix in ("ffn_down_exps.weight", "ffn_up_exps.weight",
                            "ffn_gate_exps.weight")]
    names.extend(f"other.registered.{i:03d}" for i in range(613))
    assert len(names) == cert.TENSORS
    # Balanced deterministic mini groups, with source order retained.
    groups = [names[i * len(names) // cert.CHUNKS:
                    (i + 1) * len(names) // cert.CHUNKS] for i in range(cert.CHUNKS)]
    outputs = {}
    for i, group in enumerate(groups):
        old = {name: {"shape": [1], "tensor_type": "BF16", "raw_sha256": "a" * 64}
               for name in group}
        new = {name: {"shape": [1], "tensor_type": "Q4_K", "raw_sha256": "b" * 64}
               for name in group}
        outputs.update(new)
        entry = {"schema": cert.CHUNK_SCHEMA, "status": cert.CHUNK_STATUS,
                 "identity": {"source_sha256": SOURCE_SHA, "stage_spec_sha256": STAGE_SHA,
                              "quantizer_sha256": cert.BINARY_SHA256,
                              "mode": "Q4_K_M", "index": i, "chunk_count": cert.CHUNKS,
                              "max_mib": 1024, "slab_mib": 1, "names": group},
                 "effective_quant_command": ["quantize.exe", "--max-buffer-size", "1",
                                             "original-mini.gguf", "out.gguf", "Q4_K_M", "2"],
                 "input_tensor_provenance": old, "output_tensor_provenance": new,
                 "output_sha256": "c" * 64,
                 "source_mini_sha256": "d" * 64,
                 "output_size_bytes": 123,
                 "output": str(chunks / f"chunk-{i + 1:03d}-of-042.q4_k_m.gguf")}
        (chunks / f"chunk-{i + 1:03d}-of-042.receipt.json").write_text(json.dumps(entry))
    return chunks, names, {"source_groups": groups, "all_tensors": outputs}


def _collect(chunks, names, claims):
    return cert.collect_chunk_receipts(chunks, source_sha=SOURCE_SHA,
                                       stage_sha=STAGE_SHA,
                                       names=names, final_claims=claims)


def test_all_42_original_chunks_match_733_exact_names_and_payload_claims(tmp_path):
    chunks, names, claims = _fixture(tmp_path)
    inputs, outputs, receipts = _collect(chunks, names, claims)
    assert set(inputs) == set(names) == set(outputs)
    assert len(receipts) == 42
    assert receipts[34]["index"] == 34
    assert receipts[-1]["filename"] == "chunk-042-of-042.receipt.json"


@pytest.mark.parametrize("mutation", ["source", "stage", "quantizer", "index", "groups",
                                     "mixed_receipt", "output_proof", "source_proof",
                                     "command", "missing", "duplicate", "output_sha",
                                     "source_mini_sha", "output_bytes", "output_path"])
def test_altered_ancestry_chunk_recipe_or_inventory_fails_closed(tmp_path, mutation):
    chunks, names, claims = _fixture(tmp_path)
    target = chunks / "chunk-035-of-042.receipt.json"
    entry = json.loads(target.read_text())
    tensor = entry["identity"]["names"][0]
    if mutation == "source":
        entry["identity"]["source_sha256"] = "f" * 64
    elif mutation == "stage":
        entry["identity"]["stage_spec_sha256"] = "f" * 64
    elif mutation == "quantizer":
        entry["identity"]["quantizer_sha256"] = "f" * 64
    elif mutation == "index":
        entry["identity"]["index"] = 0
    elif mutation == "groups":
        entry["identity"]["names"] = list(reversed(entry["identity"]["names"]))
    elif mutation == "mixed_receipt":
        entry["schema"] = "not-Q4"
    elif mutation == "output_proof":
        entry["output_tensor_provenance"][tensor]["raw_sha256"] = "e" * 64
    elif mutation == "source_proof":
        entry["input_tensor_provenance"][tensor]["shape"] = [0]
    elif mutation == "command":
        entry["effective_quant_command"][-2] = "Q8_0"
    elif mutation == "output_sha":
        entry["output_sha256"] = "not-a-digest"
    elif mutation == "source_mini_sha":
        entry["source_mini_sha256"] = "not-a-digest"
    elif mutation == "output_bytes":
        entry["output_size_bytes"] = -3
    elif mutation == "output_path":
        entry["output"] = str(chunks / "someone-else.gguf")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "duplicate":
        entry["identity"]["names"].append(entry["identity"]["names"][0])
    if mutation != "missing":
        target.write_text(json.dumps(entry))
    with pytest.raises((ValueError, OSError), match="source|recipe|chunk|receipt|tensor|trust"):
        _collect(chunks, names, claims)


@pytest.mark.parametrize("mutation", ["missing_group", "missing_tensor", "reordered", "duplicate"])
def test_final_publication_claims_must_reconcile_all_chunks(tmp_path, mutation):
    chunks, names, claims = _fixture(tmp_path)
    if mutation == "missing_group":
        claims["source_groups"].pop()
    elif mutation == "missing_tensor":
        claims["all_tensors"].pop(names[0])
    elif mutation == "reordered":
        claims["source_groups"][0], claims["source_groups"][1] = (
            claims["source_groups"][1], claims["source_groups"][0])
    else:
        claims["source_groups"][0][0] = claims["source_groups"][1][0]
    with pytest.raises(ValueError, match="incomplete|chunk|duplicate"):
        _collect(chunks, names, claims)


def test_absent_full_q4_cannot_claim_independent_attestation(tmp_path):
    with pytest.raises(ValueError, match="complete independent BF16/Q4"):
        cert.certify(tmp_path / "original.gguf", tmp_path / "partial.gguf", tmp_path)


def test_independent_certificate_has_no_registration_or_selection_pass():
    assert "NOT_QUALITY_GATE" in cert.STATUS
    assert cert.SCHEMA.endswith("independent-full-tensor-certificate.v1")
