"""Small CPU-only exact data-equivalence helpers; production model is untouched."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from chowder_batch.reap_verify_pruned import (
    _compare_mapped_bf16_shard, _equal_copied_tensor, _equal_expert_rows,
    _equal_source_row, _identical_bf16_bits, _indexed, _source_runs, inspect,
)


def test_checked_expert_mapping_and_full_unmodified_tensor_equivalence(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    original = torch.arange(8 * 4 * 3, dtype=torch.float32).reshape(8, 4, 3).bfloat16()
    untouched = torch.arange(60, dtype=torch.float32).reshape(10, 6).bfloat16()
    keep = [1, 3, 5, 7]
    safetensors.save_file({"expert": original, "untouched": untouched}, str(source))
    safetensors.save_file({
        "expert": original[keep].contiguous(), "untouched": untouched.clone(),
    }, str(output))
    from safetensors import safe_open

    with safe_open(str(source), framework="pt", device="cpu") as a, safe_open(
        str(output), framework="pt", device="cpu",
    ) as b:
        for i, j in enumerate(keep):
            _equal_source_row(a, b, "expert", j, i)
        assert _equal_expert_rows(a, b, "expert", keep) == len(keep)
        _equal_copied_tensor(a, b, "untouched")
        with pytest.raises(ValueError, match="expert weight mismatch"):
            _equal_source_row(a, b, "expert", 0, 0)
    safetensors.save_file({
        "expert": original[keep].contiguous(),
        "untouched": torch.zeros_like(untouched),
    }, str(output))
    with safe_open(str(source), framework="pt", device="cpu") as a, safe_open(
        str(output), framework="pt", device="cpu",
    ) as b:
        with pytest.raises(ValueError, match="unmodified tensor bytes differ"):
            _equal_copied_tensor(a, b, "untouched")


def test_verifier_rejects_missing_unbuilt_candidate_without_source_io(tmp_path):
    with pytest.raises(ValueError, match="has not been published"):
        inspect(source=tmp_path / "input", observed=tmp_path / "obs",
                selection=tmp_path / "selection.json", candidate=tmp_path / "unbuilt")


def test_output_tensor_index_path_traversal_refused():
    with pytest.raises(ValueError, match="unsafe index"):
        _indexed({"weight_map": {"x": "../unsafe.safetensors"}})


def test_full_expert_runs_preserve_source_and_destination_bijection():
    assert _source_runs([1, 3, 4, 5, 7]) == [(0, 1, 1), (1, 3, 3), (4, 7, 1)]


def test_bitwise_comparator_detects_signed_zero_and_preserves_nan_payloads():
    a = torch.tensor([0, -32768, 32641], dtype=torch.int16).view(torch.bfloat16)
    b = a.clone()
    assert _identical_bf16_bits(a, b)
    b.view(torch.int16)[1] = 0
    assert not _identical_bf16_bits(a, b)


def test_mapped_comparison_checks_all_expert_rows_and_copied_raw_bits(tmp_path):
    from safetensors.torch import save_file

    original = tmp_path / "source.safetensors"
    pruned = tmp_path / "pruned.safetensors"
    key = "model.language_model.layers.0.mlp.gate.weight"
    source = torch.arange(8 * 6, dtype=torch.int16).reshape(8, 6).view(torch.bfloat16)
    unchanged = torch.tensor([0, -32768, 32641], dtype=torch.int16).view(torch.bfloat16)
    kept = {0: [0, 2, 4, 7]}
    save_file({key: source.clone(), "unchanged": unchanged.clone()}, str(original))
    save_file({key: source[kept[0]].contiguous(), "unchanged": unchanged.clone()},
              str(pruned))
    assert _compare_mapped_bf16_shard(original, pruned, {key, "unchanged"}, kept) == 5
    changed = source[kept[0]].clone()
    changed.view(torch.int16)[1, 1] = 12345
    save_file({key: changed, "unchanged": unchanged.clone()}, str(pruned))
    with pytest.raises(ValueError, match="expert byte mismatch"):
        _compare_mapped_bf16_shard(original, pruned, {key, "unchanged"}, kept)
    signed_zero_changed = unchanged.clone()
    signed_zero_changed.view(torch.int16)[1] = 0
    save_file({key: source[kept[0]].contiguous(),
               "unchanged": signed_zero_changed}, str(pruned))
    with pytest.raises(ValueError, match="copied BF16 tensor byte mismatch"):
        _compare_mapped_bf16_shard(original, pruned, {key, "unchanged"}, kept)


def test_mapped_comparison_rejects_invalid_header_offsets_and_symlink(tmp_path):
    import struct
    from safetensors.torch import save_file

    src = tmp_path / "src.safetensors"
    dst = tmp_path / "dst.safetensors"
    save_file({"same": torch.ones((3, 4), dtype=torch.bfloat16)}, str(src))
    dst.write_bytes(src.read_bytes())
    data = bytearray(dst.read_bytes())
    header_size = struct.unpack("<Q", data[:8])[0]
    # Corrupt the only tensor's byte range without changing file size.
    import json
    metadata = json.loads(data[8:8 + header_size])
    metadata["same"]["data_offsets"] = [0, 2]
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    if len(encoded) <= header_size:
        data[8:8 + header_size] = encoded.ljust(header_size, b" ")
        dst.write_bytes(data)
        with pytest.raises(ValueError, match="invalid tensor byte extent"):
            _compare_mapped_bf16_shard(src, dst, {"same"}, {})
