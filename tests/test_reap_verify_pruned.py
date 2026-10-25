"""Small CPU-only exact data-equivalence helpers; production model is untouched."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from chowder_batch.reap_verify_pruned import (
    _equal_copied_tensor, _equal_source_row, _identical_bf16_bits, _indexed, inspect,
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


def test_bitwise_comparator_detects_signed_zero_and_preserves_nan_payloads():
    a = torch.tensor([0, -32768, 32641], dtype=torch.int16).view(torch.bfloat16)
    b = a.clone()
    assert _identical_bf16_bits(a, b)
    b.view(torch.int16)[1] = 0
    assert not _identical_bf16_bits(a, b)
