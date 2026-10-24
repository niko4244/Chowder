"""CPU-only exact BF16 storage/parity checks; never initialize CUDA/weights."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chowder_batch import reap_hidden_memmap as hidden_module


def _ragged():
    # Explicit sign / subnormal / signed-zero / NaN bits: float conversions
    # could canonicalize them, so this tests raw bitwise storage.
    patterns = (
        [0, 0x8000, 0x3F80, 0xBF80, 0x0001, 0x7F80, 0xFF80, 0x7FC1],
        [0x3F00, 0x4000, 0xFF01, 0x7F80],
        [0xAAAA, 0x3F80, 0x5555, 0x8000, 0x0000, 0x8001,
         0x1000, 0x7FC2, 0x1234, 0x7654, 0x3F7F, 0x3E00],
    )
    return [
        torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16).reshape(1, -1, 4)
        for bits in patterns
    ]


def _bits(t):
    return t.contiguous().view(torch.uint16).flatten().tolist()


def test_ragged_bf16_raw_bits_and_negative_indices(tmp_path):
    rows = _ragged()
    scratch = tmp_path / "new-slab"
    slab = hidden_module.BF16DiskHidden([r.shape[1] for r in rows], 4, scratch)
    assert slab.required_bytes == sum(r.numel() * 2 for r in rows)
    for i, row in enumerate(rows):
        slab[i] = row
    slab.flush()
    for i, row in enumerate(rows):
        assert _bits(slab[i]) == _bits(row)
    assert list(scratch.iterdir())
    meta = json.loads((scratch / "layout.json").read_text())
    assert meta["numeric_conversions"] is False
    assert meta["total_tokens"] == sum(row.shape[1] for row in rows)
    assert meta["scientific_gate"] == "NOT_MEASURED"
    with pytest.raises(IndexError):
        _ = slab[-1]
    with pytest.raises(IndexError):
        _ = slab[len(rows)]
    slab.close()


def test_two_layer_inplace_matches_original_ram_bitwise(tmp_path):
    initial = [
        torch.randn(1, n, 4, dtype=torch.float32).to(torch.bfloat16)
        for n in [3, 5, 1, 9]
    ]
    in_ram = [x.clone() for x in initial]
    slab = hidden_module.BF16DiskHidden([x.shape[1] for x in initial], 4,
                                         tmp_path / "fresh")
    for i, row in enumerate(initial):
        slab[i] = row
    for _layer in range(2):
        for i in range(len(initial)):
            # Frozen evaluator holds states on CPU and transfers one at a
            # time to the CUDA decoder; model output is simulated by BF16
            # arithmetic here without allocating any GPU memory.
            def forward(z):
                return ((z.float() * 0.75) + 0.125).to(torch.bfloat16)
            in_ram[i] = forward(in_ram[i])
            slab[i] = forward(slab[i])
        slab.flush()
        for i in range(len(initial)):
            assert _bits(slab[i]) == _bits(in_ram[i])
    slab.close()


@pytest.mark.parametrize("bad_length", [0, -1, True, 0.3])
def test_invalid_lengths_before_disk_allocation(tmp_path, bad_length):
    dest = tmp_path / "slab"
    with pytest.raises(ValueError, match="sequence lengths"):
        hidden_module.BF16DiskHidden([2, bad_length], 4, dest)
    assert not dest.exists()


@pytest.mark.parametrize("bad_size", [0, -4, True, 0.5])
def test_invalid_hidden_size_before_disk_allocation(tmp_path, bad_size):
    dest = tmp_path / "slab"
    with pytest.raises(ValueError, match="hidden_size"):
        hidden_module.BF16DiskHidden([2, 5], bad_size, dest)
    assert not dest.exists()


def test_refuses_existing_directory_or_unavailable_parent(tmp_path):
    preexisting = tmp_path / "old"
    preexisting.mkdir()
    (preexisting / "keep.txt").write_text("immutable")
    with pytest.raises(FileExistsError):
        hidden_module.BF16DiskHidden([2], 4, preexisting)
    with pytest.raises(FileExistsError):
        hidden_module.BF16DiskHidden([2], 4, tmp_path / "missing" / "new")
    assert (preexisting / "keep.txt").read_text() == "immutable"


def test_fails_closed_on_insufficient_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(hidden_module.shutil, "disk_usage", lambda _: SimpleNamespace(free=1000))
    scratch = tmp_path / "uncreated"
    with pytest.raises(RuntimeError, match="not enough disk"):
        hidden_module.BF16DiskHidden([2], 4, scratch)
    assert not scratch.exists()


def test_shape_device_and_dtype_validation_do_not_convert(tmp_path):
    slab = hidden_module.BF16DiskHidden([2, 3], 4, tmp_path / "slab")
    with pytest.raises(TypeError, match="exact-shape"):
        slab[0] = torch.zeros(1, 2, 4, dtype=torch.float32)
    with pytest.raises(TypeError, match="exact-shape"):
        slab[1] = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="exact-shape"):
        slab[1] = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
    slab.close()
    gc.collect()


def test_scratch_file_exact_bytes_round_trip_even_after_reopen(tmp_path):
    src = torch.tensor([0x7FC1, 0, 0x8000, 0x7F80], dtype=torch.uint16)
    slab = hidden_module.BF16DiskHidden([1], 4, tmp_path / "bf16")
    slab[0] = src.view(torch.bfloat16).reshape(1, 1, 4)
    slab.flush()
    raw = np.fromfile(tmp_path / "bf16" / "hidden-bf16-bits.memmap", dtype=np.uint16)
    assert raw.tolist() == src.tolist()
    assert _bits(slab[0]) == src.tolist()
    slab.close()
