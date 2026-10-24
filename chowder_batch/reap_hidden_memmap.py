"""Lossless, file-backed BF16 sequence states for the frozen REAP observer.

This changes only how the intermediate CPU hidden tensors are stored.
Each BF16 bit pattern is reinterpreted as uint16, not converted, and row
ordering is preserved. Every sequence is processed and rewritten in place
after its GPU decoder forward, exactly as in the in-memory observer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

GIB = 1024**3
_RESERVE_BYTES = 2 * GIB


class BF16DiskHidden:
    """Mutable, row-indexed BF16 tensors backed by one sequential disk slab."""

    def __init__(self, lengths: list[int], hidden_size: int, scratch: Path) -> None:
        if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
            raise ValueError("hidden memmap requires nonempty positive sequence lengths")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden memmap requires positive hidden_size")
        if scratch.exists() or not scratch.parent.is_dir():
            raise FileExistsError("hidden scratch directory must be NEW with existing parent")
        if scratch.is_symlink():
            raise ValueError("hidden scratch must not be symlink")
        offsets = [0]
        for count in lengths:
            offsets.append(offsets[-1] + count)
        total = offsets[-1] * hidden_size * 2
        free = shutil.disk_usage(scratch.parent).free
        if free < total + _RESERVE_BYTES:
            raise RuntimeError(
                f"not enough disk for BF16 hidden slab: {total} bytes plus "
                f"{_RESERVE_BYTES} reserved, only {free} free"
            )
        self.lengths = tuple(lengths)
        self.hidden_size = hidden_size
        self.offsets = tuple(offsets)
        self.required_bytes = total
        self.scratch = scratch
        scratch.mkdir(exist_ok=False)
        (scratch / "layout.json").write_text(
            json.dumps({
                "schema": "chowder.reap-hidden-bf16-mmap.v1",
                "total_tokens": offsets[-1],
                "sequence_lengths": lengths,
                "hidden_size": hidden_size,
                "allocated_bytes": total,
                "dtype": "uint16 bits reinterpreted as BF16",
                "numeric_conversions": False,
                "scientific_gate": "NOT_MEASURED",
            }, sort_keys=True) + "\n", encoding="utf-8",
        )
        self.slab = np.memmap(
            str(scratch / "hidden-bf16-bits.memmap"), mode="w+",
            dtype=np.uint16, shape=(offsets[-1], hidden_size),
        )

    def __len__(self) -> int:
        return len(self.lengths)

    def _bounds(self, i: int) -> tuple[int, int]:
        if type(i) is not int or i < 0 or i >= len(self.lengths):
            raise IndexError("sequence index outside hidden slab")
        return self.offsets[i], self.offsets[i + 1]

    def __getitem__(self, i: int) -> torch.Tensor:
        start, stop = self._bounds(i)
        # zero-copy CPU view: reinterpret uint16's exact 16-bit payload.
        return torch.from_numpy(self.slab[start:stop]).view(torch.bfloat16).unsqueeze(0)

    def __setitem__(self, i: int, value: torch.Tensor) -> None:
        start, stop = self._bounds(i)
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.bfloat16
                or tuple(value.shape) != (1, stop - start, self.hidden_size)):
            raise TypeError(
                "hidden memmap writes must be exact-shape CPU BF16 tensors "
                "(no implicit dtype conversion or retokenization)"
            )
        bits = value[0].contiguous().view(torch.uint16).numpy()
        self.slab[start:stop] = bits

    def flush(self) -> None:
        self.slab.flush()

    def close(self) -> None:
        self.slab.flush()
        # Do not unlink here: a kill during the registered observer cannot
        # safely be treated as successful cleanup. The supervisor owns the
        # scratch folder and must verify process exit before deletion.
