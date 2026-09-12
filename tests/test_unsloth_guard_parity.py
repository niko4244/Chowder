"""The Unsloth worker's inlined adapter guard must match the real one.

`unsloth_worker.py` runs under the isolated `.chowder/envs/unsloth`
interpreter, which deliberately has no `chowder` package (see its module
docstring and docs/UNSLOTH.md), so it cannot import
`chowder.adapter_guard` and carries an inlined copy of the guard instead.
Inlined copies drift: until 2026-09-12 the worker's copy still had the
pre-audit semantics (count an unreadable B as nonzero, no key-overlap
refusal) while the shared guard had been fixed -- exactly the divergence
class this file pins.

Importing `unsloth_worker.py` by absolute file path is itself behavioral
proof of the constraint: the module top level is stdlib-only, so it loads
in this Chowder-side test process, but it does so WITHOUT adding the
`chowder` package to its namespace and while executing no
`from chowder ...` statement anywhere in the file. The parity assertions
then run identical scenarios through BOTH implementations.
"""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import pytest

import chowder
from chowder.adapter_guard import AdapterNotLiveError, assert_adapter_is_live

_SRC = Path(chowder.__file__).resolve().parent
_WORKER_PATH = _SRC / "backends" / "unsloth_worker.py"


def _load_worker_module():
    """Import the worker by file path, as the isolated interpreter does."""
    spec = importlib.util.spec_from_file_location(
        "_unsloth_worker_under_test", _WORKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker():
    return _load_worker_module()


# ---- stubs shared by both implementations --------------------------------------


class _Tensor:
    """Enough of a tensor for the guard: .detach().float().abs().max()."""

    def __init__(self, value: float) -> None:
        self._value = value

    def detach(self):
        return self

    def float(self):
        return self

    def abs(self):
        return self

    def max(self):
        return self

    def __float__(self) -> float:
        return self._value


class _Model:
    """named_parameters() from plain float values, wrapped into _Tensor."""

    def __init__(self, params: dict[str, float]) -> None:
        self._params = params

    def named_parameters(self):
        return [(n, _Tensor(v)) for n, v in self._params.items()]


def _write_adapter(directory: Path, keys: list[str]) -> Path:
    """A real safetensors file: 8-byte header length, JSON header, then data."""
    directory.mkdir(parents=True, exist_ok=True)
    header, offset = {}, 0
    for key in keys:
        header[key] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    path = directory / "adapter_model.safetensors"
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * offset)
    return directory


_TRANSFORMERS_KEYS = [
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight",
]
_LIVE_TRAINED = {
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight": 0.03,
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight": 0.07,
}
_LIVE_FRESH = {**_LIVE_TRAINED,
               "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight": 0.0}


# ---- the constraint: no chowder import, structurally ---------------------------


def test_worker_module_has_no_chowder_imports(worker):
    """The inlined copy must never reach for the chowder package -- checked on
    the loaded module's namespace AND on the source text (the source check
    mirrors what test_progress_write pins, so a refactor cannot satisfy one
    check while silently breaking the other)."""
    assert not any(name == "chowder" or name.startswith("chowder.")
                   for name in vars(worker))


def test_worker_source_mentions_no_chowder_module_path():
    text = _WORKER_PATH.read_text(encoding="utf-8")
    assert "from chowder" not in text
    assert "import chowder" not in text
    # the guard stays inlined on purpose; the comment must say why
    assert "inlined" in text and "adapter_guard" in text


# ---- parity: identical scenarios through both implementations -------------------


def _scenarios():
    """(name, model, expected buckets) run against BOTH guards."""
    base = "base_model.model.model.layers.0.linear_attn.in_proj_qkv"
    return [
        ("trained", _LIVE_TRAINED,
         {"nonzero": 1, "zero": 0, "unreadable": 0}),
        ("all_zero_B", _LIVE_FRESH,
         {"nonzero": 0, "zero": 1, "unreadable": 0}),
    ]


def test_report_buckets_match_the_shared_guard(worker, tmp_path):
    d = _write_adapter(tmp_path / "adapter", _TRANSFORMERS_KEYS)
    for name, params, buckets in _scenarios():
        ours = worker._adapter_liveness_report(_Model(params), d)
        theirs = chowder.adapter_guard.adapter_liveness_report(_Model(params), d)
        for key in ("saved_tensors", "live_lora_parameters", "matched_keys",
                    "lora_B_parameters", "lora_B_nonzero", "lora_B_zero",
                    "lora_B_unreadable"):
            assert ours[key] == theirs[key], f"{name}: {key} diverged"
        assert ours["lora_B_nonzero"] == buckets["nonzero"]
        assert ours["lora_B_zero"] == buckets["zero"]
        assert ours["lora_B_unreadable"] == buckets["unreadable"]


def test_unreadable_B_is_evidence_incomplete_not_live(worker, tmp_path):
    """The exact audit defect, on the worker's copy: an unreadable B must be
    counted as unknown and refused -- never as nonzero liveness. The refusal
    must match the shared guard's evidence-incomplete wording class."""
    d = _write_adapter(tmp_path / "adapter", _TRANSFORMERS_KEYS)

    class _Unreadable(_Tensor):
        def detach(self):
            raise RuntimeError("unreadable test tensor")

    class _M:
        def named_parameters(self):
            return [
                ("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight", _Tensor(0.1)),
                ("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight", _Unreadable(0.0)),
            ]

    ours = worker._adapter_liveness_report(_M(), d)
    assert ours["lora_B_nonzero"] == 0, "unreadable must never count as nonzero"
    assert ours["lora_B_zero"] == 0
    assert ours["lora_B_unreadable"] == 1
    assert ours["lora_B_unreadable_names"], "offending names must be recorded"
    with pytest.raises(RuntimeError, match="unreadable|incomplete"):
        worker._assert_parent_adapter_live(_M(), d)


def test_zero_key_overlap_is_refused_like_the_shared_guard(worker, tmp_path):
    """The worker's copy previously had NO key-overlap check at all; PEFT-only
    nonzero liveness would pass a mismatched-prefix adapter. The shared guard
    refuses zero overlap before anything else; the copy must agree."""
    d = _write_adapter(tmp_path / "mismatched", [
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight",
    ])
    with pytest.raises(RuntimeError, match="shares NO parameter names|NO parameter"):
        worker._assert_parent_adapter_live(_Model(_LIVE_FRESH), d)


def test_all_zero_B_refusal_message_names_the_run_consequence(worker, tmp_path):
    """Keep the worker's own diagnostic (a continued run silently starting from
    scratch) -- the shared guard's generic message is not a substitute there."""
    d = _write_adapter(tmp_path / "fresh", _TRANSFORMERS_KEYS)
    with pytest.raises(RuntimeError, match="start from scratch"):
        worker._assert_parent_adapter_live(_Model(_LIVE_FRESH), d)


def test_report_and_refusal_agree_with_the_shared_guard_end_to_end(worker, tmp_path):
    """Same verdicts on the full refusal matrix: which cases raise, and which
    report states pass, must be identical across both implementations."""
    d_ok = _write_adapter(tmp_path / "ok", _TRANSFORMERS_KEYS)
    d_fresh = _write_adapter(tmp_path / "fresh", _TRANSFORMERS_KEYS)
    cases = [(_Model(_LIVE_TRAINED), d_ok), (_Model(_LIVE_FRESH), d_fresh)]
    for model, adapter_dir in cases:
        try:
            worker._assert_parent_adapter_live(model, adapter_dir)
            worker_raised = False
        except RuntimeError:
            worker_raised = True
        try:
            assert_adapter_is_live(model, adapter_dir)
            shared_raised = False
        except AdapterNotLiveError:
            shared_raised = True
        assert worker_raised == shared_raised
