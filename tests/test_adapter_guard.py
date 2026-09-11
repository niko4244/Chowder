"""An adapter that cannot change the model must not be scored.

The case these tests encode was measured, not imagined: an Unsloth-trained adapter
loaded onto the model Chowder's evaluator builds produced a maximum logit delta of
0.000000 (against 14.5 for a Transformers-trained one) because Unsloth's saved keys
carry a `language_model.` segment the evaluator's model does not have. PEFT warned
about missing keys, left every LoRA B at zero, and the evaluation reported
`adapter_loaded: true` plus a score identical to the baseline.

No torch and no model: saved keys are read from a real safetensors header written
here with stdlib, and the model is a stub exposing `named_parameters()`.
"""

import json
import struct
from pathlib import Path

import pytest

from chowder.adapter_guard import (
    AdapterNotLiveError,
    adapter_liveness_report,
    assert_adapter_is_live,
    saved_adapter_keys,
)


class _Tensor:
    """Enough of a tensor for the guard: .detach().float().abs().max()."""

    def __init__(self, value: float) -> None:
        self._value = abs(value)

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


# Keys as each engine really writes them, from the measured diagnosis.
_TRANSFORMERS_KEYS = [
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight",
]
_UNSLOTH_KEYS = [
    "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
    "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight",
]
# What the evaluator's model exposes: text-only CausalLM, no `language_model.`
_LIVE_TRAINED = {
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight": 0.03,
    "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight": 0.07,
}
_LIVE_FRESH = {**_LIVE_TRAINED,
               "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight": 0.0}


def test_saved_adapter_keys_reads_a_real_safetensors_header(tmp_path):
    d = _write_adapter(tmp_path / "a", _TRANSFORMERS_KEYS)
    assert saved_adapter_keys(d) == set(_TRANSFORMERS_KEYS)


def test_missing_adapter_weights_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(AdapterNotLiveError, match="no adapter weights"):
        saved_adapter_keys(tmp_path / "empty")


def test_a_matching_trained_adapter_passes(tmp_path):
    d = _write_adapter(tmp_path / "ok", _TRANSFORMERS_KEYS)
    report = assert_adapter_is_live(_Model(_LIVE_TRAINED), d)
    assert report["matched_keys"] == 2
    assert report["lora_B_nonzero"] == 1


def test_the_real_unsloth_prefix_mismatch_is_refused(tmp_path):
    """The measured defect: keys carry `language_model.`, the live model does not.
    Zero overlap, every B still zero, and PEFT would have reported success."""
    d = _write_adapter(tmp_path / "unsloth", _UNSLOTH_KEYS)
    with pytest.raises(AdapterNotLiveError) as excinfo:
        assert_adapter_is_live(_Model(_LIVE_FRESH), d)
    message = str(excinfo.value)
    assert "shares NO parameter names" in message
    # the message must be diagnostic enough to act on without re-running anything
    assert "language_model" in message
    assert "ForConditionalGeneration" in message
    assert "describe the BASE model" in message


def test_an_all_zero_B_is_refused_even_when_keys_match(tmp_path):
    """Keys can match perfectly and the adapter still be an identity -- a fresh or
    never-trained adapter. Scoring it would describe the base model."""
    d = _write_adapter(tmp_path / "fresh", _TRANSFORMERS_KEYS)
    with pytest.raises(AdapterNotLiveError, match="exactly zero"):
        assert_adapter_is_live(_Model(_LIVE_FRESH), d)


def test_report_records_what_it_measured_for_provenance(tmp_path):
    d = _write_adapter(tmp_path / "ok", _TRANSFORMERS_KEYS)
    report = adapter_liveness_report(_Model(_LIVE_TRAINED), d)
    assert report["saved_tensors"] == 2
    assert report["live_lora_parameters"] == 2
    assert report["lora_B_parameters"] == 1
    assert report["example_saved_key"] and report["example_live_parameter"]
    assert Path(report["adapter_dir"]).name == "ok"


def test_unreadable_weights_do_not_fail_closed(tmp_path):
    """A measurement gap must not be reported as a dead adapter: if a B matrix
    cannot be read (quantised or exotic storage) the guard counts it as live and
    lets the key-overlap check carry the decision."""

    class _Opaque(_Tensor):
        def max(self):
            raise RuntimeError("unreadable storage")

    class _M:
        def named_parameters(self):
            return [
                ("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight", _Tensor(0.1)),
                ("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight", _Opaque(0.0)),
            ]

    d = _write_adapter(tmp_path / "opaque", _TRANSFORMERS_KEYS)
    report = assert_adapter_is_live(_M(), d)
    assert report["lora_B_nonzero"] == 1


def test_every_adapter_load_site_is_guarded():
    """Four places load an adapter; a fifth added later must not skip the check.
    unsloth_worker.py is excluded from the import-based check on purpose: its
    docstring forbids importing from the chowder package, so it carries the same
    guard inlined, asserted separately below."""
    import chowder

    src = Path(chowder.__file__).resolve().parent
    offenders = []
    for path in sorted(src.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "PeftModel.from_pretrained(" not in text:
            continue
        rel = path.relative_to(src).as_posix()
        if rel == "backends/unsloth_worker.py":
            assert "lora_B" in text and "identity" in text, "inlined guard missing"
            continue
        if "assert_adapter_is_live" not in text:
            offenders.append(rel)
    assert not offenders, f"adapter loaded without a liveness guard: {offenders}"
