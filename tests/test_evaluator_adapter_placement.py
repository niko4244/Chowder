"""Attaching an adapter must not silently un-place the model.

The defect these pin, with its evidence: the Gen-0 trusted-ancestor arm (bare
base) reported ``params on cuda=3 cpu=0 other=424`` and finished the math500
slice in 23 minutes, while the Gen-1 parent arm (base + LoRA adapter) reported
``params on cuda=0 cpu=683 other=0`` and was killed by the declared 7200 s worker
timeout still inside that same slice. Attaching a PEFT adapter re-places the
model it wraps, so the placement the worker applied to the bare base no longer
describes the model that generates.

The rule is owned by ``placement.needs_redispatch_after_adapter`` and applied by
``transformers_text_worker.placement_after_adapter``; the tests below pin both,
using the real functions rather than a restatement of them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chowder.evaluators import transformers_text_worker as worker
from chowder.evaluators.placement import needs_redispatch_after_adapter


@pytest.mark.parametrize(
    ("quantization", "placement", "adapter", "expected"),
    [
        # The case that cost a timed-out measurement: offload + adapter.
        ("none", "offload", True, True),
        # The bare-base arm, which is already correct and must not be re-placed.
        ("none", "offload", False, False),
        # A resident model is on the card by definition; nothing to re-assert.
        ("none", "resident", True, False),
        # 4-bit loads through a device map and never takes the offload path.
        ("4bit", "offload", True, False),
        ("4bit", "resident", True, False),
    ],
)
def test_only_an_offloaded_adapter_run_needs_the_placement_re_applied(
    quantization: str, placement: str, adapter: bool, expected: bool
) -> None:
    assert (
        needs_redispatch_after_adapter(
            quantization=quantization, placement=placement, adapter=adapter
        )
        is expected
    )


def test_the_placement_is_re_applied_to_the_model_the_wrapper_generates_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-application must target the base the adapter was injected into."""
    placed: list[tuple[object, str]] = []
    monkeypatch.setattr(
        worker,
        "dispatch_offloaded",
        lambda model, device_name: placed.append((model, device_name)) or model,
    )
    base = object()
    spec = SimpleNamespace(quantization="none", placement="offload")

    returned = worker.placement_after_adapter(base, spec=spec, device_name="cuda:0")

    assert returned is base
    assert placed == [(base, "cuda:0")]


def test_a_resident_adapter_run_is_not_re_placed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared resident run keeps its placement: no second dispatch."""
    placed: list[object] = []
    monkeypatch.setattr(
        worker,
        "dispatch_offloaded",
        lambda model, device_name: placed.append(model) or model,
    )
    base = object()
    spec = SimpleNamespace(quantization="none", placement="resident")

    assert worker.placement_after_adapter(base, spec=spec, device_name="cuda:0") is base
    assert placed == []


class _RecordingPeft:
    calls: list[tuple[object, object, dict]] = []

    @classmethod
    def from_pretrained(cls, base, adapter_dir, **kwargs):  # noqa: ANN001
        cls.calls.append((base, adapter_dir, kwargs))
        return "peft-model"


@pytest.mark.parametrize(("placement", "expects_folder"), [("offload", True), ("resident", False)])
def test_an_offloaded_adapter_load_gives_peft_an_offload_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path, placement: str, expects_folder: bool
) -> None:
    """Gen-2 attempt-16: PEFT re-inferred a device map from the RAM free at load
    time, sent tail modules to "disk", and raised for want of an offload folder,
    after the attempt had trained all 140 steps."""
    import sys
    import types

    _RecordingPeft.calls = []
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=_RecordingPeft))
    monkeypatch.setenv("CHOWDER_EVAL_OFFLOAD_DIR", str(tmp_path / "offload"))
    spec = SimpleNamespace(placement=placement, adapter_dir="adapter")

    assert worker.load_adapter("base", spec) == "peft-model"
    (base, adapter_dir, kwargs), = _RecordingPeft.calls
    assert (base, adapter_dir, kwargs["is_trainable"]) == ("base", "adapter", False)
    assert ("offload_folder" in kwargs) is expects_folder
    if expects_folder:
        assert Path(kwargs["offload_folder"]).is_dir()


@pytest.mark.parametrize(
    ("placement", "adapter_dir", "expect_dispatch", "expect_to"),
    [
        ("offload", "adapter", False, False),  # deferred: dispatched once, after the adapter
        ("offload", None, True, False),       # bare base: dispatched now
        ("resident", "adapter", False, True),
        ("resident", None, False, True),
    ],
)
def test_an_offloaded_base_is_not_dispatched_before_its_adapter(
    monkeypatch: pytest.MonkeyPatch, placement, adapter_dir, expect_dispatch, expect_to
) -> None:
    """Gen-2 attempt-22: a pre-dispatched base let PEFT re-plan onto disk, and the
    declared re-dispatch then failed on a meta (disk-offloaded) parameter."""
    dispatched: list[object] = []
    monkeypatch.setattr(worker, "dispatch_offloaded", lambda m, d: dispatched.append(m) or m)

    class _Base:
        moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    base = _Base()
    spec = SimpleNamespace(quantization="none", placement=placement, adapter_dir=adapter_dir)
    assert worker.place_before_adapter(base, spec=spec, device_name="cuda:0") is base
    assert (dispatched == [base]) is expect_dispatch
    assert (base.moved_to == "cuda:0") is expect_to
