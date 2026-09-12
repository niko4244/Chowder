"""An evaluation must record its own VRAM footprint.

It did not, and that cost a verdict. `PRUNED_9B_REAL_TRAINING_PREREG.md`
pre-registered "peak VRAM under 15.93 GiB" and listed oversubscription as a FAIL
"judged by headroom and step-time blowup". Neither quantity existed in the
evaluation leg's artifacts, so the only proxy was `nvidia-smi` -- which measures the
whole machine. It read 559 MiB free during the candidate eval and the run was
recorded FAIL on oversubscription, while controlled re-measurement afterwards put
the evaluation process's own footprint at ~6.6 GiB with ~8.8 GiB of card free.

A busy desktop must not be able to fail an experiment.
"""

from __future__ import annotations

from pathlib import Path

from chowder.evaluators.vram import peak_vram


def test_cpu_reports_not_measured_rather_than_zero() -> None:
    """Unknown is not zero. A CPU run reporting 0.0 GiB would read as 'used no
    memory', the same trap `target_coverage` avoids with status 'not_reported'."""
    assert peak_vram("cpu") == {"peak_vram_gb": None, "peak_vram_reserved_gb": None}


def test_both_keys_are_always_present() -> None:
    """Callers splat this into a runtime dict, so the shape must not vary."""
    for device in ("cpu", "cuda", "cuda:0", "mps", ""):
        assert set(peak_vram(device)) == {"peak_vram_gb", "peak_vram_reserved_gb"}


def test_it_never_raises_when_torch_does(monkeypatch) -> None:
    """Telemetry must not be able to kill a run -- the rule progress_write.py exists
    to enforce, learned when a telemetry rename discarded 323 steps.

    Driven by making torch raise, not by passing a bogus device index: the first
    version of this test asserted `peak_vram("cuda:99")` returns None, which passed
    in the full suite and FAILED standalone, because torch only rejects an
    out-of-range index once CUDA has been initialised by some earlier test. An
    order-dependent test is worse than none.
    """
    try:
        import torch
    except ImportError:
        return

    def boom(*_args, **_kwargs):
        raise RuntimeError("invalid device ordinal")

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", boom)
    assert peak_vram("cuda:0") == {"peak_vram_gb": None, "peak_vram_reserved_gb": None}


def test_reports_both_allocated_and_reserved_when_cuda_is_present() -> None:
    """The distinction carried today's investigation: allocated is what the model
    needed, reserved is what torch took from the driver and therefore what decides
    whether the run fits alongside anything else."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    got = peak_vram("cuda:0")
    assert isinstance(got["peak_vram_gb"], float)
    assert isinstance(got["peak_vram_reserved_gb"], float)
    assert got["peak_vram_reserved_gb"] >= got["peak_vram_gb"]


def test_both_evaluator_workers_report_their_footprint() -> None:
    """BOTH arms, or a baseline-vs-candidate VRAM comparison is still impossible
    from run artifacts -- which is the defect, not merely a missing field."""
    import chowder

    evaluators = Path(chowder.__file__).resolve().parent / "evaluators"
    for name in ("transformers_text_worker.py", "base_text_worker.py"):
        source = (evaluators / name).read_text(encoding="utf-8")
        assert "from .vram import peak_vram" in source, f"{name} does not import the helper"
        assert "**_peak_vram(device_name)" in source, (
            f"{name} does not splat its VRAM footprint into the runtime block, so a "
            "pre-registered peak-VRAM condition stays undecidable for this leg"
        )
