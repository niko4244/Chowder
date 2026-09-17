"""The Memory Fabric acceptance judge's verdict logic.

``docs/quals/memory_fabric_acceptance_qualification.py`` encodes the
acceptance sentence mechanically: a genuine resident CUDA OOM under a real
allocator-level VRAM constraint, a genuinely succeeding offload run of the
same model/recipe/GPU, with real measured peaks — and N/A is never
converted into a pass. These tests pin that logic so the qualification
artifact's verdict cannot silently loosen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUAL = HERE.parent / "docs" / "quals" / "memory_fabric_acceptance_qualification.py"

_spec = importlib.util.spec_from_file_location("memfab_qualification", QUAL)
module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("memfab_qualification", module)
_spec.loader.exec_module(module)

judge = module.judge


def _passing(**overrides):
    base = dict(
        resident_attempted=True,
        resident_oom=True,
        offload_attempted=True,
        offload_succeeded=True,
        same_model_recipe=True,
        vram_fraction_set=True,
        resident_peak_gb=18.698,
        offload_peak_gb=9.273,
        vram_ceiling_gb=15.13,
        attempts_resident=1,
        attempts_offload=1,
    )
    base.update(overrides)
    return judge(**base)


def test_attempt5_demonstrated_result_accepts():
    verdict, reasons = _passing()
    assert verdict == "ACCEPTED"
    assert any("acceptance sentence holds" in r for r in reasons)


def test_no_resident_oom_rejects():
    verdict, reasons = _passing(resident_oom=False)
    assert verdict == "REJECTED"
    assert any("did not raise a clean" in r for r in reasons)


def test_offload_failure_rejects_with_honest_reason():
    verdict, reasons = _passing(offload_succeeded=False)
    assert verdict == "REJECTED"
    assert any("WDDM flakiness or real failure" in r for r in reasons)


def test_missing_vram_fraction_rejects_as_budget_lie():
    verdict, reasons = _passing(vram_fraction_set=False)
    assert verdict == "REJECTED"
    assert any("reported-budget lie" in r for r in reasons)


def test_different_recipe_rejects():
    verdict, _ = _passing(same_model_recipe=False)
    assert verdict == "REJECTED"


def test_unattempted_phases_infra_refuse_not_pass():
    verdict, reasons = _passing(resident_attempted=False)
    assert verdict == "INFRAREFUSED"
    assert any("N/A is not a pass" in r for r in reasons)
    verdict, reasons = _passing(offload_attempted=False)
    assert verdict == "INFRAREFUSED"
    assert any("N/A is not a pass" in r for r in reasons)


def test_missing_peaks_infra_refuse():
    verdict, reasons = _passing(offload_peak_gb=None)
    assert verdict == "INFRAREFUSED"
    assert any("not a pass" in r for r in reasons)


def test_offload_peak_over_ceiling_rejects():
    verdict, reasons = _passing(offload_peak_gb=15.5)
    assert verdict == "REJECTED"
    assert any("exceeds" in r for r in reasons)
