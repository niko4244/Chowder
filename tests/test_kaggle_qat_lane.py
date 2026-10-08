from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
sys.path.insert(0, str(ROOT / "src"))

from exp_e_confidence import margin_shift_fails_closed  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("kaggle_qat_lane", ROOT / "kaggle" / "run_qat_distill_lane.py")
lane = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lane)


def _valid_row(name: str = "repair_evolve") -> dict:
    return {
        "id": f"batch010-{name}-abc",
        "task_name": name,
        "split": "evolve",
        "harness": "state_aware",
        "green_verified": True,
        "messages": [
            {"role": "system", "content": "Files: app.py. Read listed paths; test before success."},
            {"role": "user", "content": f"Repair app.py ({name})."},
            {"role": "assistant", "content": "Verified after green tests."},
        ],
    }


def test_lane_mirror_matches_local_margin_shift_guard():
    # Keep-in-sync contract: the Kaggle mirror must agree with the local
    # exp_e_confidence guard on every branch it covers.
    for bf16, quant, tolerance in ((2.0, 2.1, 0.2), (2.0, 1.7, 0.2), (2.0, None, 0.2), (2.0, 1.9, float("nan"))):
        mirrored = lane.build_shift_report(bf16, quant, tolerance)
        local_shift = None if (bf16 is None or quant is None) else quant - bf16
        # build_shift_report returns fails_closed=True for unmeasured/invalid;
        # the local guard is the single source of truth for the verdict, fed
        # the same tolerance verbatim (it fails closed on non-finite values).
        expected = margin_shift_fails_closed(local_shift, tolerance)
        assert mirrored["quantized_margin_shift_fails_closed"] is expected, (bf16, quant, tolerance)
        assert mirrored["margin_shift"] == local_shift


def test_load_teacher_rows_accepts_valid_and_rejects_every_contaminant(tmp_path):
    path = tmp_path / "batch010.jsonl"
    path.write_text(json.dumps(_valid_row()) + "\n", encoding="utf-8")
    rows = lane.load_teacher_rows(path)
    assert len(rows) == 1 and rows[0]["task_name"] == "repair_evolve"

    def rejects(payload_rows, match):
        path.write_text("\n".join(json.dumps(row) for row in payload_rows) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            lane.load_teacher_rows(path)

    rejects([], "refusing to train on an empty dataset")
    rejects([{**_valid_row(), "split": "heldout"}], "green-verified evolve trajectory")
    rejects([{**_valid_row(), "green_verified": False}], "green-verified evolve trajectory")
    rejects([{**_valid_row(), "harness": "plain"}], "state-aware harness")
    rejects([{**_valid_row(), "messages": []}], "missing its transcript")
    rejects([{**_valid_row(), "messages": _valid_row()["messages"][:-1]}], "end in a nonempty assistant report")
    rejects(
        [{**_valid_row(), "messages": [{"role": "tool", "content": "1 passed"}] + _valid_row()["messages"]}],
        "malformed message",
    )


def test_sft_batches_supervises_assistant_tokens_only(tmp_path):
    pytest.importorskip("transformers")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    except Exception:  # offline environment: the mask logic is exercised on Kaggle
        pytest.skip("tiny test tokenizer unavailable offline")
    path = tmp_path / "batch010.jsonl"
    path.write_text(json.dumps(_valid_row()) + "\n", encoding="utf-8")
    rows = lane.load_teacher_rows(path)
    batches = list(lane.sft_batches(rows, tokenizer, max_len=4096))
    assert len(batches) == 1
    batch = batches[0]
    supervised = [position for position, label in enumerate(batch["labels"]) if label != -100]
    assert supervised, "assistant report must contribute supervised tokens"
    decoded = tokenizer.decode([batch["input_ids"][p] for p in supervised])
    assert "Verified after green tests." in decoded
    assert "Repair app.py" not in decoded


def test_install_chowder_rejects_branch_names_and_short_hashes():
    with pytest.raises(ValueError, match="40-character"):
        lane.install_chowder("main")
    with pytest.raises(ValueError, match="40-character"):
        lane.install_chowder("abc123")
