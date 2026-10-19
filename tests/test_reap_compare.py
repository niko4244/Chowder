import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
import reap_compare as rc  # noqa: E402
import reap_gates as rg  # noqa: E402


def write_arm(base, arm, suite, correct, ids=None):
    d = Path(base) / arm
    d.mkdir(parents=True, exist_ok=True)
    ids = ids or [f"{suite}-{i}" for i in range(len(correct))]
    rows = [{"id": i, "correct": bool(c), "terminated": True, "trigram_ratio": 0.9,
             "completion_tokens": 10, "seconds": 1.0, "text": "x"}
            for i, c in zip(ids, correct)]
    path = d / f"{suite}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    summary[suite] = {"n": len(rows), "accuracy": sum(map(bool, correct)) / len(rows),
                      "terminated_rate": 1.0, "degenerate_rate": 0.0,
                      "mean_completion_tokens": 10.0,
                      "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def run(tmp_path, baselines="dense", margin=None):
    out = Path(tmp_path) / "report.json"
    argv = ["--eval-dir", str(tmp_path), "--candidate", "reap128",
            "--baselines", baselines, "--out", str(out)]
    if margin is not None:
        argv += ["--margin", str(margin)]
    code = rc.main(argv)
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_clear_win(tmp_path):
    n = 200
    write_arm(tmp_path, "reap128", "gsm8k", [1] * n)
    write_arm(tmp_path, "dense", "gsm8k", [0] * n)
    code, report = run(tmp_path, margin=0.1)
    row = report["rows"][0]
    assert code == 0
    assert report["version"] == "reap-compare-v1"
    assert (row["candidate_accuracy"], row["baseline_accuracy"]) == (1.0, 0.0)
    assert row["delta"] == 1.0 and row["lower"] == 1.0 and row["upper"] == 1.0
    assert row["lower"] == pytest.approx(rg.lower_bound([1.0] * n))
    assert row["verdict"] == "BEATS" and report["verdicts"] == {"dense": "BEATS"}
    assert report["candidate_beats_all"] is True
    assert report["inputs"]["reap128"]["gsm8k.jsonl"] == \
        hashlib.sha256((Path(tmp_path) / "reap128" / "gsm8k.jsonl").read_bytes()).hexdigest()


def test_clear_loss(tmp_path):
    n = 200
    write_arm(tmp_path, "reap128", "humaneval", [0] * n)
    write_arm(tmp_path, "dense", "humaneval", [1] * n)
    code, report = run(tmp_path)
    row = report["rows"][0]
    assert code == 1
    assert row["delta"] == -1.0 and row["upper"] == -1.0
    assert row["verdict"] == "LOSES"
    assert report["verdicts"] == {"dense": "LOSES"}
    assert report["candidate_beats_all"] is False


def test_tie_within_margin(tmp_path):
    n = 40
    write_arm(tmp_path, "reap128", "gsm8k", [i % 2 for i in range(n)])
    write_arm(tmp_path, "dense", "gsm8k", [(i + 1) % 2 for i in range(n)])
    code, report = run(tmp_path)
    row = report["rows"][0]
    assert code == 1
    assert row["delta"] == 0.0
    assert row["lower"] < 0.0 < row["upper"]
    assert row["verdict"] == "TIES"


def test_mismatched_item_ids_refused(tmp_path):
    write_arm(tmp_path, "reap128", "gsm8k", [1, 1, 1])
    write_arm(tmp_path, "dense", "gsm8k", [1, 1, 1], ids=["gsm8k-0", "gsm8k-1", "gsm8k-9"])
    code, report = run(tmp_path)
    assert code == 2
    assert report is None  # nothing written on refusal


def test_prediction_sha256_mismatch_refused(tmp_path):
    write_arm(tmp_path, "reap128", "gsm8k", [1, 0, 1])
    write_arm(tmp_path, "dense", "gsm8k", [1, 1, 1])
    summary_path = Path(tmp_path) / "reap128" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gsm8k"]["file_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    code, report = run(tmp_path)
    assert code == 2
    assert report is None


def test_suite_only_in_one_arm_is_not_compared(tmp_path):
    write_arm(tmp_path, "reap128", "gsm8k", [1] * 10)
    write_arm(tmp_path, "reap128", "humaneval", [1] * 10)
    write_arm(tmp_path, "dense", "gsm8k", [0] * 10)
    code, report = run(tmp_path)
    assert code == 0
    assert [r["suite"] for r in report["rows"]] == ["gsm8k"]
    assert report["verdicts"] == {"dense": "BEATS"}
    assert report["not_compared"] == [
        {"baseline": "dense", "suite": "humaneval", "candidate_has": True, "baseline_has": False}]


def test_suite_of_another_baseline_is_not_attributed(tmp_path):
    # math500 exists only in "other", so it is not a suite either "dense" pair lacks together
    write_arm(tmp_path, "reap128", "gsm8k", [1] * 10)
    write_arm(tmp_path, "dense", "gsm8k", [0] * 10)
    write_arm(tmp_path, "other", "gsm8k", [0] * 10)
    write_arm(tmp_path, "other", "math500", [1] * 10)
    code, report = run(tmp_path, baselines="dense,other")
    assert code == 0
    assert report["not_compared"] == [
        {"baseline": "other", "suite": "math500", "candidate_has": False, "baseline_has": True}]


def test_overall_mixed_beats_one_ties_other(tmp_path):
    n = 40
    write_arm(tmp_path, "reap128", "gsm8k", [1] * n)
    write_arm(tmp_path, "dense", "gsm8k", [0] * n)
    write_arm(tmp_path, "reap128", "humaneval", [i % 2 for i in range(n)])
    write_arm(tmp_path, "dense", "humaneval", [(i + 1) % 2 for i in range(n)])
    code, report = run(tmp_path)
    assert code == 1
    assert {r["suite"]: r["verdict"] for r in report["rows"]} == \
        {"gsm8k": "BEATS", "humaneval": "TIES"}
    assert report["verdicts"] == {"dense": "MIXED"}


def test_cli_exit_codes_via_subprocess(tmp_path):
    write_arm(tmp_path, "reap128", "gsm8k", [1] * 20)
    write_arm(tmp_path, "dense", "gsm8k", [0] * 20)
    out = Path(tmp_path) / "cli.json"
    done = subprocess.run(
        [sys.executable, str(ROOT / "chowder_batch" / "reap_compare.py"),
         "--eval-dir", str(tmp_path), "--candidate", "reap128",
         "--baselines", "dense", "--out", str(out)],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text())["candidate_beats_all"] is True
