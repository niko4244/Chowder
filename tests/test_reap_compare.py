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
import reap_select as rs  # noqa: E402


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


def _full_v2_quality_fixture(tmp_path):
    for suite, (size, _) in rc.FROZEN_QUALITY_COHORTS.items():
        write_arm(tmp_path, "reap128_v2", suite, [True] * size,
                  ids=([f"gsm8k-{i}" for i in range(size)] if suite == "gsm8k"
                       else [f"HumanEval/{i}" for i in range(size)]))
        write_arm(tmp_path, "dense", suite, [False] * size,
                  ids=([f"gsm8k-{i}" for i in range(size)] if suite == "gsm8k"
                       else [f"HumanEval/{i}" for i in range(size)]))
    folder = tmp_path / "reap128_v2"
    model = {"gguf": "I:/registered-example.gguf", "gguf_sha256": "a" * 64,
             "flags": rs.FROZEN_SERVE_FLAGS}
    (folder / "serve.json").write_text(json.dumps(model), encoding="utf-8")
    for suite, (size, _) in rc.FROZEN_QUALITY_COHORTS.items():
        witness = {"schema": rs.FROZEN_INPUT_SCHEMA, "model": model,
                   "suite": suite, "source_items_sha256": rg.FROZEN_SOURCE_SHA256[suite],
                   "source_items_n": size,
                   "decode": {"system": "You are a helpful assistant.",
                              "temperature": 0.0, "top_k": 1, "seed": 7,
                              "max_tokens": rs.FROZEN_MAX_TOKENS[suite],
                              "enable_thinking": False},
                   "scoring": rs.FROZEN_SCORING}
        (folder / f"{suite}.inputs.json").write_text(json.dumps(witness), encoding="utf-8")


def test_v2_descriptive_comparison_requires_both_full_registered_suites(tmp_path):
    # Before hardening, an apparently successful 4-item or GSM8K-only v2
    # comparison was reportable as a complete head-to-head win.
    write_arm(tmp_path, "reap128_v2", "gsm8k", [True] * 4)
    write_arm(tmp_path, "dense", "gsm8k", [False] * 4)
    with pytest.raises(rc.CompareError, match="source-bound comparison input|full registered"):
        rc.compare(tmp_path, "reap128_v2", ["dense"])


@pytest.mark.parametrize("tamper", ["source", "cohort", "correct"])
def test_v2_comparison_refuses_resealed_source_population_or_invalid_rows(tmp_path, tamper):
    _full_v2_quality_fixture(tmp_path)
    directory = tmp_path / "reap128_v2"
    if tamper == "source":
        path = directory / "gsm8k.inputs.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["source_items_sha256"] = "b" * 64
        path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        path = directory / "gsm8k.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        if tamper == "cohort":
            rows[0]["id"] = "gsm8k-not-frozen"
        else:
            rows[0]["correct"] = 2
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        summary["gsm8k"]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(rc.CompareError, match="source|cohort|correctness|nonbinary"):
        rc.compare(tmp_path, "reap128_v2", ["dense"])


def test_v2_comparison_valid_full_source_witness_is_descriptive_only(tmp_path):
    _full_v2_quality_fixture(tmp_path)
    result = rc.compare(tmp_path, "reap128_v2", ["dense"])
    assert result["candidate_beats_all"] is True
    assert {row["suite"]: row["n"] for row in result["rows"]} == {
        "gsm8k": 1319, "humaneval": 164}
    assert result["version"] == "reap-compare-v1"
    for name in ("serve.json", "gsm8k.inputs.json", "humaneval.inputs.json"):
        assert result["inputs"]["reap128_v2"][name] == hashlib.sha256(
            (tmp_path / "reap128_v2" / name).read_bytes()).hexdigest()
    assert result["qualification"] == "DESCRIPTIVE_PAIRED_COMPARISON_NOT_REGISTERED_SCIENTIFIC_GATE"


def test_new_v2_uses_frozen_full_gsm8k_only_random128_negative_control(tmp_path):
    _full_v2_quality_fixture(tmp_path)
    write_arm(tmp_path, "random128", "gsm8k", [False] * 1319,
              ids=[f"gsm8k-{i}" for i in range(1319)])
    result = rc.compare(tmp_path, "reap128_v2", ["dense", "random128"])
    assert result["candidate_beats_all"] is True
    assert {row["baseline"]: row["n"] for row in result["rows"]
            if row["suite"] == "gsm8k"} == {"dense": 1319, "random128": 1319}
    assert result["not_compared"] == [{"baseline": "random128",
                                       "suite": "humaneval", "candidate_has": True,
                                       "baseline_has": False}]
    assert result["qualification"].endswith("NOT_REGISTERED_SCIENTIFIC_GATE")


def test_v2_refuses_truncated_random128_negative_control(tmp_path):
    _full_v2_quality_fixture(tmp_path)
    write_arm(tmp_path, "random128", "gsm8k", [False] * 200)
    with pytest.raises(rc.CompareError, match="noncanonical or partial"):
        rc.compare(tmp_path, "reap128_v2", ["random128"])


@pytest.mark.parametrize("changed", ["serve.json", "gsm8k.inputs.json", "gsm8k.jsonl"])
def test_v2_paired_comparison_refuses_mutating_source_bound_inputs(
    tmp_path, monkeypatch, changed,
):
    _full_v2_quality_fixture(tmp_path)
    witness = tmp_path / "reap128_v2" / changed
    original_bounds = rc._bounds
    altered = []

    def change_during_statistics(values):
        if not altered:
            # A concurrent process alters the input after all earlier source
            # and summary checks, but before the signed descriptive result.
            with witness.open("ab") as stream:
                stream.write(b"\n")
            altered.append(True)
        return original_bounds(values)

    monkeypatch.setattr(rc, "_bounds", change_during_statistics)
    with pytest.raises(rc.CompareError, match="changed during scoring"):
        rc.compare(tmp_path, "reap128_v2", ["dense"])


@pytest.mark.parametrize("margin", [float("nan"), -0.1, float("inf")])
def test_comparison_refuses_invalid_margin_without_writing_report(tmp_path, margin):
    with pytest.raises(rc.CompareError, match="finite nonnegative"):
        rc.compare(tmp_path, "reap128_v2", ["dense"], margin)
