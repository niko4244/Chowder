"""Head-to-head comparison of one REAP arm against named baselines, paired per item.

  python chowder_batch/reap_compare.py --eval-dir D --candidate reap128 \
      --baselines dense,random128 --out compare.json [--margin 0.0]

For every baseline, and every suite both arms share, this pairs the per-item
``correct`` flags on identical ids and reports the accuracy of each arm, the
paired delta, and a 95% interval. The interval is the one-sided frozen
statistic from ``reap_gates.py`` (``mean - 1.645 * sd / sqrt(n)``) mirrored
upwards as ``mean + 1.645 * sd / sqrt(n)``, so the lower bound printed here is
exactly the Gate 1 / Gate 2 statistic.

Verdicts use ``--margin`` (default 0.0) as an equivalence band:

  BEATS  lower bound >  margin
  LOSES  upper bound < -margin
  TIES   otherwise

Per baseline: BEATS only if every shared suite BEATS (and at least one suite is
shared), LOSES if any suite LOSES, MIXED otherwise.

Exit codes: 0 candidate BEATS every baseline, 1 otherwise, 2 integrity error
(arm or suite missing, prediction sha256 not matching ``summary.json``, item ids
differing between the paired arms, duplicate ids, a paired sample too small for
a bound). On exit 2 no report is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

try:  # script mode (python chowder_batch/reap_compare.py)
    from reap_gates import lower_bound
except ImportError:  # imported as chowder_batch.reap_compare
    from chowder_batch.reap_gates import lower_bound

REPORT_VERSION = "reap-compare-v1"


class CompareError(Exception):
    """Refusal: the inputs cannot be compared or are not what they claim to be."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(eval_dir, arm: str) -> dict:
    path = Path(eval_dir) / arm / "summary.json"
    if not path.is_file():
        raise CompareError(f"{arm}: missing {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise CompareError(f"{arm}: {path} is not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise CompareError(f"{arm}: {path} is not a suite->stats object")
    return data


def _rows(eval_dir, arm: str, suite: str, entry: dict) -> dict:
    """Per-item rows for one arm/suite, refusing unless summary.json vouches for them."""
    pred = Path(eval_dir) / arm / f"{suite}.jsonl"
    if not pred.is_file():
        raise CompareError(f"{arm}: summary.json lists {suite} but {pred} is missing")
    expected = entry.get("file_sha256")
    actual = _sha256(pred)
    if expected != actual:
        raise CompareError(
            f"{arm}/{suite}: prediction sha256 {actual} != summary.json file_sha256 {expected}"
        )
    rows = {}
    for line in pred.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row["id"]
        if rid in rows:
            raise CompareError(f"{arm}/{suite}: duplicate item id {rid!r}")
        rows[rid] = row
    if entry.get("n") != len(rows):
        raise CompareError(
            f"{arm}/{suite}: summary.json n={entry.get('n')} but {len(rows)} rows on disk"
        )
    return rows


def _bounds(deltas: list[float]) -> tuple[float, float]:
    """95% paired interval; the lower leg is reap_gates' frozen statistic."""
    lb = lower_bound(deltas)
    if lb is None:
        raise CompareError(f"n={len(deltas)}: need at least 2 paired items for a bound")
    mean = statistics.fmean(deltas)
    return lb, 2 * mean - lb  # == mean + 1.645 * sd / sqrt(n)


def _verdict(lb: float, ub: float, margin: float) -> str:
    if lb > margin:
        return "BEATS"
    if ub < -margin:
        return "LOSES"
    return "TIES"


def _overall(verdicts: list[str]) -> str:
    if not verdicts:
        return "MIXED"  # nothing shared; cannot claim a win
    if "LOSES" in verdicts:
        return "LOSES"
    if all(v == "BEATS" for v in verdicts):
        return "BEATS"
    return "MIXED"


def compare(eval_dir, candidate: str, baselines: list[str], margin: float = 0.0) -> dict:
    summaries = {candidate: _summary(eval_dir, candidate)}
    for arm in baselines:
        summaries[arm] = _summary(eval_dir, arm)

    # Read (and integrity-check) each input exactly once.
    rows: dict[tuple[str, str], dict] = {}
    for arm, summary in summaries.items():
        for suite in summary:
            rows[(arm, suite)] = _rows(eval_dir, arm, suite, summary[suite])

    all_suites: list[str] = []
    for summary in summaries.values():
        all_suites += [s for s in summary if s not in all_suites]

    compared, verdicts, not_compared = [], {}, []
    for arm in baselines:
        cand_suites, base_suites = set(summaries[candidate]), set(summaries[arm])
        shared = [s for s in all_suites if s in cand_suites & base_suites]
        for suite in all_suites:
            if suite not in shared:
                not_compared.append({"baseline": arm, "suite": suite,
                                     "candidate_has": suite in cand_suites,
                                     "baseline_has": suite in base_suites})
        arm_verdicts = []
        for suite in shared:
            cand, base = rows[(candidate, suite)], rows[(arm, suite)]
            if set(cand) != set(base):
                only_c = sorted(set(cand) - set(base))
                only_b = sorted(set(base) - set(cand))
                raise CompareError(
                    f"{candidate}/{arm} {suite}: item ids differ — "
                    f"{len(only_c)} only in candidate (e.g. {only_c[:3]}), "
                    f"{len(only_b)} only in baseline (e.g. {only_b[:3]})"
                )
            ids = sorted(cand)
            deltas = [float(cand[i]["correct"]) - float(base[i]["correct"]) for i in ids]
            lb, ub = _bounds(deltas)
            delta = statistics.fmean(deltas)
            verdict = _verdict(lb, ub, margin)
            arm_verdicts.append(verdict)
            compared.append({
                "baseline": arm, "suite": suite, "n": len(ids),
                "candidate_accuracy": sum(float(cand[i]["correct"]) for i in ids) / len(ids),
                "baseline_accuracy": sum(float(base[i]["correct"]) for i in ids) / len(ids),
                "delta": delta, "lower": lb, "upper": ub, "verdict": verdict,
            })
        verdicts[arm] = _overall(arm_verdicts)

    inputs = {arm: {"summary.json": _sha256(Path(eval_dir) / arm / "summary.json")}
              for arm in summaries}
    for arm, suite in rows:
        inputs[arm][f"{suite}.jsonl"] = _sha256(Path(eval_dir) / arm / f"{suite}.jsonl")

    return {
        "version": REPORT_VERSION,
        "eval_dir": str(Path(eval_dir)),
        "candidate": candidate,
        "baselines": baselines,
        "margin": margin,
        "suites": all_suites,
        "inputs": inputs,
        "rows": compared,
        "verdicts": verdicts,
        "not_compared": not_compared,
        "candidate_beats_all": all(v == "BEATS" for v in verdicts.values()),
    }


def format_table(report: dict) -> str:
    lines = [f"{'baseline':<16} {'suite':<10} {'n':>5} {'cand':>7} {'base':>7} "
             f"{'delta':>8} {'95% low':>8} {'95% high':>8}  verdict"]
    for r in report["rows"]:
        lines.append(f"{r['baseline']:<16} {r['suite']:<10} {r['n']:>5} "
                     f"{r['candidate_accuracy']:>7.4f} {r['baseline_accuracy']:>7.4f} "
                     f"{r['delta']:>+8.4f} {r['lower']:>+8.4f} {r['upper']:>+8.4f}  {r['verdict']}")
    for nc in report["not_compared"]:
        have = "candidate" if nc["candidate_has"] else "baseline"
        lines.append(f"{nc['baseline']:<16} {nc['suite']:<10}  not compared ({have} only)")
    for arm, verdict in report["verdicts"].items():
        lines.append(f"overall: {arm} {verdict}")
    lines.append(f"candidate beats all baselines: {'YES' if report['candidate_beats_all'] else 'NO'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--baselines", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--margin", type=float, default=0.0)
    a = p.parse_args(argv)

    baselines = [b.strip() for b in a.baselines.split(",") if b.strip()]
    try:
        if not baselines:
            raise CompareError("--baselines is empty")
        if a.candidate in baselines:
            raise CompareError(f"candidate {a.candidate!r} also listed as a baseline")
        report = compare(a.eval_dir, a.candidate, baselines, a.margin)
    except CompareError as e:
        print(f"reap_compare: refusing: {e}", file=sys.stderr)
        return 2

    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_table(report))
    return 0 if report["candidate_beats_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
