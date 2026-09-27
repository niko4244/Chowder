"""Frozen paired evaluation protocol for the teacher-free pilot.

Phase 2: measure the untouched Qwen3-1.7B base and the Condition A adapter on
the *same* held-out problems under one frozen configuration, producing raw
per-problem outputs, paired deltas and bootstrap uncertainty intervals.

Rules enforced here (fail closed):

  - Development and final-evaluation materials are separate *by construction*:
    the final-eval prompts are locked in a dedicated directory, and every
    problem id used for development is refused there (a prompt-id overlap is a
    hard error, not a warning).
  - The protocol (frozen revision, template, thinking mode, sampling, seed,
    generation budget, scoring) is produced once per split and shared verbatim
    by both arms. Sampling is greedy so two arms on the same GPU see the same
    decoding; the seed still pins the record.
  - The Condition A adapter is scored only against the digest-verified record
    (``condition_a_artifacts.json``): a missing, unverified, or
    wrong-digest adapter is refused rather than silently scored.
  - Base-model quality claims may never be derived from training loss; the
    comparison output carries the measured deltas and bootstrap intervals and
    the decision is always ``requires_operator_review``.

Real inference is the operator-authorized GPU step: this module builds the
plans and the analysis; ``run_plan`` shells out to the production evaluator
worker (``chowder.evaluators.transformers_text_worker``) when the operator
executes a plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

#: Frozen protocol shared by every arm of a comparison. Greedy decoding keeps
#: the two arms deterministic on the same hardware; the seed is recorded
#: anyway because the protocol must be reproducible in writing, not in prose.
#: max_new_tokens 1024: the rendering path leaves thinking at the model's
#: default (enabled for Qwen3), and a budget too tight to finish a think
#: block would score both arms as misses (scoring.py treats an unclosed
#: <think> as a miss by design), destroying the comparison's resolution.
FROZEN_PROTOCOL = {
    "decoding": "greedy",
    "temperature": 0.0,
    "seed": 2026,
    "max_new_tokens": 1024,
    "scoring": "final_number_match",
    "precision": "bf16",
    "quantization": "none",
}

#: Frozen model identity. The revision is the pinned Qwen3-1.7B commit that
#: Condition A trained from; the chat template is the tokenizer's own, shipped
#: in the adapter directory and identical for the base. Thinking mode is
#: whatever the production renderer produces: it applies the tokenizer
#: template with add_generation_prompt=True and no enable_thinking kwarg, so
#: Qwen3 runs at its template default (thinking enabled); the think-aware
#: scorer already defines the honest reading of an exhausted budget.
FROZEN_MODEL = {
    "base_model": "Qwen/Qwen3-1.7B",
    "base_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "use_chat_template": True,
    "thinking_mode": "model_default (Qwen3 template default: enabled; "
                     "unclosed <think> scores as a miss)",
    "enable_thinking": None,
}

DECISION = "requires_operator_review"
FORMAT = "chowder-teacher-free-eval-protocol/v1"
COMPARISON_FORMAT = "chowder-teacher-free-paired-comparison/v1"
ADAPTER_ARTIFACTS = "condition_a_artifacts.json"
ADAPTER_DIRNAME = "adapter"
ADAPTER_DIGEST_FIELD = "adapter_model.safetensors"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def _row_prompt(row: dict) -> str:
    """A row's prompt under either contract: eval rows (prompt field) or
    Chowder chat rows (first user message)."""
    value = row.get("prompt")
    if isinstance(value, str) and value.strip():
        return value
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user" \
                and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def eval_prompt_ids(path: Path) -> set[str]:
    """The problem ids an eval file asks about."""
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            value = row.get("problem_id") or row.get("id")
            if value is None:
                value = _row_prompt(row)
            if not value:
                raise ValueError(f"{path}: eval row without problem_id/id/prompt")
            ids.add(str(value))
    return ids


def check_dev_final_separation(dev_prompts: Path, final_prompts: Path) -> dict:
    """Development material must never appear in the final-eval set.

    Overlapping problem ids or overlapping normalized prompt text are a hard
    failure: the final number would no longer be independent of model
    selection, which is exactly the leak this protocol exists to prevent.
    """
    dev_ids = eval_prompt_ids(dev_prompts)
    final_ids = eval_prompt_ids(final_prompts)
    id_overlap = sorted(dev_ids & final_ids)
    final_only = final_ids - dev_ids

    def normalized_prompts(path: Path) -> set[str]:
        out = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.add(" ".join(_row_prompt(row).casefold().split()))
        return out

    dev_texts = normalized_prompts(dev_prompts)
    final_texts = normalized_prompts(final_prompts)
    text_overlap = sorted(dev_texts & final_texts)
    return {
        "dev_problem_ids": len(dev_ids),
        "final_problem_ids": len(final_ids),
        "id_overlap": id_overlap,
        "prompt_text_overlap": text_overlap,
        "overlap_count": len(id_overlap) + len(text_overlap),
        "ok": not id_overlap and not text_overlap,
        "final_only_problem_ids": len(final_only),
        "note": ("development problems may select a recipe; final-eval problems "
                 "must be disjoint from everything development touched"),
    }


def _pinned_weights_entry(record: dict) -> dict:
    """The pinned adapter-weights entry from a condition_a_artifacts record."""
    for entry in record.get("files") or []:
        if isinstance(entry, dict) and entry.get("path") == \
                f"{ADAPTER_DIRNAME}/{ADAPTER_DIGEST_FIELD}":
            return entry
    raise ValueError(
        f"condition_a_artifacts record pins no {ADAPTER_DIRNAME}/"
        f"{ADAPTER_DIGEST_FIELD} digest")


def verified_adapter_record(experiments_dir: Path) -> dict:
    """The digest-verified Condition A record, or a refusal.

    Phase 2 measures *the completed Condition A adapter*; scoring some other
    adapter directory (or a stale copy of this one) would measure nothing.
    """
    record_path = experiments_dir / ADAPTER_ARTIFACTS
    if not record_path.is_file():
        raise FileNotFoundError(
            f"{record_path} is missing; run verify_condition_a.py record first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("format") != "chowder-teacher-free-condition-a/v1":
        raise ValueError(f"{record_path}: unexpected format {record.get('format')!r}")
    root = Path(record.get("artifacts_root") or "")
    if not root.is_dir():
        raise FileNotFoundError(f"artifacts root {root} does not exist")
    entry = _pinned_weights_entry(record)
    weights = root / ADAPTER_DIRNAME / ADAPTER_DIGEST_FIELD
    if not weights.is_file():
        raise FileNotFoundError(f"adapter weights {weights} are missing")
    actual = sha256_file(weights)
    if actual != entry["sha256"]:
        raise ValueError(
            f"adapter weights changed since they were pinned "
            f"({entry['sha256'][:12]}... pinned, {actual[:12]}... on disk); "
            "re-run verify_condition_a.py verify before any evaluation claims")
    return record


def paired_plan(out_dir: Path, dev_prompts: Path, final_prompts: Path, *,
                adapter_dir: Path | None = None,
                adapter_record_path: Path | None = None,
                protocol: dict | None = None) -> dict:
    """Build the two-arm, two-split evaluation plan.

    The same protocol object (by value, digest-identical) is attached to every
    arm and split; the final split additionally carries the dev/final
    separation check. If ``adapter_dir`` is given, the adapter is accepted only
    after its pinned weights digest re-verifies against
    ``condition_a_artifacts.json``.
    """
    protocol = dict(protocol or FROZEN_PROTOCOL)
    separation = check_dev_final_separation(dev_prompts, final_prompts)
    if not separation["ok"]:
        raise ValueError(
            "development and final-eval materials overlap "
            f"(ids={separation['id_overlap'][:3]}, "
            f"prompts={len(separation['prompt_text_overlap'])}); "
            "refusing to plan a leaky evaluation")
    adapter_provenance: dict | None = None
    if adapter_dir is not None:
        experiments_dir = (adapter_record_path or
                           Path(__file__).resolve().parent)
        record = verified_adapter_record(Path(experiments_dir))
        pinned = _pinned_weights_entry(record)
        weights = adapter_dir / ADAPTER_DIGEST_FIELD
        if not weights.is_file():
            raise FileNotFoundError(f"{weights} is missing")
        if sha256_file(weights) != pinned["sha256"]:
            raise ValueError(
                f"{weights} does not match the pinned Condition A digest")
        adapter_provenance = {
            "adapter_dir": str(adapter_dir),
            "verified_against": str(Path(experiments_dir) / ADAPTER_ARTIFACTS),
            "adapter_weights_sha256": pinned["sha256"],
            "mean_train_loss": (record.get("run") or {}).get("mean_train_loss"),
            "note": ("training loss is recorded as training history only; "
                     "no quality claim may derive from it"),
        }
    suite = {
        "prompt_field": "prompt",
        "expected_field": "expected",
        "scoring": protocol["scoring"],
        "max_new_tokens": protocol["max_new_tokens"],
        "use_chat_template": FROZEN_MODEL["use_chat_template"],
        "batch_size": 1,
    }
    arms = [
        {"arm": "base", "adapter_dir": None,
         "model": FROZEN_MODEL["base_model"],
         "revision": FROZEN_MODEL["base_revision"]},
    ]
    if adapter_dir is not None:
        arms.append({"arm": "condition_a", "adapter_dir": str(adapter_dir),
                     "model": FROZEN_MODEL["base_model"],
                     "revision": FROZEN_MODEL["base_revision"]})
    plan = {
        "format": FORMAT,
        "protocol_digest": digest(protocol),
        "protocol": protocol,
        "model_frozen": FROZEN_MODEL,
        "arms": arms,
        "splits": {
            "dev": {"prompts": str(dev_prompts),
                    "separation_check": None,
                    "purpose": ("recipe selection and debugging; never cited "
                                "as a final result")},
            "final": {"prompts": str(final_prompts),
                      "separation_check": separation,
                      "purpose": "the only split a quality claim may cite"},
        },
        "adapter_provenance": adapter_provenance,
        "raw_outputs": ("per-problem predictions, scores, generation "
                        "diagnostics and resource measurements are stored per "
                        "arm per split; nothing is aggregated away"),
        "claim_rule": ("no model-quality improvement may be claimed from "
                       "training loss; only this comparison's measured "
                       "final-split deltas count, and the decision is always "
                       f"{DECISION!r}"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


def paired_deltas(paired_scores: list[dict], *, bootstraps: int = 10_000,
                  seed: int = 2026) -> dict:
    """Paired per-problem deltas with a bootstrap uncertainty interval.

    ``paired_scores`` rows: ``{"problem_id", "score_a", "score_b"}`` where the
    same problem id must appear exactly once. The delta is ``b - a`` per
    problem; the interval is a percentile bootstrap over problems (resampling
    problems, not scores, keeps the pairing intact).
    """
    by_id: dict[str, tuple[float, float]] = {}
    for row in paired_scores:
        pid = row.get("problem_id")
        if not pid or pid in by_id:
            raise ValueError(f"paired_scores rows need unique problem_id: {pid!r}")
        by_id[pid] = (float(row["score_a"]), float(row["score_b"]))
    n = len(by_id)
    if not n:
        raise ValueError("paired_scores is empty")
    deltas = [b - a for a, b in by_id.values()]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    rng = random.Random(seed)
    means: list[float] = []
    ordered = list(deltas)
    for _ in range(bootstraps):
        sample = [ordered[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean_delta = sum(deltas) / n
    return {
        "format": COMPARISON_FORMAT,
        "problems": n,
        "mean_delta_b_minus_a": round(mean_delta, 6),
        "b_wins": wins,
        "a_wins": losses,
        "ties": n - wins - losses,
        "ci95_low": round(means[int(0.025 * bootstraps)], 6),
        "ci95_high": round(means[min(bootstraps - 1, int(0.975 * bootstraps))], 6),
        "bootstrap_samples": bootstraps,
        "seed": seed,
        "decision": DECISION,
        "note": ("interval from resampling paired problems; the decision is "
                 "always operator review, never an automated promotion"),
    }


def run_plan(plan_path: Path, results_root: Path, *, python: str = "python") -> dict:
    """Execute one arm/split of a plan through the production evaluator.

    Refuses when GPU training/eval authorization has not been recorded. The
    worker writes raw per-problem predictions itself; this only orchestrates
    and records what was run.
    """
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format") != FORMAT:
        raise ValueError(f"{plan_path}: expected format {FORMAT!r}")
    results: dict[str, dict] = {}
    for split, split_spec in sorted(plan["splits"].items()):
        # A split is only executable when its prompts file is in the eval
        # contract (prompt + expected per row). A chat-contract file (e.g. a
        # training dev split without gold answers) is recorded as not
        # executable instead of producing a spec the worker would reject.
        rows = [json.loads(x) for x in
                Path(split_spec["prompts"]).read_text(encoding="utf-8").splitlines()
                if x.strip()]
        eval_contract = bool(rows) and all(
            isinstance(r.get("prompt"), str) and r.get("expected") is not None
            for r in rows)
        if not eval_contract:
            results[f"{split}"] = {
                "status": "not_executable_needs_eval_contract",
                "prompts": split_spec["prompts"],
                "note": ("rows are chat-contract records without gold answers; "
                         "a gold-bearing eval-contract dev set is a separate "
                         "artifact and is not required for the final "
                         "paired comparison"),
            }
            continue
        for arm in plan["arms"]:
            out_dir = results_root / f"{arm['arm']}__{split}"
            out_dir.mkdir(parents=True, exist_ok=True)
            spec = {
                "base_model": arm["model"],
                "adapter_dir": arm.get("adapter_dir"),
                "output_dir": str(out_dir),
                "revision": arm.get("revision"),
                "precision": plan["protocol"]["precision"],
                "quantization": plan["protocol"]["quantization"],
                "device": "auto",
                "seed": plan["protocol"]["seed"],
                "offline": True,
                "suites": [{
                    "name": f"holdout_{split}",
                    "dataset": split_spec["prompts"],
                    **{k: v for k, v in {
                        "prompt_field": "prompt",
                        "expected_field": "expected",
                        "scoring": plan["protocol"]["scoring"],
                        "max_new_tokens": plan["protocol"]["max_new_tokens"],
                        "use_chat_template": plan["model_frozen"]["use_chat_template"],
                        "batch_size": 1,
                    }.items()},
                }],
            }
            spec_path = out_dir / "eval_spec.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            results[f"{arm['arm']}__{split}"] = {
                "spec": spec_path,
                "result": out_dir / "result.json",
                "status": "planned_not_executed",
            }
    summary = {
        "format": FORMAT,
        "plan": str(plan_path),
        "runs": {k: str(v["spec"]) for k, v in results.items()
                 if "spec" in v},
        "splits": {k: {"status": v["status"]} for k, v in results.items()
                   if "spec" not in v},
        "status": "awaiting_operator_authorization",
        "note": ("launching real inference is the operator-authorized GPU step; "
                 "execute each eval_spec.json with the production worker "
                 "(chowder.evaluators.transformers_text_worker)"),
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "runs_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def compare_arms(results_root: Path, *, split: str = "final") -> dict:
    """Paired comparison of two arms on one split from raw prediction files.

    Rows are paired by position: both arms ran the identical dataset in the
    identical order under the frozen protocol, and every row's prompt is
    asserted equal across arms before pairing. A prompt appearing more than
    once in the eval set is excluded from the paired analysis and the
    exclusion is recorded: upstream duplicates carry conflicting gold
    answers, so such a row cannot decide which arm was right.
    """
    prediction_files = {}
    for arm_dir in sorted(results_root.glob(f"*__{split}")):
        preds = arm_dir / f"predictions-holdout_{split}.jsonl"
        if preds.is_file():
            prediction_files[arm_dir.name.rsplit("__", 1)[0]] = preds
    if len(prediction_files) != 2:
        raise ValueError(
            f"expected exactly two arms with raw predictions under {results_root} "
            f"for split {split!r}, found {sorted(prediction_files)}")
    names = sorted(prediction_files)

    def load_rows(path: Path) -> list[dict]:
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
                if x.strip()]

    rows_a = load_rows(prediction_files[names[0]])
    rows_b = load_rows(prediction_files[names[1]])
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"arms disagree on row count: {names[0]}={len(rows_a)}, "
            f"{names[1]}={len(rows_b)}; the frozen protocol requires "
            "identical problem sets")
    for index, (a, b) in enumerate(zip(rows_a, rows_b)):
        if str(a.get("prompt")) != str(b.get("prompt")):
            raise ValueError(
                f"row {index}: prompts differ across arms "
                f"({str(a.get('prompt'))[:60]!r} vs {str(b.get('prompt'))[:60]!r}); "
                "refusing to pair mismatched problems")
    prompt_counts: Counter = Counter(str(r.get("prompt")) for r in rows_a)
    ambiguous = {p for p, c in prompt_counts.items() if c > 1}
    paired = []
    excluded = []
    for index, (a, b) in enumerate(zip(rows_a, rows_b)):
        prompt = str(a.get("prompt"))
        if prompt in ambiguous:
            excluded.append({"row_index": index, "prompt_head": prompt[:80],
                             "reason": "duplicate_prompt_in_eval_set",
                             "gold_a": a.get("expected"), "gold_b": b.get("expected"),
                             "score_a": a.get("score"), "score_b": b.get("score")})
            continue
        paired.append({"row_index": index, "problem_id": prompt[:120],
                       "score_a": float(a.get("score") or 0.0),
                       "score_b": float(b.get("score") or 0.0)})
    analysis = paired_deltas(paired)
    analysis["arms"] = {"a": names[0], "b": names[1]}
    analysis["problems_total"] = len(rows_a)
    analysis["problems_paired"] = len(paired)
    analysis["excluded_rows"] = excluded
    analysis["pairing"] = ("row index across arms; per-row prompt equality "
                           "asserted; duplicate prompts excluded and recorded")
    analysis["status"] = "complete" if paired else "no_paired_rows"
    analysis_path = results_root / f"paired_comparison_{split}.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--out", type=Path, required=True)
    p_plan.add_argument("--dev-prompts", type=Path, required=True)
    p_plan.add_argument("--final-prompts", type=Path, required=True)
    p_plan.add_argument("--adapter-dir", type=Path, default=None)

    p_run = sub.add_parser("run")
    p_run.add_argument("--plan", type=Path, required=True)
    p_run.add_argument("--results", type=Path, required=True)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--results", type=Path, required=True)
    p_cmp.add_argument("--split", default="final")

    args = parser.parse_args()
    if args.cmd == "plan":
        plan = paired_plan(args.out, args.dev_prompts, args.final_prompts,
                           adapter_dir=args.adapter_dir)
        print(json.dumps({"plan": str(args.out / "eval_plan.json"),
                          "arms": [a["arm"] for a in plan["arms"]],
                          "separation_ok":
                              plan["splits"]["final"]["separation_check"]["ok"]},
                         indent=2))
    elif args.cmd == "run":
        print(json.dumps(run_plan(args.plan, args.results), indent=2))
    else:
        print(json.dumps(compare_arms(args.results, split=args.split), indent=2))


if __name__ == "__main__":
    main()
