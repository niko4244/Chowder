"""Seed the frontier reference database with cited published numbers.

Lane D of the Gen-0 closeout. The database previously held zero references
with a documented rationale (`docs/FRONTIER_REFERENCE_SEED_2026-09-17.md`).
This builder imports real, citable, first-party numbers for the two
benchmarks Generation 0 actually measured, and marks every one of them with
the comparability confidence its protocol earns.

Judgment calls, stated plainly:

* **Nothing here is HIGH.** Chowder's frozen protocol
  (``gen0-freeze-protocol-v1``) is lm-eval ``minerva_math`` limited to 4
  samples per subtask (28 items) and lm-eval ``mgsm_direct_en`` limited to 24
  items, 0-shot, chat template applied, greedy. Every published number found
  uses a different subset, shot count, prompt format, or sampling regime.
  ``compare_protocol`` therefore refuses all of them, which is the correct
  outcome -- an importable number is not a comparable number.
* **The one MEDIUM row is the interesting one.** `lm-evaluation-harness`
  issue #2646 reports Llama-3.1-8B-Instruct at ``minerva_math`` 4-shot
  ``exact_match`` 0.3456 -- the same task and metric Chowder's adapter uses,
  with a fully specified command. It is still not comparable (4-shot vs
  Chowder's 0-shot chat template; full split vs a 28-item limit), but it is
  direct evidence that the *same model* scores 0.346 under this harness while
  its publisher reports 0.519 on MATH and 0.928-family numbers appear on
  MATH-500. Protocol, not capability, dominates this benchmark family at
  small subsets. That is precisely why Chowder's Gen-0 math/mgsm zeros must
  not be read as a capability floor.
* **Blanks stay blank.** No LEVEL_4 row exists for MGSM because no
  first-party absolute-frontier MGSM number was found. Filling it to avoid a
  blank would be the failure mode the seed doc already warns about.

Usage:
    PYTHONPATH=<worktree>/src python seed_frontier_references.py \\
        --root <freeze-root> --snapshot-id gen0-frontier-context-2026-09-17
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src").as_posix()))

from chowder.growth.frontier_reference import (  # noqa: E402
    LEVEL_1_COMPARABLE_PEER,
    LEVEL_2_OPEN_WEIGHT_FRONTIER,
    LEVEL_3_STRETCH,
    LEVEL_4_ABSOLUTE_FRONTIER,
    ChowderScore,
    FrontierDatabase,
    ReferenceScore,
    SnapshotStore,
    context_rows,
    gap_rows,
)

DEEPSEEK_R1_REPO = "https://github.com/deepseek-ai/deepseek-r1"
QWEN25_BLOG = "https://qwenlm.github.io/blog/qwen2.5-llm/"
LM_EVAL_ISSUE_2646 = "https://github.com/EleutherAI/lm-evaluation-harness/issues/2646"

# Chowder's frozen Gen-0 protocol, quoted in every note so a reader never has
# to reconstruct why these rows are inert.
CHOWDER_MATH_PROTOCOL = (
    "Chowder gen0-freeze-protocol-v1 measured this row as lm-eval 0.4.12 "
    "`minerva_math` with limit=4 per subtask (28 items), 0-shot, chat template "
    "applied, greedy."
)
CHOWDER_MGSM_PROTOCOL = (
    "Chowder gen0-freeze-protocol-v1 measured this row as lm-eval `mgsm_direct_en` "
    "with limit=24 (English, direct answer, no chain-of-thought), 0-shot, chat "
    "template applied, greedy."
)

DEEPSEEK_MATH_PROTOCOL = (
    "DeepSeek official evaluation: MATH-500 (the 500-problem Lightman subset), "
    "pass@1 with temperature 0.6, top-p 0.95, 64 sampled responses, max generation "
    "32768 tokens."
)

QWEN_MGSM_PROTOCOL = (
    "Qwen official evaluation: 'Multi-Mathematics' from the MGSM 8-shot "
    "chain-of-thought suite, averaged across the MGSM languages; base (not "
    "instruction-tuned) checkpoints."
)


def _math500(
    model: str,
    score: float,
    level: str,
    *,
    first_party: bool = True,
    harness: str = "DeepSeek official eval (MATH-500 pass@1)",
    notes: str = "",
) -> ReferenceScore:
    return ReferenceScore(
        model=model,
        benchmark_qualified_id="math500@2024-04",
        score=score,
        level=level,
        date="2025-01-22",
        source_url=DEEPSEEK_R1_REPO,
        harness=harness,
        tool_setting="none",
        reasoning_setting="extended-thinking",
        first_party=first_party,
        comparability_confidence="LOW",
        notes=f"{DEEPSEEK_MATH_PROTOCOL} {CHOWDER_MATH_PROTOCOL} {notes}".strip(),
    )


def _mgsm(model: str, score: float, level: str, notes: str = "") -> ReferenceScore:
    return ReferenceScore(
        model=model,
        benchmark_qualified_id="mgsm@2022-11",
        score=score,
        level=level,
        date="2024-09-19",
        source_url=QWEN25_BLOG,
        harness="Qwen official eval (MGSM 8-shot CoT, multilingual mean)",
        tool_setting="none",
        reasoning_setting="chain-of-thought",
        first_party=True,
        comparability_confidence="LOW",
        notes=f"{QWEN_MGSM_PROTOCOL} {CHOWDER_MGSM_PROTOCOL} {notes}".strip(),
    )


def reference_scores() -> tuple[ReferenceScore, ...]:
    """The seed set: cited first-party numbers, honestly labeled."""
    return (
        # --- math500@2024-04 -------------------------------------------------
        _math500("DeepSeek-R1-Distill-Qwen-7B", 0.928, LEVEL_1_COMPARABLE_PEER),
        _math500("DeepSeek-R1-Distill-Llama-8B", 0.891, LEVEL_1_COMPARABLE_PEER),
        _math500("DeepSeek-R1-Distill-Qwen-32B", 0.943, LEVEL_2_OPEN_WEIGHT_FRONTIER),
        _math500("QwQ-32B-Preview", 0.906, LEVEL_2_OPEN_WEIGHT_FRONTIER),
        _math500("DeepSeek-R1-Distill-Llama-70B", 0.945, LEVEL_3_STRETCH),
        _math500("OpenAI o1-1217", 0.964, LEVEL_4_ABSOLUTE_FRONTIER),
        _math500("DeepSeek-R1", 0.973, LEVEL_4_ABSOLUTE_FRONTIER),
        # The one row that shares Chowder's task family. Third-party run, but
        # fully specified in the report, so it earns the higher of the two
        # non-decision grades.
        ReferenceScore(
            model="Meta-Llama-3.1-8B-Instruct",
            benchmark_qualified_id="math500@2024-04",
            score=0.3456,
            level=LEVEL_1_COMPARABLE_PEER,
            date="2025-01-21",
            source_url=LM_EVAL_ISSUE_2646,
            harness="lm-eval-harness hf backend, `minerva_math`, --num_fewshot 4, batch auto (v1 task)",
            tool_setting="none",
            reasoning_setting="direct",
            first_party=False,
            comparability_confidence="MEDIUM",
            notes=(
                "Reported run: `accelerate launch -m lm_eval --model hf --tasks "
                "minerva_math --num_fewshot 4`, full split, exact_match 0.3456 "
                "(stderr 0.0063). Same task and metric family as Chowder's row -- "
                "but 4-shot with no chat template on the full MATH test set, versus "
                "Chowder's 0-shot chat-templated 28-item limit. Same-model context: "
                "the publisher reports MATH 0.519 for this model and MATH-500 "
                "numbers for this 7-9B class run 0.89-0.93, so the gap between "
                "'published' and 'same-harness' is protocol, not capability. "
                f"{CHOWDER_MATH_PROTOCOL}"
            ),
        ),
        # --- mgsm@2022-11 ----------------------------------------------------
        _mgsm("Qwen2.5-7B", 0.578, LEVEL_1_COMPARABLE_PEER, "Qwen2-7B-Multi-Mathematics for comparison: 0.575."),
        _mgsm("Gemma2-9B", 0.530, LEVEL_1_COMPARABLE_PEER),
        _mgsm("Llama3-8B", 0.363, LEVEL_1_COMPARABLE_PEER, "Weakest peer row; kept so the peer band is not a single point."),
        _mgsm("Qwen2.5-14B", 0.685, LEVEL_2_OPEN_WEIGHT_FRONTIER),
        _mgsm("Qwen2.5-32B", 0.737, LEVEL_2_OPEN_WEIGHT_FRONTIER),
        _mgsm("Qwen2.5-72B", 0.767, LEVEL_2_OPEN_WEIGHT_FRONTIER),
        _mgsm("Llama-3-70B", 0.671, LEVEL_3_STRETCH),
        # LEVEL_4 deliberately absent: no first-party absolute-frontier MGSM
        # number was found. Unmeasured stays unmeasured.
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="freeze root to write into")
    parser.add_argument("--snapshot-id", default="gen0-frontier-context-2026-09-17")
    parser.add_argument("--date", default="2026-09-17")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print without writing",
    )
    args = parser.parse_args()

    root = Path(args.root)
    scores = reference_scores()

    # Validate every row through the production database first: unpinned
    # benchmark names, out-of-range scores, missing provenance and duplicates
    # must fail before anything is written.
    if args.dry_run:
        import tempfile

        database = FrontierDatabase(Path(tempfile.mkdtemp(prefix="seed-check-")))
    else:
        root.mkdir(parents=True, exist_ok=True)
        database = FrontierDatabase(root)
    for score in scores:
        try:
            database.add(score)
        except ValueError as error:
            print(f"REFUSED seed row: {error}", file=sys.stderr)
            return 2

    chowder_rows = (
        ChowderScore(
            generation_version="gen0",
            benchmark_qualified_id="math500@2024-04",
            score=0.0,
            tool_setting="none",
            reasoning_setting="chat_template",
        ),
        ChowderScore(
            generation_version="gen0",
            benchmark_qualified_id="mgsm@2022-11",
            score=0.0,
            tool_setting="none",
            reasoning_setting="chat_template",
        ),
    )

    comparable_gaps = 0
    for chowder in chowder_rows:
        gaps = gap_rows(database, chowder)
        comparable_gaps += sum(1 for row in gaps if row.comparability == "COMPARABLE")
        context = context_rows(database, chowder)
        print(f"\n== {chowder.benchmark_qualified_id} (Chowder {chowder.score}) ==")
        print(f"gate-eligible gap rows: {len(gaps)} (comparable: {sum(1 for r in gaps if r.comparability == 'COMPARABLE')})")
        for row in context:
            print(
                f"  {row.level}: {row.model} {row.reference_score:.3f} "
                f"[{row.comparability_confidence}] blocked by {', '.join(row.divergence) or 'nothing'}"
            )
    if comparable_gaps:
        print(
            f"\n{comparable_gaps} comparable gap row(s) exist -- the seed claims a "
            "protocol match it should be able to defend.",
            file=sys.stderr,
        )
    else:
        print(
            "\nNo comparable gap rows: every seeded reference is context, not a "
            "decision input. This is the intended, honest outcome."
        )

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    store = SnapshotStore(root)
    if args.snapshot_id in _existing_snapshot_ids(root):
        print(f"snapshot {args.snapshot_id} already exists; never rewritten", file=sys.stderr)
        return 2
    store.freeze(args.snapshot_id, args.date, scores)
    print(f"wrote {len(scores)} references and snapshot {args.snapshot_id} under {root}")
    return 0


def _existing_snapshot_ids(root: Path) -> set[str]:
    """Snapshot ids already frozen at ``root`` (they are never rewritten)."""
    path = root / "frontier_snapshots.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["snapshot_id"] for item in payload}


if __name__ == "__main__":
    raise SystemExit(main())
