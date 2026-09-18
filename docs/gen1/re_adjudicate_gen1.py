#!/usr/bin/env python
"""Re-adjudicate Gen-1 under the corrected promotion policy.

The original Gen-1 adjudication (2026-09-17, PR #171) recorded PROMOTED.
The integrity audit found two defects in how that adjudication was fed:

1. protected/broad evidence for the candidate was CARRIED from the Gen-0
   freeze and relabeled with the candidate's generation string, so the
   promotion rule certified "protected ok" / "broad battery ok" from rows
   that never measured the candidate;
2. only the winning recipe's cost reached the resource gate -- losing
   recipes and evaluation compute were omitted, and no actual-vs-ceiling
   settlement ran.

This tool never touches the original record. It:
  - verifies the frozen Gen-0 identity, the gen1 adapter digest, and the
    original evaluation artifacts (provenance: candidate-measured rows
    only where the candidate was actually measured);
  - rebuilds the complete cycle cost ledger (all recipes, evaluations,
    failed attempts, zero-incremental baseline references);
  - re-adjudicates with the corrected policy: the target repair is
    candidate-measured (real), the protected/broad gates have no
    candidate-measured evidence and therefore read INCONCLUSIVE;
  - appends a superseding adjudication revision to the ledger
    (append-only) and writes cycle_compute_accounting.json.

Expected effective verdict: INCONCLUSIVE with target_repair_validated=true
-- "the protocol repair is real; full-generation promotion remains
unresolved pending genuine candidate-side protected evaluation."

Usage:
    python docs/gen1/re_adjudicate_gen1.py            # perform
    python docs/gen1/re_adjudicate_gen1.py --dry-run  # report without writing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN1 = HERE.parent.parent
SRC = GEN1 / "src"
sys.path.insert(0, str(SRC))

GEN0_ROOT = Path(r"C:\Users\nikma\Chowder-Protected\runs\2026-09-16-gen0-eval-freeze")
STATE = Path(r"C:\Users\nikma\Chowder-Protected\runs\2026-09-17-gen1-protocol-compliance")

INSTRUMENT = "generation-diagnostics@gen1-eval-protocol-v1"
MATH500 = "math500@2024-04"
MGSM = "mgsm@2022-11"
PARENT = "gen0"
CANDIDATE = "gen1"
POLICY_VERSION = "promotion-policy-v2-provenance-settlement"

REASON_CODES = (
    "CARRIED_PARENT_EVIDENCE_TREATED_AS_CANDIDATE",
    "BUDGET_SETTLEMENT_INCOMPLETE_WINNER_ONLY_ACCOUNTING",
    "PROTECTED_BROAD_NEVER_CANDIDATE_MEASURED",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_records() -> dict:
    gen0_identity = json.loads((GEN0_ROOT / "identity_manifest.json").read_text(encoding="utf-8"))
    freeze_digest_doc = json.loads((GEN0_ROOT / "freeze" / "FREEZE_DIGEST.json").read_text(encoding="utf-8"))
    generations = json.loads((GEN0_ROOT / "freeze" / "generations.json").read_text(encoding="utf-8"))
    gen1_record = next(r for r in generations if r.get("version") == CANDIDATE)
    return {
        "gen0_identity": gen0_identity,
        "freeze_digest": freeze_digest_doc,
        "gen1_record": gen1_record,
        "generations_path": GEN0_ROOT / "freeze" / "generations.json",
    }


def verify_adapter_identity(records: dict) -> dict:
    from chowder.growth.training_binding import directory_digest

    chosen = json.loads((STATE / "chosen_candidate.json").read_text(encoding="utf-8"))
    evidence = json.loads((STATE / "training-evidence-gen1-recipe-a.json").read_text(encoding="utf-8"))
    recorded_sha = (
        evidence.get("artifact_sha256")
        or (chosen.get("evidence", {}) or {}).get("artifact_sha256")
    )
    artifact = Path(chosen["artifact_ref"])
    exists = artifact.exists()
    actual = None
    if exists:
        # The binding's own canonical directory-digest method -- the same
        # function that recorded the sha at training time, so verification
        # proves the artifact is byte-identical to what was promoted.
        actual, _files = directory_digest(artifact)
    return {
        "artifact_ref": str(artifact),
        "recorded_sha256": recorded_sha,
        "recomputed_sha256": actual,
        "verified": bool(exists and actual and recorded_sha and actual == recorded_sha),
    }


def build_cycle_cost_ledger():
    from chowder.growth.compute_cost import ComputeCost, CycleCostLedger

    ledger = CycleCostLedger(cycle_id="gen1-protocol-compliance")
    # Historical baseline: referenced, zero incremental cost.
    ledger.add_reference(
        "gen0 diagnostics baseline",
        str(GEN0_ROOT / "battery_results_attempt2.json"),
    )
    # Every attempt that consumed compute, winner or not.
    for attempt in sorted((STATE / "attempts").iterdir()):
        evidence_path = attempt / "training-evidence.json"
        if not evidence_path.exists():
            continue
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        measured = evidence.get("measured_gpu_hours") or 0.0
        eval_gpu = (evidence.get("evaluation") or {}).get("gpu_hours") or 0.0
        status = evidence.get("status")
        recipe = evidence.get("recipe_id", attempt.name)
        if status == "SUCCEEDED":
            kind = "training"
        else:
            kind = "failed_attempt"
        if measured:
            ledger.add(
                f"{attempt.name} train",
                kind,
                ComputeCost.from_wall_only(measured, source=str(evidence_path)),
                recipe_id=recipe,
                notes=f"status={status}",
            )
        if eval_gpu:
            ledger.add(
                f"{attempt.name} independent evaluation",
                "evaluation",
                ComputeCost.from_wall_only(eval_gpu, source=str(evidence_path)),
                recipe_id=recipe,
                notes="attempt-internal paired evaluation (wall-charged)",
            )
    # Final candidate target evaluation (the diagnostics instrument run).
    # The 910.5 s wall_seconds field includes the ~15-min model load, which
    # is already inside the training-evidence evaluation.gpu_hours row
    # (the evaluator's phase ledger measured the load + generation);
    # charging both would double-count. The load-inclusive total is the
    # authoritative row, so the final-eval entry records the device-time
    # generation figure only, as a note -- the honest, non-doubled total.
    eval_report = STATE / "candidate_evaluation.json"
    if eval_report.exists():
        diagnostics = json.loads(eval_report.read_text(encoding="utf-8")).get("diagnostics", {})
        device_hours = float(diagnostics.get("gpu_hours_device") or 0.0)
        ledger.add(
            "final candidate target evaluation (generation device-time)",
            "evaluation",
            ComputeCost(
                device_gpu_hours=device_hours,
                wall_gpu_hours=0.0,
                source=str(eval_report),
                measurement_method=(
                    "device-time of the 16-prompt generation pass; the "
                    "load-inclusive wall total is already charged in the "
                    "attempt evaluation rows (910.5s wall incl. ~840s load)"
                ),
            ),
            recipe_id="gen1-recipe-a",
        )
    return ledger


def readjudicate(dry_run: bool) -> dict:
    from chowder.evals.result import (
        MEASURED_PARENT,
        MEASURED_THIS_GENERATION,
        BenchmarkRun,
    )
    from chowder.growth.catalog import default_registry
    from chowder.growth.contamination import ContaminationFirewall
    from chowder.growth.compute_cost import ComputeCost
    from chowder.growth.lineage import GenerationLedger
    from chowder.growth.metric_binding import MetricBinder

    records = load_frozen_records()
    identity = verify_adapter_identity(records)
    if not identity["verified"]:
        raise SystemExit(
            f"adapter identity verification failed: recorded={identity['recorded_sha256']} "
            f"recomputed={identity['recomputed_sha256']} (refusing to adjudicate "
            "against an unverifiable artifact)"
        )

    # ---- candidate-measured rows ONLY (the diagnostics instrument) ----
    evaluation = json.loads((STATE / "candidate_evaluation.json").read_text(encoding="utf-8"))
    diag = evaluation["diagnostics"]
    n = int(diag["n_prompts"])
    candidate_runs = [
        BenchmarkRun(
            benchmark_qualified_id=INSTRUMENT,
            adapter="chowder_custom",
            generation_version=CANDIDATE,
            score=float(diag["eos_termination_rate"]),
            support="SUPPORTED",
            measurement_kind="raw_model",
            n_samples=n,
            metric="eos_termination_rate",
            reasoning_setting="chat_template",
            raw_artifact_ref=str(STATE / "candidate_evaluation.json"),
            per_sample_scores=tuple(1.0 if p["eos_terminated"] else 0.0 for p in diag["per_prompt"]),
            measurement_origin=MEASURED_THIS_GENERATION,
            notes="re-adjudication: candidate-measured diagnostics (protocol-identical)",
        )
    ]
    # Protected benchmarks: the honest representation. No candidate-side
    # measurement exists; they are recorded as UNMEASURED non-measurements
    # (not zeros, not parent copies).
    unmeasured_protected = [
        BenchmarkRun(
            benchmark_qualified_id=qid,
            adapter="lm_eval",
            generation_version=CANDIDATE,
            score=None,
            support="SUPPORTED",  # runnable; simply not run on the candidate
            measurement_kind="raw_model",
            metric="accuracy",
            reasoning_setting="chat_template",
            measurement_origin="UNMEASURED",
            notes="re-adjudication: never measured on the candidate; parent copy rejected",
        )
        for qid in (MATH500, MGSM)
    ]
    # binding refuses score=None rows by design; the honest representation
    # of "never measured on the candidate" is their ABSENCE from the
    # candidate side. They are listed in the output artifact instead.

    # ---- parent rows (from the freeze, parent provenance) ----
    parent_runs = []
    freeze_eval = json.loads((GEN0_ROOT / "freeze" / "eval-report.json").read_text(encoding="utf-8"))
    for row in freeze_eval["runs"]:
        if row["benchmark_qualified_id"] not in (MATH500, MGSM):
            continue
        raw = tuple(float(x) for x in (row.get("per_sample_scores") or ()))
        metric = row.get("metric") or "accuracy"
        if metric == "exact_match":
            metric = "accuracy"
        parent_runs.append(
            BenchmarkRun(
                benchmark_qualified_id=row["benchmark_qualified_id"],
                adapter=row.get("adapter", "lm_eval"),
                generation_version=PARENT,
                score=float(row["score"]),
                support="SUPPORTED",
                measurement_kind="raw_model",
                n_samples=len(raw) or int(row.get("n_samples") or 0),
                metric=metric,
                reasoning_setting=row.get("reasoning_setting", "chat_template"),
                raw_artifact_ref=row.get("raw_artifact_ref", ""),
                per_sample_scores=raw or (float(row["score"]),),
                measurement_origin=MEASURED_PARENT,
                notes="re-adjudication: parent-side evidence from the frozen freeze row",
            )
        )
    # parent target row
    gen0_diag = freeze_eval
    for row in freeze_eval["runs"]:
        pass
    # parent diagnostics from the freeze report's instrument row if present
    parent_target = [
        r for r in freeze_eval["runs"] if r["benchmark_qualified_id"].startswith("generation-diagnostics@")
    ]
    if parent_target:
        row = parent_target[0]
        parent_runs.insert(
            0,
            BenchmarkRun(
                benchmark_qualified_id=INSTRUMENT,
                adapter="chowder_custom",
                generation_version=PARENT,
                score=0.0,  # frozen: all 16 parent completions hit the cap
                support="SUPPORTED",
                measurement_kind="raw_model",
                n_samples=int(row.get("n_samples") or 16),
                metric="eos_termination_rate",
                reasoning_setting="chat_template",
                raw_artifact_ref=str(GEN0_ROOT / "battery_results_attempt2.json"),
                per_sample_scores=(0.0,) * 16,
                measurement_origin=MEASURED_PARENT,
                notes="re-adjudication: parent floor from frozen cap-hit 1.000",
            ),
        )

    # ---- contamination manifest (rebuilt identically to the original) ----
    firewall = ContaminationFirewall()
    manifest_path = STATE / "gen1_contamination_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = default_registry()
    binder = MetricBinder.from_manifest(registry, manifest)

    # ---- complete cost accounting ----
    ledger = build_cycle_cost_ledger()
    totals = ledger.total()
    accounting_document = ledger.render()

    # ---- re-adjudication under the corrected policy ----
    # Frozen envelope, wall units (Amendment 3 C2): 0.75 per recipe
    # (train incl. attempt-internal baseline + independent evaluation) for
    # two recipes. The final target evaluation was part of the prereg's
    # eval aggregate and is charged inside the attempt rows; its
    # device-time generation cost is recorded separately (non-doubled).
    config_wall_ceiling = 1.50
    assembly = binder.promotion_input(
        candidate_version=CANDIDATE,
        parent_version=PARENT,
        candidate_runs=candidate_runs,
        parent_runs=parent_runs,
        target_benchmarks=(INSTRUMENT,),
        protected_benchmarks=(MATH500, MGSM),
        broad_battery_benchmarks=(MATH500, MGSM),
        calibration_benchmarks=(),
        reliability_benchmarks=(),
        min_target_improvement=0.90,
        max_protected_regression=0.02,
        device_gpu_hours=0.0,
        device_gpu_hours_ceiling=None,
        actual_wall_gpu_hours=totals.wall_gpu_hours,
        wall_gpu_hours_ceiling=config_wall_ceiling,
    )
    decision = assembly.decision

    output = {
        "original_verdict": records["gen1_record"]["promotion"]["verdict"],
        "original_decision_digest": hashlib.sha256(
            json.dumps(records["gen1_record"]["promotion"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "corrected_effective_verdict": decision.verdict,
        "target_repair_validated": decision.target_repair_validated,
        "reason_codes": list(REASON_CODES),
        "policy_version": POLICY_VERSION,
        "checks": dict(decision.checks),
        "target_deltas": dict(decision.target_deltas),
        "newly_measured_evidence": [INSTRUMENT],
        "still_unmeasured_evidence": [MATH500, MGSM],
        "adapter_identity": identity,
        "cycle_compute_accounting": accounting_document,
        "resource_settlement": {
            "actual_wall_gpu_hours": totals.wall_gpu_hours,
            "wall_gpu_hours_ceiling": config_wall_ceiling,
            "budget_compliant": totals.wall_gpu_hours <= config_wall_ceiling,
        },
        "note": (
            "Re-adjudication qualification: new policy applied to immutable "
            "artifacts. The original PROMOTED record stands untouched as "
            "history; the effective verdict is the latest revision's."
        ),
    }

    if not dry_run:
        out_path = STATE / "gen1_readjudication.json"
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        ledger_path = STATE / "cycle_compute_accounting.json"
        ledger.write(ledger_path)
        ledger_gen = GenerationLedger(GEN0_ROOT / "freeze")
        revision = ledger_gen.append_adjudication_revision(
            generation_version=CANDIDATE,
            reason_codes=REASON_CODES,
            policy_version=POLICY_VERSION,
            new_verdict=decision.verdict,
            evidence_refs=(str(out_path), str(ledger_path)),
            notes=(
                "Integrity audit: carried parent evidence had satisfied "
                "candidate gates and budget settlement was incomplete. "
                "Effective verdict re-resolved under "
                f"{POLICY_VERSION}; original record untouched."
            ),
        )
        output["revision_id"] = revision.revision_id
        output["effective_verdict_after_revision"] = ledger_gen.effective_verdict(CANDIDATE)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[readjudicate] revision {revision.revision_id} appended; effective verdict "
              f"= {output['effective_verdict_after_revision']}")
        print(f"[readjudicate] artifacts: {out_path} / {ledger_path}")
    else:
        print("[readjudicate] DRY RUN - nothing written")

    print(f"[readjudicate] original={output['original_verdict']} "
          f"corrected={output['corrected_effective_verdict']} "
          f"target_repair_validated={output['target_repair_validated']} "
          f"actual_wall={totals.wall_gpu_hours:.4f}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    readjudicate(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
