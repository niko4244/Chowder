"""Build the Generation-0 evaluation freeze from the attempt-2 battery.

Gated builder: it refuses unless ``battery_results_attempt2.json`` exists
with ``verdict == COMPLETE`` and every measured row satisfies Amendment 1
(``mgsm_direct_en`` scope, ``chat_template_applied=True``, sub-budgets
enforced). Attempt-1's ``REFUSED_BUDGET`` stands in its own result doc;
this builder only ever consumes attempt 2.

Outputs (under ``freeze/``), all produced through the merged growth-system
APIs on ``main`` (PR #167 is NOT merged; nothing here imports beyond it):

- eval-report.json            EvalReport (registry-qualified IDs only)
- capability_profile.json     build_profile over measured battery scores
- contamination_manifest.json firewall manifest (no training sources exist)
- frontier_snapshots.json     gen0-frontier snapshot (empty reference set:
                              frontier comparison was preregistered out of
                              scope; the snapshot preserves *that* honestly)
- generations.json            GenerationLedger record for gen0
- scoreboard.md               Scoreboard.render with the contamination manifest
- FREEZE_DIGEST.json          sha256 of every artifact (freeze content identity)

Run:  PYTHONPATH=<worktree>/src python freeze_builder.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "freeze"
WORKTREE_SRC = Path(r"F:\chowder-worktrees\gen0\src")

sys.path.insert(0, str(WORKTREE_SRC))

from chowder.evals.result import SUPPORTED, BenchmarkRun, EvalReport  # noqa: E402
from chowder.evals.runner import Scoreboard  # noqa: E402
from chowder.growth.capability import build_profile  # noqa: E402
from chowder.growth.catalog import default_registry  # noqa: E402
from chowder.growth.contamination import ContaminationFirewall  # noqa: E402
from chowder.growth.frontier_reference import SnapshotStore  # noqa: E402
from chowder.growth.lineage import GenerationLedger  # noqa: E402
from chowder.growth.promotion import PromotionDecision  # noqa: E402

GENERATION_VERSION = "gen0"
CYCLE_ID = "gen0-eval-freeze-2026-09-16"
SNAPSHOT_ID = "gen0-frontier"
PREREG = "GEN0_EVAL_PREREG_2026-09-16.md"
AMENDMENT = "GEN0_EVAL_AMENDMENT1_2026-09-17.md"
RESULT_ATTEMPT1 = "battery_results.json"
RESULT_ATTEMPT2 = "battery_results_attempt2.json"
DIAGNOSTICS_ID = "generation-diagnostics@gen0-freeze-protocol-v1"


def die(msg: str) -> None:
    print(f"FREEZE REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_json(name: str) -> dict:
    path = HERE / name
    if not path.exists():
        die(f"missing artifact: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("== gen0 freeze builder ==")

    # ---- gate 1: attempt-1 refusal must still be on disk, untouched ----
    attempt1 = load_json(RESULT_ATTEMPT1)
    if attempt1.get("verdict") != "REFUSED_BUDGET":
        die(f"attempt-1 verdict is {attempt1.get('verdict')!r}, expected REFUSED_BUDGET")
    print(f"attempt-1 preserved: REFUSED_BUDGET at {attempt1.get('aggregate_gpu_hours')} GPU-h")

    # ---- gate 2: attempt-2 must be COMPLETE under the amendment ----
    attempt2 = load_json(RESULT_ATTEMPT2)
    verdict = attempt2.get("verdict")
    if verdict != "COMPLETE":
        die(f"attempt-2 verdict is {verdict!r}; the freeze requires COMPLETE")
    for key in ("aggregate_gpu_hours", "aggregate_ceiling_gpu_hours", "load_gpu_hours"):
        if key not in attempt2:
            die(f"attempt-2 result missing {key}")
    if attempt2["aggregate_gpu_hours"] > attempt2["aggregate_ceiling_gpu_hours"]:
        die("attempt-2 aggregate exceeds its ceiling")
    if not attempt2.get("load_within_budget", False):
        die("attempt-2 load cost exceeded its sub-budget")
    print(
        f"attempt-2 COMPLETE: aggregate {attempt2['aggregate_gpu_hours']} GPU-h"
        f" <= {attempt2['aggregate_ceiling_gpu_hours']} ceiling"
    )

    # ---- gate 3: amendment contract on every measured row ----
    for row in attempt2["measured"]:
        qid = row["benchmark_qualified_id"]
        if row.get("chat_template_applied") is not True:
            die(f"{qid}: chat_template_applied is not True (Amendment A2)")
        if qid == "mgsm@2022-11" and row.get("lm_eval_task") != "mgsm_direct_en":
            die(f"{qid}: task is {row.get('lm_eval_task')!r}, Amendment A1 froze mgsm_direct_en")
        if row.get("score") is None:
            die(f"{qid}: measured row carries no score")
        print(
            f"  measured {qid}: score={row['score']:.4f} n={row.get('n_samples')}"
            f" {row['gpu_hours_device']} GPU-h (cap {row['sub_budget_gpu_hours']})"
        )
    if attempt2.get("diagnostics") is None:
        die("attempt-2 diagnostics were demoted; the prereg requires generation sanity")
    diag = attempt2["diagnostics"]
    print(
        f"  diagnostics: eos={diag['eos_termination_rate']} cap={diag['max_token_cap_rate']}"
        f" trigram={diag['distinct_trigram_ratio_mean']} loops={diag['obvious_loop_count']}"
    )

    # ---- gate 4: identity + load-policy evidence ----
    identity = load_json("identity_manifest.json")
    if not identity.get("model_content_digest"):
        die("identity manifest has no model_content_digest")
    probe = load_json("load_probe.json")
    peak_gib = probe.get("peak_allocated_gb")
    if not peak_gib or float(peak_gib) > 14.5:
        die(f"load probe peak {peak_gib} missing or above the 14.5 GB refusal ceiling")
    print(f"identity: {identity['model_content_digest'][:16]}…; load probe peak {peak_gib} GiB")

    # ---- build the EvalReport ----
    registry = default_registry()
    registry_ids: list[str] = []
    runs: list[BenchmarkRun] = []

    for row in attempt2["measured"]:
        qid = row["benchmark_qualified_id"]
        if registry.get(qid) is None:
            die(f"measured benchmark {qid} is not in the registry (vague or unknown IDs refused)")
        registry_ids.append(qid)
        runs.append(
            BenchmarkRun(
                benchmark_qualified_id=qid,
                adapter="lm_eval",
                generation_version=GENERATION_VERSION,
                score=float(row["score"]),
                support=SUPPORTED,
                measurement_kind="raw_model",
                n_samples=int(row.get("n_samples") or 0),
                per_sample_scores=tuple(row.get("per_sample_scores") or ()),
                metric=str(row.get("metric") or "accuracy"),
                reasoning_setting="chat_template",
                raw_artifact_ref=str(HERE / RESULT_ATTEMPT2),
                notes=(
                    f"gen0 freeze attempt 2; task={row.get('lm_eval_task')} limit={row.get('limit')};"
                    f" {row['gpu_hours_device']} GPU-h vs cap {row['sub_budget_gpu_hours']}"
                ),
                metadata={"lm_eval_task": row.get("lm_eval_task"), "limit": row.get("limit")},
            )
        )

    runs.append(
        BenchmarkRun(
            benchmark_qualified_id=DIAGNOSTICS_ID,
            adapter="chowder_custom",
            generation_version=GENERATION_VERSION,
            score=None,  # a diagnostics suite has no normalized scalar score
            support=SUPPORTED,
            measurement_kind="raw_model",
            n_samples=int(diag.get("n_prompts") or 0),
            reasoning_setting="chat_template",
            raw_artifact_ref=str(HERE / RESULT_ATTEMPT2),
            notes=(
                "generation sanity: eos_termination_rate={eos}, max_token_cap_rate={cap},"
                " distinct_trigram_ratio_mean={tri}, obvious_loop_count={loops}"
                " (rung-4 thresholds recorded as context, not gates)"
            ).format(
                eos=diag["eos_termination_rate"],
                cap=diag["max_token_cap_rate"],
                tri=diag["distinct_trigram_ratio_mean"],
                loops=diag["obvious_loop_count"],
            ),
            metadata={
                "eos_termination_rate": diag["eos_termination_rate"],
                "max_token_cap_rate": diag["max_token_cap_rate"],
                "distinct_trigram_ratio_mean": diag["distinct_trigram_ratio_mean"],
                "distinct_trigram_ratio_min": diag["distinct_trigram_ratio_min"],
                "obvious_loop_count": diag["obvious_loop_count"],
                "compression_ratio_chars_per_word_mean": diag[
                    "compression_ratio_chars_per_word_mean"
                ],
            },
        )
    )

    for row in attempt2["unmeasured"]:
        qid = row["benchmark_qualified_id"]
        if registry.get(qid) is not None:
            registry_ids.append(qid)
        runs.append(
            BenchmarkRun(
                benchmark_qualified_id=qid,
                adapter="none",
                generation_version=GENERATION_VERSION,
                score=None,
                support="UNKNOWN",
                notes=f"UNMEASURED — {row['reason']}",
                metadata={
                    "demoted_raw": row.get("demoted_raw"),
                    "prereg": PREREG,
                    "amendment": AMENDMENT,
                },
            )
        )

    report = EvalReport(
        generation_version=GENERATION_VERSION,
        runs=tuple(runs),
        hardware={
            "device": "NVIDIA GeForce RTX 5060 Ti (device 0)",
            "vram_total_gib": 16311 / 1024,
            "load_policy": (
                "bf16; 32 decoder layers CPU-resident via accelerate dispatch_model;"
                " root modules cuda:0; transient per-forward copies"
            ),
            "load_probe_peak_allocated_gib": peak_gib,
            "load_gpu_hours_attempt2": attempt2["load_gpu_hours"],
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "batch_size": attempt2.get("batch_size"),
            "seed": attempt2.get("seed"),
        },
        date=datetime.now(timezone.utc).date().isoformat(),
    )

    # ---- contamination manifest (no training sources exist) ----
    firewall = ContaminationFirewall()
    contamination = firewall.manifest(evaluated_benchmarks=sorted(set(registry_ids)))

    # ---- capability profile from declared registry skills ----
    measured_scores = {
        run.benchmark_qualified_id: run.score
        for run in runs
        if run.support == SUPPORTED and run.score is not None and registry.get(
            run.benchmark_qualified_id
        )
    }
    skill_weights: dict[str, dict[str, float]] = {}
    for qid in measured_scores:
        entry = registry.require(qid)
        for skill in entry.skills:
            skill_weights.setdefault(skill, {})[qid] = 1.0
    profile = build_profile(
        model_version=GENERATION_VERSION,
        raw_scores=measured_scores,
        skill_weights=skill_weights,
        unsupported=tuple(
            run.benchmark_qualified_id
            for run in runs
            if run.support == "UNKNOWN" and registry.get(run.benchmark_qualified_id)
        ),
    )

    # ---- frontier snapshot: empty, honestly ----
    FREEZE.mkdir(parents=True, exist_ok=True)
    snapshot_store = SnapshotStore(FREEZE)
    snapshot = snapshot_store.freeze(SNAPSHOT_ID, report.date, scores=())
    print(f"frontier snapshot {snapshot.snapshot_id} frozen with empty reference set (out of prereg scope)")

    # ---- ledger record ----
    ledger = GenerationLedger(FREEZE)
    record = ledger.record(
        version=GENERATION_VERSION,
        parent_version=None,
        cycle_id=CYCLE_ID,
        base_model={
            "path": identity.get("model_dir"),
            "model_content_digest": identity.get("model_content_digest"),
            "dtype": "bfloat16",
            "tokenizer": identity.get("tokenizer"),
            "generation_config_defaults": identity.get("generation_config_defaults"),
        },
        dataset_manifest_ref="N/A (evaluation-only generation; no training data)",
        curriculum_manifest_ref="N/A (evaluation-only generation)",
        recipe={"freeze": "evaluation-only", "prereg": PREREG, "amendment": AMENDMENT},
        training_evidence_ref="N/A (no training performed)",
        evaluation_report_ref=str(FREEZE / "eval-report.json"),
        promotion=PromotionDecision(
            verdict="PROMOTED",
            reasons=(
                "Generation-0 baseline establishment: the dense abliterated parent is"
                " the lineage root by preregistration; no candidate competition"
                " occurred and no training was run."
            ),
            checks={"baseline_establishment": "ok", "candidate_comparison": "not_applicable"},
        ),
        adapter_ref=None,
        checkpoint_ref=None,
        required_probes=(),
        frontier_snapshot_id=snapshot.snapshot_id,
        notes=(
            "Evaluation-only freeze of the dense Qwen3.8-9B-abliterated parent."
            " The HotCore experimental derivative is NOT generation 0."
        ),
    )
    print(f"ledger record: {record.version} (parent={record.parent_version})")

    # ---- persist and render ----
    report.save(FREEZE / "eval-report.json")
    (FREEZE / "capability_profile.json").write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (FREEZE / "contamination_manifest.json").write_text(
        json.dumps(contamination, indent=2, sort_keys=True), encoding="utf-8"
    )
    scoreboard = Scoreboard(registry)
    (FREEZE / "scoreboard.md").write_text(
        scoreboard.render(report, contamination=contamination), encoding="utf-8"
    )

    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(FREEZE.iterdir())
        if path.is_file()
    }
    freeze_digest = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(artifacts.items())).encode()
    ).hexdigest()
    (FREEZE / "FREEZE_DIGEST.json").write_text(
        json.dumps(
            {
                "freeze_digest": freeze_digest,
                "artifacts": artifacts,
                "built_utc": datetime.now(timezone.utc).isoformat(),
                "gates": {
                    "attempt1_preserved": "REFUSED_BUDGET",
                    "attempt2_verdict": "COMPLETE",
                    "amendment_contract_verified": True,
                    "identity_digest": identity.get("model_content_digest"),
                    "load_probe_peak_gib": peak_gib,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"\nfreeze digest: {freeze_digest[:16]}…")
    print("\n" + (FREEZE / "scoreboard.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
