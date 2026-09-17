#!/usr/bin/env python3
"""Stage the rung-5 campaign, or a rung-4 dry-run campaign for the judge.

Two modes:

``--dryrun-rung4 <out-dir>``
    Build a campaign root in the rung-5 layout whose four arm directories hold
    only the *copyable* artifacts of the rung-4 run. The rung-5 judge must
    refuse it for the reasons the prereg changes the contract (no saturation
    instrument, no ``gate_initialization`` block, no new spec knobs in the run
    spec) while still deciding the clauses that read unchanged evidence -- in
    particular the control clause N6, which must PASS on rung-4's real 480 and
    its real layer-31 zero-gradient record. Discrimination, not greenness.

``--arms <out-dir>``
    Build the four frozen arm directories the run will use: one project file
    per arm differing only in the two declared knobs, each beside the pinned
    corpus and holdout, with the pinned SHA-256 values re-verified after
    staging. Refuses if any pin drifts.

Nothing here is a judge; it only writes the frozen inputs, and it never
overwrites an existing non-empty arm directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

RUNG4_RUN = Path(r"C:/Users/nikma/Chowder-Protected/runs/2026-09-16-router-healing-rung4")
CORPUS = "router-corpus-9b-pilot.txt"
HOLDOUT = "router-holdout-independent-9b.txt"
CORPUS_SHA256 = "15d5f5f51a739ceee2712fe5b7b550982aba7272f633781f06d5a7ea64f47941"
HOLDOUT_SHA256 = "2e99668207319a1d2b702408bcc659a525fd226d027eebeaebea918e2ad21e97"
BASE_MANIFEST_SHA256 = "77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4"
CENSUS_SHA256 = HOLDOUT_SHA256

ARMS = {
    "arm-A": {"gate_initialization": "artifact", "router_logit_scale": None, "router_logit_soft_cap": None},
    "arm-B": {"gate_initialization": "small_normal", "router_logit_scale": None, "router_logit_soft_cap": None},
    "arm-C": {"gate_initialization": "artifact", "router_logit_scale": 128.0, "router_logit_soft_cap": 30.0},
    "arm-D": {"gate_initialization": "small_normal", "router_logit_scale": 128.0, "router_logit_soft_cap": 30.0},
}
GATE_INIT_STD = 1.0e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_arm_artifacts(source_work: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_work / "runs.db", target / "runs.db")
    for kind in ("runs", "evals"):
        for directory in sorted((source_work / ".chowder" / kind).glob("*")):
            if not directory.is_dir():
                continue
            destination = target / ".chowder" / kind / directory.name
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("worker-result.json", "run-spec.json", "chowder-identity.json"):
                candidate = directory / name
                if candidate.is_file():
                    shutil.copy2(candidate, destination / name)


def dryrun_rung4(out: Path) -> int:
    if (out / "arm-A").exists():
        raise SystemExit(f"refusing to overwrite an existing campaign: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for name in (CORPUS, HOLDOUT):
        shutil.copy2(RUNG4_RUN / name, out / name)
    for arm in sorted(ARMS):
        copy_arm_artifacts(RUNG4_RUN / "work", out / arm)
    print(f"staged a rung-4 dry-run campaign at {out} (arms: {', '.join(sorted(ARMS))})")
    return 0


def arms_project(arm: str, arm_dir: Path, corpus: Path, holdout: Path) -> dict:
    knobs = {
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "holdout_corpus_sha256": HOLDOUT_SHA256,
        "device": "cuda",
        "load_policy": "bf16-offload-transient",
        "max_steps": 48,
        "learning_rate": 0.05,
        "seq_len": 64,
        "batch_size": 2,
        "seed": 1,
        "probe_window": 2,
        "eval_batches": 2,
        "paired_arms": True,
        "max_load_seconds": 20.0,
        "max_gpu_hours": 0.0710,
        "sub_budget_gpu_hours": {"loads": 0.0057, "steps": 0.0610, "generations": 0.0043},
        "census_corpus_path": str(holdout),
        "census_corpus_sha256": CENSUS_SHA256,
        "census_blocks": 2,
        "gate_initialization": ARMS[arm]["gate_initialization"],
        "router_logit_scale": ARMS[arm]["router_logit_scale"],
        "router_logit_soft_cap": ARMS[arm]["router_logit_soft_cap"],
    }
    if ARMS[arm]["gate_initialization"] == "small_normal":
        knobs["gate_init_std"] = GATE_INIT_STD
    return {
        "schema_version": 1,
        "name": f"router-healing-9b-pilot-rung5-numerics-{arm}",
        "work_dir": str(arm_dir / "work"),
        "registry_path": str(arm_dir / "work" / "runs.db"),
        "seed": 1,
        "goal": {
            "metrics": [{"name": "holdout_loss", "direction": "minimize"}],
            "gpu_hour_budget": 0.0710,
            "max_parallel_candidates": 1,
            "minimum_promotion_gain": 0.0,
        },
        "baseline": {"mode": "auto"},
        "experiment": {
            "experiment_id": "router-pilot-9b",
            "estimated_gpu_hours": 0.0541,
            "config_patch": {"router_healing": dict(knobs)},
        },
        "config": {
            "backend": {
                "type": "router-healing",
                "router_healing": dict(
                    knobs,
                    base_model_dir=r"F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176",
                    corpus_path=str(corpus),
                    holdout_corpus_path=str(holdout),
                ),
            },
            "evaluation": {"type": "router-healing"},
        },
    }


def build_arms(out: Path) -> int:
    if (out / "arm-A").exists():
        raise SystemExit(f"refusing to overwrite an existing campaign: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for name in (CORPUS, HOLDOUT):
        shutil.copy2(RUNG4_RUN / name, out / name)
    corpus, holdout = out / CORPUS, out / HOLDOUT
    if sha256_file(corpus) != CORPUS_SHA256 or sha256_file(holdout) != HOLDOUT_SHA256:
        raise SystemExit("the staged corpora do not hash to their pins")
    for arm in sorted(ARMS):
        arm_dir = out / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        project = arms_project(arm, arm_dir, corpus, holdout)
        (arm_dir / "project.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
        print(f"wrote {arm_dir / 'project.json'} ({ARMS[arm]})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dryrun-rung4", metavar="OUT", type=Path)
    mode.add_argument("--arms", metavar="OUT", type=Path)
    args = parser.parse_args()
    if args.dryrun_rung4:
        return dryrun_rung4(args.dryrun_rung4.resolve())
    return build_arms(args.arms.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
