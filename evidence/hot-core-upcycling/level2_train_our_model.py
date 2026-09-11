"""Level 2: can Chowder train OUR model? Through its real lifecycle, both engines.

Level 1 proved the plumbing on a toy model -- and for the Transformers engine it
did so on CPU in fp32 with no quantisation, so it never touched the GPU/4-bit path.
This runs the real thing: `chowder.project_runner.run_project` -- the same entry
point the smoke tests use, so registry, automatic baseline, protocol binding,
training worker, evaluation worker and the promotion gate are all production code
-- on the model the measurements recommend:

  F:\\llm-models\\Qwen3.8-9B-Pruned-CW-3456
  qwen3_5 hybrid (24 linear_attn + 8 full-attention layers), FFN pruned to 3,456
  channels, 5.937B params, nf4.

The task is chosen so that "it trained" is unambiguous rather than inferred from a
loss curve: INVENTED facts the base model cannot know. Baseline exact-match is
therefore ~0 by construction; if the adapter actually changes behaviour, exact
match rises. Training and evaluation use the same items on purpose -- this tests
whether training takes effect through Chowder, not whether it generalises, and it
is reported as such.

Target modules were first left at the engine default, which ANSWERED part of the
question by failing: PEFT has no auto-detection mapping for `qwen3_5`, so
`target_modules=None` raises "Please specify `target_modules`", and the curated
`attention_and_mlp` preset has no entry for this model_type either. The automatic
baseline had already evaluated the model successfully at that point, so the
GPU/4-bit evaluation path was proven before training was.

They are now explicit, covering all three module families the hybrid actually has:
full attention (8 layers), linear_attn / Mamba-style (24 layers), and the FFN (all
32). PEFT reports back what it actually matched, so this run also VERIFIES those
names against a real loaded model -- which is the precondition the worker's own
comment sets before an architecture may be added to the curated preset.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

WORKTREE_SRC = r"C:\Users\nikma\Chowder-router-healing\src"
sys.path.insert(0, WORKTREE_SRC)

MODEL = r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3456"
UNSLOTH_ROOT = Path(r"C:\Users\nikma\Chowder-Protected\unsloth-real-smoke")

SYLLABLES = ["zor", "vel", "quin", "thar", "plo", "mek", "rua", "sib", "dran", "oth", "wex", "kli"]
SUBJECTS = ["falcon", "lantern", "harbor", "violin", "glacier", "orchard", "compass",
            "meteor", "saddle", "thimble", "quarry", "beacon", "tundra", "ember",
            "marble", "pylon", "sparrow", "canyon", "zephyr", "lattice"]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def invented_letters(seed: int = 11) -> list[tuple[str, str]]:
    """A one-TOKEN answer per subject, so greedy decoding with max_new_tokens=1
    can match exactly. Chance accuracy is ~1/4 rather than 0, which is fine and
    reported: the gate compares a measured candidate against a measured baseline."""
    rng = random.Random(seed)
    return [(subject, rng.choice("ABCD")) for subject in SUBJECTS]


def invented_facts(seed: int = 7) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    facts, used = [], set()
    for subject in SUBJECTS:
        while True:
            word = "".join(rng.choice(SYLLABLES) for _ in range(3))
            if word not in used:
                used.add(word)
                break
        facts.append((subject, word))
    return facts


def build_project(work: Path, engine: str) -> Path:
    from chowder.project import write_project

    facts = invented_letters()
    (work / "train.jsonl").write_text("".join(
        json.dumps({"text": f"Q: Which letter is assigned to {s}? A: {w}"}) + "\n"
        for s, w in facts), encoding="utf-8")
    (work / "eval.jsonl").write_text("".join(
        json.dumps({"prompt": f"Q: Which letter is assigned to {s}? A:", "expected": w}) + "\n"
        for s, w in facts), encoding="utf-8")

    backend = {
        "schema_version": 1,
        "type": "peft",
        "engine": engine,
        "base_model": MODEL,
        "dataset": "train.jsonl",
        "text_field": "text",
        "max_length": 64,
        "quantization": "4bit",
        # Explicit, because PEFT has no auto-detection mapping for qwen3_5 and the
        # curated "attention_and_mlp" preset has no entry either. This list covers
        # all three module families the hybrid actually has: full attention (8
        # layers), linear_attn / Mamba-style (24 layers), and the FFN (all 32).
        "lora": {"r": 16, "alpha": 32, "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ]},
        "training": {
            "epochs": 10.0,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
        },
        "runtime": {"timeout_seconds": 3600.0},
    }
    if engine == "transformers":
        backend["precision"] = "bf16"
        backend["trust_remote_code"] = False
        backend["training"]["gradient_checkpointing"] = True
        backend["runtime"]["active_accelerator_count"] = 1

    project = work / "project.json"
    write_project(project, {
        "schema_version": 1,
        "name": f"level-2: train the pruned 9B through chowder ({engine})",
        "work_dir": str(work),
        "registry_path": ".chowder/level2.db",
        "seed": 123,
        "goal": {
            "metrics": [{"name": "quality", "minimum": 0.0, "direction": "maximize",
                         "regression_tolerance": 1.0}],
            "gpu_hour_budget": 2.0,
            "max_parallel_candidates": 1,
            "minimum_promotion_gain": 0.2,
            "require_protocol_match": True,
        },
        "baseline": {"mode": "auto"},
        "experiment": {
            "experiment_id": f"level2-{engine}",
            "estimated_gpu_hours": 0.5,
            "hypothesis": {
                "observation": "base model cannot know invented codewords",
                "suspected_cause": "the facts do not exist outside this dataset",
                "intervention": "LoRA SFT on the facts through chowder",
                "expected_deltas": {"quality": 0.5},
            },
            "config_patch": {},
            "tags": ["level-2", "real-ml", engine],
        },
        "config": {
            "seed": 123,
            "backend": backend,
            "evaluation": {
                "type": "transformers-text",
                "estimated_gpu_hours": 0.1,
                "precision": "bf16",
                "quantization": "4bit",
                "device": "cuda",
                "trust_remote_code": False,
                "runtime": {"timeout_seconds": 1800.0},
                "suites": [{
                    "name": "quality",
                    "dataset": "eval.jsonl",
                    "prompt_field": "prompt",
                    "expected_field": "expected",
                    "scoring": "normalized_exact_match",
                    "max_new_tokens": 1,
                    "use_chat_template": False,
                }],
            },
        },
    })
    return project


def link_unsloth_env(work: Path) -> None:
    from chowder.unsloth_env import unsloth_env_dir, unsloth_python
    persistent = unsloth_env_dir(UNSLOTH_ROOT)
    if not unsloth_python(persistent).is_file():
        raise SystemExit(f"no Unsloth env at {persistent}")
    target = unsloth_env_dir(work)
    target.parent.mkdir(parents=True, exist_ok=True)
    import _winapi
    _winapi.CreateJunction(str(persistent), str(target))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["transformers", "unsloth"], required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    import chowder
    assert Path(chowder.__file__).resolve().is_relative_to(Path(WORKTREE_SRC)), chowder.__file__
    from chowder.project_runner import run_project
    from chowder.registry import RunRegistry

    work = Path(args.work)
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"work dir not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    if args.engine == "unsloth":
        link_unsloth_env(work)
    project = build_project(work, args.engine)
    log(f"engine={args.engine}  model={MODEL}")
    log(f"chowder from {chowder.__file__}")

    events: list = []
    started = time.time()
    outcome = run_project(project, on_event=events.append)
    wall = time.time() - started
    log(f"run_project returned in {wall/60:.1f} min")

    cand = outcome.generation.candidates[0]
    report: dict = {"engine": args.engine, "model": MODEL, "wall_seconds": wall,
                    "candidate_error": cand.error}
    if cand.error:
        log(f"CANDIDATE ERROR: {cand.error[:1500]}")
    art = cand.artifact
    if art is not None:
        tel = dict(art.telemetry)
        ev = dict(art.evidence)
        report["telemetry"] = tel
        cfg = adapter_cfg = Path(art.artifact_ref) / "adapter_config.json"
        # Unsloth stores target_modules as a regex STRING, transformers as a list;
        # sorting a string yields characters, so keep the string intact.
        if cfg.is_file():
            tm = json.loads(cfg.read_text(encoding="utf-8"))["target_modules"]
            report["resolved_target_modules"] = tm if isinstance(tm, str) else sorted(tm)
        else:
            report["resolved_target_modules"] = ev.get("resolved_target_modules")
        report["artifact_ref"] = art.artifact_ref
        adapter = Path(art.artifact_ref)
        report["adapter_files"] = sorted(p.name for p in adapter.iterdir()) if adapter.is_dir() else None
        log(f"train: steps={tel.get('global_step')} loss={tel.get('train_loss')} "
            f"peak_vram={tel.get('peak_vram_gb')} GB runtime={tel.get('train_runtime_seconds')}s")
        log(f"resolved target modules: {ev.get('resolved_target_modules')}")
        log(f"adapter files: {report['adapter_files']}")
    losses = [e for e in events if type(e).__name__ == "TrainingProgressEvent"]
    report["progress_losses"] = [getattr(e, "loss", None) for e in losses]
    if losses:
        seq = [x for x in report["progress_losses"] if x is not None]
        if seq:
            log(f"loss trajectory: first {seq[0]:.4f} -> last {seq[-1]:.4f} over {len(seq)} logged steps")

    registry = RunRegistry(str(work / ".chowder" / "level2.db"))
    results = {r.experiment_id: r for r in registry.list_results()}
    report["registry_results"] = {k: {"metrics": dict(v.metrics)} for k, v in results.items()}
    for k, v in results.items():
        log(f"registry result {k}: {dict(v.metrics)}")
    if cand.result is not None:
        report["candidate_metrics"] = dict(cand.result.metrics)
    # The gate's verdict is on the generation, not the candidate: `promoted` is
    # None when the gain did not clear goal.minimum_promotion_gain.
    promoted = outcome.generation.promoted
    report["promoted_experiment_id"] = outcome.promoted_experiment_id
    report["gate_promoted"] = promoted is not None
    log(f"GATE: promoted={outcome.promoted_experiment_id!r} "
        f"(None means the gain did not clear minimum_promotion_gain)")

    out = work / "level2-report.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"wrote {out}")
    return 0 if not cand.error else 2


if __name__ == "__main__":
    raise SystemExit(main())
