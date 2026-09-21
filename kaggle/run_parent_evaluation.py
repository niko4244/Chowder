"""Kaggle notebook script: evaluate one acquired parent under protocol v2.

Run this AFTER `bootstrap_environment.py` and `acquire_parent.py` for the
same parent, in a GPU-enabled Kaggle session (T4 accelerator). Used for
both the Phase 3 equivalence-qualification re-run of Parent A and the
Phase 4 real evaluation of Parents C and D -- the same script, the same
reused code path, just a different `--parent`.

Reuses, unchanged:
- `parent_tournament.LocalParent` / `.verify_parent_integrity` /
  `.tokenizer_evidence`
- `parent_eval.ParentTokenizerEvidence` / `.ensure_parent_tokenizer_compatible`
- `parent_suite_content.build_tournament_spec` (the frozen suite's
  reference `ParentEvalSpec` -- the exact function
  `parent_tournament.run_tournament` itself calls)
- `chowder.kaggle_launcher.run_kaggle_parent_evaluation` (the one-field
  precision adaptation + VRAM preflight documented there), which itself
  calls `parent_tournament.evaluate_parent` unchanged.

This script implements no evaluation, scoring, or tokenizer logic of its
own -- it only wires paths and CLI arguments to the functions above.

Usage
-----
    python run_parent_evaluation.py \\
        --parent C \\
        --model-dir /kaggle/input/obliteratus-qwen38-27b \\
        --manifest /kaggle/input/obliteratus-qwen38-27b.manifest.json \\
        --suite-root /kaggle/input/qwen38-protected-suite-v1 \\
        --reference-tokenizer /kaggle/input/parent-a-tokenizer-evidence.json \\
        --registry /kaggle/working/kaggle-tournament.registry.db \\
        --output-root /kaggle/working/runs

`--reference-tokenizer` is a small JSON file with the reference parent's
(normally Parent A's) `{tokenizer_class, vocab_size, identity_sha256}` --
produce it once locally with `parent_tournament.tokenizer_evidence(parent_a())`
and copy just that JSON (not any model weights) into the Kaggle dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--suite-root", required=True, help="Private Kaggle-mounted frozen protected suite root")
    parser.add_argument(
        "--reference-tokenizer", required=True,
        help="JSON file: {tokenizer_class, vocab_size, identity_sha256} for the comparability gate",
    )
    parser.add_argument("--registry", required=True, help="Path to this Kaggle session's own sqlite registry")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    args = parser.parse_args(argv)

    from chowder.kaggle_launcher import run_kaggle_parent_evaluation
    from chowder.parent_eval import ParentTokenizerEvidence, ensure_parent_tokenizer_compatible
    from chowder.parent_suite_content import build_tournament_spec
    from chowder.parent_tournament import LocalParent, tokenizer_evidence, verify_parent_integrity
    from chowder.qwen38_acquisition import ALL_PARENT_PINS, PARENT_LABELS
    from chowder.registry import RunRegistry

    pin = ALL_PARENT_PINS[args.parent]
    parent = LocalParent(
        label=PARENT_LABELS[args.parent],
        revision=pin.revision,
        local_path=args.model_dir,
        manifest_path=args.manifest,
    )

    print("Verifying local checkpoint integrity before any GPU work...")
    integrity = verify_parent_integrity(parent)
    print(json.dumps(integrity, indent=2))

    print("Measuring tokenizer identity...")
    candidate_tokenizer = tokenizer_evidence(parent)
    reference_data = json.loads(Path(args.reference_tokenizer).read_text(encoding="utf-8"))
    reference_tokenizer = ParentTokenizerEvidence(**reference_data)
    ensure_parent_tokenizer_compatible(reference_tokenizer, candidate_tokenizer)
    print("Tokenizer identity comparability gate passed.")

    reference_spec = build_tournament_spec(args.suite_root)

    import torch

    def _free_vram_gib() -> float:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return free_bytes / 2**30

    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    total_weight_bytes = sum(f["size_bytes"] for f in manifest_data.get("weight_files", []))

    with RunRegistry(args.registry) as registry:
        result, kaggle_spec = run_kaggle_parent_evaluation(
            registry,
            parent,
            reference_spec,
            tokenizer=candidate_tokenizer,
            total_weight_bytes=total_weight_bytes,
            free_vram_gib_fn=_free_vram_gib,
            output_root=args.output_root,
            seed=args.seed,
            timeout_seconds=args.timeout_seconds,
        )

    print(f"Kaggle protocol digest: {kaggle_spec.digest()}")
    print(f"Evaluation run recorded: {result.evaluation_run_id}")
    print(json.dumps(result.report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
