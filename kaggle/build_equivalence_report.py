"""Build the local-vs-Kaggle equivalence report and qualification record
(Phase 3, the mandatory gate before Kaggle-evaluated C/D results may
enter the four-parent tournament).

This script has no Kaggle-specific dependency itself -- it can run
anywhere both sides' evidence/predictions files are available (locally,
after downloading the Kaggle side's output; or on Kaggle, after copying
the local side's evidence in). Only the earlier acquisition/evaluation
steps needed to run on Kaggle.

Reuses `chowder.kaggle_equivalence.build_equivalence_report` /
`.qualify_backend` unchanged -- no comparison logic lives in this file.

Usage
-----
    python build_equivalence_report.py \\
        --parent-label parent-a-qwen38-27b-official \\
        --local-report local-parent-a-report.json \\
        --kaggle-report kaggle-parent-a-report.json \\
        --local-suite-evidence local-suite-evidence.json \\
        --kaggle-suite-evidence kaggle-suite-evidence.json \\
        --local-predictions-dir /path/to/local/run/parent-a-qwen38-27b-official \\
        --kaggle-predictions-dir /path/to/downloaded/kaggle/run/parent-a-qwen38-27b-official \\
        --local-tokenizer local-tokenizer.json \\
        --kaggle-tokenizer kaggle-tokenizer.json \\
        --local-environment local-environment.json \\
        --kaggle-environment kaggle-environment.json \\
        --qualification-id local-rtx5060ti_vs_kaggle-t4x2 \\
        --report-output equivalence-report.json \\
        --record-output qualification-record.json

Each `*-report.json` is a `ParentEvalReport.to_dict()` dump (the nested
`evidence["parent_eval_report"]` value from that side's persisted
`evaluation_runs` row, or a fresh `result.report.to_dict()`).
`*-suite-evidence.json` is that same row's `evidence["suite_evidence"]`.
`*-tokenizer.json` is `{tokenizer_class, vocab_size, identity_sha256}`.
`*-environment.json` is a `BackendFingerprint.to_dict()` dump (what
`bootstrap_environment.py` writes).

Only pass `--declared-digest-divergence-reason` (repeatable) when you
have a real, citable reason two protocol digests are expected to differ
-- e.g. `chowder.kaggle_launcher.PRECISION_DIVERGENCE_REASON` for the
known T4 bf16 incompatibility. An undeclared digest mismatch still fails
qualification, by design.

Exit code is 0 when qualified (with or without acknowledged differences),
2 when not qualified -- so this script's own exit status can gate a CI
job or a human checklist without parsing its output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent-label", required=True)
    parser.add_argument("--local-report", required=True)
    parser.add_argument("--kaggle-report", required=True)
    parser.add_argument("--local-suite-evidence", required=True)
    parser.add_argument("--kaggle-suite-evidence", required=True)
    parser.add_argument("--local-predictions-dir", required=True)
    parser.add_argument("--kaggle-predictions-dir", required=True)
    parser.add_argument("--local-tokenizer", required=True)
    parser.add_argument("--kaggle-tokenizer", required=True)
    parser.add_argument("--local-environment", required=True)
    parser.add_argument("--kaggle-environment", required=True)
    parser.add_argument("--declared-digest-divergence-reason", action="append", default=[])
    parser.add_argument("--qualification-id", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--record-output", required=True)
    args = parser.parse_args(argv)

    from chowder.kaggle_equivalence import BackendFingerprint, build_equivalence_report, qualify_backend

    report = build_equivalence_report(
        parent_label=args.parent_label,
        local_report=_load_json(args.local_report),
        kaggle_report=_load_json(args.kaggle_report),
        local_suite_evidence=_load_json(args.local_suite_evidence),
        kaggle_suite_evidence=_load_json(args.kaggle_suite_evidence),
        local_predictions_dir=args.local_predictions_dir,
        kaggle_predictions_dir=args.kaggle_predictions_dir,
        local_tokenizer=_load_json(args.local_tokenizer),
        kaggle_tokenizer=_load_json(args.kaggle_tokenizer),
        local_environment=BackendFingerprint.from_dict(_load_json(args.local_environment)),
        kaggle_environment=BackendFingerprint.from_dict(_load_json(args.kaggle_environment)),
        declared_digest_divergence_reasons=tuple(args.declared_digest_divergence_reason),
    )
    record = qualify_backend(report, qualification_id=args.qualification_id)

    Path(args.report_output).write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    Path(args.record_output).write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    print(f"Qualification status: {record.status}")
    print(f"Item score agreement: {record.item_score_agreement_count}/{record.item_total_count}")
    print(f"Detail: {record.detail}")
    print(f"Equivalence report written to {args.report_output}")
    print(f"Qualification record written to {args.record_output}")
    return 0 if record.is_qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
