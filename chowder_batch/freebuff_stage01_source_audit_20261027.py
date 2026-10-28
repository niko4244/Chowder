"""Read-only, SHA-pinned audit of the October 14 Freebuff codec experiments.

The recorded checkpoints are Qwen2.5-1.5B and OLMoE-1B-7B, not REAP128_v2.
This script audits existing holdout scoring receipts; it performs no model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

DOMAINS = ("gsm8k-numeric", "math500-numeric", "mmlu-lettered")
STAGE0_SHA = {
    "stage_0_report.json": "aa8fe621eb8ee5f5fc4158e3b4b4ddd62f1dd468dec43536c8017dd59cf4c0ec",
    "rows.jsonl": "48676aedb333fb5a8b8c4bee94182eb796e83ff3c688bf9537b145adc4f4e362",
    "block_rate_curves.json": "6397b93cfda076906038113ee19a7aa3ab4d2023b5a05e2bc399438652c9abf8",
    "blocks.json": "b4673a87b72ce537c66c16d96fd5fa8552118e9ee7d58c67933a281f716051bd",
    "suites.json": "18f65891dbbd9895474393e87daba2b769df9a2fce9d6f05478eb83f892fcea1",
    "allocation_plan.json": "e3b8e5ffd6a246ee3c6e8bcf44c7f4a5ba5ebbd62fa58d0497353d2c7159b038",
    "arm-int8/arm_manifest.json": "a4189aa1069d5419a0f0f7d2a80f6e41562bbb6366c34470c07ac82b3c1c0ed9",
    "arm-nf4/arm_manifest.json": "ac1d31872ac7e5f64112e99f54e2e5387906ac525a988a992cacd29d1ef8af34",
}
STAGE1_SHA = {
    "ledger.jsonl": "78cef1e07014c9d1a8e5f82dca69d4d571af0c7adfd4603919b9a22847ce7f51",
    "rows.jsonl": "55ecaf2b3ed396dfc68d81748bb1c3404a3f04198a689a08aa22b75cd5c43028",
    "rows_partial_derived.jsonl": "8b383bb6e8d54b67a5fd68dffe0d564a791106a615828d46ff185c7e02eeda91",
    "blocks.json": "2b1bf4c788c9dc70f23a6b15065623c9db391d0d4b534f4b9ed6423372c39c6c",
    "blocks_partial.json": "4539fedb29e8496a0e9654ab2c29df46dedcf89e6b16dd4dc932c4874a6cec8e",
    "block_rate_curves.json": "be0c46be888fae8ae2f7e8d5ab3a650cee83bc7eb9441cb13ab9d2c54fd99280",
    "../finisher.log": "08ad0256c19d4bb50c924683cbc42544a4e72d2909b73c4989533cfe89b3c0fb",
    "../profiler.log": "babacdd1d9179441cf159469cabf6c32e7e703584a88102c53ef1f204f56d4d2",
    "../criticality.json": "a8f7748a55b3e1cd27d596ef869429182952bc3f137aae664630f7c92e2c9c48",
    "../calibration/corpus_build_report.json": "40e339b74b331540b12497614496f19246faae7613c742500b54a8de205288b9",
}


class EvidenceError(ValueError):
    """An expected receipt, population or score contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        target = root / relative
        require(target.is_file(), f"missing source: {relative}")
        require(sha256_file(target) == digest, f"source SHA256 mismatch: {relative}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            require(bool(line.strip()), f"empty row {path.name}:{index}")
            row = json.loads(line)
            require(isinstance(row, dict), f"non-object row {path.name}:{index}")
            rows.append(row)
    return rows


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"non-object: {path.name}")
    return value


def group_scores(rows: list[dict]) -> dict[tuple[int, int, str, str], dict[int, float]]:
    grouped = defaultdict(dict)
    for pos, row in enumerate(rows):
        require(set(row) == {"layer", "expert", "domain", "codec", "item", "score"}, f"row shape: {pos}")
        require(type(row["layer"]) is int and type(row["expert"]) is int, f"block type: {pos}")
        require(row["domain"] in DOMAINS and type(row["codec"]) is str, f"domain/codec: {pos}")
        require(type(row["item"]) is int and row["item"] >= 0, f"item type: {pos}")
        require(type(row["score"]) in (int, float) and row["score"] in (0, 1), f"score type: {pos}")
        key = row["layer"], row["expert"], row["domain"], row["codec"]
        require(row["item"] not in grouped[key], f"duplicate scoring item {key}:{row['item']}")
        grouped[key][row["item"]] = float(row["score"])
    for key, scores in grouped.items():
        require(set(scores) == set(range(48)), f"non-complete 48-item holdout: {key}")
    return dict(grouped)


def verify_curves(grouped: dict, curves: dict, expected_rows: int) -> None:
    require(curves["row_count"] == expected_rows, "curve row_count inconsistent")
    require(tuple(curves["domains"]) == DOMAINS, "curve domain labels inconsistent")
    baselines = curves["baseline_codec"]
    observed_signed = set()
    observed_options = set()
    for curve in curves["curves"]:
        block = (curve["layer"], curve["expert"])
        for option in curve["options"]:
            codec = option["codec"]
            require((block, codec) not in observed_options, "duplicate curve codec")
            observed_options.add((block, codec))
            require(math.isclose(option["bits_per_param"], curves["codecs"][codec]), "wrong codec bits")
            damage = []
            for domain in DOMAINS:
                base = grouped[(*block, domain, baselines)]
                scored = grouped[(*block, domain, codec)]
                signed = (sum(scored.values()) - sum(base.values())) / 48.0
                actual_damage = max(0.0, -signed)
                require(math.isclose(actual_damage, option["domain_damage"][domain], abs_tol=1e-12), "curve damage mismatch")
                damage.append(actual_damage)
                recorded = [
                    x for x in curves["signed_deltas"]
                    if (x["layer"], x["expert"], x["domain"], x["codec"]) == (*block, domain, codec)
                ]
                require(len(recorded) == 1, "missing/duplicate signed curve row")
                require(math.isclose(recorded[0]["signed_delta"], signed, abs_tol=1e-12), "signed delta mismatch")
                observed_signed.add((*block, domain, codec))
            require(math.isclose(option["mean_damage"], sum(damage) / 3, abs_tol=1e-12), "mean damage mismatch")
    require(len(observed_signed) == len(curves["signed_deltas"]), "extra signed delta rows")


def summarize(grouped: dict) -> list[dict]:
    return [
        {"layer": k[0], "expert": k[1], "domain": k[2], "codec": k[3], "correct": int(sum(v.values())), "total": 48}
        for k, v in sorted(grouped.items())
    ]


def audit_stage0(root: Path) -> dict:
    verify_sources(root, STAGE0_SHA)
    report = load_json(root / "stage_0_report.json")
    suites = load_json(root / "suites.json")
    blocks = load_json(root / "blocks.json")
    rows = load_jsonl(root / "rows.jsonl")
    grouped = group_scores(rows)
    require("Qwen2.5-1.5B-Instruct" in report["base_checkpoint"], "wrong stage0 model")
    require(report["probe"] is False and report["placement"] == "cuda", "unexpected stage0 mode")
    require(len(rows) == report["rows_written"] == 864, "stage0 row count")
    require(len(grouped) == 18 and suites["per_domain_items"] == 48, "stage0 group size")
    require(tuple(sorted((b["layer"], b["expert"]) for b in blocks["blocks"])) == ((0, 0), (27, 0)), "dense block labels")
    require({key[3] for key in grouped} == {"bf16", "int8", "nf4"}, "codec set stage0")
    for (_, expert, domain, codec), scores in grouped.items():
        require(grouped[(0, expert, domain, codec)] == scores, "dense representative rows differ")
        require(grouped[(27, expert, domain, codec)] == scores, "dense representative rows differ")
    require(report["under_ceiling"] is False, "unexpected reported ceiling status")
    require(report["wall_total_seconds"] > report["ceiling_seconds"], "missing stage0 overrun")
    verify_curves(grouped, load_json(root / "block_rate_curves.json"), 864)
    require(math.isclose(load_json(root / "allocation_plan.json")["achieved_average_bits"], 6), "stage0 plan bits changed")
    for codec in ("int8", "nf4"):
        manifest = load_json(root / f"arm-{codec}" / "arm_manifest.json")
        require(manifest["verified"] is True and manifest["codec"] == codec, "unverified stage0 arm manifest")
    return {
        "model": "Qwen2.5-1.5B-Instruct (dense; NOT REAP128_v2)",
        "source": "G:/freebuff-stage0/run",
        "sha256": STAGE0_SHA,
        "status": "completed_holdout_with_runtime_ceiling_failure",
        "raw_rows": 864,
        "unique_logical_model_domain_codec_items": 432,
        "duplicated_dense_block_labels": [0, 27],
        "holdout_items_per_domain": 48,
        "wall_seconds": report["wall_total_seconds"],
        "ceiling_seconds": report["ceiling_seconds"],
        "summary_scores": [v for v in summarize(grouped) if v["layer"] == 0],
        "manifest_integrity_note": "Manifest verified flags/source hashes observed; large serialized model payloads were not independently rehashed.",
    }


def audit_stage1(root: Path) -> dict:
    verify_sources(root, STAGE1_SHA)
    require(not (root / "allocation_plan.json").exists(), "stage1 allocation plan unexpectedly present")
    rows = load_jsonl(root / "rows.jsonl")
    partial = load_jsonl(root / "rows_partial_derived.jsonl")
    ledger = load_jsonl(root / "ledger.jsonl")
    full = load_json(root / "blocks.json")
    partial_blocks = load_json(root / "blocks_partial.json")
    grouped = group_scores(rows)
    partial_grouped = group_scores(partial)
    require(len(full["blocks"]) == 16 and len({(b["layer"], b["expert"]) for b in full["blocks"]}) == 16, "stage1 frozen blocks")
    blocks = {(b["layer"], b["expert"]) for b in full["blocks"]}
    require(all(b[0] == 8 for b in blocks), "stage1 wrong frozen layer")
    require(len(rows) == 2736 and len(grouped) == 57, "stage1 raw row count")
    require(len(partial) == 432 and len(partial_grouped) == 9, "stage1 derived row count")
    require(partial == [row for row in rows if row["layer"] == 8 and row["expert"] == 54], "derived rows differ from raw subset")
    require([b["expert"] for b in partial_blocks["blocks"]] == [54], "more than one paired block claimed")
    require({(l, e) for l, e, _, _ in grouped} == blocks, "stage1 baseline block scope")
    for layer, expert in blocks:
        for domain in DOMAINS:
            require((layer, expert, domain, "bf16") in grouped, "missing baseline group")
    measured = {(8, 54, "pruned"), (8, 54, "crit2bit"), (8, 49, "pruned")}
    altered = {(l, e, c) for (l, e, _, c) in grouped if c != "bf16"}
    require(altered == measured, "unexpected measured arm population")
    events = [(x["event"], x.get("arm")) for x in ledger]
    require(events == [
        ("campaign_started", None), ("baseline_started", None), ("baseline_measured", None),
        ("pass_started", "pruned-l8e54"), ("arm_measured", "pruned-l8e54"),
        ("pass_started", "crit2bit-l8e54"), ("arm_measured", "crit2bit-l8e54"),
        ("pass_started", "pruned-l8e49"), ("arm_measured", "pruned-l8e49"),
        ("pass_started", "crit2bit-l8e49"),
    ], "stage1 ledger changed")
    require("OLMoE-1B-7B-0924" in ledger[0]["base_checkpoint"], "wrong stage1 model")
    require(ledger[2]["rows"] == 2304, "stage1 baseline measured count")
    require(all(x["rows"] == 144 for x in ledger if x["event"] == "arm_measured"), "stage1 arm measured count")
    verify_curves(partial_grouped, load_json(root / "block_rate_curves.json"), 432)
    finisher = (root.parent / "finisher.log").read_text(encoding="utf-8", errors="replace")
    profiler = (root.parent / "profiler.log").read_text(encoding="utf-8", errors="replace")
    require("no feasible global bit allocation" in finisher and "landing exit: 1" in finisher, "stage1 landing failure not verified")
    require("decode profile exit: 1" in profiler and "block probe exit: 139" in profiler, "stage1 profiling failures not verified")
    return {
        "model": "allenai/OLMoE-1B-7B-0924 (NOT REAP128_v2)",
        "source": "G:/freebuff-stage1/run",
        "sha256": STAGE1_SHA,
        "status": "incomplete_campaign_no_global_allocation_or_verified_native_profile",
        "raw_rows": 2736,
        "baseline_rows": 2304,
        "completed_altered_arms": ["pruned-l8e54", "crit2bit-l8e54", "pruned-l8e49"],
        "started_without_scored_rows": ["crit2bit-l8e49"],
        "fully_paired_two_codec_experts": [54],
        "frozen_block_count": 16,
        "derived_subset_rows_not_additional_tests": 432,
        "holdout_items_per_domain": 48,
        "summary_scores_measured_blocks": [v for v in summarize(grouped) if v["expert"] in (49, 54)],
        "global_target_bits": 1.0,
        "global_allocation": "refused_no_feasible_plan",
        "post_estate_decode_profile": "exit_1",
        "real_width_moe_block_profile": "exit_139_segmentation_fault",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "chowder-freebuff-stage01-external-source-locked-audit-v1",
        "scope": "archived October 14 codec methodology holdouts only; not original REAP128_v2 or its eleven frontiers",
        "stage0": audit_stage0(args.stage0_root),
        "stage1": audit_stage1(args.stage1_root),
        "reap_gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
