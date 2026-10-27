"""Prepare a PRIVATE Q8 HLE replay of the exact completed Q4 text-only IDs.

This is research, not official HLE LLM-judge scoring or the frozen Gen-0 gate.
The original authorized pinned HLE protocol/source and private 128-row Q4
output are read-only. Build a separately named Kaggle T4x2 runner; never copy
GGUF weights or private rows to public Git, and never replace an existing kernel.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

Q4_SOURCE = Path(r"F:\reap-v2-q4-hle-original-pulled-private-20261026\reap128v2-q4-hle-textonly-20261026.py")
Q4_ROWS_DIR = Path(r"F:\reap-v2-q4-hle-complete-private-recovery-20261026")
Q4_ROWS_SHA256 = "3cdc96a324ef3fb4d2fa6f0a7a30ed83c6ec7084cca8b654a66469e2b9f1ceeb"
Q4_SOURCE_SHA256 = "717ac6a3cd190e7c66bb53f48b50e2b2a9be4485239b5fe629c182b18145ca4c"
Q8_SHA256 = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
Q8_BYTES = 19748435968
Q8_FILENAME = "reap128_v2-q8_0-20261025.gguf"
KAGGLE_ID = "nikmarco/reap128v2-q8-matched128-hle-textonly-20261026"
KAGGLE_INPUTS = (
    "nikmarco/reap-hle-private-input-20261026",
    "nikmarco/reap128v2-hle-protocol-20261026",
    "nikmarco/reap128v2-q8-original-20261026",
)
TOOLCHAIN_KERNEL = "nikmarco/chowder-reap128v2-cuda-toolchain-20261025"


def _once(src: str, before: str, after: str) -> str:
    if src.count(before) != 1:
        raise ValueError(f"source anchor count !=1: {before[:90]!r}")
    return src.replace(before, after)


def _source_bound_rows() -> list[str]:
    files = list(Q4_ROWS_DIR.glob("reap-q4-hle-textonly-*.jsonl"))
    if len(files) != 1:
        raise ValueError("exactly one recovered Q4 HLE source file required")
    raw = files[0].read_bytes()
    if hashlib.sha256(raw).hexdigest() != Q4_ROWS_SHA256:
        raise ValueError("Q4 matched source row SHA mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    ids = [x.get("id") for x in rows]
    if (len(ids) != 128 or len(set(ids)) != 128 or
            any(not isinstance(x, str) or len(x) > 256 for x in ids) or
            any(x.get("graded") is not True for x in rows)):
        raise ValueError("Q4 matched source is not 128 distinct graded rows")
    return sorted(ids)


def build(out: Path) -> dict:
    if not out.is_absolute() or out.exists() or out.is_symlink():
        raise ValueError("output must be exclusive new absolute private Kaggle directory")
    source_bytes = Q4_SOURCE.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != Q4_SOURCE_SHA256:
        raise ValueError("exact actual executed Q4 Kaggle source SHA changed")
    # Windows-saved original has CRLF. Normalize only the new private copy;
    # the original SHA and the imported frozen HLE protocol remain unchanged.
    source = source_bytes.decode("utf8").replace("\r\n", "\n")
    for anchor in (
        "Q4_GGUF_SHA256 = '2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7'",
        "Q4_GGUF_BYTES = 11563915264",
        "P.Q4_GGUF_FILENAME",
        "R['data'] = {'q4_gguf_sha256': Q4_GGUF_SHA256",
        "ordered = P.canonical_order(text_rows)[:P.PLANNED_MAX_SAMPLES]",
        "if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib'] < 8:",
        "R['seconds_until_ready'] = round(time.monotonic() - boot, 1)",
        "P.MAX_COMPLETION_TOKENS != 8192 or P.TEMPERATURE != 0",
    ):
        if source.count(anchor) != 1:
            raise ValueError("original HLE source/protocol anchor changed")
    ids = _source_bound_rows()
    matched_literal = json.dumps(ids, ensure_ascii=True, separators=(",", ":"))
    source = _once(source, "Q4_GGUF_SHA256 = '2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7'",
                   f"Q8_GGUF_SHA256 = '{Q8_SHA256}'")
    source = _once(source, "Q4_GGUF_BYTES = 11563915264", f"Q8_GGUF_BYTES = {Q8_BYTES}")
    if "PROTOCOL_FILE_SHA256 = 'cbacf0bb2215fce041ca62e991002d2772bee04891bc5e1773d3b0e2c44c49b2'" not in source or "PROTOCOL_BYTES = 13041" not in source:
        raise ValueError("Q4 actually executed and current-mounted HLE protocol mismatch")
    source = _once(source, "weights = unique(P.Q4_GGUF_FILENAME, 'frozen Q4_K_M GGUF')",
                   f"weights = unique('{Q8_FILENAME}', 'original Q8_0 GGUF')")
    source = source.replace('Q4_GGUF_SHA256', 'Q8_GGUF_SHA256').replace('Q4_GGUF_BYTES', 'Q8_GGUF_BYTES')
    source = _once(source, "ordered = P.canonical_order(text_rows)[:P.PLANNED_MAX_SAMPLES]",
                   "ordered = [x for x in P.canonical_order(text_rows) if x['id'] in MATCHED_Q4_IDS]\n"
                   "    if len(ordered) != 128 or {x['id'] for x in ordered} != MATCHED_Q4_IDS:\n"
                   "        raise RuntimeError('matched Q4 HLE 128 IDs missing or duplicated in frozen dataset')")
    source = _once(source, 'PORT = 8791', 'PORT = 8792')
    source = _once(source, "if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib'] < 8:",
                   "if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib'] < 10:")
    source = _once(source, "raise RuntimeError('8 GiB cgroup headroom floor breached')",
                   "raise RuntimeError('Q8 conservative 10 GiB cgroup headroom floor breached')")
    source = _once(source, "R['data'] = {'q4_gguf_sha256': Q8_GGUF_SHA256",
                   "R['data'] = {'q8_gguf_sha256': Q8_GGUF_SHA256")
    source = _once(source, "R['seconds_until_ready'] = round(time.monotonic() - boot, 1)",
                   "R['seconds_until_ready'] = round(time.monotonic() - boot, 1)\n"
                   "        # Proven, nonmutating postload clean GGUF file-cache advisory from the original complete Q8 T4x2 run.\n"
                   "        if not (hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED')):\n"
                   "            raise RuntimeError('Q8 postload clean file cache advisory unavailable')\n"
                   "        R['before_postload_reclaim'] = host()\n"
                   "        with weights.open('rb', buffering=0) as r_handle:\n"
                   "            os.posix_fadvise(r_handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)\n"
                   "        time.sleep(5)\n"
                   "        R['after_postload_reclaim'] = host()\n"
                   "        check_runtime(R['after_postload_reclaim'])\n"
                   "        if R['after_postload_reclaim']['cgroup_unused_gib'] < 14:\n"
                   "            raise RuntimeError('Q8 postload cache headroom too low for matched 128')\n"
                   "        native_start = log.read_bytes().decode('utf-8', errors='replace')\n"
                   "        if 'offloaded 41/41 layers to GPU' not in native_start:\n"
                   "            raise RuntimeError('Q8 full 41/41 GPU offload was not observed')\n"
                   "        R['verified_native_offload_41_of_41'] = True")
    source = source.replace('reap-q4-hle-', 'reap-q8-matched128-hle-')
    source = source.replace('reap-v2-q4-hle-', 'reap-v2-q8-matched128-hle-')
    source = source.replace('Q4_K_M text-only HLE', 'Q8_0 matched-text HLE')
    source = source.replace('Q4_K_M GGUF', 'Q8_0 GGUF')
    source = source.replace('Q4_K_M', 'Q8_0')
    source = source.replace('Q4 HLE', 'Q8 matched HLE')
    source = source.replace('Q4_HLE_', 'Q8_HLE_')
    source = source.replace('COMPLETE_Q4_HLE_', 'COMPLETE_Q8_MATCHED_HLE_')
    source = source.replace('BUDGET_TRUNCATED_Q4_HLE_', 'BUDGET_TRUNCATED_Q8_MATCHED_HLE_')
    source = source.replace("'schema': 'reap-v2-q4-k-m-text-only-hle-research.v1'",
                            "'schema': 'reap-v2-q8-matched128-text-only-hle-research.v1'")
    # The original protocol remains SHA-bound and unchanged; only the GGUF is
    # replaced. Its 600-task budget becomes an exact 128-ID matched subset.
    source = source.replace("'official_hle_score_claimed': False,",
                            "'official_hle_score_claimed': False,\n"
                            "    'matched_q4_row_sha256': '" + Q4_ROWS_SHA256 + "',\n"
                            "    'matched_q4_id_set_sha256': '" + hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() + "',")
    source = _once(source, "R = {\n", "MATCHED_Q4_IDS = set(" + matched_literal + ")\n\nR = {\n")
    if 'Q4_GGUF_SHA256' in source or 'P.Q4_GGUF_FILENAME' in source or 'reap-q4-hle-' in source:
        raise ValueError("unconverted Q4 source anchor")
    tree = ast.parse(source, filename="private_q8_matched_hle.py")
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "MATCHED_Q4_IDS" for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Call):
        raise ValueError("private Q8 notebook missing exact matched Q4 128-ID set")
    if set(ast.literal_eval(assignments[0].value.args[0])) != set(ids):
        raise ValueError("embedded HLE task IDs differ from authenticated Q4 rows")
    compile(tree, "private_q8_matched_hle.py", "exec")
    out.mkdir(parents=True, exist_ok=False)
    src_file = out / "reap128v2-q8-matched128-hle-20261026.py"
    src_file.write_text(source, encoding="utf8")
    metadata = {
        "id": KAGGLE_ID, "title": "reap128v2 q8 matched128 hle textonly 20261026",
        "code_file": src_file.name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": list(KAGGLE_INPUTS), "kernel_sources": [TOOLCHAIN_KERNEL],
        "competition_sources": [], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    return {
        "status": "PRIVATE_KAGGLE_Q8_MATCHED_128_PREPARED_NOT_EXECUTED",
        "kernel_id": KAGGLE_ID, "q8_model_sha256": Q8_SHA256,
        "q4_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(src_file.read_bytes()).hexdigest(),
        "matched_count": len(ids), "original_q4_rows_sha256": Q4_ROWS_SHA256,
        "matched_id_set_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "private_directory": str(out), "original_scientific_gen0": "BLOCKED",
        "official_hle_score_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
