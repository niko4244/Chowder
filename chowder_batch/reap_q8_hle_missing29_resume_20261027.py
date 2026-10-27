"""Private original-Q8 HLE missing-29 preparation and two-session audit.

No Kaggle submission, inference, scoring implementation or official HLE judge
is invoked here. Never merge native session identifiers or regenerate the 99.
Only aggregate non-official, source-attested heuristic research may be reported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

from chowder_batch import reap_q8_matched_hle_kaggle_20261026 as frozen

V4 = Path(r"F:\reap-v2-q8-hle-matched128-v4-kaggle-output-20261027-0137")
V3_SCRIPT = Path(r"F:\reap-v2-q8-matched128-hle-private-20261026-v3\reap128v2-q8-matched128-hle-20261026.py")
V3_SCRIPT_SHA = "46816ed05d95e3171f4eb8153fda49ddfd832359e1ad245f6b8a936b69572828"
V4_RECEIPT_SHA = "864e2ac980e36aadb912a278ad0964b4d2ca49dc42cef74cf300946b36740d2f"
V4_ROWS_SHA = "585e4dc6b21b932d8ef17bdad6a4a6ec36860cf07a1458b83291e8b3be63afbe"
V4_LOG_SHA = "cf77eb2e51da53deaf617e0ce9faefdc9f37e525c1f071b37626759d74f7dc2b"
V4_SESSION = "9bd025598c0643f585f1f171fec91d2a"
MISSING_IDS_SHA = "568f0ab76ce7bb9636452d14c9ba2d44d9c67bbc11f340a88862bd5301263ea3"
Q4_SESSION_SHA = "3cdc96a324ef3fb4d2fa6f0a7a30ed83c6ec7084cca8b654a66469e2b9f1ceeb"
SOURCE_SET_SHA = "65a8f15e747c2a00577aa0a5c5c06ec7724e4c31ec27a99c6a416c3f587acbea"
PROTOCOL_SHA = "ebdb19e818257c21335889f9bea51c9939293444d315ba0211e21d2c599e70d8"
PROTOCOL_FILE_SHA = "cbacf0bb2215fce041ca62e991002d2772bee04891bc5e1773d3b0e2c44c49b2"
PARQUET_SHA = "6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a"
SERVER_SHA = "f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d"
FLAGS = ["--fit", "on", "-np", "2", "-c", "16384", "--jinja"]
V4_RECEIPT = "reap-v2-q8-matched128-hle-textonly-research.json"
NEW_RECEIPT = "reap-v2-q8-missing29-hle-textonly-research.json"
OLD_LOG = "reap-v2-q8-matched128-hle-server.log"
NEW_LOG = "reap-v2-q8-missing29-hle-server.log"
OLD_PATTERN = "reap-q8-matched128-hle-textonly-*.jsonl"
NEW_PATTERN = "reap-q8-missing29-hle-textonly-*.jsonl"
KAGGLE_ID = "nikmarco/reap128v2-q8-hle-missing29-private-20261027"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(raw: bytes) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        out: dict = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out

    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _read(path: Path, *, max_bytes: int = 6_000_000, expected: str | None = None) -> bytes:
    if (path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= max_bytes):
        raise ValueError("private source file missing, unsafe or oversized")
    raw = path.read_bytes()
    if expected is not None and sha(raw) != expected:
        raise ValueError("SHA-pinned original evidence changed")
    return raw


def _object(path: Path, *, expected: str | None = None) -> dict:
    obj = _json_bytes(_read(path, expected=expected))
    if type(obj) is not dict:
        raise ValueError("expected JSON object")
    return obj


def _only(folder: Path, pattern: str) -> Path:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("private source directory missing or symlink")
    matches = list(folder.glob(pattern))
    if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
        raise ValueError("private session row/summary population missing or ambiguous")
    return matches[0]


def _rows(folder: Path, pattern: str, *, expected: str | None = None) -> tuple[list[dict], str]:
    raw = _read(_only(folder, pattern), expected=expected)
    if not raw.endswith(b"\n"):
        raise ValueError("incomplete JSONL row boundary")
    rows = [_json_bytes(line) for line in raw.splitlines()]
    if not rows or any(type(x) is not dict for x in rows):
        raise ValueError("empty/invalid JSONL rows")
    return rows, sha(raw)


def _ids(rows: list[dict], *, n: int, session: str | None = None) -> set[str]:
    ids: set[str] = set()
    sessions: set[str] = set()
    for row in rows:
        if (type(row.get("id")) is not str or not row["id"] or len(row["id"]) > 256
                or row["id"] in ids or type(row.get("correct")) is not bool
                or row.get("graded") is not True or row.get("truncated") is not False
                or type(row.get("completion_tokens")) is not int
                or not 0 <= row["completion_tokens"] < 8191
                or type(row.get("answer_type")) is not str
                or row["answer_type"] not in ("exactMatch", "multipleChoice")
                or type(row.get("eval_session_id")) is not str
                or not row["eval_session_id"]
                or type(row.get("response_chars")) is not int
                or row["response_chars"] < 0
                or type(row.get("seconds")) not in (int, float)
                or not math.isfinite(row["seconds"]) or row["seconds"] < 0):
            raise ValueError("HLE scored row malformed/truncated/ungraded")
        ids.add(row["id"])
        sessions.add(row["eval_session_id"])
    if len(rows) != n or len(ids) != n or len(sessions) != 1:
        raise ValueError("HLE session population/identity mismatch")
    if session is not None and sessions != {session}:
        raise ValueError("HLE row session disagrees with bound native session")
    return ids


def _source_ids() -> list[str]:
    ids = frozen._source_bound_rows()  # Rechecks immutable original Q4 128 JSONL SHA.
    if (len(ids) != 128 or len(set(ids)) != 128 or
            sha(json.dumps(ids, separators=(",", ":")).encode()) != SOURCE_SET_SHA):
        raise ValueError("original Q4 128-ID set changed")
    q4 = _object(frozen.Q4_ROWS_DIR / "reap-v2-q4-hle-textonly-research.json")
    if (q4.get("status") != "BUDGET_TRUNCATED_Q4_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE"
            or q4.get("protocol_sha256") != PROTOCOL_SHA
            or q4.get("protocol_file_sha256") != PROTOCOL_FILE_SHA
            or q4.get("data", {}).get("q4_gguf_sha256") !=
            "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
            or q4.get("official_hle_score_claimed") is not False
            or q4.get("session_id") in (None, "")):
        raise ValueError("original Q4 source/receipt scientific identity changed")
    q4_rows, digest = _rows(frozen.Q4_ROWS_DIR, "reap-q4-hle-textonly-*.jsonl",
                            expected=Q4_SESSION_SHA)
    if digest != frozen.Q4_ROWS_SHA256 or _ids(q4_rows, n=128, session=q4["session_id"]) != set(ids):
        raise ValueError("original Q4 cohort cannot be source-attested")
    return ids


def _protocol(receipt: dict) -> None:
    data = receipt.get("data")
    proto = receipt.get("protocol")
    if (type(data) is not dict or type(proto) is not dict
            or receipt.get("protocol_sha256") != PROTOCOL_SHA
            or receipt.get("protocol_file_sha256") != PROTOCOL_FILE_SHA
            or data.get("q8_gguf_sha256") != frozen.Q8_SHA256
            or data.get("hle_parquet_sha256") != PARQUET_SHA
            or data.get("server_binary_sha256") != SERVER_SHA
            or data.get("text_only_n") != 2158
            or proto.get("serving_flags") != FLAGS
            or proto.get("temperature") != 0 or proto.get("sampler_seed") != 0
            or proto.get("max_completion_tokens") != 8192
            or proto.get("concurrency") != 2
            or proto.get("wall_budget_seconds") != 10800
            or receipt.get("official_hle_score_claimed") is not False
            or receipt.get("scientific_selection") != "BLOCKED"
            or receipt.get("amendment4_gate_eligible") is not False
            or receipt.get("release_approved") is not False
            or receipt.get("chosen") is not None
            or receipt.get("verified_native_offload_41_of_41") is not True):
        raise ValueError("HLE original Q8 model/protocol/offload gate mismatch")
    cmd = receipt.get("cmd")
    if (type(cmd) is not list or len(cmd) != 12
            or cmd[1:8] != FLAGS or cmd[8] != "--port"
            or cmd[10] != "-m" or not isinstance(cmd[11], str)
            or not cmd[11].endswith(frozen.Q8_FILENAME)):
        raise ValueError("native serving command changed")
    if (type(receipt.get("server_pid")) is not int
            or receipt["server_pid"] <= 0
            or receipt.get("server_version", {}).get("code") != 0
            or type(receipt.get("session_id")) is not str
            or len(receipt["session_id"]) != 32):
        raise ValueError("native session/PID unavailable")


def _native(folder: Path, receipt: dict, name: str) -> str:
    raw = _read(folder / name, max_bytes=16_000_000,
                expected=receipt.get("server_log_sha256"))
    log = raw.decode("utf-8", errors="replace")
    if (receipt.get("server_log_bytes") != len(raw)
            or "offloaded 41/41 layers to GPU" not in log
            or "n_ctx_seq             = 8192" not in log
            or len(receipt.get("samples", [])) < 1):
        raise ValueError("native offload/context/log/sampling not independently attested")
    initial = receipt.get("initial")
    postload = receipt.get("after_postload_reclaim")
    if (type(initial) is not dict or type(postload) is not dict
            or type(initial.get("physical_free_gib")) not in (int, float)
            or type(initial.get("cgroup_unused_gib")) not in (int, float)
            or not math.isfinite(initial["physical_free_gib"])
            or not math.isfinite(initial["cgroup_unused_gib"])
            or initial["physical_free_gib"] < 24
            or initial["cgroup_unused_gib"] < 24
            or type(initial.get("gpu_free_mib")) is not list
            or len(initial["gpu_free_mib"]) != 2
            or any(type(g) is not int or g < 12288 for g in initial["gpu_free_mib"])
            or type(postload.get("cgroup_unused_gib")) not in (int, float)
            or not math.isfinite(postload["cgroup_unused_gib"])
            or postload["cgroup_unused_gib"] < 14):
        raise ValueError("original Q8 GPU/cgroup startup or postload admission missing")
    for sample in receipt["samples"]:
        if (type(sample) is not dict
                or type(sample.get("physical_free_gib")) not in (int, float)
                or type(sample.get("cgroup_unused_gib")) not in (int, float)
                or not math.isfinite(sample["physical_free_gib"])
                or not math.isfinite(sample["cgroup_unused_gib"])
                or sample["physical_free_gib"] < 7
                or sample["cgroup_unused_gib"] < 10
                or type(sample.get("gpu_free_mib")) is not list
                or len(sample["gpu_free_mib"]) != 2
                or any(type(g) is not int or g < 4096 for g in sample["gpu_free_mib"])):
            raise ValueError("original Q8 runtime floor/cgroup/GPU sample breached")
    return sha(raw)


def _summary(folder: Path, rows: list[dict], digest: str, receipt: dict, *, n: int,
             planned: int, not_attempted: int, prefix: str) -> None:
    sid = receipt["session_id"]
    s = _object(_only(folder, f"{prefix}{sid}-summary.json"))
    counts = {
        kind: {"n": sum(row["answer_type"] == kind for row in rows),
               "correct": sum(row["answer_type"] == kind and row["correct"] for row in rows)}
        for kind in ("exactMatch", "multipleChoice")
    }
    correct = sum(row["correct"] is True for row in rows)
    if (s != receipt.get("summary") or s.get("session_id") != sid
            or s.get("row_file_sha256") != digest
            or s.get("planned_n") != planned or s.get("completed_n") != n
            or s.get("graded_n") != n or s.get("not_attempted_n") != not_attempted
            or s.get("ungradeable_n") != 0 or s.get("truncated_n") != 0
            or s.get("item_errors") != [] or s.get("official_judge_score") != "UNAVAILABLE"
            or s.get("metric") != "deterministic_non_official_normalized_match"
            or s.get("correct") != correct or s.get("accuracy") != correct / n
            or s.get("correct_among_untruncated") != correct / n
            or s.get("population_text_only_n") != 2158
            or s.get("completion_fraction_of_population") != n / 2158
            or s.get("by_answer_type") != counts
            or s.get("generation_wall_seconds") != round(sum(x["seconds"] for x in rows), 3)
            or s.get("completion_tokens") != sum(row["completion_tokens"] for row in rows)):
        raise ValueError("HLE summary cannot be independently reconciled")


def original_and_missing(v4_dir: Path = V4) -> tuple[list[str], list[str], str]:
    ids = _source_ids()
    v4 = _object(v4_dir / V4_RECEIPT, expected=V4_RECEIPT_SHA)
    _protocol(v4)
    if (v4.get("status") !=
            "BUDGET_TRUNCATED_Q8_MATCHED_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE"
            or v4["session_id"] != V4_SESSION
            or v4.get("matched_q4_row_sha256") != frozen.Q4_ROWS_SHA256
            or v4.get("matched_q4_id_set_sha256") != SOURCE_SET_SHA
            or v4.get("population", {}).get("planned_n") != 128):
        raise ValueError("original Q8 v4 run identity/status changed")
    rows, digest = _rows(v4_dir, OLD_PATTERN, expected=V4_ROWS_SHA)
    existing = _ids(rows, n=99, session=v4["session_id"])
    _summary(v4_dir, rows, digest, v4, n=99, planned=128, not_attempted=29,
             prefix="reap-q8-matched128-hle-textonly-")
    _native(v4_dir, v4, OLD_LOG)
    if v4["server_log_sha256"] != V4_LOG_SHA:
        raise ValueError("original native session log SHA changed")
    if not existing <= set(ids):
        raise ValueError("v4 contains HLE IDs not in original Q4 cohort")
    missing = sorted(set(ids) - existing)
    if (len(missing) != 29 or
            sha(json.dumps(missing, separators=(",", ":")).encode()) != MISSING_IDS_SHA):
        raise ValueError("source-locked missing29 complement changed")
    return sorted(existing), missing, digest


def _replace(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("pinned private source anchor missing or ambiguous")
    return source.replace(old, new)


def prepare(out: Path, *, v4_dir: Path = V4) -> dict:
    """Write only a NEW private Kaggle package. Does not launch inference."""
    if not out.is_absolute() or out.exists() or out.is_symlink():
        raise ValueError("must supply unused absolute private output directory")
    original99, missing29, prior_sha = original_and_missing(v4_dir)
    raw = _read(V3_SCRIPT, max_bytes=60_000, expected=V3_SCRIPT_SHA)
    source = raw.decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(source)
    embedded = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "MATCHED_Q4_IDS" for t in n.targets)]
    if (len(embedded) != 1 or
            set(ast.literal_eval(embedded[0].value.args[0])) !=
            set(original99) | set(missing29)):
        raise ValueError("prior matched128 script changed its embedded Q4 IDs")
    if "Q8_GGUF_SHA256 = '" + frozen.Q8_SHA256 + "'" not in source:
        raise ValueError("prior Q8 model not pinned")
    if "if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib'] < 10:" not in source:
        raise ValueError("Q8 10-GiB cgroup runtime stop missing")
    if "if R['after_postload_reclaim']['cgroup_unused_gib'] < 14:" not in source:
        raise ValueError("Q8 14-GiB postload cgroup stop missing")
    if "if 'offloaded 41/41 layers to GPU' not in native_start:" not in source:
        raise ValueError("Q8 41/41 native offload stop missing")
    source = _replace(source, "PORT = 8792", "PORT = 8793")
    source = _replace(source, "LLAMA_ARG_LOG_VERBOSITY='3'",
                      "LLAMA_ARG_LOG_VERBOSITY='4'")
    select = (
        "ordered = [x for x in P.canonical_order(text_rows) if x['id'] in MATCHED_Q4_IDS]\n"
        "    if len(ordered) != 128 or {x['id'] for x in ordered} != MATCHED_Q4_IDS:\n"
        "        raise RuntimeError('matched Q8 matched HLE 128 IDs missing or duplicated in frozen dataset')"
    )
    guarded = (
        "if (len(MATCHED_Q4_IDS) != 128 or len(PRIOR_Q8_IDS) != 99 or\n"
        "        len(MISSING_Q8_IDS) != 29 or PRIOR_Q8_IDS & MISSING_Q8_IDS or\n"
        "        PRIOR_Q8_IDS | MISSING_Q8_IDS != MATCHED_Q4_IDS):\n"
        "        raise RuntimeError('original Q8 v4 99 and missing29 are not an exact partition')\n"
        "    ordered = [x for x in P.canonical_order(text_rows) if x['id'] in MISSING_Q8_IDS]\n"
        "    if len(ordered) != 29 or {x['id'] for x in ordered} != MISSING_Q8_IDS:\n"
        "        raise RuntimeError('missing29 canonical original HLE prompts absent or duplicated')"
    )
    source = _replace(source, select, guarded)
    constants = ("PRIOR_Q8_IDS = set(" + json.dumps(original99, separators=(",", ":")) +
                 ")\nMISSING_Q8_IDS = set(" + json.dumps(missing29, separators=(",", ":")) +
                 ")\n\n")
    source = _replace(source, "R = {\n", constants + "R = {\n")
    source = _replace(source, "    'samples': [],",
                      "    'samples': [],\n"
                      "    'resume_generator_template_sha256': '" + V3_SCRIPT_SHA + "',\n"
                      "    'resume_prior_v4_rows_sha256': '" + prior_sha + "',\n"
                      "    'resume_prior_v4_session_id': '" + V4_SESSION + "',\n"
                      "    'resume_q4_original_rows_sha256': '" + Q4_SESSION_SHA + "',\n"
                      "    'resume_missing29_ids_sha256': '" + MISSING_IDS_SHA + "',")
    source = source.replace("reap-q8-matched128-hle-", "reap-q8-missing29-hle-")
    source = source.replace("reap-v2-q8-matched128-hle-", "reap-v2-q8-missing29-hle-")
    source = _replace(
        source, "COMPLETE_Q8_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE",
        "COMPLETE_Q8_MISSING29_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE")
    source = _replace(
        source, "BUDGET_TRUNCATED_Q8_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE",
        "BUDGET_TRUNCATED_Q8_MISSING29_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE")
    tree = ast.parse(source)
    compile(tree, "private-original-q8-hle-missing29.py", "exec")
    for name, count in (("PRIOR_Q8_IDS", 99), ("MISSING_Q8_IDS", 29)):
        found = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        if len(found) != 1 or len(ast.literal_eval(found[0].value.args[0])) != count:
            raise ValueError("private missing29 source ID proof failed")
    out.mkdir(parents=True, exist_ok=False)
    script = out / "reap128v2-q8-hle-missing29-private-20261027.py"
    script.write_bytes(source.encode("utf-8"))
    meta = {
        "id": KAGGLE_ID, "title": "original q8 hle missing29 private 20261027",
        "code_file": script.name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": list(frozen.KAGGLE_INPUTS),
        "kernel_sources": [frozen.TOOLCHAIN_KERNEL],
        "competition_sources": [], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"status": "PRIVATE_Q8_MISSING29_PREPARED_NOT_SUBMITTED",
            "private_source_sha256": sha(script.read_bytes()),
            "original_q8_v4_rows_sha256": prior_sha,
            "original_v4_session": V4_SESSION,
            "original_q4_rows_sha256": frozen.Q4_ROWS_SHA256,
            "missing29_id_set_sha256": MISSING_IDS_SHA,
            "missing_n": 29, "existing_untouched_n": 99,
            "q8_original_model_sha256": frozen.Q8_SHA256,
            "native_flags": FLAGS, "official_hle_judge_score": "UNAVAILABLE",
            "scientific_selection": "BLOCKED", "chosen": None,
            "release_approved": False}


def audit_two_sessions(fresh_dir: Path, *, v4_dir: Path = V4) -> dict:
    """Two separately attested native sessions; NEVER forge one-session receipt."""
    old99, missing29, old_sha = original_and_missing(v4_dir)
    original = set(_source_ids())
    v4 = _object(v4_dir / V4_RECEIPT, expected=V4_RECEIPT_SHA)
    fresh = _object(fresh_dir / NEW_RECEIPT)
    _protocol(fresh)
    if (fresh.get("status") !=
            "COMPLETE_Q8_MISSING29_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE"
            or fresh.get("resume_generator_template_sha256") != V3_SCRIPT_SHA
            or fresh.get("resume_prior_v4_rows_sha256") != old_sha
            or fresh.get("resume_prior_v4_session_id") != V4_SESSION
            or fresh.get("resume_q4_original_rows_sha256") != Q4_SESSION_SHA
            or fresh.get("resume_missing29_ids_sha256") != MISSING_IDS_SHA
            or fresh.get("protocol") != v4.get("protocol")
            or fresh.get("session_id") == v4.get("session_id")
            or fresh.get("population", {}).get("planned_n") != 29):
        raise ValueError("second native session or missing29 source/protocol attestations absent")
    v4_rows, _ = _rows(v4_dir, OLD_PATTERN, expected=V4_ROWS_SHA)
    fresh_rows, fresh_sha = _rows(fresh_dir, NEW_PATTERN)
    new_ids = _ids(fresh_rows, n=29, session=fresh["session_id"])
    if (new_ids != set(missing29) or new_ids & set(old99)
            or set(old99) | new_ids != original):
        raise ValueError("missing29 overlaps, omits or substitutes original HLE IDs")
    if (set(fresh_rows[0]) != set(v4_rows[0])
            or any(set(x) != set(v4_rows[0]) for x in fresh_rows)):
        raise ValueError("new Q8 row schema differs from original Q8 source scorer")
    _summary(fresh_dir, fresh_rows, fresh_sha, fresh, n=29, planned=29,
             not_attempted=0, prefix="reap-q8-missing29-hle-textonly-")
    fresh_log_sha = _native(fresh_dir, fresh, NEW_LOG)
    if fresh_log_sha == V4_LOG_SHA:
        raise ValueError("fresh29 cannot reuse previous native session log")
    q4_rows, _ = _rows(frozen.Q4_ROWS_DIR, "reap-q4-hle-textonly-*.jsonl",
                        expected=frozen.Q4_ROWS_SHA256)
    q4 = {r["id"]: r for r in q4_rows}
    q8 = {r["id"]: r for r in v4_rows + fresh_rows}
    if len(q8) != 128:
        raise ValueError("two-session Q8 result does not contain 128 original rows")
    pairs = {"both_correct": 0, "q8_only_correct": 0,
             "q4_only_correct": 0, "both_incorrect": 0}
    for id_ in original:
        a, b = q4[id_], q8[id_]
        if a["answer_type"] != b["answer_type"]:
            raise ValueError("original Q4 vs Q8 answer type diverged")
        if a["correct"] and b["correct"]:
            pairs["both_correct"] += 1
        elif b["correct"]:
            pairs["q8_only_correct"] += 1
        elif a["correct"]:
            pairs["q4_only_correct"] += 1
        else:
            pairs["both_incorrect"] += 1
    qc4 = sum(x["correct"] is True for x in q4_rows)
    qc8 = sum(x["correct"] is True for x in q8.values())
    if qc4 != 3 or len(q4) != 128 or sum(pairs.values()) != 128:
        raise ValueError("original Q4 heuristic research cohort changed")
    return {
        "status": "TWO_NATIVE_SESSIONS_PAIRED_HEURISTIC_RESEARCH_NOT_OFFICIAL_HLE",
        "original_q4_n": 128, "q8_first_session_n": 99, "q8_second_session_n": 29,
        "combined_source_ids_n": 128,
        "first_session_id": V4_SESSION, "second_session_id": fresh["session_id"],
        "first_session_rows_sha256": old_sha, "second_session_rows_sha256": fresh_sha,
        "first_native_log_sha256": V4_LOG_SHA, "second_native_log_sha256": fresh_log_sha,
        "q4_correct_claimed_nonofficial": qc4, "q8_correct_claimed_nonofficial": qc8,
        "paired_nonofficial_partition": pairs,
        "q8_minus_q4_percentage_points_nonofficial": 100 * (qc8 - qc4) / 128,
        "independent_raw_response_rescore": False,
        "metric": "existing_original_runner_deterministic_nonofficial_heuristic_bits",
        "official_hle_judge_score": "UNAVAILABLE",
        "single_native_session_claim": False,
        "scientific_selection": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    make = action.add_parser("prepare", help="CPU-only private Kaggle package, no submission")
    make.add_argument("--out", type=Path, required=True)
    check = action.add_parser("audit", help="CPU-only genuine 99+29 native sessions")
    check.add_argument("--fresh", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.out) if args.action == "prepare" else audit_two_sessions(args.fresh)
    except (OSError, ValueError, TypeError, KeyError, SyntaxError,
            json.JSONDecodeError) as ex:
        print(json.dumps({"status": "BLOCKED_UNVERIFIED_ORIGINAL_HLE",
                          "reason": type(ex).__name__, "official_hle_judge_score": "UNAVAILABLE",
                          "scientific_selection": "BLOCKED", "release_approved": False}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
