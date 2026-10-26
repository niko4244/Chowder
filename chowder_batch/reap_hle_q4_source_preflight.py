"""Read-only, fail-closed admission for a future private original-Q4 HLE study.

This file never downloads HLE examples, starts a kernel, loads a model, invokes
a judge, executes generated code, or writes weights or benchmark predictions.
The frozen manifest describes a separately registered HLE *research* protocol,
not an amendment to the original Windows/Gen-0 scientific selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

HLE_REPO = "cais/hle"
HLE_COMMIT = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
HLE_PARQUET = "data/test-00000-of-00001.parquet"
HLE_URL = f"https://huggingface.co/datasets/{HLE_REPO}/resolve/{HLE_COMMIT}/{HLE_PARQUET}"

HLE_EVAL_REPO = "centerforaisafety/hle"
HLE_EVAL_COMMIT = "22ed3074b1e7b134bcbc09028d0ba320839b0655"
PREDICTION_SCRIPT = "hle_eval/run_model_predictions.py"
PREDICTION_SCRIPT_SHA = "d2062d8afcfb8968f350bbae953e897d4303a6f72a4ae7d97b7db0137a333b17"
JUDGE_SCRIPT = "hle_eval/run_judge_results.py"
JUDGE_SCRIPT_SHA = "a796e74d6e58beab3bd88f46182f599443a102308b2d1c7e0b4b22f578468446"
JUDGE_MODEL = "o3-mini-2025-01-31"
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
Q4_DATASET = "nikmarco/reap128v2-q4-registered-20261025"
Q4_GGUF_FILENAME = "reap128-v2-q4_k_m-20261025.gguf"
Q4_GGUF_BYTES = 11563915264
Q4_GGUF_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
Q4_SIDECAR_SHA = "cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c"
PINNED_LLAMA_CPP_COMMIT = "7609846557c50f9d984719a9e1e8c5f3d02f807b"
PINNED_LLAMA_SERVER_SHA = "f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d"
Q4_COMPLETED_KERNEL = "nikmarco/reap128v2-q4-full-evaluation-20261026"
PROTECTED_ACTIVE_KERNEL = "nikmarco/reap128v2-random128-full-postload"
Q8_KERNEL = "nikmarco/reap128v2-q8-full-postload-reclaim"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
QUOTA_GPU_RE = re.compile(
    r"^\s*GPU\s+([0-9]+(?:\.[0-9]+)?)h\s+"
    r"([0-9]+(?:\.[0-9]+)?)h\s+([0-9]+(?:\.[0-9]+)?)h\b"
)
KERNEL_RE = re.compile(r'KernelWorkerStatus\.([A-Z_]+)')
MIN_REMAINING_GPU_HOURS = Decimal("2.0")


@dataclass(frozen=True)
class FrozenHLEProtocol:
    dataset: str = HLE_REPO
    dataset_revision: str = HLE_COMMIT
    dataset_file: str = HLE_PARQUET
    subset: str = "test/text-only (empty image field only)"
    max_samples: int = 1
    tools: str = "none"
    model_sha256: str = Q4_GGUF_SHA
    model_bytes: int = Q4_GGUF_BYTES
    model_dataset: str = Q4_DATASET
    model_sidecar_sha256: str = Q4_SIDECAR_SHA
    llama_cpp_source_commit: str = PINNED_LLAMA_CPP_COMMIT
    llama_server_binary_sha256: str = PINNED_LLAMA_SERVER_SHA
    operating_system: str = "Kaggle Linux; not Windows-equivalent"
    serving_flags: tuple[str, ...] = ("--fit", "on", "-np", "1", "-c", "16384", "--jinja")
    max_completion_tokens: int = 8192
    temperature: int = 0
    sampler_seed: int = 0
    system_prompt: str = SYSTEM_PROMPT
    input_template: str = "{question}"
    selection: str = "source order, first eligible text-only question, no post-hoc changes"
    prediction_source_git: str = HLE_EVAL_COMMIT
    prediction_source_sha256: str = PREDICTION_SCRIPT_SHA
    judge_source_git: str = HLE_EVAL_COMMIT
    judge_source_sha256: str = JUDGE_SCRIPT_SHA
    judge_model: str = JUDGE_MODEL
    grading: str = "official HLE judge only; no accuracy until independent validated judgment"
    output_policy: str = "separate private Kaggle output and distinct single-serving-session receipts"
    original_scientific_gate_eligible: bool = False


def frozen_manifest() -> dict[str, Any]:
    raw = asdict(FrozenHLEProtocol())
    raw["serving_flags"] = list(raw["serving_flags"])
    raw["system_prompt_sha256"] = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    encoded = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"protocol": raw, "protocol_sha256": hashlib.sha256(encoded).hexdigest()}


def text_only_ids(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Reject ambiguous image-bearing questions; never emit question or answer text."""
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        item_id = row.get("id")
        question, answer = row.get("question"), row.get("answer")
        img = row.get("image")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("missing or duplicate HLE id")
        seen.add(item_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError("missing HLE question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("missing HLE answer")
        if img is None or img == "":
            result.append(item_id)
        elif not isinstance(img, (str, bytes)):
            raise ValueError("unrecognized HLE image field type")
    return result


def parsed_quota(output: str) -> Decimal | None:
    for line in output.splitlines():
        match = QUOTA_GPU_RE.match(line)
        if not match:
            continue
        try:
            used, remaining, total = (Decimal(v) for v in match.groups())
        except InvalidOperation:
            return None
        if used < 0 or remaining < 0 or total <= 0 or used + remaining > total + Decimal(".02"):
            return None
        return remaining
    return None


def observed_kernel_state(text: str) -> str | None:
    matches = KERNEL_RE.findall(text)
    return matches[0] if len(matches) == 1 else None


def verified_hle_head(head: Callable[[str], Mapping[str, Any]], expected_blob: str) -> tuple[bool, str]:
    try:
        result = head(HLE_URL)
    except Exception:
        return False, "HLE_AUTH_HEAD_UNAVAILABLE"
    status = result.get("status")
    if status in (401, 403):
        return False, "HLE_GATED_ACCESS_DENIED"
    if status not in (200, 302, 307):
        return False, "HLE_AUTH_HEAD_UNVERIFIED"
    if not HEX64.fullmatch(expected_blob):
        return False, "HLE_BLOB_SHA_NOT_FROZEN"
    if result.get("revision") != HLE_COMMIT:
        return False, "HLE_SOURCE_REVISION_MISMATCH"
    if result.get("blob_sha256") != expected_blob:
        return False, "HLE_DATA_BLOB_IDENTITY_MISMATCH"
    if not isinstance(result.get("size"), int) or result["size"] <= 0:
        return False, "HLE_FILE_LENGTH_MISSING"
    return True, "HLE_AUTHORIZED_EXACT_BLOB_HEAD"


def read_only_hf_head(url: str) -> Mapping[str, Any]:
    """One metadata-only authenticated HEAD; never follow a redirect to data."""
    import requests
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        return {"status": 401}
    response = requests.head(
        url, headers={"Authorization": f"Bearer {token}"},
        allow_redirects=False, timeout=20,
    )
    headers = response.headers
    length = headers.get("X-Linked-Size") or headers.get("Content-Length")
    blob = (headers.get("X-Linked-Etag") or headers.get("Etag") or "").strip('"')
    return {
        "status": response.status_code,
        "revision": headers.get("X-Repo-Commit"),
        "blob_sha256": blob,
        "size": int(length) if length and length.isdecimal() else None,
    }


def check_public_source_bytes(fetch: Callable[[str], bytes]) -> list[str]:
    """Only fetch two small public pinned evaluation scripts, never the dataset."""
    problems = []
    for filename, expected in ((PREDICTION_SCRIPT, PREDICTION_SCRIPT_SHA), (JUDGE_SCRIPT, JUDGE_SCRIPT_SHA)):
        url = f"https://raw.githubusercontent.com/{HLE_EVAL_REPO}/{HLE_EVAL_COMMIT}/{filename}"
        try:
            data = fetch(url)
            if not isinstance(data, bytes) or len(data) > 128 * 1024 or hashlib.sha256(data).hexdigest() != expected:
                problems.append("PUBLIC_EVALUATOR_SHA_MISMATCH")
        except Exception:
            problems.append("PUBLIC_EVALUATOR_SOURCE_UNAVAILABLE")
    return sorted(set(problems))


def default_public_fetch(url: str) -> bytes:
    import requests

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    if len(response.content) > 128 * 1024:
        raise ValueError("public evaluator unexpectedly large")
    return response.content


def cli_read(args: list[str]) -> str:
    completed = subprocess.run(args, capture_output=True, text=True, timeout=45, check=True)
    return completed.stdout


def check_kaggle(kaggle_read: Callable[[list[str]], str]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    states: dict[str, Any] = {}
    try:
        remaining = parsed_quota(kaggle_read(["kaggle", "quota"]))
        states["gpu_remaining_hours"] = str(remaining) if remaining is not None else None
        if remaining is None or remaining < MIN_REMAINING_GPU_HOURS:
            errors.append("GPU_QUOTA_UNVERIFIED_OR_INSUFFICIENT")
    except Exception:
        states["gpu_remaining_hours"] = None
        errors.append("GPU_QUOTA_UNAVAILABLE")
    for slug in (Q4_COMPLETED_KERNEL, PROTECTED_ACTIVE_KERNEL, Q8_KERNEL):
        try:
            state = observed_kernel_state(kaggle_read(["kaggle", "kernels", "status", slug]))
        except Exception:
            state = None
        states[slug] = state
        if state != "COMPLETE":
            errors.append("PROTECTED_KAGGLE_JOB_ACTIVE_OR_UNVERIFIED")
    # A known three-kernel sample is never a complete inventory of all GPUs.
    # Require separate externally proven clean GPU inventory before a launch.
    errors.append("FULL_GPU_JOB_INVENTORY_NOT_ATTESTED")
    return sorted(set(errors)), states


def source_preflight(
    *, expected_blob: str, head: Callable[[str], Mapping[str, Any]],
    kaggle_read: Callable[[list[str]], str], public_fetch: Callable[[str], bytes],
) -> dict[str, Any]:
    manifest = frozen_manifest()
    allowed, explanation = verified_hle_head(head, expected_blob)
    blockers: list[str] = [] if allowed else [explanation]
    blockers.extend(check_public_source_bytes(public_fetch))
    kaggle_blockers, states = check_kaggle(kaggle_read)
    blockers.extend(kaggle_blockers)
    # Source-only stage cannot by itself prove runtime CUDA, mount, serving PID,
    # input-parquet full bytes, output isolation or an authorized judge call.
    blockers.extend(("RUNTIME_GPU_MOUNT_AND_MEMORY_NOT_ATTESTED", "JUDGE_RUNTIME_NOT_ATTESTED"))
    return {
        "schema": "reap128v2-private-hle-q4-source-admission.v1",
        "status": "REFUSED" if blockers else "SOURCE_ONLY_NOT_RUNTIME_ADMITTED",
        "gpu_job_launched": False,
        "generated_programs_executed": 0,
        "dataset_rows_read": 0,
        "scientific_selection": "BLOCKED",
        "chosen": None,
        "release_approved": False,
        "runtime_one_item_approved": False,
        "full_hle_score_available": False,
        "blockers": sorted(set(blockers)),
        "hle_access_result": explanation,
        "observation": states,
        **manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-blob-sha256", default="", help="Already independently verified LFS SHA; never guessed")
    args = parser.parse_args()
    result = source_preflight(
        expected_blob=args.dataset_blob_sha256,
        head=read_only_hf_head, kaggle_read=cli_read, public_fetch=default_public_fetch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "SOURCE_ONLY_NOT_RUNTIME_ADMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
