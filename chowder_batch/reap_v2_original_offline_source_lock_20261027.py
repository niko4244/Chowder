"""Independent original frozen HF source admission, never model inference.

Loads the original evaluator's exact source-construction function with offline
HF environment enforced *before importing datasets*, then verifies each
canonical prompt, answer and HumanEval executable test by full source digest.
The first200 GSM8K subset is the first 200 IDs of that same full cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from chowder_batch import reap_eval

SCHEMA = "chowder.reap-v2-original-offline-frozen-source-admission.v1"
GSM = (1319, "913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89")
HUMAN = (164, "49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420")
EVAL_SHA256 = "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1"


class FrozenSourceRefused(ValueError):
    """A frozen canonical population is absent, changed or not reproducible."""


def prove_items(suite: str, items: list[dict]) -> dict:
    if suite not in ("gsm8k", "humaneval"):
        raise FrozenSourceRefused("unregistered original source suite")
    count, expected = GSM if suite == "gsm8k" else HUMAN
    if not isinstance(items, list) or len(items) != count:
        raise FrozenSourceRefused("original source population size drift")
    ids = [item.get("id") if isinstance(item, dict) else None for item in items]
    expected_ids = ([f"gsm8k-{n}" for n in range(count)] if suite == "gsm8k"
                    else [f"HumanEval/{n}" for n in range(count)])
    if ids != expected_ids or len(set(ids)) != count:
        raise FrozenSourceRefused("original canonical ordered item IDs drift")
    if any(not isinstance(item.get("prompt"), str) or not item["prompt"]
           for item in items):
        raise FrozenSourceRefused("missing frozen original prompt")
    if suite == "gsm8k":
        if any(not isinstance(item.get("expected"), str) or not item["expected"]
               for item in items):
            raise FrozenSourceRefused("missing original GSM8K answer")
    elif any(not isinstance(item.get("test"), str) or not item["test"]
             or not isinstance(item.get("header"), str) or not item["header"]
             or not isinstance(item.get("entry_point"), str) or not item["entry_point"]
             for item in items):
        raise FrozenSourceRefused("missing canonical HumanEval test, header or function")
    digest = reap_eval._digest(items)
    if digest != expected:
        raise FrozenSourceRefused("original complete rendered prompt/answer/test SHA drift")
    return {"suite": suite, "n": count, "full_rendered_source_sha256": digest,
            "first_item_id": ids[0], "last_item_id": ids[-1],
            "full_population_and_order_verified": True,
            "frozen_first200_subset_sha256": (
                reap_eval._digest(items[:200]) if suite == "gsm8k" else None
            ),
            "models_invoked": False,
            "humaneval_generated_code_executed": False}


def verify_current_offline() -> dict:
    from importlib.metadata import version

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise FrozenSourceRefused("offline source evaluation requires both HF offline flags")
    source = Path(reap_eval.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != EVAL_SHA256:
        raise FrozenSourceRefused("original registered evaluator implementation SHA drift")
    observed = {}
    for suite in ("gsm8k", "humaneval"):
        observed[suite] = prove_items(suite, reap_eval._load(suite))
    return {"schema": SCHEMA,
            "status": "COMPLETE_ORIGINAL_FROZEN_OFFLINE_SOURCES_MATCHED",
            "evaluator_sha256": EVAL_SHA256,
            "python": sys.version.split()[0],
            "datasets_version": version("datasets"),
            "cohorts": observed,
            "network_or_model_called": False,
            "official_quality_scored": False,
            "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        print(json.dumps({"status": "REFUSED_EXISTING_OR_INVALID_OUTPUT"}))
        return 2
    # Isolate the dataset import and any incidental hub-client effects from
    # the parent's environment. All load_dataset() calls then read local cache.
    if os.environ.get("REAP_ORIGINAL_SOURCE_OFFLINE_CHILD") != "1":
        child_env = {**os.environ, "REAP_ORIGINAL_SOURCE_OFFLINE_CHILD": "1",
                     "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        completed = subprocess.run([
            sys.executable, "-m",
            "chowder_batch.reap_v2_original_offline_source_lock_20261027",
            "--out", str(args.out),
        ],
                                   env=child_env, check=False, timeout=120)
        return completed.returncode
    try:
        result = verify_current_offline()
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"status": result["status"],
                          "gsm8k": result["cohorts"]["gsm8k"]["n"],
                          "humaneval": result["cohorts"]["humaneval"]["n"],
                          "official_quality_scored": False}, sort_keys=True))
        return 0
    except (FrozenSourceRefused, ValueError, OSError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "REFUSED_ORIGINAL_FROZEN_SOURCE",
                          "reason": str(error)[:300]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
