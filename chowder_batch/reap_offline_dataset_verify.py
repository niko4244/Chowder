"""Re-read full frozen REAP evaluation inputs from the genuine offline HF cache.

This preflight does not serve a model, score responses, touch GPU resources,
or certify any quality/selection gate. It independently verifies the prompt,
answer and test-code rendering used by the registered evaluator against the
unchangeable source population hashes and canonical item IDs. For example:

  python -m chowder_batch.reap_offline_dataset_verify --out docs/quals/sources.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from chowder_batch import reap_eval, reap_gates

SCHEMA = "chowder.reap-v2-full-offline-benchmark-inputs.v1"
STATUS = "FULL_FROZEN_OFFLINE_DATASET_CONTENT_MATCH_NOT_MODEL_QUALITY"
SIZES = {"gsm8k": 1319, "humaneval": 164}


def verify(*, loader=None) -> dict:
    """Require the full canonical populations and production prompt rendering."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    loader = reap_eval._load if loader is None else loader
    cohorts = {}
    for suite, count in SIZES.items():
        items = loader(suite)
        if not isinstance(items, list) or len(items) != count:
            raise ValueError(f"{suite}: registered offline population must contain {count} items")
        ids = [item.get("id") if isinstance(item, dict) else None for item in items]
        expected = ([f"gsm8k-{i}" for i in range(count)] if suite == "gsm8k"
                    else [f"HumanEval/{i}" for i in range(count)])
        if ids != expected:
            raise ValueError(f"{suite}: frozen ordered benchmark IDs differ from official cohort")
        if suite == "gsm8k":
            valid = all(isinstance(item.get("prompt"), str) and item["prompt"]
                        and isinstance(item.get("expected"), str) and item["expected"]
                        for item in items)
        else:
            valid = all(all(isinstance(item.get(key), str) and item[key]
                            for key in ("prompt", "header", "test", "entry_point"))
                        for item in items)
        if not valid:
            raise ValueError(f"{suite}: missing actual prompt, expected answer or executable test")
        computed = reap_eval._digest(items)
        if computed != reap_gates.FROZEN_SOURCE_SHA256[suite]:
            raise ValueError(f"{suite}: altered rendered prompts/answers/tests differ from frozen SHA256")
        cohorts[suite] = {"n": count, "source_items_sha256": computed,
                          "first_id": ids[0], "last_id": ids[-1],
                          "canonical_ids_exact": True}
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "mode": "HF_HUB_OFFLINE=1;HF_DATASETS_OFFLINE=1",
        "registered_evaluator_sha256": hashlib.sha256(Path(reap_eval.__file__).read_bytes()).hexdigest(),
        "registered_gates_sha256": hashlib.sha256(Path(reap_gates.__file__).read_bytes()).hexdigest(),
        "cohorts": cohorts,
        "model_inference_performed": False,
        "quality_gate": "NOT_EVALUATED",
        "release_approved": False,
    }


def publish_exclusive(path: Path, result: dict) -> None:
    """Publish complete fsynced evidence; never clobber a previous witness."""
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise FileExistsError("official dataset receipt requires a fresh regular destination")
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with pending.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            pending.rename(path)  # Windows rename refuses an existing file.
        else:
            os.link(pending, path)  # POSIX rename would overwrite an owner.
    finally:
        pending.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify()
        publish_exclusive(args.out, result)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"frozen offline benchmark input verification refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": STATUS, "cohorts": result["cohorts"],
                      "receipt": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
