"""Kaggle CLI script: publish the frozen protected suite as a PRIVATE
Kaggle Dataset. This is the one step in this whole workflow that moves
protected evaluation content off the local machine.

Run it only when you have deliberately decided Kaggle should hold a
private copy of the frozen v1 suite, and verify the created dataset's
visibility is actually Private on kaggle.com afterward -- this script
defaults to private and refuses to publish publicly without an explicit,
separate acknowledgement flag, but a human check is still worth it for
anything protected-content-adjacent.

What is uploaded
-----------------
Exactly the materialized suite directory the local tournament already
uses read-only (e.g. ``C:\\Users\\nikma\\Chowder-Protected\\suites\\v1``):
the dataset JSONL files and their hash-only fingerprint indexes. This
script never reads, prints, or logs a single row of that content -- only
byte counts and the `kaggle` CLI's own stdout/stderr (dataset name/URL,
upload progress), which never includes prompt/expected text.

Usage
-----
    python upload_protected_suite.py \\
        --suite-root "C:\\Users\\nikma\\Chowder-Protected\\suites\\v1" \\
        --kaggle-dataset-slug <your-kaggle-username>/qwen38-protected-suite-v1

Requires the `kaggle` CLI configured (``~/.kaggle/kaggle.json``); this
script never reads or logs that file's contents beyond what the `kaggle`
CLI itself already does on the user's behalf.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--kaggle-dataset-slug", required=True, help="e.g. yourusername/qwen38-protected-suite-v1")
    parser.add_argument(
        "--publish-publicly",
        action="store_true",
        help="DANGEROUS: publish the protected suite as a PUBLIC Kaggle dataset instead of private. "
        "Refused unless --i-understand-this-exposes-protected-content is also passed.",
    )
    parser.add_argument("--i-understand-this-exposes-protected-content", action="store_true")
    args = parser.parse_args(argv)

    if args.publish_publicly and not args.i_understand_this_exposes_protected_content:
        print(
            "Refusing to publish publicly without --i-understand-this-exposes-protected-content.",
            file=sys.stderr,
        )
        return 1

    suite_root = Path(args.suite_root)
    if not suite_root.is_dir():
        print(f"suite root does not exist: {suite_root}", file=sys.stderr)
        return 1

    metadata_path = suite_root / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "title": "qwen38-protected-suite-v1",
                "id": args.kaggle_dataset_slug,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    command = ["kaggle", "datasets", "create", "-p", str(suite_root), "-r", "zip"]
    if not args.publish_publicly:
        command.append("--private")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print(
        "Dataset created. VERIFY ITS VISIBILITY IS PRIVATE at "
        f"https://www.kaggle.com/datasets/{args.kaggle_dataset_slug} before relying on it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
