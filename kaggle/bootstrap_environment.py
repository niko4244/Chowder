"""Kaggle notebook script: reproducible environment bootstrap + fingerprint.

Run this FIRST, before acquisition or evaluation, in a private Kaggle
GPU notebook (T4 x2 accelerator selected; this script itself does not
require the GPU, but later steps do).

What this does
---------------
1. Installs Chowder from the exact pinned Git commit given by
   ``--commit`` (never a branch, never "main") with the extras needed for
   4-bit protected-parent evaluation (`train`, `qlora`). This is the
   "prefer the exact pinned code over hand-copied notebook cells"
   requirement -- every later script in this directory imports
   `chowder.*` rather than reimplementing anything.
2. Verifies the installed `chowder` package really did resolve to that
   commit (`pip show`/`importlib.metadata` records the VCS ref; this
   script cross-checks it and refuses to silently continue on a mismatch,
   e.g. if pip served a cached wheel from a previous, different commit).
3. Captures and writes a full `BackendFingerprint` JSON (Python,
   torch/transformers/bitsandbytes/accelerate versions, CUDA runtime, GPU
   model(s) and count, device-map summary, quantization/dtype policy) --
   the exact environment-evidence fields the mission requires recorded
   for every later Kaggle run. Uses `chowder.kaggle_launcher.
   capture_environment_fingerprint` unchanged.

Usage
-----
    python bootstrap_environment.py --commit <40-char-sha> --output /kaggle/working/environment.json

Never prints or writes any Hugging Face token; this script does not
touch Kaggle Secrets at all.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _install_chowder_at_commit(commit_sha: str) -> None:
    if len(commit_sha) != 40 or any(c not in "0123456789abcdef" for c in commit_sha.lower()):
        raise ValueError(
            f"--commit must be a full 40-character git commit sha, not a branch name "
            f"or short hash: {commit_sha!r}"
        )
    spec = f"chowder-ai[train,qlora] @ git+https://github.com/niko4244/Chowder.git@{commit_sha}"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", spec], check=True)


def _installed_chowder_commit() -> str | None:
    """Best-effort read of the resolved VCS commit from package metadata.

    pip records a `direct_url.json` alongside the installed distribution
    for a VCS install; its `vcs_info.commit_id` is the exact resolved
    sha, independent of what `--commit` asked for -- the honest
    cross-check that pip actually installed that commit rather than
    silently reusing a cached wheel.
    """
    import importlib.metadata

    try:
        dist = importlib.metadata.distribution("chowder-ai")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        return None
    direct_url = json.loads(direct_url_text)
    return (direct_url.get("vcs_info") or {}).get("commit_id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", required=True, help="Exact 40-character Chowder commit sha to install")
    parser.add_argument("--output", required=True, help="Path to write the environment fingerprint JSON")
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip the pip install step (chowder already installed at the right commit)",
    )
    args = parser.parse_args(argv)

    if not args.skip_install:
        _install_chowder_at_commit(args.commit)

    resolved = _installed_chowder_commit()
    if resolved is not None and resolved != args.commit:
        print(
            f"REFUSING TO CONTINUE: requested commit {args.commit} but the installed "
            f"chowder-ai package resolved to {resolved} (a cached wheel or a prior "
            "install may be in the way). Re-run with --skip-install removed, or in a "
            "fresh Kaggle session.",
            file=sys.stderr,
        )
        return 1

    from chowder.kaggle_launcher import capture_environment_fingerprint

    fingerprint = capture_environment_fingerprint(
        chowder_commit_sha=resolved or args.commit,
        tokenizer_identity_sha256=None,  # filled in per-parent by run_parent_evaluation.py
        quantization="4bit",
        dtype="float16",  # T4 cannot run bf16 -- see chowder.kaggle_launcher.PRECISION_DIVERGENCE_REASON
        device_map_summary='{"": 0}',  # single-GPU pin, matching base_text_worker.evaluate's own convention
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fingerprint.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Environment fingerprint written to {output_path}")
    print(json.dumps(fingerprint.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
