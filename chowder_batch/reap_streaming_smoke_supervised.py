"""Run verified BF16 40-layer CPU inference with own-child resource safeguards.

Read-only admission is the default. An actual run requires --execute, a new
exclusive log, and a full source-tensor equivalence receipt. Only the child
spawned by this supervisor may be terminated on a RAM/commit/disk breach.
GPU visibility is disabled by the shared owned-child runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from chowder_batch.reap_prune_supervised import _run_owned, host_state

START_RAM_GIB = 16.0
START_COMMIT_GIB = 24.0
START_DISK_GIB = 8.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def preadmit(*, model: Path, verification: Path, receipt: Path) -> dict:
    model, verification, receipt = (p.resolve() for p in (model, verification, receipt))
    if not model.is_dir() or not verification.is_file():
        raise ValueError("a published model and full-equivalence receipt are required")
    if receipt.exists() or not receipt.parent.is_dir() or receipt.is_relative_to(model):
        raise ValueError("new smoke receipt must be outside the frozen model")
    proof = json.loads(verification.read_text(encoding="utf-8"))
    if (proof.get("schema") != "chowder.reap128-v2-pruned-bf16-verification.v1"
            or proof.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
            or proof.get("candidate") != str(model)
            or proof.get("shards") != 26
            or not isinstance(proof.get("candidate_shard_sha256"), dict)
            or len(proof["candidate_shard_sha256"]) != 26
            or proof.get("candidate_index_sha256")
            != _sha(model / "model.safetensors.index.json")
            or proof.get("candidate_prune_receipt_sha256")
            != _sha(model / "reap-prune.json")):
        raise ValueError("full source-exact BF16 proof is incomplete or does not bind this model")
    state = host_state(model.parent)
    report_disk = host_state(receipt.parent)
    problems = []
    if state.ram_available_gib < START_RAM_GIB:
        problems.append("available RAM below 16 GiB")
    if state.commit_available_gib < START_COMMIT_GIB:
        problems.append("Windows commit headroom below 24 GiB")
    if (state.output_disk_free_gib < START_DISK_GIB
            or report_disk.output_disk_free_gib < START_DISK_GIB):
        problems.append("model/report disk safety reserve below 8 GiB")
    return {
        "status": "REFUSED" if problems else "ADMITTED_NOT_EXECUTED",
        "reasons": problems,
        "model": str(model),
        "verification": str(verification),
        "verification_sha256": _sha(verification),
        "intended_smoke_receipt": str(receipt),
        "host": vars(state),
        "report_disk_free_gib": report_disk.output_disk_free_gib,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="The answer is")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--progress-dir", type=Path,
                        help="existing external directory for source-bound layer-activation receipts")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    try:
        decision = preadmit(
            model=args.model, verification=args.verification, receipt=args.out,
        )
        print(json.dumps(decision, sort_keys=True), flush=True)
        if decision["reasons"]:
            return 2
        if not args.execute:
            return 0
        if args.log is None:
            raise ValueError("executing requires a new explicit --log")
        log_path = args.log.resolve()
        proof = json.loads(args.verification.read_text(encoding="utf-8"))
        original_source = proof.get("original_source")
        protected_roots = [args.model.resolve()]
        if isinstance(original_source, str) and original_source:
            protected_roots.append(Path(original_source).resolve())
        progress_dir = args.progress_dir.resolve() if args.progress_dir else None
        if progress_dir is not None and (
                not progress_dir.is_dir()
                or any(progress_dir.is_relative_to(root) or root.is_relative_to(progress_dir)
                       for root in protected_roots)
                or args.out.resolve().is_relative_to(progress_dir)
                or log_path.is_relative_to(progress_dir)):
            raise ValueError("activation progress dir must be existing and disjoint from all models/logs/receipts")
        if (log_path.exists() or not log_path.parent.is_dir()
                or any(log_path.is_relative_to(root) for root in protected_roots)
                or log_path == args.out.resolve()):
            raise ValueError("owned log must be exclusive and outside original/model/receipt")
        # Re-admit immediately before spawning one owned child.
        fresh = preadmit(model=args.model, verification=args.verification, receipt=args.out)
        if fresh["reasons"]:
            raise ValueError("resource availability changed before spawn: "
                             + "; ".join(fresh["reasons"]))
        command = [
            sys.executable, "-m", "chowder_batch.reap_streaming_bf16_smoke",
            "--model", str(args.model.resolve()), "--prompt", args.prompt,
            "--max-tokens", str(args.max_tokens), "--out", str(args.out.resolve()),
        ]
        if progress_dir is not None:
            command.extend([
                "--progress-dir", str(progress_dir),
                "--full-verification-sha256", fresh["verification_sha256"],
            ])
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            log.write(json.dumps(fresh, sort_keys=True) + "\n")
            log.flush()
            code = _run_owned(command, out=args.out.resolve(), log=log)
            log.write(json.dumps({
                "event": "OWNED_SMOKE_EXIT", "returncode": code,
                "receipt_exists": args.out.is_file(),
            }) + "\n")
        if code == 0 and not args.out.is_file():
            raise ValueError("owned smoke exited successfully without published logits receipt")
        return code
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"BF16 CPU inference admission refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
