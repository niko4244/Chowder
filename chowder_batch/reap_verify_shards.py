"""Resumable, owned CPU-only BF16 source equivalence, one shard per subprocess.

--execute is explicit; dry-run is the default. Each completed shard receives
an immutable source-bound certificate so a host interruption never requires
rechecking earlier shards. Only the supervisor's own child is controllable.
No GPU, model service, frozen source, or previous receipt is modified.

--finalize independently checks the full 26-certificate set and rehashes all
original/source and candidate shard bytes before publishing the standard
FULL_SOURCE_TENSOR_EQUIVALENCE_PASS certificate. It is separately resource
admitted and does not rely on elapsed-time or file-size-only attestation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chowder_batch.reap_hidden_checkpoint import sha256_file
from chowder_batch.reap_prune_supervised import _run_owned, host_state
from chowder_batch.reap_verify_pruned import _indexed, _load

START_RAM_GIB = 12.0
# File-backed byte comparison uses O(header + bounded chunk), unlike the
# old tensor-materializing verifier. Maintain >=8 GiB runtime commit floor
# PLUS a conservative 8 GiB prelaunch margin; original host memory/commit
# admission for pruning and scientific GPU jobs is unchanged.
START_COMMIT_GIB = 16.0
MIN_DISK_GIB = 8.0


def _preflight(source: Path, observed: Path, selection: Path, candidate: Path,
               receipts_dir: Path) -> dict:
    source, observed, selection, candidate, receipts_dir = [
        p.resolve() for p in (source, observed, selection, candidate, receipts_dir)
    ]
    if not all(p.is_dir() for p in (source, observed, candidate, receipts_dir)):
        raise ValueError("all existing source, observation, candidate, and receipt directories required")
    if not selection.is_file():
        raise ValueError("registered selection is missing")
    if (receipts_dir.is_relative_to(source) or receipts_dir.is_relative_to(observed)
            or receipts_dir.is_relative_to(candidate)
            or source.is_relative_to(receipts_dir) or candidate.is_relative_to(receipts_dir)):
        raise ValueError("receipt directory must be disjoint from source and all model input trees")
    index = _indexed(_load(candidate / "model.safetensors.index.json"))
    if len(index) != 26:
        raise ValueError("full original Qwen BF16 model requires 26 indexed shards")
    host = host_state(candidate.parent)
    witness = host_state(receipts_dir)
    reasons = []
    if host.ram_available_gib < START_RAM_GIB:
        reasons.append("RAM below 12 GiB")
    if host.commit_available_gib < START_COMMIT_GIB:
        reasons.append("Windows commit below 16 GiB (8 GiB runtime floor + 8 GiB start reserve)")
    if host.output_disk_free_gib < MIN_DISK_GIB or witness.output_disk_free_gib < MIN_DISK_GIB:
        reasons.append("model or receipt volume below 8 GiB reserve")
    return {
        "status": "REFUSED" if reasons else "ADMITTED_NOT_EXECUTED",
        "reasons": reasons,
        "model": str(candidate),
        "source": str(source),
        "observed": str(observed),
        "selection": str(selection),
        "receipts": str(receipts_dir),
        "shards": sorted(index),
        "host": vars(host),
        "receipt_volume_free_gib": witness.output_disk_free_gib,
    }


def _certificate_name(shard: str) -> str:
    return shard.removesuffix(".safetensors") + ".verification.json"


def _validate_certificate(certificate: dict, shard: str, admission: dict) -> None:
    if (certificate.get("schema") != "chowder.reap128-v2-pruned-bf16-verification.v1"
            or certificate.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_SHARD_PASS"
            or certificate.get("individual_shard") != shard
            or certificate.get("candidate") != admission["model"]
            or certificate.get("original_source") != admission["source"]
            or certificate.get("shards") != 1
            or certificate.get("selection_sha256") != sha256_file(Path(admission["selection"]))
            or certificate.get("original_observation_sha256")
            != sha256_file(Path(admission["observed"]) / "observation.json")
            or certificate.get("candidate_index_sha256")
            != sha256_file(Path(admission["model"]) / "model.safetensors.index.json")
            or certificate.get("candidate_prune_receipt_sha256")
            != sha256_file(Path(admission["model"]) / "reap-prune.json")
            or not isinstance(certificate.get("candidate_shard_sha256"), dict)
            or set(certificate["candidate_shard_sha256"]) != {shard}
            or not isinstance(certificate.get("original_shard_sha256"), str)
            or certificate.get("routed_tensors", -1) < 0
            or certificate.get("row_or_tensor_checks", 0) <= 0):
        raise ValueError(f"missing or mismatched source-bound shard certificate: {shard}")


def consolidate(admission: dict, *, target: Path) -> dict:
    if target.exists() or not target.parent.is_dir():
        raise ValueError("final verification receipt must be new")
    source, candidate = Path(admission["source"]), Path(admission["model"])
    original = _load(Path(admission["observed"]) / "observation.json")
    expected_shas = {
        key.removeprefix("weights/"): digest
        for key, digest in original["checkpoint_receipt"]["source_identity"]["artifacts"].items()
        if key.startswith("weights/")
    }
    if set(expected_shas) != set(admission["shards"]):
        raise ValueError("registered upstream shard identity differs from candidate index")
    totals = {"source_non_mtp_tensor_bytes": 0, "pruned_tensor_bytes": 0,
              "row_or_tensor_checks": 0, "routed_tensors": 0}
    attested = {}
    for shard in admission["shards"]:
        receipt = Path(admission["receipts"]) / _certificate_name(shard)
        if not receipt.is_file():
            raise ValueError(f"missing per-shard bitwise proof: {shard}")
        certificate = _load(receipt)
        _validate_certificate(certificate, shard, admission)
        # Full current-state rehash closes the check-to-use gap, including
        # same-size mutations since the durable per-shard certificate.
        if (certificate["original_shard_sha256"] != expected_shas[shard]
                or sha256_file(source / shard) != expected_shas[shard]
                or sha256_file(candidate / shard) != certificate["candidate_shard_sha256"][shard]):
            raise ValueError(f"live source or pruned shard differs from attested bytes: {shard}")
        for key in totals:
            value = certificate.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid {key} per-shard quantity: {shard}")
            totals[key] += value
        attested[shard] = certificate["candidate_shard_sha256"][shard]
    published = _load(candidate / "reap-prune.json")
    index = _load(candidate / "model.safetensors.index.json")
    if (totals["routed_tensors"] != 120
            or totals["pruned_tensor_bytes"] != 37_981_137_632
            or totals["pruned_tensor_bytes"] != published.get("total_bytes")
            or totals["pruned_tensor_bytes"] != index.get("metadata", {}).get("total_size")):
        raise ValueError("26-shard geometry sum disagrees with registered 40x128 model")
    return {
        "schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
        "status": "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS",
        "candidate": str(candidate),
        "original_source": str(source),
        "selection_sha256": sha256_file(Path(admission["selection"])),
        "original_observation_sha256": sha256_file(Path(admission["observed"]) / "observation.json"),
        "candidate_index_sha256": sha256_file(candidate / "model.safetensors.index.json"),
        "candidate_prune_receipt_sha256": sha256_file(candidate / "reap-prune.json"),
        "candidate_shard_sha256": attested,
        "shards": len(attested),
        **totals,
        "no_gpu_loaded": True,
        "qualifies_scientific_accuracy": False,
        "partial_shard_proofs": {
            shard: sha256_file(Path(admission["receipts"]) / _certificate_name(shard))
            for shard in sorted(attested)
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "observed", "selection", "candidate", "receipts-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize", type=Path, help="new complete proof; refuses incomplete shards")
    parser.add_argument("--max-new-shards", type=int, default=26)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.max_new_shards <= 26:
            raise ValueError("max-new-shards must be between 1 and 26")
        admission = _preflight(args.source, args.observed, args.selection,
                               args.candidate, args.receipts_dir)
        if args.finalize is not None:
            if admission["reasons"]:
                raise ValueError("full consolidation resource admission failed: "
                                 + "; ".join(admission["reasons"]))
            result = consolidate(admission, target=args.finalize.resolve())
            with args.finalize.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
            print(json.dumps({"status": result["status"], "receipt": str(args.finalize),
                              "shards": result["shards"]}), flush=True)
            return 0
        completed = []
        remaining = []
        for shard in admission["shards"]:
            file = Path(admission["receipts"]) / _certificate_name(shard)
            if file.is_file():
                _validate_certificate(_load(file), shard, admission)
                completed.append(shard)
            else:
                remaining.append(shard)
        print(json.dumps({"admission": admission, "completed": completed,
                          "remaining": remaining}, sort_keys=True), flush=True)
        if admission["reasons"]:
            return 2
        if not args.execute:
            return 0
        if args.log is None:
            raise ValueError("--execute requires a new exclusive --log")
        log_path = args.log.resolve()
        roots = [Path(admission[key]) for key in ("model", "source", "observed", "receipts")]
        if (log_path.exists() or not log_path.parent.is_dir()
                or any(log_path.is_relative_to(root) for root in roots)):
            raise ValueError("log must be new and outside every model/input/receipt directory")
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            for shard in remaining[:args.max_new_shards]:
                fresh = _preflight(args.source, args.observed, args.selection,
                                   args.candidate, args.receipts_dir)
                if fresh["reasons"]:
                    raise ValueError("host resources changed before shard: "
                                     + "; ".join(fresh["reasons"]))
                output = Path(admission["receipts"]) / _certificate_name(shard)
                if output.exists():
                    raise ValueError("another writer created shard certificate")
                command = [
                    sys.executable, "-m", "chowder_batch.reap_verify_pruned",
                    "--full-compare", "--shard", shard, "--source", admission["source"],
                    "--observed", admission["observed"], "--selection", admission["selection"],
                    "--candidate", admission["model"], "--out", str(output),
                ]
                log.write(json.dumps({"event": "OWNED_FULL_SHARD_START", "shard": shard}) + "\n")
                log.flush()
                returncode = _run_owned(command, out=output, log=log)
                if returncode or not output.is_file():
                    raise RuntimeError(f"owned BF16 source verification stopped for {shard}: "
                                       f"exit={returncode}; previous shard proofs preserved")
                _validate_certificate(_load(output), shard, admission)
                log.write(json.dumps({"event": "OWNED_FULL_SHARD_PASS", "shard": shard,
                                      "receipt_sha256": sha256_file(output)}) + "\n")
                log.flush()
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"BF16 resumable equivalence refused: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
