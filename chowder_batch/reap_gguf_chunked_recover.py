"""Recover a *finished and sealed* Q8 assembly after interrupted publication.

The primary Q8 build deletes its source chunks on I: only after producing a
full, SHA-verified merged model on F:. A process interruption in that narrow
publication phase leaves some sealed chunk payloads intentionally absent and
the ordinary chunk builder cannot resume. This recovery route checks the
original BF16 and 26-shard ancestry, all 42 group receipts, the actual merged
733-tensor GGUF, exact chunk-deletion audit events, and cross-volume bytes.

No quantizer is ever launched. No missing tensor or receipt is reconstructed.
The registered scientific quality and GPU gates remain NOT_EVALUATED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from shutil import disk_usage

from chowder_batch import reap_gguf_chunked as core

MODE = "Q8_0"
Q8_FINAL_SCHEMA = "chowder.reap128-v2-gguf-q8-chunked-final.v1"
QUANTIZER_SHA = "eeb75877bd41d8d7b51d18129d873e6220014613baea349fa18e6b5d36c987de"
FROZEN_QUANTIZER_PATH = Path("C:/llama.cpp/bin/llama-quantize.exe")
FROZEN_BF16_SHA = "80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6"
FROZEN_SPEC_SHA = "7f9065d8036f7bd04c19a15f87540ad195658f46a80a1a09aa407b44c4836a94"
FROZEN_PROOF_SHA = "0746c8c06764f784fa7134ecdc4a18954a761372f865256c9e0120588cfbbc7d"
CERTIFIED_ORPHAN_STATUS = "GGUF_CHUNKED_TENSOR_INDEPENDENT_ORPHAN_ATTESTED_NOT_QUALITY_QUALIFIED"


def _hash_span(path: Path, offset: int, nbytes: int) -> str:
    """Hash a GGUF tensor from bounded *file reads*, not mmap resident pages."""
    if (type(offset) is not int or offset < 0 or type(nbytes) is not int or nbytes <= 0
            or offset > path.stat().st_size
            or nbytes > path.stat().st_size - offset):
        raise ValueError("GGUF raw tensor span exceeds existing file bounds")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = nbytes
        while remaining:
            block = stream.read(min(core.HASH_BLOCK, remaining))
            if not block:
                raise ValueError("GGUF tensor raw byte span truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _independent_volume_plan(source: Path, output: Path, scratch: Path, chunks: Path) -> bool:
    return (source.parent == scratch.parent and output.parent == chunks.parent
            and source.drive.lower() != output.drive.lower())


def _small_json(path: Path, limit: int = 4 * core.MIB):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing or oversized certified JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"certified JSON must contain an object: {path}")
    return data


def _only_match(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise ValueError(f"expected exactly one owned and sealed {pattern} in {directory}")
    return candidates[0]


def _cleanup_events(log: Path) -> dict[str, str]:
    if log.is_symlink() or not log.is_file() or log.stat().st_size > 64 * core.MIB:
        raise ValueError("original exclusive Q8 supervision log unavailable")
    deleted = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # an interrupted, unsealed event is not evidence
        if not isinstance(event, dict) or event.get("event") != "EXACT_SOURCE_BOUND_PART_CLEANED":
            continue
        name, sha = event.get("path"), event.get("sha256")
        if (not isinstance(name, str) or not isinstance(sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                or name in deleted):
            raise ValueError("ambiguous original Q8 chunk deletion audit")
        deleted[name] = sha
    return deleted


def _resume_owned_copy(assembly: Path, output: Path, sha256: str,
                       resource_problems) -> None:
    """Resume at most one original/recovery-owned I: transfer, never guess bytes.

    The original Q8 writer can be stopped by its physical-RAM guard after
    deleting all 42 already-certified chunks. Its partial `.chunked-owned-`
    copy is valuable, but must not be treated as a sealed model. Under the
    exclusive source-chunk lock, compare *every existing prefix byte* with
    the full independently reattested F: assembly before appending.
    """
    prefix = "." + output.stem
    pattern = re.compile(re.escape(prefix) +
                         r"\.(?:chunked|recovery)-owned-[0-9a-f]{32}\.gguf\Z")
    candidates = sorted(path for path in output.parent.iterdir()
                        if path.name.startswith(prefix + ".")
                        and (".chunked-owned-" in path.name
                             or ".recovery-owned-" in path.name))
    if len(candidates) > 1:
        raise ValueError("multiple Q8 private transfers prevent deterministic recovery")
    if candidates:
        transfer = candidates[0]
        if (not pattern.fullmatch(transfer.name) or transfer.is_symlink()
                or not transfer.is_file()):
            raise ValueError("untrusted Q8 private transfer candidate")
        offset = transfer.stat().st_size
        if offset > assembly.stat().st_size:
            raise ValueError("Q8 partial transfer longer than independently sealed assembly")
    else:
        transfer = output.with_name(prefix + ".recovery-owned-" +
                                    uuid.uuid4().hex + ".gguf")
        offset = 0
    if disk_usage(output.parent).free < assembly.stat().st_size - offset + 6 * core.GIB:
        raise ValueError("Q8 recovery lacks remaining copy bytes plus protected reserve")
    with assembly.open("rb") as src, transfer.open("r+b" if candidates else "xb") as dst:
        remaining = offset
        while remaining:
            piece = min(core.HASH_BLOCK, remaining)
            if src.read(piece) != dst.read(piece):
                raise ValueError("Q8 partial transfer failed exact assembled-prefix comparison")
            remaining -= piece
        dst.seek(offset)
        for block in iter(lambda: src.read(core.HASH_BLOCK), b""):
            if resource_problems([assembly, transfer]):
                raise RuntimeError("Q8 resumed transfer refused by original host guard")
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    if core.hash_file(transfer) != sha256:
        raise ValueError("Q8 completed owned transfer differs from original F: SHA")
    transfer.rename(output)


def _verify_source(source: Path, verification: Path, source_sha: str):
    from gguf import GGUFReader

    if source_sha != FROZEN_BF16_SHA:
        raise ValueError("registered original REAP128_v2 BF16 SHA required")
    stage = _small_json(source.with_name(source.name + ".reap-stage.json"))
    full = _small_json(verification)
    if (stage.get("schema") != "chowder.reap-gguf-stage.v1"
            or stage.get("phase") != 0
            or stage.get("output") != str(source)
            or stage.get("size_bytes") != source.stat().st_size
            or stage.get("sha256") != source_sha
            or stage.get("spec_sha256") != FROZEN_SPEC_SHA
            or stage.get("input_sha256") != core.hash_file(verification)
            or stage["input_sha256"] != FROZEN_PROOF_SHA
            or stage.get("status") != "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED"
            or full.get("schema") != "chowder.reap128-v2-pruned-bf16-verification.v1"
            or full.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
            or len(full.get("candidate_shard_sha256", {})) != 26
            or core.hash_file(source) != source_sha):
        raise ValueError("original published BF16 and all 26 source shards not attested")
    original = GGUFReader(str(source), "r")
    for key, expected in (("general.file_type", 32),
                          ("qwen35moe.block_count", 40),
                          ("qwen35moe.expert_count", 128),
                          ("qwen35moe.expert_used_count", 8)):
        if (original.get_field(key) is None
                or original.get_field(key).contents() != expected):
            raise ValueError("recovery source violates frozen 40x128/top-k 8 geometry")
    if len(original.tensors) != 733:
        raise ValueError("recovery source is not full 733-tensor GGUF")
    return original, stage


def _recheck_owned_assembly(
    source: Path, scratch: Path, chunks: Path, original, stage: dict,
    source_sha: str, supervision_log: Path, *, max_mib: int, slab_mib: int,
    admit_complete_unsealed_orphan: bool = False,
) -> tuple[Path, dict, list[list[str]], dict]:
    from gguf import GGUFReader

    assembly = _only_match(scratch, "merged-q8-*.gguf")
    assembly_path = assembly.with_name(assembly.name + ".receipt.json")
    if assembly_path.is_symlink():
        raise ValueError("Q8 merged worker receipt cannot be a symlink")
    unsealed = not assembly_path.exists()
    if unsealed and not admit_complete_unsealed_orphan:
        raise ValueError("Q8 finished merge is missing independent or original seal")
    if unsealed:
        if not (supervision_log.is_file() and not supervision_log.is_symlink()
                and supervision_log.stat().st_size <= 64 * core.MIB
                and "OWNED_CONVERSION_ABORT: insufficient unowned physical RAM"
                in supervision_log.read_text(encoding="utf-8", errors="replace")):
            raise ValueError("unsealed Q8 orphan is not bound to the original guarded RAM abort")
        # Never claim that the original guarded worker issued a receipt.
        # Certification must reconstruct the entire SHA/tensor proof below.
        assembly_receipt = {"status": CERTIFIED_ORPHAN_STATUS,
                            "mode": MODE, "source": str(source),
                            "output": str(assembly), "tensors": 733}
    else:
        assembly_receipt = _small_json(assembly_path)
    if (assembly_receipt.get("status") not in {
                "GGUF_CHUNKED_TENSOR_BITWISE_MERGE_NOT_QUALITY_QUALIFIED",
                CERTIFIED_ORPHAN_STATUS,
            }
            or assembly_receipt.get("mode") != MODE
            or assembly_receipt.get("source") != str(source)
            or assembly_receipt.get("output") != str(assembly)
            or assembly_receipt.get("tensors") != 733
            or (not unsealed and assembly_receipt.get("output_sha256") != core.hash_file(assembly))):
        raise ValueError("Q8 owned assembled GGUF has no valid finished merge witness")
    groups = core.groups_for(original, max_mib=max_mib)
    signature = {
        "source_sha256": source_sha, "stage_spec_sha256": stage["spec_sha256"],
        "quantizer_sha256": QUANTIZER_SHA, "mode": MODE,
        "slab_mib": slab_mib, "max_mib": max_mib,
        "chunk_count": len(groups),
    }
    deletion_events = _cleanup_events(supervision_log)
    expected_tensors = {}
    named_original = {t.name: t for t in original.tensors}
    for index, names in enumerate(groups):
        prefix = f"chunk-{index+1:03d}-of-{len(groups):03d}"
        part = chunks / (prefix + ".q8_0.gguf")
        if part.is_symlink():
            raise ValueError("Q8 chunk payload cannot be redirected through symlink")
        receipt = _small_json(chunks / (prefix + ".receipt.json"))
        identity = {**signature, "index": index, "names": names}
        command = [
            str(FROZEN_QUANTIZER_PATH.resolve()), "--max-buffer-size", str(slab_mib),
            str(scratch / (prefix + ".bf16.gguf")), str(part), MODE, "2",
        ]
        if (receipt.get("schema") != "chowder.reap128-v2-gguf-chunk.v1"
                or receipt.get("identity") != identity
                or receipt.get("status") != "CHUNK_Q8_BITWISE_WITNESSED_NOT_QUALITY_GATE"
                or receipt.get("output") != str(part)
                or receipt.get("effective_quant_command") != command
                or not isinstance(receipt.get("output_tensor_provenance"), dict)
                or set(receipt["output_tensor_provenance"]) != set(names)
                or set(receipt.get("input_tensor_provenance", {})) != set(names)):
            raise ValueError(f"chunk {index+1} source-bound receipt mismatch")
        if part.exists():
            core._load_chunk_record(chunks / (prefix + ".receipt.json"),
                                    part, expected=identity)
        elif unsealed:
            raise ValueError(f"unsealed Q8 F assembly requires every original part, missing {index+1}")
        elif deletion_events.get(str(part)) != receipt.get("output_sha256"):
            raise ValueError(f"missing chunk {index+1} lacks exact owned deletion event")
        for name in names:
            original_tensor = named_original[name]
            original_receipt = receipt["input_tensor_provenance"][name]
            if (original_receipt.get("shape") != list(map(int, original_tensor.shape))
                    or original_receipt.get("tensor_type") != original_tensor.tensor_type.name
                    or original_receipt.get("raw_sha256") !=
                    _hash_span(source, int(original_tensor.data_offset),
                               int(original_tensor.n_bytes))):
                raise ValueError(f"chunk source tensor differs from published BF16: {name}")
            if name in expected_tensors:
                raise ValueError("overlapping Q8 chunk tensor receipts")
            expected_tensors[name] = receipt["output_tensor_provenance"][name]
    # A guarded merge can abort before producing any assembly, leaving its
    # small merge-input-UUID.json behind. A later, byte-identical retry writes
    # another immutable roster. Requiring *exactly one* would strand the valid
    # completed assembly after post-merge chunk deletion. Admit multiple only
    # when every existing source-bound roster is independently identical.
    rosters = sorted(scratch.glob("merge-input-*.json"))
    if not 1 <= len(rosters) <= 128:
        raise ValueError("missing or unbounded Q8 merge-input roster history")
    for roster in rosters:
        if _small_json(roster) != expected_tensors:
            raise ValueError("saved merge tensor roster differs from original chunk receipts")
    merged = GGUFReader(str(assembly), "r")
    if (merged.get_field("general.file_type").contents() != 7
            or merged.get_field("qwen35moe.block_count").contents() != 40
            or merged.get_field("qwen35moe.expert_count").contents() != 128
            or len(merged.tensors) != 733
            or [t.name for t in merged.tensors] != [t.name for t in original.tensors]):
        raise ValueError("merged GGUF is not registered 733-tensor 40x128 Q8 model")
    for source_tensor, t in zip(original.tensors, merged.tensors):
        claimed = expected_tensors[t.name]
        input_shape = tuple(int(v) for v in source_tensor.shape)
        actual_shape = tuple(int(v) for v in t.shape)
        original_is_f32 = source_tensor.tensor_type.name == "F32"
        if original_is_f32:
            while len(input_shape) > len(actual_shape) and input_shape[-1] == 1:
                input_shape = input_shape[:-1]
        matches = input_shape == actual_shape
        if original_is_f32:
            matches = (matches and t.tensor_type.name == "F32"
                       and _hash_span(source, int(source_tensor.data_offset),
                                      int(source_tensor.n_bytes))
                       == claimed.get("raw_sha256"))
        if (not matches
                or claimed.get("shape") != list(map(int, t.shape))
                or claimed.get("tensor_type") != t.tensor_type.name
                or claimed.get("raw_sha256") !=
                _hash_span(assembly, int(t.data_offset), int(t.n_bytes))):
            raise ValueError(f"merged Q8 tensor provenance or payload differs: {t.name}")
    # File digest after every reconstructed chunk/tensor witness, without
    # mmap paging of the 18+ GiB output into the resident process working set.
    final_sha = core.hash_file(assembly)
    if unsealed:
        assembly_receipt["output_sha256"] = final_sha
        assembly_receipt["origin"] = (
            "independent 733-tensor reattestation of guarded merge orphan; "
            "original merge worker aborted without certifying a receipt"
        )
        assembly_receipt["supervision_log_sha256"] = core.hash_file(supervision_log)
    elif assembly_receipt.get("output_sha256") != final_sha:
        raise ValueError("assembled GGUF full SHA changed during full source reattestation")
    return assembly, assembly_receipt, groups, expected_tensors


def recover_final(
    *, source: Path, output: Path, scratch: Path, chunks: Path,
    full_verification: Path, supervision_log: Path, quantizer: Path,
    source_sha: str, max_mib: int = 1024, slab_mib: int = 1,
    execute: bool = False, seal_unsealed_orphan_only: bool = False,
) -> dict:
    from chowder_batch.reap_gguf_supervised import resource_problems

    paths = [source, output, scratch, chunks, full_verification, supervision_log, quantizer]
    source, output, scratch, chunks, full_verification, supervision_log, quantizer = [
        p.resolve() for p in paths
    ]
    if (not scratch.is_dir() or not chunks.is_dir()
            or scratch.is_symlink() or chunks.is_symlink()
            or not _independent_volume_plan(source, output, scratch, chunks)
            or quantizer != FROZEN_QUANTIZER_PATH.resolve()
            or output.is_symlink() or output.with_name(output.name + ".reap-chunked.json").exists()
            or core.hash_file(quantizer) != QUANTIZER_SHA):
        raise ValueError("Q8 recovery requires original exclusive cross-volume workspaces")
    with core._exclusive_chunk_owner(chunks):
        if resource_problems([source, output, scratch], initial=True):
            raise ValueError("recovery host RAM/commit/disk admission failed")
        original, stage = _verify_source(source, full_verification, source_sha)
        assembly, sealed, groups, tensor_proofs = _recheck_owned_assembly(
            source, scratch, chunks, original, stage, source_sha, supervision_log,
            max_mib=max_mib, slab_mib=slab_mib,
            admit_complete_unsealed_orphan=seal_unsealed_orphan_only,
        )
        report = {
            "schema": Q8_FINAL_SCHEMA,
            "status": "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED",
            "source_gguf_sha256": source_sha,
            "output": str(output),
            "output_sha256": sealed["output_sha256"],
            "output_size_bytes": assembly.stat().st_size,
            "frozen_stage_spec_sha256": stage["spec_sha256"],
            "quantizer_binary_sha256": QUANTIZER_SHA,
            "chunk_count": len(groups),
            "chunked_reference_receipts": {
                "all_tensors": tensor_proofs, "source_groups": groups,
            },
            "quality_gate": "NOT_EVALUATED",
            "gpu_fit_gate": "NOT_EVALUATED",
        }
        if not execute:
            return {"status": "ORPHAN_FULL_TENSOR_PREFLIGHT_PASS_NOT_PUBLISHED"
                    if seal_unsealed_orphan_only else "RECOVERY_PREFLIGHT_PASS_NOT_PUBLISHED",
                    "source_sha256": source_sha,
                    "output_sha256": sealed["output_sha256"],
                    "tensors": len(tensor_proofs), "chunks": len(groups)}
        if seal_unsealed_orphan_only:
            stage_receipt = assembly.with_name(assembly.name + ".receipt.json")
            if stage_receipt.exists() or stage_receipt.is_symlink():
                raise ValueError("source-recovered orphan must not overwrite an existing merge witness")
            with stage_receipt.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(sealed, sort_keys=True, indent=2,
                                        allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return {"status": CERTIFIED_ORPHAN_STATUS,
                    "assembled_output": str(assembly),
                    "output_sha256": sealed["output_sha256"],
                    "tensors": len(tensor_proofs), "chunks": len(groups),
                    "final_I": "NOT_PUBLISHED", "quality_gate": "NOT_EVALUATED"}
        if output.exists():
            # Crash after cross-volume rename but before exclusive sidecar.
            if (output.stat().st_size != assembly.stat().st_size
                    or core.hash_file(output) != sealed["output_sha256"]):
                raise ValueError("existing unsealed final target has different bytes")
        else:
            _resume_owned_copy(assembly, output, sealed["output_sha256"],
                               resource_problems)
        sidecar = output.with_name(output.name + ".reap-chunked.json")
        with sidecar.open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
        return {"status": report["status"], "output": str(output),
                "output_sha256": report["output_sha256"],
                "tensors": len(tensor_proofs), "chunks": len(groups)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "out", "scratch", "chunks", "quantizer",
                "full-verification", "supervision-log"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--parent-source-sha256", required=True)
    parser.add_argument("--max-mib", type=int, default=1024)
    parser.add_argument("--slab-mib", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seal-unsealed-orphan-only", action="store_true",
                        help="independently certify an aborted but complete F: assembly; never delete/copy chunks")
    a = parser.parse_args(argv)
    try:
        result = recover_final(
            source=a.source, output=a.out, scratch=a.scratch, chunks=a.chunks,
            quantizer=a.quantizer, full_verification=a.full_verification,
            supervision_log=a.supervision_log, source_sha=a.parent_source_sha256,
            max_mib=a.max_mib, slab_mib=a.slab_mib, execute=a.execute,
            seal_unsealed_orphan_only=a.seal_unsealed_orphan_only,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(f"Q8 final publication recovery REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
