"""Independently attest the complete published REAP128_v2 mixed-K GGUF.

This is an expensive, CPU-only *post-publication* source-bound audit: it
rehashes the entire original BF16 source, all 733 original and mixed-K
tensor payloads, the complete final GGUF, every immutable chunk receipt,
and the publication sidecar. Neither sampled accuracy nor this certificate
is a registered GSM8K/HumanEval, GPU-fit or deployment gate.

Do not invoke while an exclusive Q4 constructor/merge is still running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from chowder_batch import reap_gguf_chunked as core
from chowder_batch import reap_gguf_quant_sample as quant
from chowder_batch.reap_gguf_q4_chunked import BINARY_SHA256, _publish_exclusive

SCHEMA = "chowder.reap128-v2-q4-independent-full-tensor-certificate.v1"
STATUS = "INDEPENDENT_SOURCE_AND_733_MIXED_K_PAYLOADS_MATCH_NOT_QUALITY_GATE"
CHUNKS = 42
TENSORS = 733
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")
FINAL_SCHEMA = "chowder.reap128-v2-gguf-q4-chunked-final.v1"
CHUNK_SCHEMA = "chowder.reap128-v2-gguf-chunk.v1"
CHUNK_STATUS = "CHUNK_Q4_K_M_BITWISE_WITNESSED_NOT_QUALITY_GATE"
FINAL_STATUS = "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"
ALLOWED = {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "F32", "F16", "BF16"}


def _json_receipt(path: Path, max_bytes: int = 4 * 1024 * 1024) -> dict:
    if (path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= max_bytes):
        raise ValueError(f"missing or untrusted certificate input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object certificate: {path}")
    return value


def collect_chunk_receipts(
    chunks: Path, *, source_sha: str, stage_sha: str,
    names: list[str], final_claims: dict,
) -> tuple[dict, dict, list[dict]]:
    """Reconcile 42 immutable mini-GGUF receipts with the final publication.

    Safe, receipt-only preparatory work: never map or read massive model
    files. Actual full tensor/source bytes are verified in ``certify``.
    """
    if len(names) != TENSORS or len(set(names)) != TENSORS:
        raise ValueError("expected complete unique frozen 733 tensor names")
    groups = final_claims.get("source_groups")
    proofs = final_claims.get("all_tensors")
    if (not isinstance(groups, list) or len(groups) != CHUNKS
            or not isinstance(proofs, dict) or set(proofs) != set(names)):
        raise ValueError("published Q4 has incomplete 42-group/733-tensor witness")
    seen: set[str] = set()
    source_proofs: dict = {}
    receipt_digests: list[dict] = []
    for i, group in enumerate(groups):
        name = f"chunk-{i + 1:03d}-of-{CHUNKS:03d}.receipt.json"
        path = chunks / name
        record = _json_receipt(path, 2 * 1024 * 1024)
        identity = record.get("identity")
        if (record.get("schema") != CHUNK_SCHEMA
                or record.get("status") != CHUNK_STATUS
                or not isinstance(record.get("source_mini_sha256"), str)
                or not HEX_SHA.fullmatch(record["source_mini_sha256"])
                or not isinstance(record.get("output_sha256"), str)
                or not HEX_SHA.fullmatch(record["output_sha256"])
                or type(record.get("output_size_bytes")) is not int
                or record["output_size_bytes"] <= 0
                or record.get("output") != str(chunks / name.replace(".receipt.json", ".q4_k_m.gguf"))
                or not isinstance(identity, dict)
                or identity.get("source_sha256") != source_sha
                or identity.get("stage_spec_sha256") != stage_sha
                or identity.get("quantizer_sha256") != BINARY_SHA256
                or identity.get("chunk_count") != CHUNKS
                or type(identity.get("index")) is not int or identity["index"] != i
                or identity.get("mode") != "Q4_K_M"
                or identity.get("max_mib") != 1024
                or identity.get("slab_mib") != 1
                or not isinstance(group, list) or not group
                or identity.get("names") != group):
            raise ValueError(f"chunk {i + 1}: immutable BF16-source/recipe identity mismatch")
        old = record.get("input_tensor_provenance")
        new = record.get("output_tensor_provenance")
        if (not isinstance(old, dict) or not isinstance(new, dict)
                or set(old) != set(group) or set(new) != set(group)
                or seen.intersection(group)):
            raise ValueError(f"chunk {i + 1}: missing/duplicate source tensor provenance")
        command = record.get("effective_quant_command")
        if (not isinstance(command, list) or len(command) < 6
                or command[-2:] != ["Q4_K_M", "2"]
                or command[1:3] != ["--max-buffer-size", "1"]):
            raise ValueError(f"chunk {i + 1}: original bounded quantization command changed")
        for tensor in group:
            a, b = old[tensor], new[tensor]
            if (not isinstance(a, dict) or not isinstance(b, dict)
                    or set(a) != {"shape", "tensor_type", "raw_sha256"}
                    or set(b) != {"shape", "tensor_type", "raw_sha256"}
                    or a.get("tensor_type") not in ("BF16", "F32")
                    or b.get("tensor_type") not in ALLOWED
                    or any(not isinstance(proof.get("shape"), list)
                           or not proof["shape"] or any(type(v) is not int or v <= 0
                                                           for v in proof["shape"])
                           or not isinstance(proof.get("raw_sha256"), str)
                           or not HEX_SHA.fullmatch(proof["raw_sha256"])
                           for proof in (a, b))
                    or b != proofs[tensor]):
                raise ValueError(f"chunk {i + 1}: source/quantized tensor witness changed: {tensor}")
            source_proofs[tensor] = a
        seen.update(group)
        receipt_digests.append({"index": i, "filename": name,
                                "sha256": core.hash_file(path),
                                "output_sha256": record.get("output_sha256")})
    if seen != set(names) or [name for group in groups for name in group] != names:
        raise ValueError("42 chunk receipts do not cover the exact ordered BF16 tensor inventory")
    return source_proofs, proofs, receipt_digests


def certify(bf16: Path, model: Path, chunks: Path) -> dict:
    """Actually rehash original, final, all 733 payloads and chunk receipts."""
    from gguf import GGUFReader

    sidecar = model.with_name(model.name + ".reap-chunked.json")
    if any(p.is_symlink() or not p.is_file() for p in (bf16, model, sidecar)):
        raise ValueError("complete independent BF16/Q4/sidecar model evidence required")
    stage = quant._load_stage(bf16, 0)
    if (stage["sha256"] != quant.FROZEN_SOURCE_SHA256
            or stage["spec_sha256"] != quant.FROZEN_SPEC_SHA256
            or core.hash_file(bf16) != stage["sha256"]):
        raise ValueError("actual original frozen BF16 GGUF whole SHA256 differs")
    side = _json_receipt(sidecar)
    if (side.get("schema") != FINAL_SCHEMA
            or side.get("status") != FINAL_STATUS
            or side.get("source_gguf_sha256") != stage["sha256"]
            or side.get("frozen_stage_spec_sha256") != stage["spec_sha256"]
            or side.get("quantizer_binary_sha256") != BINARY_SHA256
            or side.get("quality_gate") != "NOT_EVALUATED"
            or side.get("gpu_fit_gate") != "NOT_EVALUATED"
            or side.get("chunk_count") != CHUNKS
            or side.get("output") != str(model)
            or side.get("output_size_bytes") != model.stat().st_size
            or side.get("output_sha256") != core.hash_file(model)):
        raise ValueError("Q4 published full-model original ancestry/whole SHA differs")
    source = GGUFReader(str(bf16), "r")
    candidate = GGUFReader(str(model), "r")
    if (len(source.tensors) != TENSORS or len(candidate.tensors) != TENSORS
            or any(reader.get_field(field) is None
                   or reader.get_field(field).contents() != expected
                   for reader, kind in ((source, 32), (candidate, 15))
                   for field, expected in (("general.file_type", kind),
                                           ("qwen35moe.block_count", 40),
                                           ("qwen35moe.expert_count", 128),
                                           ("qwen35moe.expert_used_count", 8)))):
        raise ValueError("original or final GGUF topology/mode does not match frozen 40×128")
    names = [t.name for t in source.tensors]
    if [t.name for t in candidate.tensors] != names:
        raise ValueError("Q4 GGUF tensor names/order differ from original BF16")
    orig_proofs, quant_proofs, receipts = collect_chunk_receipts(
        chunks, source_sha=stage["sha256"], stage_sha=stage["spec_sha256"],
        names=names, final_claims=side.get("chunked_reference_receipts", {}),
    )
    types: dict[str, int] = {}
    for left, right in zip(source.tensors, candidate.tensors):
        previous, current = orig_proofs[left.name], quant_proofs[right.name]
        if (previous["shape"] != list(map(int, left.shape))
                or previous["tensor_type"] != left.tensor_type.name
                or previous["raw_sha256"] != core.hash_tensor_span(
                    bf16, int(left.data_offset), int(left.n_bytes))
                or current["shape"] != list(map(int, right.shape))
                or current["tensor_type"] != right.tensor_type.name
                or current["raw_sha256"] != core.hash_tensor_span(
                    model, int(right.data_offset), int(right.n_bytes))
                or not core._quantized_shape_matches(left, right)):
            raise ValueError(f"independent original/quantized tensor payload invalid: {left.name}")
        if right.tensor_type.name in ("F32", "F16", "BF16") and (
            left.tensor_type != right.tensor_type
                or previous["raw_sha256"] != current["raw_sha256"]
        ):
            raise ValueError(f"original copied tensor changed under Q4: {left.name}")
        types[right.tensor_type.name] = types.get(right.tensor_type.name, 0) + 1
    if types.get("Q4_K", 0) == 0 or sum(types.values()) != TENSORS:
        raise ValueError("real mixed-K Q4 tensor inventory is missing")
    return {"schema": SCHEMA, "status": STATUS,
            "source_bf16_sha256": stage["sha256"],
            "published_gguf_sha256": side["output_sha256"],
            "published_sidecar_sha256": core.hash_file(sidecar),
            "quantizer_sha256": BINARY_SHA256,
            "frozen_stage_spec_sha256": stage["spec_sha256"],
            "source_tensor_count": TENSORS, "verified_tensor_count": TENSORS,
            "sealed_chunk_receipts": receipts, "chunk_count": CHUNKS,
            "mixed_tensor_types": types,
            "topology": {"layers": 40, "experts_per_layer": 128, "routed_top_k": 8},
            "quality_gate": "NOT_EVALUATED", "gpu_fit_gate": "NOT_EVALUATED",
            "release_approved": False,
            "limitations": "Independent construction and exact tensor ancestry only, "
                           "not GSM8K/HumanEval, quantization sample, GPU fit or Gen-0 selection."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
            raise ValueError("independent Q4 certificate requires new exclusive output")
        result = certify(args.bf16, args.q4, args.chunks)
        _publish_exclusive(args.out, result)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"Q4 independent source cert refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in
                      ("status", "published_gguf_sha256", "mixed_tensor_types",
                       "verified_tensor_count", "chunk_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
