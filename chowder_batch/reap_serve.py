"""Serve one REAP arm with the pre-registered llama-server flags, then run reap_eval.

Usage:
  python chowder_batch/reap_serve.py --gguf <path> --arm <name> --port 8790 --out <eval dir> \
      -- --port 8790 --arm <name> --suites gsm8k --out <eval dir>

Flags are frozen by the prereg (section 3): ``--fit on -np 4 -c 16384 --jinja``.
Creates <out>/<arm>/server.log and <out>/<arm>/serve.json exclusively. Resumed
sessions retain these original receipts and use separate per-session records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SERVER = r"C:\llama.cpp\bin\llama-server.exe"
FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
HEALTH_TIMEOUT = 900
REGISTERED_V2_CONSTRUCTION = {
    "reap128_v2": ("chowder.reap128-v2-gguf-q8-chunked-final.v1",
                   "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"),
    "reap128_v2_q4": ("chowder.reap128-v2-gguf-q4-chunked-final.v1",
                      "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"),
}
REGISTERED_BF16_SHA256 = "80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6"
REGISTERED_STAGE_SHA256 = "7f9065d8036f7bd04c19a15f87540ad195658f46a80a1a09aa407b44c4836a94"
REGISTERED_QUANTIZER_SHA256 = "eeb75877bd41d8d7b51d18129d873e6220014613baea349fa18e6b5d36c987de"
REGISTERED_Q8_GGUF_SHA256 = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
REGISTERED_Q4_GGUF_SHA256 = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"

_KEEP = re.compile(r"llama_params_fit|common_fit|n_cpu_moe|overflowing|offloaded \d+/\d+ layers|exps.*(?:overridden|CPU)", re.I)
_CPU_MOE = re.compile(r"n_cpu_moe\s*[=:]?\s*(\d+)", re.I)
_OVERFLOW = re.compile(r"\((\d+) overflowing\)")
_OVERRIDE = re.compile(r"blk\.(\d+)\..*exps.*CPU", re.I)
_ALL_LAYERS = re.compile(r"offloaded (\d+)/(\d+) layers to GPU", re.I)
# llama-server logs neither the fit decision nor layer placement at its default
# verbosity (3); 4 does, at ~30 lines per request. Set through the environment so
# the pre-registered FLAGS (and inference) stay unchanged.
LOG_VERBOSITY = "4"
REGISTERED_V2_EVALUATION_ARMS = {
    "reap128_v2", "reap128_v2_q4", "reap160_v2", "reap160_v2_q4",
    "dense_v2_q4", "dense_v2", "random128_v2",
}
REGISTERED_DENSE_Q4_SHA256 = "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049"
REGISTERED_QUALITY_REFERENCE_SHA256 = {
    "dense_v2": "2ff57a74a36939c5baabf40a0d3eb06d4424ecc94cf7d4ef04b3ac02059097c2",
    "random128_v2": "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b",
    "dense_v2_q4": REGISTERED_DENSE_Q4_SHA256,
}


def parse_offload(text: str) -> tuple[list[str], int | None]:
    lines = [l.strip() for l in text.splitlines() if _KEEP.search(l)]
    moe = [int(m.group(1)) for l in lines if (m := _CPU_MOE.search(l))]
    if moe:
        return lines, moe[-1]
    over = [int(m.group(1)) for l in lines if (m := _OVERFLOW.search(l))]
    if over:
        return lines, sum(over)
    layers = {m.group(1) for l in lines if (m := _OVERRIDE.search(l))}
    if layers:
        return lines, len(layers)
    # A clean fit with every layer on the GPU and no expert tensor overridden to CPU
    # is a measured zero, not an unknown (reap160_q4 read UNKNOWN for this reason).
    full = [m for l in lines if (m := _ALL_LAYERS.search(l))]
    fitted = any("successfully fit" in l for l in lines)
    if fitted and full and full[-1].group(1) == full[-1].group(2):
        return lines, 0
    return lines, None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _publish_initial_serve_exclusive(path: Path, receipt: dict) -> None:
    """Publish only an entirely fsynced, never-replaced initial serve receipt.

    After a crash during JSON writing, an incomplete `.serve.json` could
    otherwise masquerade as authoritative session evidence. Private pending
    siblings may survive hard kills but cannot be mistaken for a receipt.
    Re-entry with such an orphan still refuses, pending a manual owner audit.
    """
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        raw = (json.dumps(receipt, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
        with pending.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            pending.rename(path)
        else:
            os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _verify_registered_v2_construction(gguf: Path, arm: str, digest: str) -> None:
    """Do not serve an unregistered/partial v2 GGUF merely because it hashes.

    This binds the actual model file SHA (already freshly measured by the
    serving wrapper) to its published source-bound 42-chunk construction
    evidence. This is a necessary preflight, not independent full raw-tensor
    or model-quality proof. The historical REAP160 and random128 arms are
    intentionally unchanged and must never acquire retroactive witnesses.
    """
    required = REGISTERED_V2_CONSTRUCTION.get(arm)
    if required is None:
        return
    receipt = gguf.with_name(gguf.name + ".reap-chunked.json")
    if (receipt.is_symlink() or not receipt.is_file()
            or receipt.stat().st_size > 4 * 1024 * 1024):
        raise ValueError(f"{arm}: complete registered source-bound GGUF receipt missing")
    try:
        claimed = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{arm}: registered construction receipt cannot be parsed") from exc
    if not isinstance(claimed, dict):
        raise ValueError(f"{arm}: registered construction receipt must be an object")
    provenance = claimed.get("chunked_reference_receipts")
    groups = provenance.get("source_groups") if isinstance(provenance, dict) else None
    proofs = provenance.get("all_tensors") if isinstance(provenance, dict) else None
    if (claimed.get("schema") != required[0] or claimed.get("status") != required[1]
            or claimed.get("output") != str(gguf)
            or claimed.get("output_sha256") != digest
            or claimed.get("output_size_bytes") != gguf.stat().st_size
            or claimed.get("source_gguf_sha256") != REGISTERED_BF16_SHA256
            or (arm == "reap128_v2" and digest != REGISTERED_Q8_GGUF_SHA256)
            or (arm == "reap128_v2_q4" and digest != REGISTERED_Q4_GGUF_SHA256)
            or claimed.get("frozen_stage_spec_sha256") != REGISTERED_STAGE_SHA256
            or claimed.get("quantizer_binary_sha256") != REGISTERED_QUANTIZER_SHA256
            or claimed.get("chunk_count") != 42
            or claimed.get("quality_gate") != "NOT_EVALUATED"
            or claimed.get("gpu_fit_gate") != "NOT_EVALUATED"
            or not isinstance(groups, list) or len(groups) != 42
            or not isinstance(proofs, dict) or len(proofs) != 733):
        raise ValueError(f"{arm}: incomplete or foreign BF16-derived v2 construction")
    names = [name for group in groups if isinstance(group, list) for name in group]
    if len(names) != 733 or len(set(names)) != 733 or set(names) != set(proofs):
        raise ValueError(f"{arm}: registered 733-tensor GGUF inventory invalid")
    for name, record in proofs.items():
        if (not isinstance(record, dict)
                or not isinstance(record.get("shape"), list)
                or not record["shape"]
                or any(type(n) is not int or n <= 0 for n in record["shape"])
                or record.get("tensor_type") not in {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "F32", "F16", "BF16"}
                or not isinstance(record.get("raw_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["raw_sha256"])):
            raise ValueError(f"{arm}: incomplete registered GGUF tensor proof: {name}")
    for layer in range(40):
        for suffix in ("ffn_down_exps.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight"):
            if f"blk.{layer}.{suffix}" not in proofs:
                raise ValueError(f"{arm}: frozen 40-layer routed expert inventory absent")


def _digest_json(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _startup_log(path: Path, size: int) -> bytes:
    if (not isinstance(size, int) or isinstance(size, bool)
            or not 1 <= size <= 64 * 1024 * 1024):
        raise ValueError("REAP serving startup log length is invalid")
    with path.open("rb") as stream:
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("REAP serving startup log is truncated")
    return raw


def _verify_offload_receipt(receipt: dict, path: Path) -> None:
    """Recompute the original fit assertion from the same serving startup log."""
    if "server_log_startup_sha256" in receipt:
        raw = _startup_log(path, receipt.get("server_log_startup_bytes"))
        if hashlib.sha256(raw).hexdigest() != receipt["server_log_startup_sha256"]:
            raise ValueError("REAP serving startup log digest disagrees with receipt")
    else:
        # Older original receipts lack the prefix digest; they cannot gain one
        # retrospectively. Their offload assertion must still match raw log.
        raw = path.read_bytes()
    lines, count = parse_offload(raw.decode("utf-8", errors="replace"))
    measured = receipt.get("measured_experts_offloaded", receipt.get("experts_offloaded"))
    if (lines != receipt.get("offload_log_lines") or count != measured
            or (receipt.get("experts_offloaded") is not None
                and receipt["experts_offloaded"] != count)):
        raise ValueError("REAP serving offload receipt disagrees with original startup log")


def _certify_q4_speed_session(arm_dir: Path, *, eval_args: list[str], receipt: dict,
                              session_id: str, port: int, server_pid: int) -> dict | None:
    """Certify exactly one new, witnessed first-200 GSM8K session, or UNKNOWN."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port")
    parser.add_argument("--arm")
    parser.add_argument("--out")
    parser.add_argument("--suites", default="gsm8k,humaneval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=4)
    if not eval_args:
        return None
    args = parser.parse_args(eval_args)
    if args.suites != "gsm8k" or args.limit != 200:
        return None
    # The listener PID is independently checked on the Windows execution host.
    # Other platforms have no implemented ownership attestation in this wrapper.
    if os.name != "nt":
        return None
    if receipt.get("measured_experts_offloaded") is None:
        return None
    from chowder_batch import reap_eval

    for filename in ("gsm8k.inputs.json", "gsm8k.jsonl", "summary.json"):
        path = arm_dir / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError("REAP speed session lacks a regular result, summary or input witness")
    full_items = reap_eval._load("gsm8k")
    witness = reap_eval._bind_evaluation_inputs(arm_dir, "gsm8k", full_items)
    if witness["model"] != {key: receipt[key] for key in ("gguf", "gguf_sha256", "flags")}:
        raise ValueError("REAP speed session witness belongs to a different serving model")
    saved, _ = reap_eval._validated_saved_summary(arm_dir)
    summary = saved.get("gsm8k")
    expected_ids = {f"gsm8k-{i}" for i in range(200)}
    rows_path = arm_dir / "gsm8k.jsonl"
    rows = reap_eval._read_resume_rows(rows_path, expected_ids, arm_dir / "summary.json", "gsm8k")
    if (len(rows) != 200 or summary is None or summary.get("n") != 200
            or set(rows) != expected_ids):
        raise ValueError("REAP speed session lacks its complete same-session first-200 GSM8K rows")
    for row in rows.values():
        speed = row.get("predicted_per_second")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not 0 <= speed < float("inf")):
            raise ValueError("REAP speed session contains missing/nonfinite throughput")
    return {
        "schema": "chowder.reap-q4-same-session-speed.v1",
        "session_id": session_id, "server_pid": server_pid, "port": port,
        "suite": "gsm8k", "n": 200,
        "gguf_sha256": receipt["gguf_sha256"],
        "server_log_startup_sha256": receipt["server_log_startup_sha256"],
        "source_items_sha256": witness["source_items_sha256"],
        "inputs_sha256": _sha256(arm_dir / "gsm8k.inputs.json"),
        "rows_sha256": _sha256(rows_path),
        "summary_entry_sha256": _digest_json(summary),
    }


def _model_stat(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino)


def _wait_health(proc, port: int, timeout: int) -> None:
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.poll()}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError(f"llama-server not healthy after {timeout}s")
        time.sleep(2)


def _vram_used_mib() -> int | None:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30)
        return int(done.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _verify_eval_target(eval_args: list[str], *, port: int, arm: str, out: Path) -> None:
    """Bind evaluation to the just-launched server and its output identity.

    The evaluation CLI accepts its own independent ``--port``/``--arm``/``--out``.
    Without this check, serving one GGUF can silently score a different resident
    endpoint or append results under an unrelated arm.  Empty args are left to
    reap_eval's required-argument validation for backward compatibility.
    """
    if not eval_args:
        return
    for name in ("--port", "--arm", "--out"):
        count = sum(value == name or value.startswith(name + "=") for value in eval_args)
        if count != 1:
            raise ValueError(f"eval {name} must be supplied exactly once; found {count}")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suites")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel", type=int)
    try:
        parsed, unknown = parser.parse_known_args(eval_args)
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValueError("eval argument syntax is invalid") from exc
    if unknown:
        raise ValueError(f"eval has unrecognized arguments: {unknown!r}")
    if (parsed.port != port or parsed.arm != arm or
            parsed.out.resolve() != out.resolve()):
        raise ValueError("eval --port/--arm/--out must match the launched REAP server and output")


def _require_free_loopback_port(port: int) -> None:
    """Reject an existing listener before startup (not a post-start PID proof)."""
    if not 1 <= port <= 65535:
        raise ValueError("server port must be in 1..65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"server port {port} is in use or unavailable; refusing foreign health endpoint") from exc


def _require_owned_listener(proc, port: int) -> None:
    """On the Windows execution host, bind a healthy listener to our child PID."""
    if os.name != "nt":
        return
    script = ("$ErrorActionPreference='Stop'; "
              f"Get-NetTCPConnection -State Listen -LocalPort {port} | "
              "Where-Object { $_.LocalAddress -eq '127.0.0.1' -or "
              "$_.LocalAddress -eq '0.0.0.0' } | "
              "Select-Object -ExpandProperty OwningProcess")
    try:
        check = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                                "-Command", script], capture_output=True, text=True, timeout=20)
        owners = {int(line.strip()) for line in check.stdout.splitlines() if line.strip()}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not verify Windows REAP loopback listener owner") from exc
    if check.returncode != 0 or owners != {proc.pid}:
        raise RuntimeError("healthy REAP loopback listener is not owned by launched server PID")


def _admit_output(arm_dir: Path, *, gguf: Path, digest: str) -> bool:
    """Allow a fresh directory or an original, model-bound resumable session."""
    if arm_dir.is_symlink() or not arm_dir.is_dir():
        raise ValueError("REAP arm output must be a real directory")
    existing = [path for path in arm_dir.iterdir() if path.name != ".serve.lock"]
    if not existing:
        return False
    serve_path = arm_dir / "serve.json"
    log_path = arm_dir / "server.log"
    if (not serve_path.is_file() or not log_path.is_file()
            or serve_path.is_symlink() or log_path.is_symlink()):
        raise ValueError("REAP existing output has incomplete or unclaimed serving identity")
    try:
        serving = json.loads(serve_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("REAP existing serving identity cannot be read") from exc
    if not isinstance(serving, dict) or not isinstance(serving.get("gguf"), str):
        raise ValueError("REAP existing serving model identity is malformed")
    expected = {"gguf": serving["gguf"], "gguf_sha256": digest, "flags": FLAGS}
    if (Path(serving["gguf"]).resolve() != gguf
            or any(serving.get(k) != v for k, v in expected.items())):
        raise ValueError("REAP existing serving model identity/flags disagree; use a new output arm")
    _verify_offload_receipt(serving, log_path)

    from chowder_batch.reap_eval import IDENTITY_SCHEMA

    # Never give legacy historical results a retrospective model witness.
    for results in arm_dir.glob("*.jsonl"):
        suite = results.stem
        witness = arm_dir / f"{suite}.inputs.json"
        if results.is_symlink() or not witness.is_file() or witness.is_symlink():
            raise ValueError(f"REAP existing {suite} results lack a trustworthy input witness")
    for witness in arm_dir.glob("*.inputs.json"):
        suite = witness.name.removesuffix(".inputs.json")
        if witness.is_symlink():
            raise ValueError("REAP existing input witness cannot be a symlink")
        try:
            record = json.loads(witness.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError("REAP existing input witness is unreadable") from exc
        if (not isinstance(record, dict) or record.get("schema") != IDENTITY_SCHEMA
                or record.get("suite") != suite or record.get("model") != expected):
            raise ValueError("REAP existing input witness model identity disagrees")
    summary_path = arm_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        if not summary_path.is_file() or summary_path.is_symlink():
            raise ValueError("REAP existing evaluation summary is not a regular file")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError("REAP existing evaluation summary is unreadable") from exc
        if not isinstance(summary, dict):
            raise ValueError("REAP existing evaluation summary is malformed")
        for suite, info in summary.items():
            results = arm_dir / f"{suite}.jsonl"
            if (not isinstance(suite, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", suite)
                    or not isinstance(info, dict) or not results.is_file()
                    or not (arm_dir / f"{suite}.inputs.json").is_file()
                    or info.get("file_sha256") != _sha256(results)):
                raise ValueError("REAP existing evaluation summary is unclaimed or modified")
    session = serving.get("speed_session")
    if session is not None:
        if (not isinstance(session, dict)
                or session.get("schema") != "chowder.reap-q4-same-session-speed.v1"
                or session.get("suite") != "gsm8k" or type(session.get("n")) is not int
                or session["n"] != 200
                or not isinstance(session.get("session_id"), str) or not session["session_id"]
                or type(session.get("server_pid")) is not int or session["server_pid"] <= 0
                or type(session.get("port")) is not int or not 1 <= session["port"] <= 65535
                or session.get("gguf_sha256") != digest
                or session.get("server_log_startup_sha256") != serving.get("server_log_startup_sha256")
                or "gsm8k" not in summary
                or _digest_json(summary["gsm8k"]) != session.get("summary_entry_sha256")
                or any(_sha256(arm_dir / name) != session.get(field) for name, field in (
                    ("gsm8k.jsonl", "rows_sha256"),
                    ("gsm8k.inputs.json", "inputs_sha256"),
                ))):
            raise ValueError("REAP existing same-session speed receipt evidence changed")
        try:
            source_witness = json.loads((arm_dir / "gsm8k.inputs.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError("REAP existing speed input witness is unreadable") from exc
        if (not isinstance(source_witness, dict) or source_witness.get("source_items_sha256")
                != session.get("source_items_sha256")):
            raise ValueError("REAP existing speed source witness differs from same-session proof")
    return True


def _stop(proc) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_eval(server, command: list[str], *, env: dict | None = None) -> int:
    """Stop scoring if the model server exits before the evaluator finishes."""
    evaluator = subprocess.Popen(command, **({"env": env} if env is not None else {}))
    try:
        while True:
            try:
                code = evaluator.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if server.poll() is not None:
                    raise RuntimeError("REAP model server exited during evaluation") from None
                continue
            if server.poll() is not None:
                raise RuntimeError("REAP model server exited before evaluation completed")
            return code
    finally:
        _stop(evaluator)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ours, eval_args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--out", required=True)
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--timeout", type=int, default=HEALTH_TIMEOUT)
    args = p.parse_args(ours)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.arm):
        raise ValueError("REAP arm must be a simple name without path separators")
    gguf = Path(args.gguf).resolve(strict=True)
    if not gguf.is_file() or Path(args.gguf).is_symlink():
        raise ValueError("REAP GGUF must be an existing regular model file")
    _verify_eval_target(eval_args, port=args.port, arm=args.arm, out=Path(args.out))

    arm_dir = Path(args.out) / args.arm
    if arm_dir.is_symlink():
        raise ValueError("REAP arm output cannot be a symlink")
    arm_dir.mkdir(parents=True, exist_ok=True)
    lock_path = arm_dir / ".serve.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("REAP arm already has an active or abandoned session lock") from exc
    try:
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        model_stat = _model_stat(gguf)
        digest = _sha256(gguf)
        if _model_stat(gguf) != model_stat:
            raise RuntimeError("REAP GGUF changed while hashing; refusing serving")
        expected_reference = REGISTERED_QUALITY_REFERENCE_SHA256.get(args.arm)
        if expected_reference is not None and digest != expected_reference:
            raise ValueError("new scientific reference/control must use its original pinned GGUF SHA")
        resumed = _admit_output(arm_dir, gguf=gguf, digest=digest)
        # The official new-v2 scientific arms must use a published original
        # BF16-derived artifact, not a foreign GGUF that merely matches its
        # own claimed serve.json and evaluator input witness.
        _verify_registered_v2_construction(gguf, args.arm, digest)
        _require_free_loopback_port(args.port)
        # Every new-v2 evaluation needs an original, unique live-server
        # identity, including the first session, not only resume sessions.
        session = uuid.uuid4().hex if resumed or args.arm in REGISTERED_V2_EVALUATION_ARMS else None
        log_path = arm_dir / (f"server.resume-{session}.log" if resumed else "server.log")
        receipt_path = arm_dir / (f"serve.resume-{session}.json" if resumed else "serve.json")
        q4 = args.arm.endswith("_q4")
        # A resumed 200-row cohort is always mixed/uncertain for speed.
        may_certify_speed = q4 and not resumed and bool(eval_args)
        if may_certify_speed and any((arm_dir / file).exists() for file in (
            "gsm8k.jsonl", "gsm8k.inputs.json", "summary.json"
        )):
            may_certify_speed = False
        cmd = [args.server, *FLAGS, "--port", str(args.port), "-m", str(gguf)]
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.time()
        with log_path.open("xb") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "LLAMA_ARG_LOG_VERBOSITY": LOG_VERBOSITY})
            try:
                _wait_health(proc, args.port, args.timeout)
                if proc.poll() is not None:
                    raise RuntimeError("launched llama-server exited before evaluation")
                _require_owned_listener(proc, args.port)
                if _model_stat(gguf) != model_stat:
                    raise RuntimeError("REAP GGUF changed during server startup")
                ready = round(time.time() - t0, 1)
                startup_raw = log_path.read_bytes()
                lines, n = parse_offload(startup_raw.decode("utf-8", errors="replace"))
                receipt = {
                    "gguf": str(gguf), "gguf_sha256": digest, "flags": FLAGS,
                    "log_verbosity": LOG_VERBOSITY,
                    # Keep the speed gate UNKNOWN until this new Q4 session has
                    # produced and sealed its complete, same-session 200 rows.
                    "experts_offloaded": None if q4 else n,
                    "measured_experts_offloaded": n,
                    "offload_log_lines": lines,
                    "server_log_startup_bytes": len(startup_raw),
                    "server_log_startup_sha256": hashlib.sha256(startup_raw).hexdigest(),
                    "vram_used_mib": _vram_used_mib(), "started": started, "ready_seconds": ready,
                }
                if args.arm in REGISTERED_V2_EVALUATION_ARMS:
                    receipt.update({"eval_session_id": session, "server_pid": proc.pid,
                                    "server_port": args.port})
                if resumed:
                    receipt["session_log"] = log_path.name
                _publish_initial_serve_exclusive(receipt_path, receipt)
                eval_py = str(Path(__file__).with_name("reap_eval.py"))
                evaluator_env = None
                if args.arm in REGISTERED_V2_EVALUATION_ARMS:
                    evaluator_env = {
                        **os.environ,
                        "REAP_BOUND_EVALUATION_SESSION": json.dumps({
                            "session_id": session, "server_pid": proc.pid,
                            "port": args.port, "model_sha256": digest,
                            "receipt": receipt_path.name,
                        }, sort_keys=True),
                    }
                code = _run_eval(proc, [sys.executable, eval_py, *eval_args],
                                 **({"env": evaluator_env} if evaluator_env else {}))
                if _model_stat(gguf) != model_stat:
                    raise RuntimeError("REAP GGUF changed during evaluation")
                _verify_offload_receipt(receipt, log_path)
                if may_certify_speed and code == 0:
                    _require_owned_listener(proc, args.port)
                    proof = _certify_q4_speed_session(
                        arm_dir, eval_args=eval_args, receipt=receipt,
                        session_id=session or uuid.uuid4().hex,
                        port=args.port, server_pid=proc.pid,
                    )
                    if proof is not None:
                        provisional_receipt = receipt.copy()
                        receipt["speed_session"] = proof
                        receipt["experts_offloaded"] = n
                        # The provisional receipt is still ours under the arm
                        # lock. Seal only after source, rows, summary and log
                        # were verified; no historical receipt is replaced.
                        # A concurrent edit to the provisional receipt is a
                        # provenance violation, not permission to overwrite it.
                        if json.loads(receipt_path.read_text(encoding="utf-8")) != provisional_receipt:
                            raise ValueError("REAP provisional serving receipt changed before sealing")
                        pending = arm_dir / f".serve-pending-{uuid.uuid4().hex}.json"
                        try:
                            with pending.open("x", encoding="utf-8") as record:
                                json.dump(receipt, record, indent=2)
                                record.flush()
                                os.fsync(record.fileno())
                            if json.loads(receipt_path.read_text(encoding="utf-8")) != provisional_receipt:
                                raise ValueError("REAP provisional serving receipt changed before atomic replacement")
                            os.replace(pending, receipt_path)
                        finally:
                            if pending.exists():
                                pending.unlink()
                return code
            finally:
                _stop(proc)
    finally:
        lock.close()
        lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
