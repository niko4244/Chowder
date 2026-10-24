"""Read-only, fail-closed REAP160 inference endpoint identity admission.

This never sends a generation request or loads/unloads a model. A server's
model label is not sufficient to attribute benchmark output to a checkpoint:
we require a *currently loaded* model with the exact, existing GGUF path in
its reported llama.cpp launch arguments, and a passing local GGUF topology
check. This is an operational precondition, not a content or benchmark proof.
"""

from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
from typing import Any, Mapping

from chowder_batch.reap_gguf_header_preflight import run_preflight


class RuntimeIdentityRefusal(ValueError):
    """The listening endpoint cannot be shown to serve the requested GGUF."""


def _launched_model_path(args: Any) -> str | None:
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        return None
    declared: list[str] = []
    for i, arg in enumerate(args):
        if arg in ("--model", "-m"):
            if i + 1 >= len(args):
                return None
            declared.append(args[i + 1])
        elif arg.startswith("--model="):
            declared.append(arg.partition("=")[2])
    if len(declared) != 1:
        return None
    return declared[0]


def admit_loaded_gguf(models: Any, expected: Path) -> dict[str, Any]:
    """Refuse aliases, unloaded presets, or paths unlike the verified file."""
    if not isinstance(models, Mapping) or not isinstance(models.get("data"), list):
        raise RuntimeIdentityRefusal("/v1/models has no inspectable data list")
    if not expected.is_file():
        raise RuntimeIdentityRefusal("expected GGUF file no longer exists")
    exact = expected.resolve(strict=True)
    found: list[dict[str, Any]] = []
    for item in models["data"]:
        if not isinstance(item, Mapping):
            raise RuntimeIdentityRefusal("/v1/models contains a malformed model record")
        status = item.get("status")
        if not isinstance(status, Mapping) or status.get("value") != "loaded":
            continue  # never request an unloaded preset and trigger auto-load
        candidate = _launched_model_path(status.get("args"))
        if not candidate:
            continue  # an ID, alias, or incomplete args is not file identity
        path = Path(candidate)
        if not path.is_absolute() or not path.is_file():
            continue
        try:
            actual = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if actual == exact:
            name = item.get("id")
            if not isinstance(name, str) or not name.strip():
                raise RuntimeIdentityRefusal("matching model has no stable ID")
            found.append({"model_id": name, "reported_gguf_path": candidate})
    if len(found) != 1:
        raise RuntimeIdentityRefusal(
            f"expected exactly one loaded llama.cpp model with the existing GGUF path "
            f"{str(exact)!r}; found {len(found)}; no benchmark is authorized"
        )
    return {
        "status": "LOADED_PATH_MATCH_ONLY_NOT_WEIGHT_ATTESTED",
        **found[0],
        "loaded_gguf_file_bytes": exact.stat().st_size,
        "model_entries_seen": len(models["data"]),
        "no_generation_sent": True,
    }


def read_existing_server_models(port: int) -> dict[str, Any]:
    """GET only over IPv4 loopback; no redirects, credentials or remote URLs."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise RuntimeIdentityRefusal("port must be an integer between 1 and 65535")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        conn.request("GET", "/v1/models", headers={"Accept": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeIdentityRefusal(f"existing server GET /v1/models returned {response.status}")
        raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise RuntimeIdentityRefusal("server model metadata is unreasonably large")
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        raise RuntimeIdentityRefusal(f"no reliable existing local server metadata: {exc}") from exc
    finally:
        conn.close()
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeIdentityRefusal("server sent invalid JSON model metadata") from exc
    if not isinstance(result, dict):
        raise RuntimeIdentityRefusal("server model metadata root must be a JSON object")
    return result


def check_runtime(*, port: int, gguf: Path, bf16_model: Path) -> dict[str, Any]:
    """Combine model-shape evidence and *current* endpoint path admission."""
    metadata = run_preflight(gguf, bf16_model)
    model = admit_loaded_gguf(read_existing_server_models(port), gguf)
    return {
        "schema": "chowder.reap-runtime-identity-preflight.v1",
        "status": "READY_FOR_SEPARATELY_AUTHORIZED_BOUNDED_INFERENCE_TEST_ONLY",
        "local_port": port,
        "server": model,
        "gguf_header": metadata,
        "limits": "No GGUF weights rehashed; path and server-declared loaded state "
                  "are not independently attested model contents. No inference, "
                  "decode, VRAM, tool-call or frozen fit/speed measurement performed.",
        "new_gpu_hours": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="new receipt path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = check_runtime(port=args.port, gguf=args.gguf, bf16_model=args.bf16_model)
    except (OSError, ValueError, UnicodeError) as exc:
        report = {"schema": "chowder.reap-runtime-identity-preflight.v1",
                  "status": "REFUSED", "local_port": args.port,
                  "reason": str(exc), "no_generation_sent": True, "new_gpu_hours": 0}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded)
    return 0 if report["status"].startswith("READY_FOR") else 2


if __name__ == "__main__":
    raise SystemExit(main())
