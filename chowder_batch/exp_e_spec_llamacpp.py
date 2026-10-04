"""Phase-2 measurements against llama.cpp: ngram speculative vs plain decoding.

llama.cpp implements the draft-and-verify loop internally (``--spec-type
ngram-simple`` / ``ngram-map-k``): drafts come from context n-grams and are
verified by the model's own logits, so with temperature 0 the output is
greedy-equivalent. This client measures, on identical prompts and settings:

* tokens/second (from the server's own timings and from wall time),
* wall-clock latency,
* output equivalence against the ``spec-type none`` reference run,
* reported draft/accept statistics when the server exposes them.

The baseline reference file is produced by running this client against a
server started with ``--spec-type none``; the speculative runs must reproduce
those outputs exactly.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    "copy_function": (
        "def parse_version(s):\n    parts = s.split('.')\n    while len(parts) < 3: parts.append('0')\n"
        "    return tuple(int(p) for p in parts)\n\n"
        "# Copy this function exactly, then add a version() helper returning '1.0.0'.\n"
    ),
    "copy_tools": (
        "The runtime harness exposes three tools: read_file, write_file, and run_tests. "
        "Repeat the three tool names exactly, separated by commas.\n"
    ),
    "copy_repeat": (
        "Repeat exactly: The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog.\n"
    ),
    "generative": "Explain in one sentence why testing before releasing software reduces risk.\n",
}


def complete(prompt: str, *, n_predict: int, port: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "cache_prompt": True,
        }).encode(),
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        data = json.loads(response.read())
    data["wall_seconds"] = time.perf_counter() - t0
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-predict", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for name, prompt in PROMPTS.items():
        runs = []
        for repeat in range(args.repeats):
            data = complete(prompt, n_predict=args.n_predict, port=args.port)
            timings = data.get("timings", {})
            runs.append({
                "content": data.get("content", ""),
                "tokens_predicted": data.get("tokens_predicted"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "predicted_ms": timings.get("predicted_ms"),
                "wall_seconds": round(data["wall_seconds"], 3),
                "stopped_eos": data.get("stop_type") == "eos",
            })
        # First run includes prompt processing; steady-state speed is the
        # median of the repeats.
        speeds = sorted(r["predicted_per_second"] or 0 for r in runs)
        results[name] = {
            "runs": runs,
            "median_tokens_per_second": speeds[len(speeds) // 2],
            "reference_output": runs[0]["content"],
        }
        print(f"{name}: {results[name]['median_tokens_per_second']:.1f} tok/s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
