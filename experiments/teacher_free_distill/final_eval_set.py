"""Build the independent final-eval problem set for the paired comparison.

Selection rules (recorded, deterministic):

  - Source: OpenThoughts3 rows the Phase-1 pipeline never scanned for
    training (`ot3_sft.jsonl` rows after the chunker's scan window), so the
    problems cannot overlap training material by construction. The plan-time
    separation check in eval_protocol.py still verifies prompt-text
    disjointness against every development chunk-0 prompt before an eval plan
    is accepted.
  - Gold answers: the row's own teacher trace must end in a single, clean
    numeric final answer, extracted with the production GSM8K extraction
    convention (``#### <answer>`` marker preferred, else the last number of
    the last line). Rows whose trace ends in code, prose, or ambiguity are
    rejected; nothing is hand-authored.
  - Determinism: a seeded reservoir over the candidate rows makes the sample
    reproducible from the recorded seed; every emitted row carries its source
    row index and trace digest.
  - Immutability: writing over an existing set with different bytes is
    refused unless --allow-rewrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

FORMAT = "chowder-teacher-free-final-eval/v1"
ANSWER_RE = re.compile(r"####\s*(-?\$?[\d,]+(?:\.\d+)?%?)")
NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?%?")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_gold(trace: str) -> str | None:
    """Gold answer from a teacher trace, production GSM8K convention.

    Returns the normalized numeric string, or None when the trace does not
    end in one unambiguous number (code blocks, prose endings, no digits).
    """
    text = (trace or "").strip()
    if not text:
        return None

    def clean(raw: str) -> str | None:
        value = raw.replace(",", "").replace("$", "").replace("%", "")
        # A pattern like a lone comma matches the number regex but is not a
        # number; an unparseable gold would make the row unscoreable.
        if not value or not any(c.isdigit() for c in value):
            return None
        try:
            float(value)
        except ValueError:
            return None
        return value

    match = ANSWER_RE.search(text)
    if match:
        return clean(match.group(1))
    last_line = text.splitlines()[-1].strip() if text.splitlines() else ""
    # A fenced code block or a sentence ending in punctuation is not a
    # confident numeric answer.
    if last_line.endswith(("```", ";", ":", "{", "(")):
        return None
    numbers = NUMBER_RE.findall(last_line)
    return clean(numbers[-1]) if numbers else None


def dev_prompt_texts(pilot_dir: Path) -> set[str]:
    """Normalized chunk-0 prompts of every development example."""
    texts: set[str] = set()
    for name in ("dev.jsonl", "train.jsonl"):
        path = pilot_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                user = next((m["content"] for m in row["messages"]
                             if m["role"] == "user"), "")
                texts.add(" ".join(str(user).casefold().split()))
    return texts


def candidates(source: Path, start_row: int, limit: int) -> list[dict]:
    """Rows from `start_row` on whose traces yield a clean numeric gold."""
    found: list[dict] = []
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index < start_row:
                continue
            if not line.strip():
                continue
            row = json.loads(line)
            conversations = row.get("conversations") or []
            question = next((m["value"] for m in conversations
                             if m.get("from") == "human"), None)
            trace = next((m["value"] for m in conversations
                          if m.get("from") == "gpt"), None)
            if not isinstance(question, str) or not isinstance(trace, str):
                continue
            if len(question) > 4000:
                continue
            gold = extract_gold(trace)
            if gold is None:
                continue
            found.append({
                "source_row_index": index,
                "prompt": question.strip(),
                "expected": gold,
                "trace_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
                "domain": row.get("domain"),
                "difficulty": row.get("difficulty"),
            })
            if len(found) >= limit:
                break
    return found


def build(source: Path, pilot_dir: Path, out_dir: Path, *, size: int = 120,
          scan_limit: int = 3000, seed: int = 2026,
          allow_rewrite: bool = False) -> dict:
    out_path = out_dir / "final_eval_prompts.jsonl"
    pool = candidates(source, start_row=600, limit=scan_limit)
    if len(pool) < size:
        raise RuntimeError(
            f"only {len(pool)} candidate rows with clean numeric gold answers "
            f"in the scanned window; need {size}")
    rng = random.Random(seed)
    # Over-sample: OT3 repeats questions across rows, so some draws will
    # overlap development prompts even though this window was never scanned
    # for training. The overlap check is the leak guard; the surplus keeps
    # the emitted count at `size` without weakening it.
    sample = rng.sample(pool, min(len(pool), size * 2 + 20))
    dev_texts = dev_prompt_texts(pilot_dir)
    emitted: list[dict] = []
    rejected_overlap = 0
    for row in sample:
        if len(emitted) >= size:
            break
        if " ".join(row["prompt"].casefold().split()) in dev_texts:
            # A question identical to a development prompt is refused and
            # recorded, never emitted.
            rejected_overlap += 1
            continue
        emitted.append(row)
    if len(emitted) < size:
        raise RuntimeError(
            f"{rejected_overlap} sampled rows overlapped development "
            f"prompts; only {len(emitted)} remain, need {size}. "
            "Extend the scan window and re-run.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.is_file() and not allow_rewrite:
        prior = out_path.read_bytes()
        new_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in emitted).encode("utf-8")
        if prior and prior != new_bytes:
            raise SystemExit(
                f"refusing to rewrite the pinned eval set {out_path}; "
                "pass --allow-rewrite deliberately")
    with out_path.open("wb") as handle:
        for row in emitted:
            handle.write((json.dumps(row, ensure_ascii=False) + "\n")
                         .encode("utf-8"))
    summary = {
        "format": FORMAT,
        "source": str(source),
        "source_row_window": [600, 600 + scan_limit],
        "sample_seed": seed,
        "candidates_in_window": len(pool),
        "sampled": size,
        "rejected_dev_overlap": rejected_overlap,
        "emitted": len(emitted),
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "expected_pattern": "numeric gold extracted from the row's own "
                            "teacher trace (production GSM8K convention)",
        "note": "drawn from source rows the training pipeline never scanned; "
                "eval_protocol's separation check re-verifies disjointness "
                "at plan time",
    }
    (out_dir / "final_eval_set.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path,
                        default=Path("C:/Users/nikma/chowder_teacher_free/ot3_sft.jsonl"))
    parser.add_argument("--pilot-dir", type=Path,
                        default=Path("C:/Users/nikma/chowder_teacher_free/pilot_v4"))
    parser.add_argument("--out", type=Path,
                        default=Path("C:/Users/nikma/chowder_teacher_free/final_eval"))
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--scan-limit", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-rewrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.pilot_dir, args.out,
                           size=args.size, scan_limit=args.scan_limit,
                           seed=args.seed,
                           allow_rewrite=args.allow_rewrite), indent=2))


if __name__ == "__main__":
    main()
