"""OT3 short-trace subset builder: conclusion-boundary chunking.

OpenThoughts3 rows average ~47KB (full QwQ-32B traces), so whole-row
acceptance at SFT context sizes is ~0.1-2%. This module converts each long
teacher trace into several self-contained conversations:

  chunk 0: original question -> reasoning segment ending at the FIRST
           intermediate conclusion ("CONCLUSION: ..." line)
  chunk k: "continue" prompt identifying task + part -> next segment

Design rules (from the pilot report):
  - supervised target of every chunk is the segment text (never just
    "Okay." / "Continue." — a continuation prompt must not teach non-answers)
  - the final chunk keeps the row's real final answer whenever one exists
  - segments are stripped of partial LaTeX / dangling code fences
  - a chunk level near-dup key includes the task digest, so different tasks
    with identical "continue" prompts stay distinct
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from prepare import near_duplicate_key, normalized

CONCLUSION_RE = re.compile(r"^\s*\**CONCLUSION\b[\s:]*\**\s*(.+)$", re.IGNORECASE | re.MULTILINE)
ANSWER_MARKERS = ("the final answer is", "answer is", "answer:", "boxed{")
MIN_SEGMENT_CHARS = 400
MAX_SEGMENT_CHARS = 5500
MAX_CHUNKS_PER_ROW = 12


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def task_key(question: str, assistant: str) -> str:
    return digest({"q": normalized(question)[:4000], "a": normalized(assistant)[:4000]})


def split_segments(assistant_text: str) -> list[str]:
    """Split a long trace at CONCLUSION boundaries; fall back to paragraphs."""
    matches = list(CONCLUSION_RE.finditer(assistant_text))
    if len(matches) >= 2:
        bounds = [0]
        for m in matches[:-1]:
            bounds.append(m.end())
        bounds.append(len(assistant_text))
    else:
        bounds = [0, len(assistant_text)]
    segments = []
    for start, end in zip(bounds, bounds[1:]):
        seg = assistant_text[start:end]
        if len(seg) > MAX_SEGMENT_CHARS:
            # secondary split: double newlines only (deterministic positions)
            parts = [p for p in seg.split("\n\n") if p.strip()]
            cur = ""
            for part in parts:
                if cur and len(cur) + len(part) > MAX_SEGMENT_CHARS:
                    segments.append(cur)
                    cur = part
                else:
                    cur = f"{cur}\n\n{part}" if cur else part
            if cur.strip():
                segments.append(cur)
        else:
            segments.append(seg)
    return [s for s in (seg.strip() for seg in segments) if len(s) >= MIN_SEGMENT_CHARS]


def has_final_answer(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in ANSWER_MARKERS)


def clean_segment(text: str) -> str:
    """Trim dangling code fences so chunks stay self-contained."""
    if text.count("```") % 2 == 1:
        text = text.rsplit("```", 1)[0] + "\n```"
    return text.strip()


def chunk_row(row: dict) -> list[dict]:
    conv = row.get("conversations") or []
    question = next((m.get("value", "") for m in conv if m.get("from") == "human"), "")
    assistant = next((m.get("value", "") for m in conv if m.get("from") == "gpt"), "")
    if not question or not assistant:
        return []
    key = task_key(question, assistant)
    segments = split_segments(assistant)
    chunks = []
    for i, seg in enumerate(segments[:MAX_CHUNKS_PER_ROW]):
        is_final = i == len(segments[:MAX_CHUNKS_PER_ROW]) - 1 and i == len(segments) - 1
        if is_final and not has_final_answer(seg):
            continue  # drop trailing mid-reasoning tail without an answer
        seg = clean_segment(seg)
        if i == 0:
            prompt = question
        else:
            prompt = (f"Task {key}. Continue the reasoning from part {i} of {len(segments)}. "
                      f"Original question (excerpt): {question[:300]}")
        chunks.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": seg},
            ],
            "group": f"ot3chunk:{key}:{i}",
        })
    return chunks


def build(input_path: Path, out_path: Path, *, target: int = 5200,
          max_rows_scanned: int | None = None) -> dict:
    stats: Counter = Counter()
    near_bands: list[dict[str, str]] = [dict() for _ in range(8)]
    rows = 0
    with out_path.open("w", encoding="utf8") as out:
        for line in input_path.open(encoding="utf8"):
            if rows >= (max_rows_scanned or float("inf")) or stats["chunks_written"] >= target:
                break
            rows += 1
            try:
                chunks = chunk_row(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                stats["malformed_rows"] += 1
                continue
            for chunk in chunks:
                # Dedup on the supervised TARGET (segment text), not the prompt:
                # same-task chunks share near-identical prompts but teach
                # different segments; cross-task identical segments are real dups.
                target_text = chunk["messages"][1]["content"]
                nk = near_duplicate_key(target_text)
                prior = None
                if nk:
                    for band, value in enumerate(nk):
                        prior = near_bands[band].get(value)
                        if prior is not None:
                            break
                if prior is not None:
                    stats["near_duplicate_dropped"] += 1
                    continue
                if nk:
                    for band, value in enumerate(nk):
                        near_bands[band].setdefault(value, chunk["group"])
                out.write(json.dumps({"conversations": [
                    {"from": "human", "value": chunk["messages"][0]["content"]},
                    {"from": "gpt", "value": chunk["messages"][1]["content"]},
                ]}, ensure_ascii=False) + "\n")
                stats["chunks_written"] += 1
    stats["rows_scanned"] = rows
    return dict(stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=5200)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.out, target=args.target,
                           max_rows_scanned=args.max_rows), indent=2))
