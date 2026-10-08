#!/usr/bin/env python3
"""arXiv literature watch: query the feed against Chowder's surface areas.

This is a *fetch and render* tool, nothing more. It emits an **unvetted
candidate pool** -- titles, dates and abstracts from arXiv, grouped by the
Chowder surface a query targets. It does not judge, rank, or promote anything:
an entry in its output is a thing to read, not a finding, and nothing here
registers an intervention family or changes a maturity label.

The curated output lives in ``WATCH_LOG.md``, written by a person/agent who
read the papers and mapped them to concrete Chowder surfaces under the honesty
rules in ``README.md``. Keep that split: the tool discovers, the log decides.

Stdlib only, so it runs anywhere Python 3.10+ runs (including the light CI
images) and needs no network in CI because nothing imports it during a build.
arXiv asks for roughly one request every three seconds; the profiles are spaced
accordingly.

Usage:
    python docs/literature/watch.py                     # all profiles, 14-day window
    python docs/literature/watch.py --days 3 --max 6
    python docs/literature/watch.py --profile compression -o pool.md
    python docs/literature/watch.py --profile all --json pool.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

#: One query per Chowder surface. ``query`` is an arXiv API ``search_query``
#: string; ``why`` records what a hit here is meant to inform, so the pool is
#: read against a purpose instead of a vibe. Keep these aligned with the
#: intervention families in ``src/chowder/growth/interventions.py`` and the
#: training backends in ``docs/TRAINING_BACKENDS_UNIFIED_2026-10-05.md``.
PROFILES: tuple[dict[str, str], ...] = (
    {
        "name": "training-methods",
        "query": (
            'cat:cs.LG AND (all:"training recipe" OR all:"continued pretraining" '
            'OR all:"optimizer" OR all:"curriculum learning" OR '
            'all:"data curation" OR all:"RLVR")'
        ),
        "why": (
            "new training recipes that could enter as training.* families or "
            "change how a generation is trained"
        ),
    },
    {
        "name": "compression",
        "query": (
            'cat:cs.LG AND (all:"quantization" OR all:"pruning" OR '
            'all:"low-rank" OR all:"KV cache" OR all:"compression")'
        ),
        "why": (
            "compression.* families: ptq, low-rank-vocab, and any new "
            "checkpoint-shrinking mechanism with a measured rate-distortion"
        ),
    },
    {
        "name": "peft-adapters",
        "query": (
            'cat:cs.CL AND (all:"LoRA" OR all:"adapter" OR all:"PEFT" OR '
            'all:"parameter-efficient fine-tuning")'
        ),
        "why": (
            "training.adapter-continuation and the LoRA/QLoRA trainer the Local "
            "and Unsloth backends drive"
        ),
    },
    {
        "name": "distillation",
        "query": (
            'cat:cs.LG AND (all:"knowledge distillation" OR all:"distillation" '
            'OR all:"self-play" OR all:"teacher model")'
        ),
        "why": (
            "training.teacher-distillation and the generation-to-generation "
            "survivor transition"
        ),
    },
    {
        "name": "efficient-inference",
        "query": (
            'cat:cs.LG AND (all:"speculative decoding" OR all:"latency" OR '
            'all:"decoding throughput" OR all:"inference efficiency")'
        ),
        "why": (
            "inference.speculative and inference.retrieval families, and any "
            "runtime cost the acceptance gate should price"
        ),
    },
    {
        "name": "evaluation-integrity",
        "query": (
            'cat:cs.CL AND (all:"LLM judge" OR all:"judge reliability" OR '
            'all:"benchmark contamination" OR all:"evaluation validity")'
        ),
        "why": (
            "the judge and settlement path: ways a measured result can be an "
            "artifact rather than a capability, which the promotion gate must "
            "refuse"
        ),
    },
)


@dataclass
class Entry:
    arxiv_id: str
    title: str
    published: str
    updated: str
    primary_category: str
    categories: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    comment: str = ""
    summary: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _parse_entry(node: ET.Element) -> Entry:
    raw_id = _clean(node.findtext(f"{ATOM}id", default=""))
    arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
    base = arxiv_id.split("v")[0] if arxiv_id else ""
    primary = node.find(f"{ARXIV}primary_category")
    return Entry(
        arxiv_id=arxiv_id,
        title=_clean(node.findtext(f"{ATOM}title", default="")),
        published=_clean(node.findtext(f"{ATOM}published", default="")),
        updated=_clean(node.findtext(f"{ATOM}updated", default="")),
        primary_category=_clean(primary.get("term", "")) if primary is not None else "",
        categories=[
            _clean(c.get("term", "")) for c in node.findall(f"{ATOM}category")
        ],
        authors=[_clean(a.findtext(f"{ATOM}name", default="")) for a in node.findall(f"{ATOM}author")],
        comment=_clean(node.findtext(f"{ARXIV}comment", default="")),
        summary=_clean(node.findtext(f"{ATOM}summary", default="")),
        url=f"https://arxiv.org/abs/{base}" if base else raw_id,
    )


def fetch(query: str, *, max_results: int, timeout: float = 30.0) -> list[Entry]:
    """One arXiv API call. Raises on a network or parse failure."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(max_results),
        }
    )
    url = f"{API}?{params}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "chowder-literature-watch/1 (research intake)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    root = ET.fromstring(payload)
    return [_parse_entry(node) for node in root.findall(f"{ATOM}entry")]


def _within_window(entry: Entry, days: int, now: datetime) -> bool:
    if days <= 0:
        return True
    stamp = entry.published or entry.updated
    if not stamp:
        return True
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (now - when).total_seconds() <= days * 86400


def build_pool(
    profiles: list[dict[str, str]],
    *,
    max_results: int,
    days: int,
    spacing: float = 3.0,
) -> tuple[list[dict], list[str]]:
    """Fetch every profile; return ``(groups, errors)``. Never raises per-profile."""
    now = datetime.now(timezone.utc)
    groups: list[dict] = []
    errors: list[str] = []
    for index, profile in enumerate(profiles):
        if index:
            time.sleep(spacing)  # be polite to the arXiv API
        try:
            entries = [
                e
                for e in fetch(profile["query"], max_results=max_results)
                if _within_window(e, days, now)
            ]
        except (urllib.error.URLError, ET.ParseError, TimeoutError, OSError) as exc:
            errors.append(f"{profile['name']}: {type(exc).__name__}: {exc}")
            continue
        groups.append(
            {
                "profile": profile["name"],
                "query": profile["query"],
                "why": profile["why"],
                "entries": [e.to_dict() for e in entries],
            }
        )
    return groups, errors


def render_markdown(groups: list[dict], *, days: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# arXiv candidate pool -- {stamp}",
        "",
        "> **Unvetted.** These are search hits, not findings. Nothing here has been",
        "> read, and no claim of improvement transfers without a first-party",
        "> measurement. Curate into `WATCH_LOG.md` under the rules in `README.md`.",
        "",
        f"- window: last {days} day(s)" if days > 0 else "- window: unlimited",
        "",
    ]
    for group in groups:
        lines.append(f"## {group['profile']} ({len(group['entries'])})")
        lines.append("")
        lines.append(f"_Why:_ {group['why']}")
        lines.append("")
        if not group["entries"]:
            lines.append("_no hits in window_")
            lines.append("")
            continue
        for entry in group["entries"]:
            lines.append(f"- **{entry['title']}** -- `arXiv:{entry['arxiv_id']}`")
            lines.append(
                f"  - {entry['published'][:10]} | {entry['primary_category']} | "
                f"{entry['url']}"
            )
            summary = entry["summary"]
            if summary:
                lines.append(f"  - {summary[:320]}{'...' if len(summary) > 320 else ''}")
        lines.append("")
    return "\n".join(lines)


def _select_profiles(name: str) -> list[dict[str, str]]:
    if name == "all":
        return list(PROFILES)
    chosen = [p for p in PROFILES if p["name"] == name]
    if not chosen:
        valid = ", ".join(["all", *(p["name"] for p in PROFILES)])
        raise SystemExit(f"unknown profile {name!r}; choose one of: {valid}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chowder arXiv literature watch")
    parser.add_argument(
        "--profile",
        default="all",
        help="profile name or 'all' (default: all)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="only entries submitted within N days; 0 disables the window",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="max results fetched per profile",
    )
    parser.add_argument("-o", "--out", help="write markdown here instead of stdout")
    parser.add_argument("--json", dest="json_out", help="also write the raw pool as JSON")
    parser.add_argument(
        "--list-profiles", action="store_true", help="print the profiles and exit"
    )
    args = parser.parse_args(argv)

    if args.list_profiles:
        for profile in PROFILES:
            print(f"{profile['name']}: {profile['query']}")
        return 0

    groups, errors = build_pool(
        _select_profiles(args.profile), max_results=args.max, days=args.days
    )
    markdown = render_markdown(groups, days=args.days)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        total = sum(len(g["entries"]) for g in groups)
        print(f"wrote {args.out} ({total} entries across {len(groups)} profiles)")
    else:
        print(markdown)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"groups": groups, "errors": errors}, handle, indent=2)
        print(f"wrote {args.json_out}", file=sys.stderr)

    for error in errors:
        print(f"warning: {error}", file=sys.stderr)

    # A partial pool is still useful; only fail when nothing at all came back.
    return 0 if groups else 1


if __name__ == "__main__":
    raise SystemExit(main())
