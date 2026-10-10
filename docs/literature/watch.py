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

Two things this tool can do with a pool:

* render it (the default) -- read it yourself;
* ``--append-log`` -- screen it through each profile's two coarse gates (the
  paper's primary category, then a case-insensitive substring test for a surface
  term in its title or abstract) and append an *unvetted* drop to the watch log
  when something new matches. No ranking, no reading. A drop written this way carries
  no claim, quotes no result, and changes no registry -- it is a triage list,
  and a person still curates anything real into the log under the rules in
  ``README.md``. Nothing new means nothing written. With ``--prune-drops``,
  the same write also removes automated sections older than the horizon, so a
  weekly drop can never grow the log without bound. The recurring invocation
  lives in ``.github/workflows/literature-watch.yml``.

Usage:
    python docs/literature/watch.py                     # all profiles, 14-day window
    python docs/literature/watch.py --days 3 --max 6
    python docs/literature/watch.py --profile compression -o pool.md
    python docs/literature/watch.py --profile all --json pool.json
    python docs/literature/watch.py --append-log docs/literature/WATCH_LOG.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

#: One query per Chowder surface. ``query`` is an arXiv API ``search_query``
#: string; ``why`` records what a hit here is meant to inform, so the pool is
#: read against a purpose instead of a vibe. ``categories`` and ``keywords``
#: are the two coarse gates ``--append-log`` applies: the paper has to be
#: *primarily* in a category the surface watches -- arXiv's ``cat:`` matches
#: cross-lists too, which is how a speech or robotics paper answers an ML query
#: -- and one of the keywords has to appear in its title or abstract. Neither
#: gate decides what is true, only what is worth listing; a miss is possible,
#: which is why every run also publishes the unfiltered pool. Keep these aligned
#: with the intervention families in ``src/chowder/growth/interventions.py`` and
#: the training backends in ``docs/TRAINING_BACKENDS_UNIFIED_2026-10-05.md``.
PROFILES: tuple[dict[str, Any], ...] = (
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
        "categories": ("cs.LG", "cs.CL", "cs.AI"),
        "keywords": (
            "curriculum",
            "optimizer",
            "continued pretraining",
            "pretraining",
            "post-training",
            "rlvr",
            "reinforcement learning",
            "preference optimization",
            "data curation",
            "data selection",
            "data mixture",
            "training recipe",
            "fine-tuning",
            "scaling law",
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
        "categories": ("cs.LG", "cs.CL", "cs.AI", "cs.CV"),
        "keywords": (
            "quantiz",
            "quantis",
            "prun",
            "low-rank",
            "low rank",
            "kv cache",
            "kv-cache",
            "compress",
            "mixed precision",
            "bit-width",
            "sparsit",
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
        "categories": ("cs.CL", "cs.LG", "cs.AI", "cs.CV"),
        "keywords": (
            "lora",
            "qlora",
            "adapter",
            "peft",
            "parameter-efficient",
            "parameter efficient",
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
        "categories": ("cs.LG", "cs.CL", "cs.AI", "cs.CV"),
        "keywords": (
            "distillation",
            "distil",
            "teacher model",
            "teacher-student",
            "self-play",
            "synthetic data",
            "reward model",
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
        "categories": ("cs.LG", "cs.CL", "cs.AI", "cs.DC"),
        "keywords": (
            "speculative decoding",
            "speculative",
            "draft model",
            "drafter",
            "decoding throughput",
            "throughput",
            "latency",
            "inference efficiency",
            "serving",
            "memory footprint",
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
        "categories": ("cs.CL", "cs.LG", "cs.AI", "cs.CY"),
        "keywords": (
            "llm judge",
            "judge",
            "benchmark contamination",
            "contamination",
            "evaluation validity",
            "evaluator",
            "reward hacking",
            "leaderboard",
            "annotat",
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
    profiles: list[dict[str, Any]],
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


def matched_terms(title: str, summary: str, keywords: tuple[str, ...]) -> list[str]:
    """Which surface terms a hit trips. Coarse on purpose -- substring, no ranking."""
    haystack = f"{title} {summary}".lower()
    return [term for term in keywords if term.lower() in haystack]


def _distinct_terms(terms: list[str]) -> list[str]:
    """Drop terms another fired term subsumes ('distil' under 'distillation').

    Hyphens and spaces are ignored for the comparison, so a paper that trips both
    'low-rank' and 'low rank' reports one term. Display only: the screen ran on
    every term, and what is shown is the profile's raw term so it stays auditable.
    """
    normalized = {t: t.replace("-", " ").replace(" ", "") for t in terms}
    return [
        term
        for term in terms
        if not any(
            term != other and normalized[term] in normalized[other] for other in terms
        )
    ]


def _base_id(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0] if arxiv_id else ""


_LOGGED_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")

_DROP_HEADING_RE = re.compile(r"^## Automated pool drop -- (\d{4}-\d{2}-\d{2})\s*$")


def prune_drops(
    log_text: str, *, older_than_days: int, now: datetime | None = None
) -> tuple[str, int]:
    """Drop automated sections older than the horizon. Returns ``(text, removed)``.

    The horizon must stay longer than the fetch window (the schedule prunes at
    30 days against a 14-day window) so that pruning can never resurrect a
    listing: an id inside a pruned drop is already too old to be fetched again.
    Curated sections are never touched -- only the drops a machine wrote.
    """
    now = now or datetime.now(timezone.utc)
    lines = log_text.splitlines(keepends=True)
    kept: list[str] = []
    removed = 0
    index = 0
    while index < len(lines):
        match = _DROP_HEADING_RE.match(lines[index].rstrip("\r\n"))
        if not match:
            kept.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not lines[end].startswith("## "):
            end += 1
        try:
            stamp = datetime.strptime(match.group(1), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            kept.extend(lines[index:end])  # unreadable date: keep it for a person
            index = end
            continue
        if (now - stamp).total_seconds() > older_than_days * 86400:
            removed += 1
            index = end
            continue
        kept.extend(lines[index:end])
        index = end
    return "".join(kept), removed


def logged_ids(log_text: str) -> set[str]:
    """Every arXiv id the log already mentions -- curated sections and drops.

    This is the whole dedup story: a paper a person already wrote up, or a hit a
    previous drop already listed, is not listed again.
    """
    return {match.group(1) for match in _LOGGED_ID_RE.finditer(log_text)}


def build_append_section(
    groups: list[dict],
    *,
    log_text: str,
    days: int,
    limit: int,
    now: datetime | None = None,
) -> tuple[str, dict]:
    """Render an unvetted drop of surface-matched hits not already in the log.

    Returns ``(section, stats)``; ``section`` is empty when nothing new matched.
    The drop deliberately carries no abstract text and no number: a triage line
    has to stay unable to smuggle a claim into the curated log (README rules 5
    and 7).
    """
    now = now or datetime.now(timezone.utc)
    seen = logged_ids(log_text)
    matched = 0
    appended = 0
    truncated = 0
    out_of_scope = 0
    blocks: list[list[str]] = []
    for group in groups:
        profile = next((p for p in PROFILES if p["name"] == group["profile"]), None)
        keywords = tuple(profile["keywords"]) if profile else ()
        categories = tuple(profile.get("categories", ())) if profile else ()
        lines: list[str] = []
        for entry in group["entries"]:
            terms = matched_terms(entry["title"], entry["summary"], keywords)
            if not terms:
                continue
            if categories and entry["primary_category"] not in categories:
                out_of_scope += 1
                continue
            matched += 1
            terms = _distinct_terms(terms)
            base = _base_id(entry["arxiv_id"])
            if not base or base in seen:
                continue
            seen.add(base)
            if appended >= limit:
                truncated += 1
                continue
            appended += 1
            lines.append(
                f"- **{entry['title']}** -- `arXiv:{entry['arxiv_id']}` -- "
                f"{entry['published'][:10]} -- {entry['primary_category']} -- "
                f"matched: {', '.join(terms)}"
            )
            lines.append(f"  - {entry['url']}")
        if lines:
            blocks.append([f"### {group['profile']}", "", *lines, ""])
    stats = {
        "matched": matched,
        "appended": appended,
        "truncated": truncated,
        "already_logged": matched - appended - truncated,
        "out_of_scope": out_of_scope,
    }
    if appended == 0:
        return "", stats
    stamp = now.strftime("%Y-%m-%d")
    parts: list[str] = [
        f"## Automated pool drop -- {stamp}",
        "",
        "> **Unvetted, machine-appended -- nothing here is registered.** These are",
        "> arXiv search hits that named a surface mechanism in a watched primary",
        "> category, listed for triage. No paper here has been read, no result is",
        "> quoted, and no intervention family, maturity label, or gate is touched",
        "> (`README.md`, rule 7). Curate anything real into a dated curated section",
        "> under the rules, or delete this drop if it adds nothing.",
        "",
        f"- window: last {days} day(s) | surfaced: {matched} | new: {appended} | "
        f"already in the log: {stats['already_logged']}"
        + (f" | off-scope: {out_of_scope}" if out_of_scope else ""),
        "- source: `.github/workflows/literature-watch.yml` (`watch.py "
        "--append-log`: the primary-category gate plus the surface-term screen)",
    ]
    if truncated:
        parts.append(
            f"- {truncated} further new hit(s) exceeded the drop limit; the run's "
            "`pool.json` artifact holds the whole unfiltered pool"
        )
    parts.append("")
    for block in blocks:
        parts.extend(block)
    return "\n".join(parts).rstrip("\n") + "\n", stats


def _write_append(path: Path, base_text: str, section: str, stats: dict) -> bool:
    """Write ``base_text`` plus the drop. Returns True when the log changed.

    Nothing new means nothing written at all: no drop, no prune, no commit. A
    prune only rides along with a drop that has something to say.
    """
    if not section:
        print(
            "no new surface-matched entries "
            f"(surfaced {stats['matched']}, all already in the log); log unchanged"
        )
        return False
    separator = ""
    if base_text and not base_text.endswith("\n\n"):
        separator = "\n" if base_text.endswith("\n") else "\n\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(base_text + separator + section)
    pruned = stats.get("pruned_drops", 0)
    print(
        f"wrote an unvetted drop to {path}: {stats['appended']} new hit(s), "
        f"{stats['already_logged']} already logged, "
        f"{stats['truncated']} over the limit"
        + (f", {pruned} stale drop(s) pruned" if pruned else "")
    )
    return True


def _select_profiles(name: str) -> list[dict[str, Any]]:
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
        "--append-log",
        metavar="PATH",
        help=(
            "screen the pool through the surface gates and append an unvetted "
            "drop to this markdown log; writes nothing when nothing is new"
        ),
    )
    parser.add_argument(
        "--append-limit",
        type=int,
        default=40,
        help="max entries in one automated drop (default: 40)",
    )
    parser.add_argument(
        "--prune-drops",
        type=int,
        default=0,
        metavar="DAYS",
        help=(
            "remove automated drops older than DAYS when appending; must exceed "
            "the fetch window so a prune cannot resurrect a listing (0 = keep)"
        ),
    )
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

    # The append runs against the log as checked out, so a rerun after a merge
    # finds the ids it already listed and writes nothing.
    log_path = Path(args.append_log) if args.append_log else None
    log_text = ""
    payload: dict = {"groups": groups, "errors": errors}
    if log_path is not None:
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8")
        # Dedup reads the whole log as it stands; the prune only edits what gets
        # written back.
        base_text, pruned = (
            prune_drops(log_text, older_than_days=args.prune_drops)
            if args.prune_drops > 0
            else (log_text, 0)
        )
        section, append_stats = build_append_section(
            groups, log_text=log_text, days=args.days, limit=args.append_limit
        )
        append_stats["pruned_drops"] = pruned
        payload["append"] = append_stats

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(groups, days=args.days))
        total = sum(len(g["entries"]) for g in groups)
        print(f"wrote {args.out} ({total} entries across {len(groups)} profiles)")
    elif log_path is None:
        print(render_markdown(groups, days=args.days))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json_out}", file=sys.stderr)

    if log_path is not None:
        _write_append(log_path, base_text, section, append_stats)

    for error in errors:
        print(f"warning: {error}", file=sys.stderr)

    # A partial pool is still useful; only fail when nothing at all came back.
    return 0 if groups else 1


if __name__ == "__main__":
    raise SystemExit(main())
