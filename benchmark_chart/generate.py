"""Source-bound, CPU-only chart of the eleven DeepSeek comparison categories.

This script NEVER asks a model to answer a question or executes a generated program.
Its only numerical inputs are SHA-pinned local audits and quoted publisher tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET


ORDER = (
    "hle_no_tools", "terminal_bench_4", "deepswe_1_1", "programbench",
    "nl2repo", "automation_v1_0_6", "agents_last_exam", "apex_v1_1",
    "chartography_tools", "babyvision_tools", "zerobench_main_tools",
)
MODELS = {
    "qwen": ("Qwen/Qwen3.8-27B", "https://huggingface.co/Qwen/Qwen3.8-27B"),
    "opus": ("Claude Opus 5.5", "https://www.anthropic.com/claude-opus-5-5"),
}
ALLOWED_COMPARABILITY = {
    "same_named_benchmark", "different_cohort", "different_harness",
    "unverified_harness_match", "unverified_metric_match",
    "unverified_version_match", "unverified_tool_match",
}
MANIFEST = Path("benchmark_chart/source_manifest.json")
REFERENCES = Path("benchmark_chart/references.json")
AUDIT = Path("docs/quals/reap-v2-gen0-integrated-scientific-state-20261026.md")
PAIRED = Path("docs/quals/reap-v2-qwen38-opus55-common-test-20261026.md")
# Independently reviewed primary-source snapshot locks. A caller cannot make an
# altered audit or vendor claim legitimate merely by rewriting the JSON manifest.
SOURCE_LOCKS = {
    REFERENCES.as_posix(): "8b7da2e68aafc769da5e47546183e04cadf654c1593df40394f6898f2c2bf4b1",
    AUDIT.as_posix(): "177cabf3a73eeac23f879f4477916afa4da9c0705c9202701821858977f7dd0c",
    PAIRED.as_posix(): "29acb3dfa6aa65ec63a5c29db87ceff11906a132ecad4d5a7426744e2be508c5",
}


class EvidenceError(ValueError):
    """An evidence identity or benchmark protocol was changed or is ambiguous."""


def _read_checked(root: Path, path: Path, pinned_sha: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{64}", pinned_sha):
        raise EvidenceError(f"Invalid pinned source digest: {path}")
    raw = (root / path).read_bytes()
    # Git on Windows commonly checks text out as CRLF and Linux uses LF. The
    # source evidence is the identical Git text across either representation.
    normalized = raw.replace(b"\r\n", b"\n")
    actual = hashlib.sha256(normalized).hexdigest()
    if actual != pinned_sha:
        raise EvidenceError(f"Pinned evidence digest mismatch: {path} expected={pinned_sha} actual={actual}")
    return normalized


def _single_int(pattern: str, source: str, label: str) -> tuple[int, ...]:
    matches = list(re.finditer(pattern, source, re.MULTILINE))
    if len(matches) != 1:
        raise EvidenceError(f"Expected one source-anchored match for {label}, found {len(matches)}")
    return tuple(int(value.replace(",", "")) for value in matches[0].groups())


def _extract_local_audits(integrated: str, paired: str) -> list[dict[str, str | int]]:
    q4_gsm = _single_int(
        r"^\| GSM8K \*\*([\d,]+)/([\d,]+)\*\* \| \*\*([\d,]+) \([\d.]+%\)\*\*",
        integrated, "Q4 original GSM8K full",
    )
    q4_he = _single_int(
        r"^\| HumanEval \*\*([\d,]+)/([\d,]+)\*\* \| \*\*(\d+) \([\d.]+%\)\*\*",
        integrated, "Q4 original HumanEval full",
    )
    q8_he = _single_int(
        r"^\s*\| HumanEval 164 \| \*\*(\d+) \([\d.]+%\)\*\*",
        integrated, "Q8 original HumanEval scored population",
    )
    q8_gsm = _single_int(
        r"REAP historical full Linux quality remains Q8 \*\*([\d,]+)/([\d,]+)\*\* GSM8K",
        paired, "Q8 original GSM8K full",
    )
    q4_cross = _single_int(r"and Q4 \*\*([\d,]+)/([\d,]+)\*\* GSM8K", paired, "paired audit Q4 witness")
    if q4_gsm != (1319, 1319, 1229) or q4_he != (164, 164, 151):
        raise EvidenceError("Original Q4 audited suite identity/count unexpectedly changed")
    if q8_gsm != (1220, 1319) or q8_he != (152,) or q4_cross != (1229, 1319):
        raise EvidenceError("Original Q8 or paired Q4 audited source unexpectedly changed")
    if "independent 164-program executable replay is still PENDING" not in integrated:
        raise EvidenceError("Q8 HumanEval independent execution status not proved")
    if "No model ranking is valid" not in paired:
        raise EvidenceError("Paired study scope caveat absent")
    for competitor in ("Qwen3.8-27B", "Claude Opus 5.5"):
        expr = rf"^\| {re.escape(competitor)} \| \*\*UNKNOWN / NOT TESTED\*\*"
        if len(re.findall(expr, paired, flags=re.MULTILINE)) != 1:
            raise EvidenceError(f"Paired competitor availability changed: {competitor}")
    return [
        {"model": "REAP128_v2 Q4_K_M", "benchmark": "GSM8K original Linux", "correct": 1229, "total": 1319, "qualification": "source-audited provisional research"},
        {"model": "REAP128_v2 Q8_0", "benchmark": "GSM8K original Linux", "correct": 1220, "total": 1319, "qualification": "source-audited provisional research"},
        {"model": "REAP128_v2 Q4_K_M", "benchmark": "HumanEval original Linux", "correct": 151, "total": 164, "qualification": "historical independently re-executed; strict sandbox attestation still limited"},
        {"model": "REAP128_v2 Q8_0", "benchmark": "HumanEval original Linux", "correct": 152, "total": 164, "qualification": "original scorer only; separate independent replay PENDING"},
    ]


def _validate_reference(raw: dict) -> list[dict]:
    if set(raw) != {"schema", "screenshot_definition", "competitors", "benchmarks", "publisher_references"}:
        raise EvidenceError("Unexpected reference data or undeclared direct scores")
    if raw.get("schema") != "reap128-v2-eleven-benchmarks.v1":
        raise EvidenceError("Unsupported eleven-suite benchmark schema")
    if raw.get("screenshot_definition") != "https://api-docs.deepseek.com/updates/":
        raise EvidenceError("Screenshot benchmark source changed")
    if set(raw.get("competitors", {})) != set(MODELS):
        raise EvidenceError("Competitor set changed")
    for key, (identity, url) in MODELS.items():
        competitor = raw["competitors"][key]
        if set(competitor) != {"name", "identity", "publisher", "source"}:
            raise EvidenceError(f"Unexpected competitor fields for {key}")
        if competitor.get("identity") != identity or competitor.get("source") != url:
            raise EvidenceError(f"Exact model/publisher provenance changed for {key}")
    benchmarks = raw.get("benchmarks")
    if not isinstance(benchmarks, list) or tuple(b.get("id") for b in benchmarks) != ORDER:
        raise EvidenceError("Benchmark categories or exact order changed")
    if any(set(b) != {"id", "label", "category", "benchmark_url", "definition"} for b in benchmarks):
        raise EvidenceError("Unexpected benchmark fields, including forged direct scores")
    if any(not b.get("label") or not str(b.get("benchmark_url", "")).startswith("https://") for b in benchmarks):
        raise EvidenceError("A benchmark is missing its authoritative label or URL")
    seen: set[tuple[str, str]] = set()
    for ref in raw.get("publisher_references", []):
        if set(ref) != {"benchmark", "model", "score", "metric", "comparability", "publisher_evidence", "caveat"}:
            raise EvidenceError("Unexpected publisher score or missing provenance fields")
        pair = (ref.get("benchmark"), ref.get("model"))
        if pair in seen or pair[0] not in ORDER or pair[1] not in MODELS:
            raise EvidenceError(f"Duplicate, unknown benchmark, or model reference: {pair}")
        seen.add(pair)
        value = ref.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100:
            raise EvidenceError(f"Invalid published metric: {pair}")
        evidence = ref.get("publisher_evidence", "")
        # The first numeric field after the publisher's table separator MUST be the selected
        # model's score; this catches edits that silently swap 'Score 42.9' with Pass@1 20.4.
        matched = re.search(r"\|\s*(?:Pass@1\s*)?(\d+(?:\.\d+)?)", evidence)
        if not matched or float(matched.group(1)) != float(value):
            raise EvidenceError(f"Published score does not agree with cited publisher row: {pair}")
        if ref.get("comparability") not in ALLOWED_COMPARABILITY or not ref.get("caveat") or not ref.get("metric"):
            raise EvidenceError(f"Publisher reference lacks method/metric disclosure: {pair}")
        if ref["benchmark"] == "hle_no_tools" and ref["comparability"] != "different_cohort":
            raise EvidenceError("HLE mixed/test cohort cannot be promoted to pure-text no-tools")
        if ref["benchmark"] == "deepswe_1_1" and ref["comparability"] != "different_harness":
            raise EvidenceError("Claude Code DeepSWE cannot be conflated with Mini-SWE-V2")
        if ref["benchmark"] == "automation_v1_0_6" and ref["comparability"] != "unverified_version_match":
            raise EvidenceError("Unversioned AutomationBench cannot claim confirmed v1.0.6")
    return benchmarks


def load_evidence(root: Path) -> dict:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != "reap128-v2-chart-pinned-evidence.v1":
        raise EvidenceError("Missing checked manifest schema")
    if manifest.get("baseline_commit") != "e62cf4e2952bbd97b5b81ae9a5c81fbc2e90f6fb":
        raise EvidenceError("Unapproved REAP audit baseline commit")
    sha = manifest.get("sha256")
    if not isinstance(sha, dict) or sha != SOURCE_LOCKS:
        raise EvidenceError("Pinned inputs unexpectedly changed")
    inputs = {p.as_posix(): _read_checked(root, p, sha[p.as_posix()]) for p in (REFERENCES, AUDIT, PAIRED)}
    ref = json.loads(inputs[REFERENCES.as_posix()].decode("utf-8"))
    _validate_reference(ref)
    audited = _extract_local_audits(
        inputs[AUDIT.as_posix()].decode("utf-8"),
        inputs[PAIRED.as_posix()].decode("utf-8"),
    )
    return {"references": ref, "audits": audited, "sha256": sha, "baseline_commit": manifest["baseline_commit"]}


def rows(evidence: dict) -> list[dict]:
    ref = evidence["references"]
    published = {(r["benchmark"], r["model"]): r for r in ref["publisher_references"]}
    result = []
    for benchmark in ref["benchmarks"]:
        row = {
            "id": benchmark["id"], "benchmark": benchmark["label"],
            "category": benchmark["category"], "definition": benchmark["definition"],
            "benchmark_url": benchmark["benchmark_url"],
            "reap_direct": "N/T", "qwen_direct": "N/T", "opus_direct": "N/T",
        }
        for key in MODELS:
            claim = published.get((benchmark["id"], key))
            row[f"{key}_reference"] = f'{claim["score"]:.1f}' if claim else "N/T"
            row[f"{key}_metric"] = claim["metric"] if claim else "N/T"
            row[f"{key}_comparability"] = claim["comparability"] if claim else "N/T"
            row[f"{key}_caveat"] = claim["caveat"] if claim else ""
            row[f"{key}_source"] = ref["competitors"][key]["source"] if claim else ""
            row[f"{key}_publisher_row"] = claim["publisher_evidence"] if claim else ""
        result.append(row)
    return result


def csv_text(data: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(data[0]))
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def markdown(evidence: dict, data: list[dict]) -> str:
    lines = [
        "# REAP128_v2 — 11 benchmark comparison", "",
        "**Status:** no official same-protocol 11-suite REAP128_v2 run has been recorded. "
        "All *direct* cells are **N/T** (not tested), never zero. Published model-vendor "
        "references appear in separate columns and are not matched three-way trials.", "",
        "| Screenshot category | REAP direct | Qwen direct | Opus direct | Qwen publisher ref† | Opus publisher ref† |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for item in data:
        lines.append(f'| {item["benchmark"]} | N/T | N/T | N/T | {item["qwen_reference"]} | {item["opus_reference"]} |')
    lines += ["", "† Publisher-reported percentage, **not** a local score. A number may use a "
              "different cohort, judge, harness, version, metric, or tool set. Do not average or "
              "rank these columns. N/T means no verified value for that model and category.", "",
              "## Published reference provenance and comparability", "",
              "| Benchmark | Model | Publisher-reported | Exact limitation | Primary source |",
              "|:--|:--|--:|:--|:--|"]
    for item in data:
        for key in MODELS:
            if item[f"{key}_reference"] == "N/T":
                continue
            source = item[f"{key}_source"]
            lines.append(f'| {item["benchmark"]} | {evidence["references"]["competitors"][key]["name"]} | '
                         f'{item[f"{key}_reference"]}% ({item[f"{key}_metric"]}) | '
                         f'{item[f"{key}_comparability"]}: {item[f"{key}_caveat"]} | [Official publisher]({source}) |')
    lines += ["", "## Actual REAP measurements from immutable local audits (different benchmarks)", "",
              "These scores are **Linux research evidence**, not any of the eleven screenshot categories "
              "and not an original Windows Gen-0 release gate.", "",
              "| REAP checkpoint | Audited benchmark | Correct / total | Accuracy | Qualification |",
              "|:--|:--|--:|--:|:--|"]
    for measured in evidence["audits"]:
        accuracy = 100.0 * measured["correct"] / measured["total"]
        lines.append(f'| {measured["model"]} | {measured["benchmark"]} | '
                     f'{measured["correct"]}/{measured["total"]} | {accuracy:.2f}% | {measured["qualification"]} |')
    lines += ["", "The 16-item shared GSM8K pilot previously reported REAP Q4 and Q8 each "
              "15/16, but Qwen3.8-27B and Opus 5.5 each **N/T** owing to rejected provider "
              "requests. That slice is insufficient for a ranking.", "",
              "## Definitions and primary sources", "",
              f'- Eleven labels and special scoring constraints: [DeepSeek October 17 change log]({evidence["references"]["screenshot_definition"]}) '
              "(source of the screenshot *categories*, not scores for our three exact models).",
              f'- Exact Qwen model/official results: [{MODELS["qwen"][0]}]({MODELS["qwen"][1]}).',
              f'- Exact Opus model/official results: [Anthropic Claude Opus 5.5]({MODELS["opus"][1]}).',
              "- Benchmark definitions: " + ", ".join(f'[{item["label"]}]({item["benchmark_url"]})' for item in evidence["references"]["benchmarks"]) + ".",
              f'- Local baseline commit: `{evidence["baseline_commit"]}`. Source SHA-256 pins:', ""]
    for name, sha in evidence["sha256"].items():
        lines.append(f'  - `{name}`: `{sha}`')
    lines += ["", "All source SHA-256 pins above hash bytes with CRLF normalized to LF, "
              "so the same Git evidence works on Windows and Linux.", "",
              "Regenerate (standard-library Python only): "
              "`python -m benchmark_chart.generate` from the isolated worktree root. "
              "The generator refuses altered evidence, wrong model IDs, score/quote mismatches, "
              "unsupported category substitutions and protocol promotions. "
              "No GPU, API, generated-code execution, private output, or network call is involved.", ""]
    return "\n".join(lines)


def svg(evidence: dict, data: list[dict]) -> str:
    """Accessible publisher-reference plot: null values are rendered literally N/T."""
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    tag = lambda name: f"{{{ns}}}{name}"
    top, step, width = 170, 91, 1430
    height = top + step * len(data) + 220
    root = ET.Element(tag("svg"), {"viewBox": f"0 0 {width} {height}", "width": str(width),
                                    "height": str(height), "role": "img", "aria-labelledby": "title desc"})
    ET.SubElement(root, tag("title"), {"id": "title"}).text = "REAP128_v2 eleven-benchmark evidence chart"
    ET.SubElement(root, tag("desc"), {"id": "desc"}).text = (
        "All direct measurements for the eleven requested categories are not tested. "
        "Only source-labeled Qwen3.8-27B and Opus 5.5 publisher reference percentages are graphed; "
        "different protocols are flagged. Exact tabular evidence is in the accompanying CSV."
    )
    def box(x: float, y: float, w: float, h: float, fill: str, radius: int = 0) -> None:
        ET.SubElement(root, tag("rect"), {"x": str(x), "y": str(y), "width": str(w), "height": str(h),
                                            "fill": fill, "rx": str(radius)})
    def txt(x: float, y: float, value: str, size: int = 18, fill: str = "#e4eaf5", weight: str = "400") -> None:
        ET.SubElement(root, tag("text"), {"x": str(x), "y": str(y), "fill": fill,
                                            "font-family": "Arial, Helvetica, sans-serif",
                                            "font-size": str(size), "font-weight": weight}).text = value
    box(0, 0, width, height, "#0d1525")
    txt(44, 63, "REAP128_v2  /  11-category benchmark comparison", 29, "#ffffff", "700")
    txt(44, 93, "Direct: all N/T  ·  Colored bars: published vendor references only", 17, "#b4c4da")
    box(44, 116, 16, 16, "#7e9ff2", 3)
    txt(68, 129, "Qwen3.8-27B", 15)
    box(246, 116, 16, 16, "#d7a574", 3)
    txt(270, 129, "Claude Opus 5.5", 15)
    txt(1000, 129, "REAP128_v2: N/T on all 11", 16, "#dce5f5", "700")
    for i, item in enumerate(data):
        y = top + i * step
        if i % 2 == 0:
            box(29, y - 29, 1370, 84, "#152238", 8)
        txt(46, y - 4, item["benchmark"], 17, "#ffffff", "600")
        txt(47, y + 18, item["category"], 13, "#8da1be")
        for offset, key, color in ((0, "qwen", "#7e9ff2"), (25, "opus", "#d7a574")):
            number = item[f"{key}_reference"]
            bar_y = y - 18 + offset
            box(508, bar_y, 420, 14, "#24364e", 5)
            if number != "N/T":
                box(508, bar_y, 420 * float(number) / 100.0, 14, color, 5)
                flag = "" if item[f"{key}_comparability"] == "same_named_benchmark" else "†"
                txt(943, bar_y + 13, f"{number}%{flag}", 16, color, "700")
            else:
                txt(943, bar_y + 13, "N/T", 15, "#8193ac")
        txt(1156, y + 8, "N/T", 25, "#c3d0e4", "700")
        txt(1220, y + 8, "direct", 14, "#879bb8")
    bottom = top + len(data) * step + 12
    txt(45, bottom, "† Published score with unresolved cohort, harness, metric, version, or tools mismatch.", 15, "#c0cde0")
    txt(45, bottom + 30, "N/T = not tested or no exact reference. Zero scores are never imputed.", 15, "#c0cde0")
    txt(45, bottom + 64, "Audited outside this chart: REAP Q4 GSM8K 1229/1319; Q8 1220/1319; HumanEval Q4 151/164, Q8 152/164 (replay pending).", 14, "#dce5f5")
    txt(45, bottom + 94, "Sources: huggingface.co/Qwen/Qwen3.8-27B  ·  anthropic.com/claude-opus-5-5  ·  pinned local REAP audits", 13, "#aebdd1")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def render(root: Path, output: Path | None = None) -> dict[str, bytes]:
    evidence = load_evidence(root)
    data = rows(evidence)
    artifacts = {
        "reap128v2_11_benchmarks.csv": csv_text(data).encode("utf-8"),
        "reap128v2_11_benchmarks.md": markdown(evidence, data).encode("utf-8"),
        "reap128v2_11_benchmarks.svg": svg(evidence, data).encode("utf-8"),
    }
    ET.fromstring(artifacts["reap128v2_11_benchmarks.svg"])
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            (output / name).write_bytes(content)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    destination = arguments.output.resolve() if arguments.output else root / "benchmark_chart/output"
    artifacts = render(root, destination)
    print(f"PASS: {len(ORDER)} categories; {len(artifacts)} deterministic files at {destination}")


if __name__ == "__main__":
    main()
