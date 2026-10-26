"""Frozen, prospectively declared REAP128_v2 text-only HLE research protocol.

This module is deliberately pure: it never downloads HLE, loads a model, runs a
kernel or invokes a judge. It defines the settings a Q4 text-only HLE research
run must obey so that the protocol is fixed *before* any outcome is observed,
and it holds the deterministic helpers the Kaggle kernel uses.

Scope and honesty constraints
-----------------------------
* This is a **new research protocol**. It is not Amendment 4, not a Gen-0
  selector input, and it cannot qualify Windows Gate0/Gate1/Gate2.
* HLE's official grading is an LLM judge (``o3-mini``). No judge is invoked
  here, so this module can only produce a **deterministic non-official**
  normalized exact-match signal. An official HLE score stays UNAVAILABLE.
* Only text-only questions (empty ``image`` field) are admissible: the REAP
  artefact is a text-only GGUF and cannot answer image-bearing questions.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

HLE_REPO = "cais/hle"
HLE_COMMIT = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
HLE_PARQUET = "data/test-00000-of-00001.parquet"
HLE_PARQUET_BYTES = 274_276_147
HLE_PARQUET_SHA256 = "6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a"
HLE_EVAL_REPO = "centerforaisafety/hle"
HLE_EVAL_COMMIT = "22ed3074b1e7b134bcbc09028d0ba320839b0655"

# Official HLE answer format prompt (hle_eval/run_model_predictions.py).
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

Q4_GGUF_FILENAME = "reap128-v2-q4_k_m-20261025.gguf"
Q4_GGUF_BYTES = 11_563_915_264
Q4_GGUF_SHA256 = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
PINNED_LLAMA_SERVER_SHA256 = "f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d"
PINNED_LLAMA_CPP_COMMIT = "7609846557c50f9d984719a9e1e8c5f3d02f807b"

# Frozen serving geometry for this protocol. Two slots share the 16k context so
# each request keeps 8192 tokens of prompt+completion headroom, which HLE
# multi-step questions need; the original cohort geometry (-np 4) would leave
# only 4096 per slot and truncate legitimate reasoning.
SERVING_FLAGS = ("--fit", "on", "-np", "2", "-c", "16384", "--jinja")
MAX_COMPLETION_TOKENS = 8192
TEMPERATURE = 0
SAMPLER_SEED = 0
CONCURRENCY = 2

# Declared before execution: the run serves items in canonical order until it
# either finishes the population or reaches the wall-clock budget, then reports
# the completed prefix as a deterministic pseudorandom sample of the text-only
# population. Both the order rule and the budget are frozen here.
PLANNED_MAX_SAMPLES = 1_200
WALL_BUDGET_SECONDS = 32_400  # 9 h of a 12 h Kaggle GPU session
ORDER_RULE = "ascending sha256(id); completed prefix is an unbiased text-only sample"

SUBSET_RULE = "text-only (empty image field only)"
TOOLS = "none"
INPUT_TEMPLATE = "{question}"
GRADING = (
    "deterministic normalized exact-match / letter match only; the official "
    "HLE LLM judge is NOT invoked, so no official HLE score is claimed"
)


def frozen_manifest() -> dict[str, Any]:
    """Return the prospective protocol plus its own digest."""
    raw: dict[str, Any] = {
        "dataset": HLE_REPO,
        "dataset_revision": HLE_COMMIT,
        "dataset_file": HLE_PARQUET,
        "dataset_bytes": HLE_PARQUET_BYTES,
        "dataset_sha256": HLE_PARQUET_SHA256,
        "subset": SUBSET_RULE,
        "tools": TOOLS,
        "input_template": INPUT_TEMPLATE,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "order_rule": ORDER_RULE,
        "planned_max_samples": PLANNED_MAX_SAMPLES,
        "wall_budget_seconds": WALL_BUDGET_SECONDS,
        "model_sha256": Q4_GGUF_SHA256,
        "model_bytes": Q4_GGUF_BYTES,
        "serving_flags": list(SERVING_FLAGS),
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": TEMPERATURE,
        "sampler_seed": SAMPLER_SEED,
        "concurrency": CONCURRENCY,
        "llama_cpp_source_commit": PINNED_LLAMA_CPP_COMMIT,
        "llama_server_binary_sha256": PINNED_LLAMA_SERVER_SHA256,
        "prediction_source_git": HLE_EVAL_COMMIT,
        "grading": GRADING,
        "official_hle_score_claimed": False,
        "original_scientific_gate_eligible": False,
    }
    encoded = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"protocol": raw, "protocol_sha256": hashlib.sha256(encoded).hexdigest()}


def is_text_only(row: Mapping[str, Any]) -> bool:
    """A row is text-only only when the image field is genuinely empty."""
    image = row.get("image")
    if image is None or image == "" or image == b"":
        return True
    if isinstance(image, (str, bytes)):
        return False
    raise ValueError("unrecognized HLE image field type")


def text_only_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Validate the full population and keep only unambiguous text-only rows."""
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("missing HLE id")
        if item_id in seen:
            raise ValueError("duplicate HLE id")
        seen.add(item_id)
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("missing HLE question")
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("missing HLE answer")
        answer_type = row.get("answer_type")
        if answer_type not in ("exactMatch", "multipleChoice"):
            raise ValueError("unrecognized HLE answer_type")
        if is_text_only(row):
            result.append(row)
    return result


def canonical_order(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Order rows by ascending sha256(id) so any completed prefix is unbiased."""
    return sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode("utf-8")).hexdigest())


def build_messages(question: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("empty HLE question")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


_LATEX_WRAP = re.compile(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize_answer(text: str) -> str:
    """Conservative, non-official answer normalisation.

    Deliberately shallow: it removes presentation noise (currency/LaTeX wrappers,
    thousands separators, surrounding punctuation, case, spacing) and does not
    attempt mathematical equivalence, which only the official judge can rule on.
    """
    if not isinstance(text, str):
        raise TypeError("answer must be text")
    s = text.replace("\u00a0", " ")
    s = _LATEX_WRAP.sub(r"\1", s)
    for token in ("$", "\\(", "\\)", "\\[", "\\]", "**", "`"):
        s = s.replace(token, "")
    s = re.sub(r"\s+", " ", s).strip()
    s = _THOUSANDS.sub("", s)
    s = s.replace(" percent", "%")
    s = s.strip(" \t\"'.;:,")
    s = s.casefold().strip()
    while s.endswith("."):
        s = s[:-1].rstrip()
    return s


_LETTER = re.compile(r"\b([A-Ea-e])\b")
_ANSWER_LINE = re.compile(r"answer\s*[:\-]\s*(.*)", re.IGNORECASE)


def extract_final_answer(response: str) -> str:
    """Pull the answer out of the official Explanation/Answer/Confidence format."""
    if not isinstance(response, str) or not response.strip():
        return ""
    matches = list(_ANSWER_LINE.finditer(response))
    if matches:
        candidate = matches[-1].group(1)
    else:
        lines = [line for line in response.splitlines() if line.strip()]
        candidate = lines[-1] if lines else response
    candidate = candidate.splitlines()[0] if candidate.strip() else ""
    # Drop a trailing confidence restatement on the same line.
    candidate = re.split(r"\bconfidence\s*[:\-]", candidate, flags=re.IGNORECASE)[0]
    candidate = candidate.replace("**", "").strip()
    candidate = candidate.strip("*_` ")
    # "A" / "(A)" / "A." / "A) ..." all mean option A.
    leading = re.match(r"^\(?([A-Ea-e])\)?\s*(?:[.):\-]|$)", candidate)
    if leading:
        return leading.group(1)
    return candidate


def grade(response: str, answer: str, answer_type: str) -> dict[str, Any]:
    """Return a deterministic, non-official correctness verdict for one item."""
    if answer_type not in ("exactMatch", "multipleChoice"):
        raise ValueError("unrecognized HLE answer_type")
    predicted = extract_final_answer(response)
    if answer_type == "multipleChoice":
        letters = _LETTER.findall(predicted) or _LETTER.findall(response)
        expected = _LETTER.findall(answer)
        if len(expected) != 1:
            raise ValueError("multiple-choice reference is not a single option")
        verdict = bool(letters) and letters[-1].upper() == expected[0].upper()
        return {"correct": verdict, "predicted": predicted[:64], "reference": expected[0].upper()}
    return {
        "correct": normalize_answer(predicted) == normalize_answer(answer),
        "predicted": predicted[:64],
        "reference": normalize_answer(answer)[:64],
    }


def summarize(scored: Iterable[Mapping[str, Any]], population: int) -> dict[str, Any]:
    """Aggregate completed items. Only descriptive statistics are reported."""
    rows = list(scored)
    total = len(rows)
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_type.setdefault(row["answer_type"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(bool(row["correct"]))
    correct = sum(1 for row in rows if row["correct"])
    return {
        "population_text_only_n": population,
        "completed_n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
        "completion_fraction_of_population": (total / population) if population else None,
        "by_answer_type": by_type,
        "official_judge_score": "UNAVAILABLE",
        "metric": "deterministic_non_official_normalized_match",
    }
