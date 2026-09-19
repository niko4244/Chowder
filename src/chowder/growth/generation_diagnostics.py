"""The generation-diagnostics instrument, in production.

The campaign's declared target set is the response-surface instrument, and the
frozen judge's T1-T10 read one row of it: the completions of the declared
diagnostic prompts, plus five aggregates computed from the raw generations --
EOS termination, max-token cap, unclosed ``<think>``, obvious loops and
distinct-trigram ratio.

Those definitions used to exist only in the historical
``docs/gen1/run_gen1_cycle.py``, so the instrument could only be measured by
running that driver; every other run left T1-T10 ``UNKNOWN``. This module is
that instrument in ``src/``, so a campaign produces the evidence the judge
scores on the same path that produces every other row.

The scoring rules are **ports, not redesigns**: the same triggers, the same
denominators, computed from the same decoded completions. A gen2 diagnostic is
therefore comparable with the gen0 and gen1 diagnostics it is judged against --
changing a definition here would silently change the meaning of every threshold
the frozen judge compares against.

What it refuses is the interesting part. The aggregates are computed from facts
only the generating worker can observe (how many tokens were produced, and
whether generation stopped on EOS), so the facts are read from the worker's own
per-item rows and validated rather than inferred from the completion text: a
run that cannot say whether a generation terminated must not report a
termination rate of zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .candidate_evaluation import CandidateEvaluationRefusal

#: The evaluation named an instrument whose per-item generations do not carry
#: the facts the diagnostics are defined over.
GENERATION_DIAGNOSTICS_UNMEASURED = "GENERATION_DIAGNOSTICS_UNMEASURED"

#: The facts one generated item must carry. ``prompt``/``expected``/``prediction``
#: are the item itself; ``generated_tokens`` and ``eos_terminated`` are what the
#: worker observed while generating it -- the cap-hit rate is derived from the
#: first, so a worker cannot report one fact and disagree with the other.
_TEXT_FIELDS = ("prompt", "expected", "prediction")

#: The 16 frozen prompts of the generation-diagnostics instrument: the dataset
#: the target benchmark is measured on, in dataset order, indices 0..15.  This
#: is the production owner of the instrument's items so a campaign can *declare*
#: the dataset its evaluator measures (``evaluation_material_path``) rather than
#: leaving the target slice to be assembled by hand.  It is byte-identical to the
#: frozen judge's ``INSTRUMENT_PROMPTS``; a test asserts the two cannot drift.
INSTRUMENT_PROMPTS: tuple[tuple[str, str], ...] = (
    ("Reply with exactly: ping", "ping"),
    ("What is 17 * 23? Answer with the number only.", "391"),
    ("Name the capital of Australia in one word.", "Canberra"),
    ("Write one sentence describing rain.", "rain"),
    ("Count from 1 to 5, digits only.", "5"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Translate 'good morning' into French.", "bonjour"),
    ("Complete: The opposite of hot is", "cold"),
    ("List the first three prime numbers.", "2"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is 100 divided by 4?", "25"),
    ("Say 'done' and nothing else.", "done"),
    ("Give one synonym for 'happy'.", "joyful"),
    ("How many continents are there?", "7"),
    ("What color is a banana?", "yellow"),
    ("Answer with a single word: 2 + 2 =", "4"),
)


@dataclass(frozen=True)
class GenerationDiagnostics:
    """The frozen diagnostics of one measured set of generations."""

    n_prompts: int
    eos_termination_rate: float
    max_token_cap_rate: float
    unclosed_think_rate: float
    obvious_loop_count: int
    distinct_trigram_ratio_mean: float
    distinct_trigram_ratio_min: float
    max_new_tokens: int
    seed: int
    per_prompt: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_items(
        cls,
        items: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
        seed: int,
        source: str,
    ) -> "GenerationDiagnostics":
        """Compute the diagnostics from the worker's own per-item rows.

        ``source`` names the file the items came from, so a refusal says which
        measurement could not be diagnosed rather than only that one could not.
        """
        if not items:
            raise CandidateEvaluationRefusal(
                f"{GENERATION_DIAGNOSTICS_UNMEASURED}: {source} holds no generated "
                "item, so there is nothing to diagnose"
            )
        if max_new_tokens <= 0:
            raise CandidateEvaluationRefusal(
                f"{GENERATION_DIAGNOSTICS_UNMEASURED}: the declared protocol names no "
                "positive max_new_tokens, so a cap-hit cannot be decided"
            )

        completions: list[str] = []
        per_prompt: list[Mapping[str, Any]] = []
        terminated = 0
        capped = 0
        for index, item in enumerate(items):
            text = {field: str(item[field]) for field in _TEXT_FIELDS}
            generated = _observed_int(item, "generated_tokens", index=index, source=source)
            stopped = _observed_bool(item, "eos_terminated", index=index, source=source)
            # The gen1 rule, ported: stopping on EOS means stopping *before* the
            # cap, so a generation cannot both terminate and hit it. A worker
            # whose two facts disagree is refused rather than reconciled: one of
            # them is wrong and this module cannot know which.
            if stopped and generated >= max_new_tokens:
                raise CandidateEvaluationRefusal(
                    f"{GENERATION_DIAGNOSTICS_UNMEASURED}: {source} item {index} claims "
                    f"EOS termination after {generated} generated token(s), which is "
                    f"the declared cap ({max_new_tokens}); a terminated generation "
                    "stops short of the cap"
                )
            hit_cap = generated >= max_new_tokens
            terminated += int(stopped)
            capped += int(hit_cap)
            completions.append(text["prediction"])
            per_prompt.append(
                {
                    "prompt": text["prompt"],
                    "expected": text["expected"],
                    "completion": text["prediction"],
                    "generated_tokens": generated,
                    "eos_terminated": stopped,
                    "cap_hit": hit_cap,
                }
            )

        ratios = [_trigram_ratio(completion) for completion in completions]
        loops = sum(1 for completion in completions if _has_repeated_lines(completion))
        unclosed = sum(1 for completion in completions if _is_unclosed_think(completion))
        n = len(completions)
        return cls(
            n_prompts=n,
            eos_termination_rate=terminated / n,
            max_token_cap_rate=capped / n,
            unclosed_think_rate=unclosed / n,
            obvious_loop_count=loops,
            distinct_trigram_ratio_mean=sum(ratios) / len(ratios),
            distinct_trigram_ratio_min=min(ratios),
            max_new_tokens=max_new_tokens,
            seed=seed,
            per_prompt=tuple(per_prompt),
        )

    def to_metadata(self) -> dict[str, Any]:
        """The keys the frozen judge reads, plus the provenance of the numbers.

        Flat on purpose: the judge reads ``metadata["per_prompt"]`` and
        ``metadata["eos_termination_rate"]`` directly, so nesting these under a
        name of our own would leave every one of T1-T10 ``UNKNOWN``.
        """
        return {
            "per_prompt": [dict(entry) for entry in self.per_prompt],
            "n_prompts": self.n_prompts,
            "eos_termination_rate": self.eos_termination_rate,
            "max_token_cap_rate": self.max_token_cap_rate,
            "unclosed_think_rate": self.unclosed_think_rate,
            "obvious_loop_count": self.obvious_loop_count,
            "distinct_trigram_ratio_mean": self.distinct_trigram_ratio_mean,
            "distinct_trigram_ratio_min": self.distinct_trigram_ratio_min,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "diagnostics_rule": "gen1-instrument-ported",
        }


def _observed_int(
    item: Mapping[str, Any], field: str, *, index: int, source: str
) -> int:
    """One observed integer fact, or a refusal naming the item it is missing from."""
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateEvaluationRefusal(
            f"{GENERATION_DIAGNOSTICS_UNMEASURED}: {source} item {index} carries no "
            f"{field} (got {value!r}); the generation diagnostics are defined over "
            "what the worker observed while generating, not over the decoded text"
        )
    return int(value)


def _observed_bool(
    item: Mapping[str, Any], field: str, *, index: int, source: str
) -> bool:
    value = item.get(field)
    if not isinstance(value, bool):
        raise CandidateEvaluationRefusal(
            f"{GENERATION_DIAGNOSTICS_UNMEASURED}: {source} item {index} carries no "
            f"{field} (got {value!r}); an unevaluated {field} would be read as False "
            "and turn an unmeasured generation into a termination failure"
        )
    return value


def _trigram_ratio(text: str) -> float:
    """Distinct word-trigrams over all of them; short texts are trivially distinct."""
    words = text.split()
    if len(words) < 3:
        return 1.0
    trigrams = [tuple(words[index : index + 3]) for index in range(len(words) - 2)]
    return len(set(trigrams)) / len(trigrams)


def _has_repeated_lines(text: str) -> bool:
    """A line repeated three times in a row: the shipped loop detector."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return any(
        lines[index] == lines[index + 1] == lines[index + 2]
        for index in range(len(lines) - 2)
    )


def _is_unclosed_think(text: str) -> bool:
    return "<think>" in text and "</think>" not in text


__all__ = [
    "GENERATION_DIAGNOSTICS_UNMEASURED",
    "GenerationDiagnostics",
]
