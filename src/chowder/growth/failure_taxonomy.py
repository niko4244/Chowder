"""Failure taxonomy: automatic classification of model failures.

Every meaningful failure becomes durable evidence, classified so the
curriculum engine can aggregate by skill-deficit rather than re-reading raw
transcripts. The taxonomy is extensible: ``CATEGORY_KEYWORDS`` seeds the
classifier and new categories emerge from evidence (registered explicitly)
instead of being invented speculatively by an ad-hoc string.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

#: Seeded categories. Each maps to keyword patterns over the failure
#: evidence (verifier output, diff text, judge notes, telemetry).
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "missing_knowledge": (r"does not (know|contain)", r"no relevant (fact|information)", r"unfamiliar"),
    "arithmetic": (r"arithmetic", r"off[- ]by[- ]one", r"numerical (error|mistake)", r"miscalculat"),
    "logical_error": (r"logical (error|flaw|inconsistency)", r"invalid (inference|deduction)", r"contradict"),
    "premature_conclusion": (r"concluded (too |prematurely)", r"stopped (early|reasoning)", r"jumped to"),
    "hallucination": (r"hallucinat", r"fabricat", r"invented (a|an|the) ", r"unsupported claim"),
    "wrong_tool": (r"wrong (tool|function|api)", r"incorrect tool", r"should have called"),
    "malformed_tool_arguments": (r"schema validation", r"invalid arguments?", r"malformed (call|json)"),
    "failed_retrieval": (r"retrieval (failed|missed)", r"did not find (the|any)", r"missed (the )?(needle|document)"),
    "ignored_evidence": (r"ignored (the )?(evidence|context|document)", r"contrary to (the )?(passage|context)"),
    "coding_syntax": (r"syntax ?error", r"failed to (parse|compile)", r"indentation"),
    "coding_semantics": (r"semantic (error|bug)", r"wrong (logic|behavior)", r"incorrect (return|output value)"),
    "incomplete_patch": (r"incomplete (patch|diff|edit)", r"partial (fix|edit)", r"did not apply"),
    "failed_tests": (r"test(s)? failed", r"assertion error", r"failing test", r"unit test(s)? (fail|did not pass)"),
    "regression_introduced": (r"broke(n)? (a |an )?(existing|other|previously)", r"regression", r"previously passing"),
    "instruction_violation": (r"violated (the )?instruction", r"did not follow", r"constraint (not )?(met|satisfied)"),
    "formatting": (r"format(ting)? (error|mismatch)", r"malformed (output|json|xml)", r"wrong (format|schema)"),
    "context_loss": (r"lost (track|context)", r"forgot (the|earlier)", r"context (overflow|exceeded)"),
    "citation_error": (r"citation (missing|incorrect|invalid)", r"fabricated (citation|reference|source)"),
    "excessive_verbosity": (r"too (long|verbose)", r"unnecessary (detail|elaboration)"),
    "under_explanation": (r"too (short|terse)", r"insufficient (explanation|justification)"),
    "inability_to_abstain": (r"should have (abstained|declined)", r"answered (despite|without) (knowing|evidence)"),
    "poor_planning": (r"poor (plan|planning|decomposition)", r"no (plan|strategy)", r"disorganiz(ed|ation)"),
    "failed_recovery": (r"did not recover", r"repeated (the )?(same )?(error|mistake|tool call)", r"stuck in (a )?loop"),
    "reward_hacking": (r"gamed (the )?(test|metric|checker)", r"hard[- ]cod(ed|ing) (the )?(answer|expected)", r"exploited (the )?(grader|checker)"),
    "falsely_claimed_completion": (r"claim(ed|s) (it |the task is )?(is )?(done|complete|passing)", r"falsely (claim|report)"),
}


class FailureTaxonomy:
    """Classifies failures into named categories; new categories are
    registered explicitly with their patterns."""

    def __init__(self, categories: Mapping[str, tuple[str, ...]] | None = None) -> None:
        self._patterns: dict[str, tuple[re.Pattern[str], ...]] = {}
        for category, patterns in (categories or CATEGORY_KEYWORDS).items():
            self.register_category(category, patterns)

    def register_category(self, category: str, patterns: Iterable[str]) -> None:
        """Register (or extend) a category. Patterns are case-insensitive
        regexes matched against failure evidence text."""
        compiled = tuple(re.compile(p, re.IGNORECASE) for p in patterns)
        existing = self._patterns.get(category, ())
        self._patterns[category] = existing + compiled

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted(self._patterns))

    def classify(self, evidence: str) -> tuple[str, ...]:
        """All matching categories for one failure's evidence text, most
        specific evidence first. Empty when nothing matches (the failure is
        then recorded as ``unclassified`` -- visible, not silently dropped)."""
        if not evidence:
            return ()
        matches = [
            category
            for category, patterns in self._patterns.items()
            if any(p.search(evidence) for p in patterns)
        ]
        return tuple(sorted(matches))
