"""Chowder evaluation layer: normalized results over external harnesses.

Adapters translate Inspect AI, lm-evaluation-harness, benchmark-native agent
suites, and Chowder-native diagnostics into one schema. Scores keep their
measurement kind (raw model vs agent harness), honest non-measurements are
first-class, and the scoreboard renders deltas and frontier context without
faking comparability.
"""

from .adapters import (
    ADAPTERS,
    AdapterUnavailable,
    ChowderCustomEvalAdapter,
    InspectAdapter,
    LMEvalAdapter,
    NativeAgentBenchmarkAdapter,
    unavailable_run,
)
from .result import (
    AGENT_HARNESS,
    NOT_APPLICABLE_MODALITY,
    RAW_MODEL,
    SUPPORTED,
    UNSUPPORTED_HARNESS,
    BenchmarkRun,
    EvalAdapter,
    EvalReport,
)
from .runner import EvalRunner, RunnerHooks, Scoreboard, category_aggregates

__all__ = [
    "ADAPTERS",
    "AGENT_HARNESS",
    "AdapterUnavailable",
    "BenchmarkRun",
    "ChowderCustomEvalAdapter",
    "EvalAdapter",
    "EvalReport",
    "EvalRunner",
    "InspectAdapter",
    "LMEvalAdapter",
    "NativeAgentBenchmarkAdapter",
    "NOT_APPLICABLE_MODALITY",
    "RAW_MODEL",
    "RunnerHooks",
    "Scoreboard",
    "SUPPORTED",
    "UNSUPPORTED_HARNESS",
    "category_aggregates",
    "unavailable_run",
]
