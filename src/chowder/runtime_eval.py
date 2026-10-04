"""Live multi-task runtime evaluation for text-worker promotion evidence.

The benchmark drives a frozen model through deterministic workspaces using the
same read/write/test/final-report protocol the agent uses.  Tasks are grouped
into families (imports, stateful bugs, multi-file repairs, failed first fixes,
misleading test output) so a harness change can be judged on *where* it helps
rather than only on an aggregate reward.

Two harness mechanisms are offered, independently and in combination:

* ``state_aware`` -- the model is shown the workspace's *actual* file list,
  refreshed from live state every turn, and a missing read reports the real
  available paths.  Nothing is hard-coded: the listing comes from the workspace
  dict itself.
* ``recovery`` -- an explicit post-failure state is tracked.  After a red test
  run the harness enters recovery, tells the model a new write is required, and
  refuses (without executing) any further ``run_tests`` until the workspace
  changes.

Neither mechanism names a file or a solution.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Mapping

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a workspace file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run workspace tests.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]
_SPAN = re.compile(r"<tool_call>([a-z_]+)(.*?)</tool_call>", re.S)
_ARG = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.S)
_MISSING = re.compile(r"ERROR:\s*no such file", re.I)
_GREEN = re.compile(r"\b([1-9]\d*)\s+passed\b", re.I)
_RED = re.compile(r"\b(failed|error|traceback|exception)\b", re.I)

# Per-tool execution cost: reads are cheap, a test run actually executes code.
_TOOL_COST = {"read_file": 1, "write_file": 2, "run_tests": 5}

# Metric keys published by run_live_benchmark. Evaluators whitelist exactly
# this set so a worker result can be validated against its configured protocol.
RUNTIME_METRIC_KEYS = (
    "runtime_reward",
    "runtime_green_rate",
    "runtime_nonexistent_read_rate",
    "runtime_premature_completion_rate",
    "runtime_repeated_action_rate",
    "runtime_execution_cost",
    "policy_tokens",
    "prompt_tokens",
    "total_tokens",
    "prompt_message_chars",
    "policy_message_chars",
)


@dataclass(frozen=True)
class RuntimeTask:
    name: str
    goal: str
    initial: dict[str, str]
    target: str
    expected_fix: str
    test_count: int
    test_success: str
    family: str = "single_file"
    # When set, `check(workspace)` returns the raw test observation.  This lets
    # a task require multi-file fixes, partial-credit suites, or misleading
    # output without naming any solution in the harness.
    check: Callable[[Mapping[str, str]], str] | None = field(default=None, compare=False)


def _contains(*markers: str) -> Callable[[str], bool]:
    return lambda source: any(marker in source for marker in markers)


def _check(
    target: str,
    valid: Callable[[str], bool] | tuple[str, ...],
    *,
    success: str,
    failure: str,
    partial: str | None = None,
    partial_markers: tuple[str, ...] = (),
) -> Callable[[Mapping[str, str]], str]:
    predicate = valid if callable(valid) else _contains(*valid)

    def check(files: Mapping[str, str]) -> str:
        source = files.get(target, "")
        if predicate(source):
            return success
        if partial and any(marker in source for marker in partial_markers):
            return partial
        return failure

    return check


def _multi_check(
    success: str,
    failure: str,
    predicate: Callable[[Mapping[str, str]], bool],
    *,
    partial: str | None = None,
    partial_predicate: Callable[[Mapping[str, str]], bool] | None = None,
) -> Callable[[Mapping[str, str]], str]:
    def check(files: Mapping[str, str]) -> str:
        if predicate(files):
            return success
        if partial and partial_predicate is not None and partial_predicate(files):
            return partial
        return failure

    return check


# ---------------------------------------------------------------------------
# Development split (TASKS): the tasks harness edits are allowed to observe.
# ---------------------------------------------------------------------------

_SINGLE_FILE = (
    RuntimeTask("version_parser", "Repair version.py, run its tests, and report only after green.", {"version.py": "def parse_version(s):\n    return tuple(int(p) for p in s.split('.'))\n"}, "version.py", "while len(parts) < 3", 2, "2 passed"),
    # Explicit check: the bare marker "return sum(values)" is a substring of the
    # buggy "return sum(values) - 1", which previously made this task green with
    # no fix at all.
    RuntimeTask("sum_text", "Repair sum_text.py, run its tests, and report only after green.", {"sum_text.py": "def total(values):\n    return sum(values) - 1\n"}, "sum_text.py", "return sum(values)", 3, "3 passed", "single_file",
               _check("sum_text.py", lambda source: "return sum(values)" in source and "- 1" not in source,
                      success="3 passed", failure="FAILED 3 - total")),
    RuntimeTask("slugify", "Repair slugify.py, run its tests, and report only after green.", {"slugify.py": "def slugify(value):\n    return value\n"}, "slugify.py", ".strip().lower()", 2, "2 passed"),
)

_IMPORT_TASKS = (
    RuntimeTask(
        "import_rename",
        "Repair the failing import so main.py loads, then run the tests.",
        {
            "helpers.py": "def normalise(text):\n    return text.strip().lower()\n",
            "main.py": "from helpers import normalize\n\ndef title(text):\n    return normalize(text).title()\n",
        },
        "main.py", "", 2, "2 passed", "imports",
        _multi_check(
            "2 passed",
            "FAILED 2 - ImportError: cannot import name 'normalize'",
            lambda files: _contains("from helpers import normalise", "import normalise as normalize", "from helpers import normalise as normalize")(files.get("main.py", ""))
            or _contains("def normalize", "normalize = normalise")(files.get("helpers.py", "")),
        ),
    ),
    RuntimeTask(
        "import_missing",
        "Repair report.py so it imports what it uses, then run the tests.",
        {"report.py": "def build(rows):\n    return OrderedDict((row['id'], row) for row in rows)\n"},
        "report.py", "", 2, "2 passed", "imports",
        _check("report.py", ("from collections import OrderedDict", "import collections"),
               success="2 passed", failure="FAILED 2 - NameError: name 'OrderedDict' is not defined"),
    ),
    RuntimeTask(
        "import_relative",
        "Repair the package import in pkg/app.py, then run the tests.",
        {
            "pkg/__init__.py": "",
            "pkg/util.py": "def clamp(v, lo, hi):\n    return max(lo, min(v, hi))\n",
            "pkg/app.py": "from util import clamp\n\ndef bounded(v):\n    return clamp(v, 0, 10)\n",
        },
        "pkg/app.py", "", 2, "2 passed", "imports",
        _check("pkg/app.py", ("from .util import clamp", "from pkg.util import clamp", "import pkg.util"),
               success="2 passed", failure="FAILED 2 - ModuleNotFoundError: No module named 'util'"),
    ),
    RuntimeTask(
        "import_alias",
        "Repair parse.py so the aliased module is called correctly, then run the tests.",
        {"parse.py": "import json as jsonlib\n\ndef load(text):\n    return json.loads(text)\n"},
        "parse.py", "", 2, "2 passed", "imports",
        _check("parse.py", ("jsonlib.loads",), success="2 passed",
               failure="FAILED 2 - NameError: name 'json' is not defined"),
    ),
)

_STATEFUL_TASKS = (
    RuntimeTask(
        "mutable_default",
        "Repair the shared-default bug in tags.py, then run the tests.",
        {"tags.py": "def append_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags\n"},
        "tags.py", "", 3, "3 passed", "stateful",
        _check("tags.py", ("tags=None", "tags = None", "tags: list | None = None"),
               success="3 passed", failure="FAILED 3 - shared mutable default"),
    ),
    RuntimeTask(
        "class_counter",
        "Repair the counter so each instance keeps its own total, then run the tests.",
        {"counter.py": "class Counter:\n    total = 0\n\n    def add(self, n):\n        Counter.total += n\n        return Counter.total\n"},
        "counter.py", "", 2, "2 passed", "stateful",
        _check("counter.py", ("self.total", "def __init__"),
               success="2 passed", failure="FAILED 2 - shared class state"),
    ),
    RuntimeTask(
        "generator_reuse",
        "Repair stream.py so twice() returns all values, then run the tests.",
        {"stream.py": "def numbers():\n    return (i for i in range(3))\n\ndef twice():\n    gen = numbers()\n    return list(gen) + list(gen)\n"},
        "stream.py", "", 2, "2 passed", "stateful",
        _check("stream.py", ("list(range(3))", "return [0, 1, 2]", "tuple(range(3))", "list(numbers())"),
               success="2 passed", failure="FAILED 2 - exhausted generator"),
    ),
    RuntimeTask(
        "shared_memo",
        "Repair the cache key in memo.py, then run the tests.",
        {"memo.py": "CACHE = {}\n\ndef scale(value, factor):\n    if value in CACHE:\n        return CACHE[value]\n    CACHE[value] = value * factor\n    return CACHE[value]\n"},
        "memo.py", "", 2, "2 passed", "stateful",
        _check("memo.py", ("CACHE[(value, factor)]", "(value, factor) in CACHE", "value, factor in CACHE"),
               success="2 passed", failure="FAILED 2 - stale cache entry"),
    ),
    RuntimeTask(
        "accumulator_reset",
        "Repair record() so each call is independent, then run the tests.",
        {"stats.py": "TOTALS = []\n\ndef record(n):\n    TOTALS.append(n)\n    return sum(TOTALS)\n"},
        "stats.py", "", 2, "2 passed", "stateful",
        _check("stats.py", (r"return n\b",), success="2 passed",
               failure="FAILED 2 - accumulator not reset"),
    ),
)

_MULTI_FILE_TASKS = (
    RuntimeTask(
        "two_file_fix",
        "Repair api.py and whatever it depends on so quadruple is correct, then run the tests.",
        {
            "mathx.py": "def double(n):\n    return n + 2\n",
            "api.py": "from mathx import double\n\ndef quadruple(n):\n    return double(double(n))\n",
        },
        "api.py", "", 2, "2 passed", "multi_file",
        _multi_check(
            "2 passed",
            "FAILED 2 - double is wrong",
            lambda files: "return n * 2" in files.get("mathx.py", "") and "quadruple" in files.get("api.py", ""),
            partial="1 failed, 1 passed",
            partial_predicate=lambda files: "quadruple" in files.get("api.py", ""),
        ),
    ),
    RuntimeTask(
        "helper_and_caller",
        "Repair invoice.py so the converted amount is correct, then run the tests.",
        {
            "fmt.py": "def money(cents):\n    return '$' + f'{cents / 100:.2f}'\n",
            "invoice.py": "from fmt import money\n\ndef line(cents):\n    return money(cents * 100)\n",
        },
        "invoice.py", "", 2, "2 passed", "multi_file",
        _check("invoice.py", ("money(cents)",), success="2 passed",
               failure="FAILED 2 - cents scaled twice"),
    ),
    RuntimeTask(
        "schema_validator",
        "Repair validate.py and its schema so required fields are enforced, then run the tests.",
        {
            "schema.py": "FIELDS = ('id', 'name')\n",
            "validate.py": "from schema import FIELDS\n\ndef valid(row):\n    return all(row)\n",
        },
        "validate.py", "", 2, "2 passed", "multi_file",
        _multi_check(
            "2 passed",
            "FAILED 2 - schema/validator mismatch",
            lambda files: "'email'" in files.get("schema.py", "") and "for key in FIELDS" in files.get("validate.py", ""),
            partial="1 failed, 1 passed",
            partial_predicate=lambda files: "'email'" in files.get("schema.py", ""),
        ),
    ),
    RuntimeTask(
        "config_loader",
        "Repair loader.py and its defaults so missing keys are filled, then run the tests.",
        {
            "defaults.py": "DEFAULTS = {'retries': 2}\n",
            "loader.py": "from defaults import DEFAULTS\n\ndef load(cfg):\n    return cfg\n",
        },
        "loader.py", "", 2, "2 passed", "multi_file",
        _multi_check(
            "2 passed",
            "FAILED 2 - defaults not merged",
            lambda files: "**DEFAULTS" in files.get("loader.py", "") and "'timeout'" in files.get("defaults.py", ""),
            partial="1 failed, 1 passed",
            partial_predicate=lambda files: "**DEFAULTS" in files.get("loader.py", ""),
        ),
    ),
)

_FAILED_FIRST_FIX_TASKS = (
    RuntimeTask(
        "duration_parser",
        "Repair parse_duration in duration.py so plain seconds and '1m30s' both parse, then run the tests.",
        {"duration.py": "def parse_duration(text):\n    return int(text)\n"},
        "duration.py", "", 3, "3 passed", "failed_first_fix",
        _check("duration.py", ("'m' in text", "* 60", "split('m')", "// 60"),
               success="3 passed", failure="FAILED 3 - duration",
               partial="1 failed, 2 passed", partial_markers=("text[:-1]", "int(text[")),
    ),
    RuntimeTask(
        "url_join",
        "Repair join() in url.py so path separators are handled, then run the tests.",
        {"url.py": "def join(base, path):\n    return base + path\n"},
        "url.py", "", 3, "3 passed", "failed_first_fix",
        _check("url.py", ("rstrip('/')", "lstrip('/')", "strip('/')"),
               success="3 passed", failure="FAILED 3 - url",
               partial="1 failed, 2 passed", partial_markers=("'.join", ".join([")),
    ),
    RuntimeTask(
        "csv_escape",
        "Repair escape() in csvx.py so quoted fields survive, then run the tests.",
        {"csvx.py": "def escape(value):\n    return value\n"},
        "csvx.py", "", 3, "3 passed", "failed_first_fix",
        _check("csvx.py", ("import csv", "replace("),
               success="3 passed", failure="FAILED 3 - csv escape"),
    ),
    RuntimeTask(
        "line_wrap",
        "Repair wrap() in wrap.py so long lines split at spaces, then run the tests.",
        {"wrap.py": "def wrap(text, width):\n    return text\n"},
        "wrap.py", "", 2, "2 passed", "failed_first_fix",
        _check("wrap.py", ("textwrap", "split()", "for word in"),
               success="2 passed", failure="FAILED 2 - wrap",
               partial="1 failed, 1 passed", partial_markers=("[:width]",)),
    ),
)

_MISLEADING_TASKS = (
    RuntimeTask(
        "misleading_zero_passed",
        "Repair add() in calc.py, then run the tests.",
        {"calc.py": "def add(a, b):\n    return a - b\n"},
        "calc.py", "", 2, "2 passed", "misleading_output",
        _check("calc.py", ("return a + b",), success="2 passed",
               failure="0 passed, 2 failed"),
    ),
    RuntimeTask(
        "misleading_partial_pass",
        "Repair peek() in queue.py so it does not consume the item, then run the tests.",
        {"queue.py": "def peek(items):\n    return items.pop(0)\n"},
        "queue.py", "", 2, "2 passed", "misleading_output",
        _check("queue.py", ("items[0]",), success="2 passed",
               failure="1 passed, 2 failed\nsee the traceback above"),
    ),
    RuntimeTask(
        "misleading_error_trace",
        "Repair parse() in loader_bug.py so empty input is handled, then run the tests.",
        {"loader_bug.py": "def parse(text):\n    return int(text)\n"},
        "loader_bug.py", "", 2, "2 passed", "misleading_output",
        _check("loader_bug.py", ("if not text", "strip()", "try:"),
               success="2 passed",
               failure="2 passed\nERROR collecting extra_tests.py\nTraceback (most recent call last):"),
    ),
    RuntimeTask(
        "misleading_stale_cache",
        "Repair put() and get() in cache.py, then run the tests.",
        {"cache.py": "STORE = {}\n\ndef put(key, value):\n    return None\n\ndef get(key):\n    return STORE[key]\n"},
        "cache.py", "", 2, "2 passed", "misleading_output",
        _multi_check(
            "2 passed",
            "1 failed, 1 passed\n(report generated from a stale cache)",
            lambda files: "STORE[key] = value" in files.get("cache.py", "") and "STORE.get(key)" in files.get("cache.py", ""),
            partial="1 failed, 1 passed\n(report generated from a stale cache)",
            partial_predicate=lambda files: "STORE.get(key)" in files.get("cache.py", ""),
        ),
    ),
)

def _wrong_second_fix_check(
    target: str,
    *,
    success: str,
    partial: str,
    failure: str,
    correct: Callable[[str], bool],
    common_wrong: Callable[[str], bool],
) -> Callable[[Mapping[str, str]], str]:
    """Model a partial first patch whose red boundary test requires a correction."""
    def check(files: Mapping[str, str]) -> str:
        source = files.get(target, "")
        if correct(source):
            return success
        if common_wrong(source):
            return partial
        return failure

    return check


_WRONG_SECOND_FIX_TASKS = (
    RuntimeTask(
        "retry_boundary_second_fix",
        "Repair should_retry() in retry.py. It must permit retries below the configured limit and stop at the limit; run tests and report only after green.",
        {"retry.py": "def should_retry(attempt, limit):\n    return attempt > limit\n"},
        "retry.py", "", 3, "3 passed", "wrong_second_fix",
        _wrong_second_fix_check(
            "retry.py",
            success="3 passed",
            partial="FAILED 1, 2 passed - retry boundary",
            failure="FAILED 3 - retry policy",
            correct=lambda source: "attempt < limit" in source or bool(re.search(r"attempt\s*<=\s*limit\s*-\s*1", source)),
            common_wrong=lambda source: "attempt <= limit" in source and not bool(re.search(r"attempt\s*<=\s*limit\s*-\s*1", source)),
        ),
    ),
    RuntimeTask(
        "inclusive_window_second_fix",
        "Repair window() in window.py so the requested end index is included, including at the final element; run tests and report only after green.",
        {"window.py": "def window(values, start, end):\n    return values[start:end]\n"},
        "window.py", "", 3, "3 passed", "wrong_second_fix",
        _wrong_second_fix_check(
            "window.py",
            success="3 passed",
            partial="FAILED 1, 2 passed - final endpoint",
            failure="FAILED 3 - inclusive window",
            correct=lambda source: "end + 1" in source or "end+1" in source,
            common_wrong=lambda source: "min(end, len(values))" in source or "min(end,len(values))" in source,
        ),
    ),
)

TASKS = _SINGLE_FILE + _IMPORT_TASKS + _STATEFUL_TASKS + _MULTI_FILE_TASKS + _FAILED_FIRST_FIX_TASKS + _MISLEADING_TASKS + _WRONG_SECOND_FIX_TASKS

# ---------------------------------------------------------------------------
# Held-out split (HELDOUT_TASKS): different names, files, bug shapes, and test
# vocabularies. Never allowed to leak into a harness proposal.
# ---------------------------------------------------------------------------

_HELDOUT_SINGLE = (
    RuntimeTask("config_defaults", "Repair config_defaults.py and verify its behavior.", {"config_defaults.py": "def with_defaults(config):\n    return {}\n"}, "config_defaults.py", "setdefault", 2, "2 passed"),
    RuntimeTask("retry_budget", "Repair retry_policy.py and verify its retry budget.", {"retry_policy.py": "def should_retry(attempt):\n    return attempt == 0\n"}, "retry_policy.py", "attempt < 3", 3, "3 passed"),
    RuntimeTask("csv_active_rows", "Repair csv_filter.py and verify active-row filtering.", {"csv_filter.py": "def active_rows(rows):\n    return list(rows)\n"}, "csv_filter.py", "row.get('active')", 2, "2 passed"),
    RuntimeTask("clamp_ratio", "Repair clamp_ratio.py so values stay in range.", {"ratio.py": "def clamp_ratio(value):\n    return value\n"}, "ratio.py", "", 2, "2 passed", "single_file",
               _check("ratio.py", ("min(", "max("), success="2 passed", failure="FAILED 2 - ratio out of range")),
)

_HELDOUT_IMPORTS = (
    RuntimeTask(
        "import_pkg_reexport",
        "Repair lib/__init__.py so the package exports shout, then run the tests.",
        {"lib/__init__.py": "", "lib/strings.py": "def shout(text):\n    return text.upper()\n", "app2.py": "import lib\n\ndef banner(text):\n    return lib.shout(text)\n"},
        "lib/__init__.py", "", 2, "2 passed", "imports",
        _check("lib/__init__.py", ("from .strings import shout", "from lib.strings import shout", "import strings"),
               success="2 passed", failure="FAILED 2 - AttributeError: module 'lib' has no attribute 'shout'"),
    ),
    RuntimeTask(
        "import_typo",
        "Repair the misspelled import in summary.py, then run the tests.",
        {"maths.py": "def mean(values):\n    return sum(values) / len(values)\n", "summary.py": "from maths import mena\n\ndef average(values):\n    return mena(values)\n"},
        "summary.py", "", 2, "2 passed", "imports",
        _multi_check(
            "2 passed",
            "FAILED 2 - ImportError: cannot import name 'mena'",
            lambda files: "import mean" in files.get("summary.py", "") and "mena" not in files.get("summary.py", ""),
        ),
    ),
)

_HELDOUT_STATEFUL = (
    RuntimeTask("mutable_kwargs", "Repair build() in opts.py so options are not shared.", {"opts.py": "def build(options={}):\n    options['ready'] = True\n    return options\n"}, "opts.py", "", 2, "2 passed", "stateful",
               _check("opts.py", ("options=None", "options = None"), success="2 passed", failure="FAILED 2 - shared options")),
    RuntimeTask("shared_buffer", "Repair collect() in buffer.py so out is not shared.", {"buffer.py": "def collect(items, out=[]):\n    out.extend(items)\n    return out\n"}, "buffer.py", "", 2, "2 passed", "stateful",
               _check("buffer.py", ("out=None", "out = None"), success="2 passed", failure="FAILED 2 - shared buffer")),
)

_HELDOUT_MULTI = (
    RuntimeTask(
        "handler_and_registry",
        "Repair the registry so decorated handlers are actually stored, then run the tests.",
        {"registry.py": "HANDLERS = {}\n\ndef register(name):\n    def deco(fn):\n        return fn\n    return deco\n", "handler.py": "from registry import register\n\n@register('ping')\ndef ping():\n    return 'pong'\n"},
        "handler.py", "", 2, "2 passed", "multi_file",
        _multi_check("2 passed", "FAILED 2 - handler not registered",
                     lambda files: "HANDLERS[name] = fn" in files.get("registry.py", ""),
                     partial="1 failed, 1 passed",
                     partial_predicate=lambda files: "deco" in files.get("registry.py", "")),
    ),
    RuntimeTask(
        "template_and_renderer",
        "Repair render.py so the placeholder is filled, then run the tests.",
        {"template.py": "TEMPLATE = 'Hello {name}'\n", "render.py": "from template import TEMPLATE\n\ndef render(name):\n    return TEMPLATE\n"},
        "render.py", "", 2, "2 passed", "multi_file",
        _multi_check("2 passed", "FAILED 2 - placeholder not filled",
                     lambda files: ".format(" in files.get("render.py", ""),
                     partial="1 failed, 1 passed",
                     partial_predicate=lambda files: "TEMPLATE" in files.get("render.py", "")),
    ),
    RuntimeTask(
        "merge_and_validate",
        "Repair merge.py and its defaults so configs are merged, then run the tests.",
        {"defaults2.py": "DEFAULTS = {}\n", "merge.py": "from defaults2 import DEFAULTS\n\ndef merge(cfg):\n    return cfg\n"},
        "merge.py", "", 2, "2 passed", "multi_file",
        _multi_check("2 passed", "FAILED 2 - merge",
                     lambda files: ("update(" in files.get("merge.py", "") or "**DEFAULTS" in files.get("merge.py", "")) and "'level'" in files.get("defaults2.py", ""),
                     partial="1 failed, 1 passed",
                     partial_predicate=lambda files: "**DEFAULTS" in files.get("merge.py", "")),
    ),
)

_HELDOUT_FAILED_FIRST_FIX = (
    RuntimeTask(
        "size_parser",
        "Repair parse_size in size.py so bytes and kilobytes both parse, then run the tests.",
        {"size.py": "def parse_size(text):\n    return int(text)\n"},
        "size.py", "", 3, "3 passed", "failed_first_fix",
        _check("size.py", ("* 1024", "'kb' in text", "lower()", "// 1024"),
               success="3 passed", failure="FAILED 3 - size",
               partial="1 failed, 2 passed", partial_markers=("text[:-1]", "int(text[")),
    ),
)

_HELDOUT_MISLEADING = (
    RuntimeTask(
        "misleading_stale_report",
        "Repair summarize() in report2.py so only active rows are counted, then run the tests.",
        {"report2.py": "def summarize(rows):\n    return len(rows)\n"},
        "report2.py", "", 2, "2 passed", "misleading_output",
        _check("report2.py", ("row.get('active')", "if row.get('active')"),
               success="2 passed", failure="1 failed, 1 passed\n(report generated from a stale cache)"),
    ),
)

_HELDOUT_WRONG_SECOND_FIX = (
    RuntimeTask(
        "retry_cap_second_fix",
        "Repair permitted() in retry_policy2.py so retries continue below the cap but stop at it; run tests and report only after green.",
        {"retry_policy2.py": "def permitted(retries, max_retries): return retries < 0"},
        "retry_policy2.py", "", 3, "3 passed", "wrong_second_fix",
        _wrong_second_fix_check(
            "retry_policy2.py",
            success="3 passed",
            partial="FAILED 1, 2 passed - retry cap boundary",
            failure="FAILED 3 - retry cap",
            correct=lambda source: "retries < max_retries" in source,
            common_wrong=lambda source: "retries <= max_retries" in source,
        ),
    ),
    RuntimeTask(
        "inclusive_prefix_second_fix",
        "Repair prefix_through() in prefix.py so the requested final index is included; run tests and report only after green.",
        {"prefix.py": "def prefix_through(items, index): return items[:index]"},
        "prefix.py", "", 3, "3 passed", "wrong_second_fix",
        _wrong_second_fix_check(
            "prefix.py",
            success="3 passed",
            partial="FAILED 1, 2 passed - inclusive prefix boundary",
            failure="FAILED 3 - prefix endpoint",
            correct=lambda source: bool(re.search(r"index\s*\+\s*1", source)),
            common_wrong=lambda source: "min(index, len(items))" in source or "min(index,len(items))" in source,
        ),
    ),
)

HELDOUT_TASKS = (
    _HELDOUT_SINGLE + _HELDOUT_IMPORTS + _HELDOUT_STATEFUL + _HELDOUT_MULTI
    + _HELDOUT_FAILED_FIRST_FIX + _HELDOUT_MISLEADING + _HELDOUT_WRONG_SECOND_FIX
)


def parse_tool_call(text: str) -> tuple[str, dict[str, str]] | None:
    spans = _SPAN.findall(text)
    if len(spans) != 1:
        return None
    name, body = spans[0]
    return name, {key: value for key, value in _ARG.findall(body)}


def _is_green(observation: str) -> bool:
    """A passing suite must report at least one pass and no failure signal.

    Deliberately stricter than a bare ``\\d+ passed`` match: "0 passed" and
    "1 failed, 1 passed" are not green, and an error trace suppresses green even
    when a partial count appears.
    """
    if _RED.search(observation):
        return False
    return bool(_GREEN.search(observation))


def _evaluate_task(task: RuntimeTask, files: Mapping[str, str]) -> str:
    if task.check is not None:
        return task.check(files)
    source = files.get(task.target, "")
    if task.expected_fix and task.expected_fix in source:
        return task.test_success
    return f"FAILED {task.test_count} - implementation"


def _action_signature(name: str, args: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((str(key), str(value)) for key, value in args.items()))


def _score(trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    tools = [row for row in trace if row.get("role") == "tool"]
    reads = [row for row in tools if row.get("tool") == "read_file"]
    writes = [row for row in tools if row.get("tool") == "write_file"]
    tests = [row for row in tools if row.get("tool") == "run_tests"]
    nonexistent = sum(bool(_MISSING.search(str(row.get("observation", "")))) for row in reads)
    green = False
    seen_paths: Counter[str] = Counter()
    action_rewards: list[dict[str, Any]] = []
    repeated = 0
    signatures: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
    execution_cost = 0
    green_seen_so_far = False
    premature = False
    report_after_green = False
    for row in trace:
        role = row.get("role")
        if role == "final_report":
            if green_seen_so_far:
                report_after_green = True
            else:
                premature = True
            continue
        if role == "early_report":
            if not green_seen_so_far:
                premature = True
            continue
        if role != "tool":
            continue
        name = str(row.get("tool", ""))
        observation = str(row.get("observation", ""))
        path = str(row.get("args", {}).get("path", ""))
        # A blocked (synthetic) call never executed: it must not be charged as
        # execution cost or counted as a repeated real action.
        if not row.get("synthetic"):
            signature = _action_signature(name, row.get("args", {}) or {})
            if signatures[signature] > 0:
                repeated += 1
            signatures[signature] += 1
            execution_cost += int(row.get("cost", _TOOL_COST.get(name, 0)))
        if name == "write_file" and row.get("workspace_changed"):
            green_seen_so_far = False
        if name == "read_file" and _MISSING.search(observation):
            event, reward = "nonexistent_read", -4.0
        elif name == "read_file" and seen_paths[path] > 0:
            event, reward = "repeated_read", -2.0
        elif name == "write_file":
            event, reward = "write", 4.0
        elif name == "run_tests" and _is_green(observation):
            event, reward = "observed_green", 8.0
            green_seen_so_far = True
        elif name == "run_tests" and row.get("synthetic"):
            event, reward = "blocked_run", -1.0
        elif name == "run_tests":
            event, reward = "red_test", -1.0
            green_seen_so_far = False
        else:
            event, reward = "neutral", 0.0
        if name == "read_file":
            seen_paths[path] += 1
        action_rewards.append({"tool": name, "event": event, "reward": reward})
    green = green_seen_so_far
    terms = [8.0 if green else -10.0, 4.0 if writes else -8.0]
    if report_after_green:
        terms.extend((2.0, 1.0))
    if nonexistent:
        terms.append(-4.0 * nonexistent)
    if premature:
        terms.append(-3.0)
    if repeated:
        terms.append(-1.0 * min(repeated, 4))
    return {
        "reward": sum(terms),
        "green_seen": green_seen_so_far,
        "nonexistent_reads": nonexistent,
        "premature_completion": premature,
        "repeated_actions": repeated,
        "tool_calls": sum(1 for row in tools if not row.get("synthetic")),
        "execution_cost": execution_cost,
        "action_rewards": action_rewards,
    }


_HARNESS_FEATURES: dict[str, frozenset[str]] = {
    "plain": frozenset(),
    "guarded": frozenset({"guarded"}),
    "state_aware_legacy": frozenset({"state_aware", "state_aware_legacy"}),
    "state_aware": frozenset({"state_aware", "state_aware_compact"}),
    "state_aware+recovery_legacy": frozenset({"state_aware", "state_aware_legacy", "recovery"}),
    "recovery": frozenset({"recovery"}),
    "state_aware+recovery": frozenset({"state_aware", "state_aware_compact", "recovery"}),
}


def _features(harness: str) -> frozenset[str]:
    try:
        return _HARNESS_FEATURES[harness]
    except KeyError:
        raise ValueError(f"unknown runtime harness: {harness!r}") from None


def _workspace_listing(files: Mapping[str, str]) -> str:
    return ", ".join(sorted(files)) or "(no files)"


def _state_message(files: Mapping[str, str], *, compact: bool = True) -> dict[str, str]:
    listing = _workspace_listing(files)
    if compact:
        content = f"Files: {listing}. Read listed paths; test before success."
    else:
        content = (
            "Workspace state: the files that currently exist are "
            f"{listing}. Read only paths from this list, and verify with "
            "run_tests before reporting success."
        )
    return {"role": "system", "content": content}


def _run_task(
    generate: Callable[[list[dict[str, str]]], str],
    task: RuntimeTask,
    max_turns: int,
    *,
    harness: str = "plain",
) -> dict[str, Any]:
    features = _features(harness)
    files = dict(task.initial)
    messages: list[dict[str, str]] = []
    if "state_aware" in features:
        messages.append(
            _state_message(
                files,
                compact=("state_aware_compact" in features or "state_aware_legacy" not in features),
            )
        )
    elif "guarded" in features:
        messages.append({
            "role": "system",
            "content": (
                "Runtime repair policy: inspect only the named target file, never read tests or "
                "unknown paths, write the target before verification, and call run_tests once "
                "after the write. A success report is valid only after a passing observation."
            ),
        })
    messages.append({"role": "user", "content": f"Target file: {task.target}. {task.goal}"})
    trace: list[dict[str, Any]] = []
    green = False
    awaiting_write = False
    recovery_target = task.target
    for turn in range(max_turns):
        if "state_aware" in features and messages and messages[0].get("role") == "system":
            messages[0] = _state_message(
                files,
                compact=("state_aware_compact" in features or "state_aware_legacy" not in features),
            )
        prompt_messages = [dict(message) for message in messages]
        raw = generate(messages)
        trace.append(
            {
                "turn": turn,
                "role": "assistant",
                "text": raw,
                "prompt_messages": prompt_messages,
            }
        )
        call = parse_tool_call(raw)
        if call is None:
            if green:
                trace.append({"turn": turn, "role": "final_report", "text": raw})
                break
            if features & {"guarded", "recovery", "state_aware"}:
                trace.append({"turn": turn, "role": "early_report", "text": raw})
                messages.append({"role": "assistant", "content": raw})
                if "recovery" in features and awaiting_write:
                    messages.append({"role": "user", "content": (
                        f"Recovery state: the last write to {recovery_target} failed verification. "
                        "Issue a corrected write_file; run_tests is disabled until the file changes."
                    )})
                else:
                    messages.append({"role": "user", "content": "Harness gate: no green observation yet. Continue with the target-file repair and run_tests."})
                continue
            trace.append({"turn": turn, "role": "final_report", "text": raw})
            break
        name, args = call
        path = args.get("path", "")
        synthetic = False
        cost = _TOOL_COST.get(name, 0)
        if name == "read_file":
            path = args.get("path", "")
            if path in files:
                observation = files[path]
            else:
                if "state_aware" in features:
                    observation = (
                        f"ERROR: no such file: {path}. Files: {_workspace_listing(files)}"
                    )
                else:
                    observation = f"ERROR: no such file: {path}"
        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            changed = path not in files or files[path] != content
            files[path] = content
            observation = f"OK {path} written" if changed else f"OK {path} unchanged"
            if changed:
                green = False
                awaiting_write = False
                recovery_target = path
        elif name == "run_tests":
            if "recovery" in features and awaiting_write:
                synthetic = True
                cost = 0
                observation = (
                    "ERROR: tests not executed -- the workspace is unchanged since the last "
                    f"failure. Write a corrected {recovery_target} before running tests again."
                )
            else:
                observation = _evaluate_task(task, files)
                green = _is_green(observation)
                if green:
                    awaiting_write = False
                else:
                    awaiting_write = "recovery" in features
        else:
            observation = f"ERROR: unknown tool {name}"
        trace.append({
            "turn": turn, "role": "tool", "tool": name, "args": args,
            "observation": observation, "cost": cost,
            **({"workspace_changed": changed} if name == "write_file" else {}),
            **({"synthetic": True} if synthetic else {}),
        })
        messages.extend(({"role": "assistant", "content": raw}, {"role": "tool", "content": observation}))

        if "recovery" in features and name == "run_tests" and awaiting_write and not synthetic:
            messages.append({"role": "user", "content": (
                f"Recovery state: verification failed after the last write to {recovery_target}. "
                "Diagnose the failure and issue a corrected write_file; run_tests is blocked until "
                "the file changes."
            )})
        elif "guarded" in features and name == "read_file" and path != task.target:
            messages.append({"role": "user", "content": f"Harness gate: read only {task.target}; do not inspect unknown or test paths."})
        elif "guarded" in features and name == "run_tests" and not _is_green(observation):
            messages.append({"role": "user", "content": "Harness gate: the suite is still red. Do not repeat run_tests; write a corrected target implementation."})
    else:
        trace.append({"turn": max_turns, "role": "budget_exhausted", "text": "Budget exhausted without a model final report."})
    scored = _score(trace)
    return {"task": task.name, "family": task.family, "trace": trace, **scored}


def make_transformers_generate(tokenizer: Any, model: Any, *, max_new_tokens: int, device: Any):
    """Bind the live benchmark to an already-loaded Transformers model."""
    def generate(messages: list[dict[str, str]]) -> str:
        import torch

        rendered = tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        generate.policy_tokens += max(0, int(generated.shape[1] - prompt_tokens))
        generate.prompt_tokens += int(encoded["input_ids"].numel())
        return tokenizer.decode(generated[0, prompt_tokens:], skip_special_tokens=False)

    generate.policy_tokens = 0
    generate.prompt_tokens = 0
    return generate


def _split_metrics(
    rows: list[Mapping[str, Any]], *, policy_tokens: int, prompt_tokens: int,
    prompt_message_chars: int, policy_message_chars: int,
) -> dict[str, float]:
    return {
        "runtime_reward": mean(float(row["reward"]) for row in rows),
        "runtime_green_rate": mean(bool(row["green_seen"]) for row in rows),
        "runtime_nonexistent_read_rate": mean(float(row["nonexistent_reads"]) for row in rows),
        "runtime_premature_completion_rate": mean(bool(row["premature_completion"]) for row in rows),
        "runtime_repeated_action_rate": mean(float(row["repeated_actions"]) for row in rows),
        "runtime_execution_cost": mean(float(row["execution_cost"]) for row in rows),
        "policy_tokens": float(policy_tokens),
        "prompt_tokens": float(prompt_tokens),
        "total_tokens": float(policy_tokens + prompt_tokens),
        "prompt_message_chars": float(prompt_message_chars),
        "policy_message_chars": float(policy_message_chars),
    }


def run_live_benchmark(
    generate: Callable[[list[dict[str, str]]], str],
    *,
    max_turns: int = 8,
    harness: str = "plain",
    tasks: tuple[RuntimeTask, ...] = TASKS,
    split: str = "evolve",
) -> dict[str, Any]:
    """Drive all deterministic workspaces and return metrics plus raw traces."""
    if max_turns < 1:
        raise ValueError("runtime max_turns must be positive")
    if not tasks:
        raise ValueError("runtime benchmark requires at least one task")
    _features(harness)  # validate before any generation
    before_tokens = int(getattr(generate, "policy_tokens", 0))
    before_prompt_tokens = int(getattr(generate, "prompt_tokens", 0))
    rows = [_run_task(generate, task, max_turns, harness=harness) for task in tasks]
    policy_tokens = int(getattr(generate, "policy_tokens", 0)) - before_tokens
    prompt_tokens = int(getattr(generate, "prompt_tokens", 0)) - before_prompt_tokens
    prompt_message_chars = sum(
        len(str(message.get("content", "")))
        for row in rows for entry in row["trace"] if entry.get("role") == "assistant"
        for message in entry.get("prompt_messages", ())
    )
    policy_message_chars = sum(
        len(str(entry.get("text", "")))
        for row in rows for entry in row["trace"] if entry.get("role") == "assistant"
    )
    families: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        families.setdefault(str(row.get("family", "unknown")), []).append(row)
    return {
        "split": split,
        "harness": harness,
        "metrics": _split_metrics(
            rows, policy_tokens=policy_tokens, prompt_tokens=prompt_tokens,
            prompt_message_chars=prompt_message_chars,
            policy_message_chars=policy_message_chars,
        ),
        "families": {
            name: {
                "tasks": len(group),
                "green_rate": mean(bool(row["green_seen"]) for row in group),
                "reward": mean(float(row["reward"]) for row in group),
                "repeated_actions": mean(float(row["repeated_actions"]) for row in group),
                "nonexistent_reads": mean(float(row["nonexistent_reads"]) for row in group),
            }
            for name, group in sorted(families.items())
        },
        "tasks": rows,
    }
