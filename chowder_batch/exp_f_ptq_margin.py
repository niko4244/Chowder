"""Experiment F: does PTQ of the small model change harness results or the
Phase-4 logprob-margin confidence signal?

Lane: ``nvidia-modelopt`` **core only** (no ``[hf]`` extra: upstream pins
``transformers>=4.57,<5.15`` while the train lane runs transformers 5.x; see
``pyproject.toml``). Quantization runs through the pure-torch API
(``modelopt.torch.quantization``), which was smoke-verified on CPU with
nvidia-modelopt 0.47.0 / torch 2.11.0+cu128 / transformers 5.16.1.

Fail-closed rules, matching the rest of ``chowder_batch``:

* Every live entry point is GPU-gated; nothing below runs on CPU silently.
* Calibration text must be supplied by the caller (real corpus/dev rows);
  the scaffold never invents calibration data.
* The comparison is paired: both arms must produce rows for exactly the same
  task set, and margins must be present or explicitly ``null`` with a
  ``logprob_error`` -- missing/NaN margins abort the comparison.
* A passing margin bound never licenses the quantized arm: the same paired
  rows carry per-task green outcomes, and the router refuses an arm that loses
  more than the declared fraction of the reference arm's greens
  (``green_retention`` in ``router_rows_from_report``).
* The report contains only measured quantities. No throughput or speedup
  claim is emitted (fake-kernel quantization on this lane does not license
  one; a real serving measurement is a separate, future experiment).

The two questions this experiment answers, in order:

1. Does an mtq-quantized arm change ``chowder.runtime_eval`` results (green
   rate, reward, policy tokens) versus the BF16 arm on identical tasks?
2. Does quantization shift the mean selected-token logprob margin that the
   Phase-4 router calibrates on? A large negative shift invalidates any
   threshold calibrated on BF16 logprobs -- that finding alone gates whether
   a quantized small model may serve the router.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

from exp_e_confidence import (  # noqa: E402
    DEFAULT_MARGIN_SHIFT_TOLERANCE,
    DEFAULT_MAX_GREEN_LOSS_FRACTION,
    calibrate_margin_threshold_per_precision,
    heldout_transfer_gate,
    quantized_arm_admission,
    validate_quantized_green_retention,
    validate_quantized_margin_shift,
    verify_arm_admission,
)

# Whitelisted mtq configs resolvable from mtq.* on nvidia-modelopt 0.47.0.
# The router tolerance these shifts feed lives in exp_e_confidence.
# Keys are experiment arm names; values are the mtq module attribute names.
PTQ_CONFIG_NAMES: dict[str | None, str | None] = {
    "bf16": None,
    "int8_smoothquant": "INT8_SMOOTHQUANT_CFG",
    "int8_weight_only": "INT8_WEIGHT_ONLY_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "nvfp4_default": "NVFP4_DEFAULT_CFG",
}


def resolve_ptq_config(config_name: str | None) -> dict[str, Any] | None:
    """Return the mtq config for an arm name, or None for the BF16 arm.

    Raises ValueError for unknown names and RuntimeError if the named config
    is absent from the installed modelopt (version drift -> fail closed).
    """
    if config_name not in PTQ_CONFIG_NAMES:
        raise ValueError(
            f"unknown PTQ arm {config_name!r}; known arms: {sorted(k for k in PTQ_CONFIG_NAMES if k)}"
        )
    attribute = PTQ_CONFIG_NAMES[config_name]
    if attribute is None:
        return None
    import modelopt.torch.quantization as mtq  # lazy: core import only

    config = getattr(mtq, attribute, None)
    if not isinstance(config, dict):
        raise RuntimeError(
            f"installed nvidia-modelopt does not expose {attribute}; "
            "refusing to substitute a different quantization recipe"
        )
    return config


def quantize_for_arm(
    model: Any, config_name: str, calib_forward_loop: Callable[[Any], None] | None
) -> Any:
    """Quantize ``model`` in place under a whitelisted config; fail closed."""
    config = resolve_ptq_config(config_name)
    if config is None:
        raise ValueError("bf16 arm must not be quantized")
    try:
        import modelopt.torch.quantization as mtq
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "nvidia-modelopt is not installed; the PTQ lane requires "
            '`pip install "nvidia-modelopt>=0.47,<0.48"`'
        ) from exc
    # mtq.quantize inserts quantizer modules in place and returns the model.
    # Callers must hand in a model instance dedicated to this arm.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtq.quantize(model, config, forward_loop=calib_forward_loop)


def _margin_value(row: Mapping[str, Any], arm: str) -> float | None:
    if "margin" not in row:
        raise ValueError(f"{arm} row is missing the margin field")
    margin = row["margin"]
    if margin is None:
        if not str(row.get("logprob_error") or "").strip():
            raise ValueError(f"{arm} margin is null without a logprob_error")
        return None
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(float(margin)):
        raise ValueError(f"{arm} margin must be a finite number or explicitly null")
    return float(margin)


def compare_margin_signal(
    bf16_rows: Sequence[Mapping[str, Any]],
    quant_rows: Sequence[Mapping[str, Any]],
    *,
    quant_config: str,
) -> dict[str, Any]:
    """Pair BF16 vs quantized margin rows and report only measured deltas.

    Each row: ``{"task": str, "margin": float | None, "logprob_error": str | None,
    "correct": bool}``. Both arms must cover exactly the same task set (paired
    design); unpaired, duplicate, or malformed rows raise ValueError.
    """
    for label, rows in (("bf16", bf16_rows), (quant_config, quant_rows)):
        if not rows:
            raise ValueError(f"{label} arm has no margin rows")
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for label, rows in (("bf16", bf16_rows), (quant_config, quant_rows)):
        table: dict[str, dict[str, Any]] = {}
        for row in rows:
            task = str(row.get("task", ""))
            if not task:
                raise ValueError(f"{label} margin row has no task name")
            if task in table:
                raise ValueError(f"{label} arm has duplicate rows for task {task}")
            table[task] = {"margin": _margin_value(row, label), "correct": bool(row.get("correct"))}
        by_arm[label] = table
    bf16_tasks, quant_tasks = set(by_arm["bf16"]), set(by_arm[quant_config])
    if bf16_tasks != quant_tasks:
        unpaired = sorted(bf16_tasks ^ quant_tasks)[:3]
        raise ValueError(f"arms are not paired over the same tasks; first mismatches: {unpaired}")

    tasks = sorted(bf16_tasks)
    per_task = []
    bf16_margins: list[float] = []
    quant_margins: list[float] = []
    bf16_correct = quant_correct = 0
    for task in tasks:
        bf16_margin = by_arm["bf16"][task]["margin"]
        quant_margin = by_arm[quant_config][task]["margin"]
        bf16_correct += by_arm["bf16"][task]["correct"]
        quant_correct += by_arm[quant_config][task]["correct"]
        if bf16_margin is not None and quant_margin is not None:
            shift = quant_margin - bf16_margin
        else:
            shift = None
        per_task.append({
            "task": task,
            "bf16_margin": bf16_margin,
            "quant_margin": quant_margin,
            "margin_shift": shift,
            # Per-arm ground truth, so downstream router rows for each precision
            # carry that arm's own correctness rather than a shared label.
            "bf16_correct": by_arm["bf16"][task]["correct"],
            "quant_correct": by_arm[quant_config][task]["correct"],
        })
        if bf16_margin is not None:
            bf16_margins.append(bf16_margin)
        if quant_margin is not None:
            quant_margins.append(quant_margin)
    n = len(tasks)
    bf16_nulls, quant_nulls = n - len(bf16_margins), n - len(quant_margins)
    return {
        "quant_config": quant_config,
        "n_tasks": n,
        "bf16_mean_margin": sum(bf16_margins) / len(bf16_margins) if bf16_margins else None,
        "quant_mean_margin": sum(quant_margins) / len(quant_margins) if quant_margins else None,
        "mean_margin_shift": (
            (sum(quant_margins) / len(quant_margins)) - (sum(bf16_margins) / len(bf16_margins))
            if bf16_margins and quant_margins else None
        ),
        "bf16_margin_nulls": bf16_nulls,
        "quant_margin_nulls": quant_nulls,
        "bf16_accuracy": bf16_correct / n,
        "quant_accuracy": quant_correct / n,
        "accuracy_delta": (quant_correct - bf16_correct) / n,
        "per_task": per_task,
    }


def _load_causal_lm(model_path: str | Path) -> Any:
    from transformers import AutoModelForCausalLM  # lazy: live-only path

    return AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch_dtype_for_device(), device_map="cuda"
    )


def torch_dtype_for_device() -> str:
    """BF16 weights for the live arms (inference lane on Ampere+)."""
    return "bfloat16"


def _calib_forward_loop(model: Any, texts: Sequence[str], tokenizer: Any) -> Callable[[Any], None]:
    def loop(m: Any) -> None:
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            m(**encoded)

    return loop


def run_ptq_margin_experiment(
    model_path: str | Path,
    *,
    calibration_texts: Sequence[str],
    benchmark_fn: Callable[[Any, str], Mapping[str, Any]],
    ptq_config: str = "int8_smoothquant",
    output_path: str | Path | None = None,
    max_calibration_texts: int = 256,
) -> dict[str, Any]:
    """Run the paired PTQ-vs-BF16 arms and return the measured report.

    ``benchmark_fn(arm_model, arm_name)`` must run the identical task suite on
    the given model and return ``{"margin_rows": [...], ...}`` (additional keys
    are passed through into the report verbatim). The BF16 arm always runs
    first on a freshly loaded model; the quantized arm reloads from
    ``model_path`` again, so mtq's in-place quantization can never contaminate
    the reference arm.
    """
    if resolve_ptq_config(ptq_config) is None:
        raise ValueError("ptq_config must name a quantized arm, not bf16")
    calib_texts = [str(text) for text in calibration_texts if str(text).strip()]
    if not calib_texts:
        raise ValueError(
            "PTQ calibration requires real calibration texts (corpus or dev rows); none supplied"
        )
    calib_texts = calib_texts[:max_calibration_texts]
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment F is a live-GPU experiment; refusing to run on CPU")

    from transformers import AutoTokenizer  # lazy: live-only path

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    def load_arm() -> Any:
        return _load_causal_lm(model_path)

    bf16_model = load_arm()
    try:
        bf16_result = dict(benchmark_fn(bf16_model, "bf16"))
    finally:
        del bf16_model
        torch.cuda.empty_cache()

    quant_model = load_arm()
    try:
        loop = _calib_forward_loop(quant_model, calib_texts, tokenizer)
        quantize_for_arm(quant_model, ptq_config, loop)
        quant_result = dict(benchmark_fn(quant_model, ptq_config))
    finally:
        del quant_model
        torch.cuda.empty_cache()

    report: dict[str, Any] = {
        "experiment": "exp_f_ptq_margin",
        "model_path": str(model_path),
        "ptq_config": ptq_config,
        "n_calibration_texts": len(calib_texts),
        "bf16": {k: v for k, v in bf16_result.items() if k != "margin_rows"},
        "quant": {k: v for k, v in quant_result.items() if k != "margin_rows"},
        "margin_comparison": compare_margin_signal(
            bf16_result["margin_rows"], quant_result["margin_rows"], quant_config=ptq_config
        ),
        "provenance": {
            "nvidia_modelopt_version": _modelopt_version(),
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "ptq_config_names": PTQ_CONFIG_NAMES,
        },
        "transfer_scope": {
            "measured_on": str(model_path),
            "ptq_config": ptq_config,
            "claim": "mean margin shift for this model under this PTQ config only",
            "transfers_to_other_architectures": False,
        },
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:  # atomic create: never overwrite
            stream.write(json.dumps(report, indent=2) + "\n")
    return report


def _modelopt_version() -> str:
    from modelopt import __version__ as version  # lazy

    return str(version)


# --------------------------------------------------------------------------
# Live benchmark: the real harness, the real prompts, real per-token margins.
# --------------------------------------------------------------------------


# Difficulty is *prompt scaffolding only*: it changes what the goal asks for,
# never the harness's grading, the workspace, or the expected fix. `hard` is
# byte-identical to every prior run, so it stays the default.
DIFFICULTIES = ("hard", "guided", "mixed")
_GUIDED_SUFFIX = " Then call run_tests and report its result."


def _difficulty_for(index: int, difficulty: str) -> str:
    """Resolve one task's difficulty under the requested mode.

    `mixed` pairs consecutive tasks rather than alternating single ones.
    `router_rows_from_report` splits the calibration and held-out halves by
    interleaving *sorted* task names -- for `exp_f_repair_{i}` that puts every
    even index in one half -- so a one-by-one alternation would hand one half
    all the guided tasks and the other all the hard ones, and the held-out
    transfer test would then be measuring the split instead of the threshold.
    Blocks of two give both halves the same composition.
    """
    if difficulty == "hard":
        return "hard"
    if difficulty == "guided":
        return "guided"
    return "guided" if (index // 2) % 2 == 0 else "hard"


def small_route_tasks(count: int, *, difficulty: str = "hard") -> list[Any]:
    """Deterministic repair tasks for the small route, as real RuntimeTasks.

    Same shape the runtime harness evaluates elsewhere: a single broken
    function whose value must change. The task set is fixed by construction so
    both arms see byte-identical prompts.

    `difficulty` selects the scaffolding, and exists because a small instruct
    model repairs the file but then narrates instead of calling `run_tests` --
    and green is only ever granted by a passing `run_tests` observation, so on
    bare goals the small route scores 0 regardless of how easy the edit is.
    `guided` states the verification step in the goal; `mixed` interleaves the
    two so the set carries both a correct and an incorrect class, which is what
    a precision threshold needs in order to be calibratable at all.
    """
    from chowder.runtime_eval import RuntimeTask

    if count < 1:
        raise ValueError("small_route_tasks requires at least one task")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    tasks = []
    for index in range(count):
        returns = 2 + index
        goal = f"Repair app.py so it returns {returns}."
        if _difficulty_for(index, difficulty) == "guided":
            goal += _GUIDED_SUFFIX
        tasks.append(RuntimeTask(
            f"exp_f_repair_{index}",
            goal,
            {"app.py": f"def f(): return {returns - 1}"},
            "app.py",
            f"return {returns}",
            1,
            "1 passed",
        ))
    return tasks


def _render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    """Render a harness turn exactly as the serving path does (tools included)."""
    from chowder.runtime_eval import TOOLS

    return tokenizer.apply_chat_template(
        [dict(message) for message in messages], tools=TOOLS, tokenize=False, add_generation_prompt=True
    )


# Qwen's native dialect: the JSON body sits inside the span instead of the
# tool name, e.g. <tool_call>{"name": "read_file", "arguments": {...}}</tool_call>
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_TOOL_NAME = re.compile(r"[a-z_]+")
TOOL_CALL_FORMATS = ("harness", "json")


# Tools that can change what the harness scores. `_evaluate_task` grades the
# workspace files, and of the three tools only `write_file` mutates them, so
# `read_file` and `run_tests` cannot move a task toward green by themselves.
_ADVANCING_TOOLS = frozenset({"write_file"})


def normalize_json_tool_calls(raw: str) -> tuple[str, int, bool]:
    """Translate a model's native JSON tool calls into the harness dialect.

    ``_render_prompt`` hands ``TOOLS`` (JSON-schema function definitions) to
    ``apply_chat_template``, so a model that follows the prompt answers with
    ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``. The harness
    parses the *other* dialect -- ``<tool_call>name`` followed by
    ``<arg_key>``/``<arg_value>`` pairs -- so without this translation the
    model's actions are never executed and a run measures prompt format rather
    than the model. This is the decode-side twin of ``_render_prompt``.

    Only calls the model already emitted are translated, and only when they are
    well-formed: malformed JSON, a non-object payload, a missing/illegal tool
    name, or non-object arguments all leave the text untouched, so no action is
    ever invented.

    The harness executes one action per turn, so a batched turn must be reduced
    to one. The rule is: the first call, unless the batch contains an advancing
    call (one that can change what ``_evaluate_task`` sees), in which case that
    call is used instead. This is a deliberate, recorded choice, not a neutral
    read of the model: keeping the first call instead would execute the read and
    discard the write, and since only a write can ever score, that truncation
    manufactures a green-less run out of a model that did propose the fix. The
    batch is the model's own, so either rule is an interpretation; the report
    carries `dropped_calls` and `reordered_turns` so the interpretation is
    auditable rather than invisible.

    Returns ``(text_for_the_harness, dropped_calls, reordered)``.
    """
    rendered: list[tuple[str, str]] = []
    for payload in _JSON_CALL.findall(raw):
        try:
            call = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if not isinstance(call, Mapping):
            continue
        name = str(call.get("name", "")).strip()
        arguments = call.get("arguments", {})
        if not name or _TOOL_NAME.fullmatch(name) is None or not isinstance(arguments, Mapping):
            continue
        body = "".join(
            f"<arg_key>{key}</arg_key><arg_value>{_argument_text(value)}</arg_value>"
            for key, value in arguments.items()
        )
        rendered.append((name, f"<tool_call>{name}{body}</tool_call>"))
    if not rendered:
        return raw, 0, False
    chosen = 0
    for index, (name, _text) in enumerate(rendered):
        if name in _ADVANCING_TOOLS:
            chosen = index
            break
    return rendered[chosen][1], len(rendered) - 1, chosen != 0


def _argument_text(value: Any) -> str:
    """Argument values reach the harness as text, exactly as it expects them."""
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list)):
        return json.dumps(value)
    return str(value)


def _greedy_turn(model: Any, tokenizer: Any, rendered: str, *, max_new_tokens: int) -> dict[str, Any]:
    """One greedy turn: decoded text, token counts, and the top-2 margin.

    The margin is computed exactly as the Kaggle lane's ``margin_probe`` does
    (softmax over the step scores, then the log gap between the top two), so a
    shift measured here and a shift measured there mean the same thing. Token
    counts come from the tensors this call actually fed and produced, so they
    cannot drift from the text the way a re-encode of decoded text can.
    """
    import torch

    encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_tokens = int(encoded["input_ids"].shape[1])
    generated_tokens = int(output.sequences.shape[1]) - prompt_tokens
    text = tokenizer.decode(output.sequences[0, prompt_tokens:], skip_special_tokens=True)
    steps: list[float] = []
    for scores in output.scores:
        probabilities = torch.softmax(scores[0].float(), dim=-1)
        top2 = torch.topk(probabilities, 2)
        if not torch.isfinite(top2.values).all() or top2.values[1] <= 0:
            continue
        steps.append(float(torch.log(top2.values[0]) - torch.log(top2.values[1])))
    return {
        "text": text,
        "steps": steps,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }


def harness_margin_benchmark(
    tasks: Sequence[Any], tokenizer: Any, *, max_new_tokens: int = 192, max_turns: int = 4,
    tool_call_format: str = "harness", task_set: str = "small_route:hard",
) -> Callable[[Any, str], dict[str, Any]]:
    """Build the ``benchmark_fn`` that runs one arm over the harness task set.

    Each task is driven through ``run_live_benchmark`` in its own run so turns
    are attributable to a task, and the harness's own green-revocation scoring
    supplies the ``correct`` ground truth -- the model is never asked whether it
    was right. A task whose turns produced no usable log-probability step gets
    ``margin: None`` with a ``logprob_error`` instead of a fabricated value.

    ``tool_call_format`` selects which dialect the harness is handed. ``"harness"``
    is the harness's own ``<tool_call>name<arg_key>..`` syntax and is what every
    other experiment uses, so it stays the default and keeps results comparable.
    ``"json"`` additionally translates the JSON dialect most instruct models
    actually emit for these tool schemas (see ``normalize_json_tool_calls``);
    without it a small model's calls are never executed and the run measures
    prompt format instead of the model. Either way the margin still comes from
    the model's untouched generation.
    """
    if tool_call_format not in TOOL_CALL_FORMATS:
        raise ValueError(f"tool_call_format must be one of {TOOL_CALL_FORMATS}")
    from chowder.runtime_eval import run_live_benchmark

    def benchmark(model: Any, arm_name: str) -> dict[str, Any]:
        margin_rows: list[dict[str, Any]] = []
        reward = 0.0
        green = 0
        tool_calls = 0
        translated_turns = 0
        dropped_calls = 0
        reordered_turns = 0
        # Summed from the tensors each turn actually used, because a per-task
        # harness outcome carries no token accounting of its own
        # (``total_tokens`` lives in the aggregate ``_split_metrics`` block,
        # not in the per-task score). Reading a missing key off the outcome
        # would silently report 0, which is the one thing a
        # measured-quantities report must never do.
        prompt_tokens = 0
        generated_tokens = 0
        for task in tasks:
            turn_margins: list[float] = []
            turns = 0

            def generate(messages: list[dict[str, str]]) -> str:
                nonlocal turns, prompt_tokens, generated_tokens
                nonlocal translated_turns, dropped_calls, reordered_turns
                result = _greedy_turn(
                    model,
                    tokenizer,
                    _render_prompt(tokenizer, messages),
                    max_new_tokens=max_new_tokens,
                )
                turn_margins.extend(result["steps"])
                prompt_tokens += int(result["prompt_tokens"])
                generated_tokens += int(result["generated_tokens"])
                turns += 1
                text = result["text"]
                if tool_call_format == "json":
                    text, dropped, reordered = normalize_json_tool_calls(text)
                    if dropped or text != result["text"]:
                        translated_turns += 1
                    dropped_calls += dropped
                    reordered_turns += int(reordered)
                return text

            outcome = run_live_benchmark(
                generate, tasks=(task,), harness="state_aware", max_turns=max_turns
            )["tasks"][0]
            # The harness's own label, not the model's: `green_seen` already
            # applies green revocation (a later write_file un-greens it), and
            # requiring a final report after that green is the same rule
            # exp_e_pipeline._verified_final_green encodes for exported
            # trajectories. A self-reported success is never evidence.
            is_green = (
                bool(outcome["green_seen"])
                and not bool(outcome["premature_completion"])
                and any(row.get("role") == "final_report" for row in outcome["trace"])
            )
            green += int(is_green)
            reward += float(outcome["reward"])
            tool_calls += int(outcome["tool_calls"])
            row = {
                "task": outcome["task"],
                "margin": (sum(turn_margins) / len(turn_margins)) if turn_margins else None,
                "logprob_error": None if turn_margins else "no usable logprob step in any turn",
                "correct": is_green,
                "n_turns": turns,
                "n_margin_steps": len(turn_margins),
            }
            margin_rows.append(row)
            # Fake-quant generation is slow on this lane (no compiled kernels),
            # so report per-task progress; a silent multi-hour run is
            # indistinguishable from a hung one.
            print(
                f"[{arm_name}] {len(margin_rows)}/{len(tasks)} {row['task']}: "
                f"margin={row['margin']} correct={is_green} turns={turns}",
                flush=True,
            )
        return {
            "arm": arm_name,
            "margin_rows": margin_rows,
            "harness": {
                "harness": "state_aware",
                "max_turns": max_turns,
                "max_new_tokens": max_new_tokens,
                "green_rate": green / len(tasks),
                "mean_runtime_reward": reward / len(tasks),
                "tool_calls": tool_calls,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "tool_call_format": tool_call_format,
                "task_set": task_set,
                "translated_turns": translated_turns,
                "dropped_calls": dropped_calls,
                "reordered_turns": reordered_turns,
            },
        }

    return benchmark


def calibration_texts_for(
    tasks: Sequence[Any], tokenizer: Any, *, max_texts: int = 64
) -> list[str]:
    """Real first-turn harness prompts, used as the PTQ calibration corpus.

    These are the actual serving prompts (not invented filler), rendered with
    the same template the arms are evaluated on.
    """
    from chowder.runtime_eval import _state_message

    texts: list[str] = []
    for task in tasks:
        files = dict(task.initial)
        messages = [
            _state_message(files),
            {"role": "user", "content": f"Target file: {task.target}. {task.goal}"},
        ]
        texts.append(_render_prompt(tokenizer, messages))
    return texts[:max_texts]


def router_rows_from_report(
    report: Mapping[str, Any], *, calibrate_fraction: float = 0.5,
    max_quantized_margin_shift: float = DEFAULT_MARGIN_SHIFT_TOLERANCE,
    max_green_loss_fraction: float = DEFAULT_MAX_GREEN_LOSS_FRACTION,
) -> dict[str, Any]:
    """Turn a measured report into the router's per-precision calibration + gate.

    The measured tasks are split once, deterministically (sorted by task name),
    into a calibration half and a held-out half. The quantized arm's held-out
    half is what the held-out transfer gate evaluates, so this single run
    produces the shift, the per-precision thresholds, and the evidence that the
    quantized threshold actually transfers -- or an explicit refusal.

    The full paired task set also yields ``green_retention``: the measured
    fraction of the reference arm's greens the quantized arm loses
    (``validate_quantized_green_retention``, counted per task, so identical
    green *totals* on different tasks still count as losses). A passing margin
    shift says nothing about that, and the router refuses an arm that fails
    this guard regardless of the shift.
    """
    if not 0.0 < calibrate_fraction < 1.0:
        raise ValueError("calibrate_fraction must be in (0, 1)")
    comparison = report["margin_comparison"]
    per_task = sorted(comparison["per_task"], key=lambda row: row["task"])
    if len(per_task) < 2:
        raise ValueError("a held-out transfer test needs at least two measured tasks")
    split = max(1, min(len(per_task) - 1, round(len(per_task) * calibrate_fraction)))
    # Interleave rather than cut: a contiguous split would hand the held-out
    # half whichever end of the margin range it landed on, so the transfer
    # test would measure the split, not the quantization. Spacing the
    # calibration tasks evenly by name keeps both halves covering the whole
    # range while staying deterministic and independent of margin.
    stride = len(per_task) / split
    calibrate_tasks = {per_task[round(index * stride)]["task"] for index in range(split)}
    quant_arm = str(report["ptq_config"])
    rows_by_precision: dict[str, list[dict[str, Any]]] = {"bf16": [], quant_arm: []}
    for row in per_task:
        task = row["task"]
        if row["bf16_margin"] is not None:
            rows_by_precision["bf16"].append(
                {"task": task, "margin": row["bf16_margin"], "correct": bool(row["bf16_correct"])}
            )
        if row["quant_margin"] is not None:
            rows_by_precision[quant_arm].append(
                {"task": task, "margin": row["quant_margin"], "correct": bool(row["quant_correct"])}
            )
    calibrations = calibrate_margin_threshold_per_precision({
        # Fit each arm's threshold on the calibration half only. Fitting on
        # every task and then gating on a subset of the same tasks would be
        # self-deception, and heldout_transfer_gate correctly refuses it.
        arm: [row for row in rows if row["task"] in calibrate_tasks]
        for arm, rows in rows_by_precision.items()
    })
    heldout_rows = [
        row for row in rows_by_precision[quant_arm] if row["task"] not in calibrate_tasks
    ]
    gate = heldout_transfer_gate(calibrations[quant_arm], heldout_rows)
    shift = validate_quantized_margin_shift(
        comparison["bf16_mean_margin"],
        comparison["quant_mean_margin"],
        max_quantized_margin_shift=max_quantized_margin_shift,
    )
    green_retention = validate_quantized_green_retention(
        [{"task": row["task"], "correct": row["bf16_correct"]} for row in per_task],
        [{"task": row["task"], "correct": row["quant_correct"]} for row in per_task],
        max_green_loss_fraction=max_green_loss_fraction,
    )
    admission = quantized_arm_admission(
        calibrations[quant_arm],
        quantized_margin_shift=shift["margin_shift"],
        max_quantized_margin_shift=max_quantized_margin_shift,
        green_loss_fraction=green_retention["green_loss_fraction"],
        max_green_loss_fraction=max_green_loss_fraction,
        heldout_gate=gate,
    )
    return {
        "calibrations": calibrations,
        "heldout_gate": gate,
        "margin_shift": shift["margin_shift"],
        "quantized_margin_shift_fails_closed": shift["quantized_margin_shift_fails_closed"],
        "green_retention": green_retention,
        "admission": admission,
        "calibrate_tasks": sorted(calibrate_tasks),
        "heldout_tasks": sorted(row["task"] for row in heldout_rows),
        "transfer_scope": {
            "measured_on": report.get("model_path"),
            "ptq_config": report["ptq_config"],
            "transfers_to_other_architectures": False,
        },
    }


def load_quantized_evidence(
    path: str | Path,
    *,
    max_quantized_margin_shift: float = DEFAULT_MARGIN_SHIFT_TOLERANCE,
    max_green_loss_fraction: float = DEFAULT_MAX_GREEN_LOSS_FRACTION,
) -> dict[str, Any]:
    """Derive the Phase-4 quantized-arm inputs from a measured Experiment F report.

    A serving lane must not be handed hand-typed scalars for a quantized arm:
    the precision name, the margin shift, the green retention, the report-side
    per-precision calibration/gate records, and the quantized arm's held-out
    rows are all computed from the report's per-task measurements through
    ``router_rows_from_report``. The derived admission artifact is verified
    before it is handed on, so an edited report fails closed here rather than
    at routing time.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("the quantized evidence file must contain one Experiment F report object")
    for key in ("ptq_config", "margin_comparison"):
        if key not in report:
            raise ValueError(f"the evidence report is missing {key!r}; refusing to derive a quantized arm from it")
    quant_arm = str(report["ptq_config"]).strip()
    if not quant_arm or quant_arm == "bf16":
        raise ValueError("the evidence report does not describe a quantized arm")
    rows = router_rows_from_report(
        report,
        max_quantized_margin_shift=max_quantized_margin_shift,
        max_green_loss_fraction=max_green_loss_fraction,
    )
    verification = verify_arm_admission(rows["admission"])
    if not verification["valid"]:
        raise ValueError(
            f"the evidence report's arm admission failed verification: {verification['reason']}"
        )
    heldout_tasks = set(rows["heldout_tasks"])
    heldout_rows = [
        {"task": row["task"], "margin": row["quant_margin"], "correct": row["quant_correct"]}
        for row in report["margin_comparison"]["per_task"]
        if row["task"] in heldout_tasks
    ]
    return {
        "source": str(path),
        "precision_arm": quant_arm,
        "margin_shift": rows["margin_shift"],
        "green_loss_fraction": rows["green_retention"]["green_loss_fraction"],
        "green_retention": rows["green_retention"],
        "heldout_rows": heldout_rows,
        "report_calibrations": rows["calibrations"],
        "report_heldout_gate": rows["heldout_gate"],
        "admission": rows["admission"],
        "admission_verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment F: PTQ vs BF16 harness + margin shift")
    parser.add_argument("--model", required=True, help="local HF model id or path for the small arm")
    parser.add_argument("--output", required=True, help="report JSON path; never overwritten")
    parser.add_argument("--ptq-config", default="int8_smoothquant", choices=sorted(
        name for name in PTQ_CONFIG_NAMES if name
    ))
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument(
        "--tool-call-format",
        default="harness",
        choices=TOOL_CALL_FORMATS,
        help=(
            "dialect handed to the harness parser. 'harness' is its own syntax "
            "and the comparable default; 'json' also translates the JSON tool "
            "calls instruct models emit for these schemas"
        ),
    )
    parser.add_argument(
        "--difficulty",
        default="hard",
        choices=DIFFICULTIES,
        help=(
            "prompt scaffolding for the small route. 'hard' is the bare goal "
            "and the comparable default; 'guided' also names the verification "
            "step; 'mixed' interleaves both so the set has a correct and an "
            "incorrect class"
        ),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tasks = small_route_tasks(args.tasks, difficulty=args.difficulty)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report = run_ptq_margin_experiment(
        args.model,
        calibration_texts=calibration_texts_for(tasks, tokenizer),
        benchmark_fn=harness_margin_benchmark(
            tasks,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            max_turns=args.max_turns,
            tool_call_format=args.tool_call_format,
            task_set=f"small_route:{args.difficulty}",
        ),
        ptq_config=args.ptq_config,
        output_path=args.output,
    )
    comparison = report["margin_comparison"]
    print(json.dumps({
        "ptq_config": report["ptq_config"],
        "tool_call_format": args.tool_call_format,
        "difficulty": args.difficulty,
        "n_tasks": comparison["n_tasks"],
        "bf16_mean_margin": comparison["bf16_mean_margin"],
        "quant_mean_margin": comparison["quant_mean_margin"],
        "mean_margin_shift": comparison["mean_margin_shift"],
        "bf16_accuracy": comparison["bf16_accuracy"],
        "quant_accuracy": comparison["quant_accuracy"],
        "bf16_harness": report["bf16"]["harness"],
        "quant_harness": report["quant"]["harness"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
