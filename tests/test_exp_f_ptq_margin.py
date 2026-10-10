from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="the PTQ lane quantizes real tiny modules")
pytest.importorskip("modelopt", reason="nvidia-modelopt is the optional 'ptq' extra")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
sys.path.insert(0, str(ROOT / "src"))

from exp_f_ptq_margin import (
    DIFFICULTIES,
    PTQ_CONFIG_NAMES,
    _greedy_turn,
    compare_margin_signal,
    harness_margin_benchmark,
    normalize_json_tool_calls,
    quantize_for_arm,
    resolve_ptq_config,
    router_rows_from_report,
    run_ptq_margin_experiment,
    small_route_tasks,
)

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _report(n_tasks: int = 16, *, shift: float = -0.2) -> dict:
    """A synthetic paired report shaped exactly like the live one.

    Margins rise with index and the low-margin tasks are the wrong ones, so a
    threshold is actually calibratable on the calibration half *and* the
    remaining half is a genuine transfer test (an alternating-correct set could
    never reach the precision target, and every record would fail closed).
    """
    per_task = []
    for index in range(n_tasks):
        bf16 = 1.0 + index * 0.1
        correct = index >= 4
        per_task.append({
            "task": f"task_{index}",
            "bf16_margin": bf16,
            "quant_margin": bf16 + shift,
            "margin_shift": shift,
            "bf16_correct": correct,
            "quant_correct": correct,
        })
    return {
        "model_path": "synthetic/model",
        "ptq_config": "int8_smoothquant",
        "margin_comparison": {
            "n_tasks": n_tasks,
            "bf16_mean_margin": sum(row["bf16_margin"] for row in per_task) / n_tasks,
            "quant_mean_margin": sum(row["quant_margin"] for row in per_task) / n_tasks,
            "per_task": per_task,
        },
    }


def test_ptq_arm_names_resolve_to_real_mtq_configs():
    assert resolve_ptq_config("bf16") is None
    for arm in ("int8_smoothquant", "int8_weight_only", "int4_awq", "nvfp4_default"):
        config = resolve_ptq_config(arm)
        assert isinstance(config, dict) and config, arm
    with pytest.raises(ValueError, match="unknown PTQ arm"):
        resolve_ptq_config("int999_make_believe")


def test_quantize_for_arm_runs_real_mtq_on_cpu_tiny_mlp():
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    calib_x = torch.randn(32, 8)
    quantized = quantize_for_arm(
        model, "int8_smoothquant", lambda m: m(calib_x)
    )
    assert quantized is model  # mtq quantizes in place
    quantizer_names = [name for name, _ in model.named_modules() if "quantizer" in name]
    assert len(quantizer_names) >= 4  # input+weight quantizers per Linear
    out = model(calib_x[:4])
    assert out.shape == (4, 4)
    with pytest.raises(ValueError, match="must not be quantized"):
        quantize_for_arm(model, "bf16", None)


def test_compare_margin_signal_reports_only_measured_deltas():
    bf16_rows = [
        {"task": "a", "margin": 3.0, "logprob_error": None, "correct": True},
        {"task": "b", "margin": 1.0, "logprob_error": None, "correct": False},
        {"task": "c", "margin": None, "logprob_error": "HTTP 503", "correct": True},
    ]
    quant_rows = [
        {"task": "a", "margin": 1.5, "logprob_error": None, "correct": True},
        {"task": "b", "margin": 0.5, "logprob_error": None, "correct": False},
        {"task": "c", "margin": None, "logprob_error": "HTTP 503", "correct": True},
    ]
    report = compare_margin_signal(bf16_rows, quant_rows, quant_config="int8_smoothquant")
    assert report["n_tasks"] == 3
    assert report["bf16_mean_margin"] == pytest.approx(2.0)  # nulls excluded
    assert report["quant_mean_margin"] == pytest.approx(1.0)
    assert report["mean_margin_shift"] == pytest.approx(-1.0)
    assert report["bf16_margin_nulls"] == 1 and report["quant_margin_nulls"] == 1
    assert report["accuracy_delta"] == pytest.approx(0.0)
    assert report["per_task"][2]["margin_shift"] is None  # null stays null, never zero


def test_compare_margin_signal_rejects_unpaired_or_malformed_rows():
    bf16_rows = [{"task": "a", "margin": 3.0, "correct": True}]
    with pytest.raises(ValueError, match="not paired"):
        compare_margin_signal(bf16_rows, [{"task": "z", "margin": 1.0, "correct": True}], quant_config="q")
    with pytest.raises(ValueError, match="duplicate"):
        compare_margin_signal(bf16_rows * 2, bf16_rows, quant_config="q")
    with pytest.raises(ValueError, match="missing the margin field"):
        compare_margin_signal([{"task": "a", "correct": True}], bf16_rows, quant_config="q")
    with pytest.raises(ValueError, match="null without a logprob_error"):
        compare_margin_signal([{"task": "a", "margin": None, "logprob_error": "", "correct": True}], bf16_rows, quant_config="q")
    with pytest.raises(ValueError, match="finite number"):
        compare_margin_signal([{"task": "a", "margin": float("nan"), "correct": True}], bf16_rows, quant_config="q")
    with pytest.raises(ValueError, match="finite number"):
        compare_margin_signal([{"task": "a", "margin": True, "correct": True}], bf16_rows, quant_config="q")
    with pytest.raises(ValueError, match="no task name"):
        compare_margin_signal([{"task": "", "margin": 1.0, "correct": True}], bf16_rows, quant_config="q")
    with pytest.raises(ValueError, match="no margin rows"):
        compare_margin_signal([], bf16_rows, quant_config="q")


def test_run_ptq_margin_experiment_fails_closed_on_cpu_or_missing_calibration():
    import torch

    def benchmark(_model, _arm):  # pragma: no cover - must never be reached
        raise AssertionError("benchmark_fn must not run on CPU")

    with pytest.raises(ValueError, match="not bf16"):
        run_ptq_margin_experiment(
            "unused", calibration_texts=["x"], benchmark_fn=benchmark, ptq_config="bf16"
        )
    with pytest.raises(ValueError, match="real calibration texts"):
        run_ptq_margin_experiment(
            "unused", calibration_texts=["  ", ""], benchmark_fn=benchmark, ptq_config="int8_smoothquant"
        )
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="live-GPU experiment"):
            run_ptq_margin_experiment(
                "unused", calibration_texts=["real calibration text"], benchmark_fn=benchmark
            )


def test_whitelisted_config_names_covers_every_quantized_arm():
    assert set(PTQ_CONFIG_NAMES) == {"bf16", "int8_smoothquant", "int8_weight_only", "int4_awq", "nvfp4_default"}


def test_small_route_tasks_are_deterministic_and_distinct():
    first = small_route_tasks(4)
    second = small_route_tasks(4)
    assert [task.name for task in first] == [task.name for task in second] == [
        f"exp_f_repair_{index}" for index in range(4)
    ]
    assert len({task.goal for task in first}) == 4
    assert all(task.target == "app.py" and task.test_count == 1 for task in first)
    with pytest.raises(ValueError, match="at least one task"):
        small_route_tasks(0)


def test_router_rows_split_tasks_and_gate_the_quantized_arm():
    rows = router_rows_from_report(_report(16), max_quantized_margin_shift=0.5)
    # Both arms get their own independent calibration, keyed by precision.
    assert set(rows["calibrations"]) == {"bf16", "int8_smoothquant"}
    for arm, record in rows["calibrations"].items():
        assert record["precision_arm"] == arm
        assert record["status"] == "calibrated"
    # The measured shift is what the router tolerance consumes.
    assert rows["margin_shift"] == pytest.approx(-0.2, abs=1e-9)
    assert rows["quantized_margin_shift_fails_closed"] is False
    # The calibration and held-out halves are disjoint task sets.
    assert not set(rows["calibrate_tasks"]) & set(rows["heldout_tasks"])
    assert len(rows["calibrate_tasks"]) + len(rows["heldout_tasks"]) == 16
    gate = rows["heldout_gate"]
    assert gate["status"] == "heldout_validated"
    assert gate["precision_arm"] == "int8_smoothquant"
    assert not set(gate["heldout_task_ids"]) & set(rows["calibrate_tasks"])
    # The report never claims its shift transfers to another architecture.
    assert rows["transfer_scope"]["transfers_to_other_architectures"] is False
    assert rows["transfer_scope"]["measured_on"] == "synthetic/model"
    # Both arms scored the same greens in the fixture: retention is measured
    # per task (0.0 lost) and does not block.
    retention = rows["green_retention"]
    assert retention["reference_greens"] == 12 and retention["retained_greens"] == 12
    assert retention["green_loss_fraction"] == 0.0
    assert retention["quantized_green_loss_fails_closed"] is False

    tight = router_rows_from_report(_report(16), max_quantized_margin_shift=0.05)
    assert tight["quantized_margin_shift_fails_closed"] is True
    with pytest.raises(ValueError, match="calibrate_fraction"):
        router_rows_from_report(_report(16), calibrate_fraction=0.0)


def test_router_rows_measure_green_retention_so_a_shift_cannot_license_an_arm():
    report = _report(16, shift=-0.1)
    # The quantized arm keeps only three reference greens and gains one wrong
    # task back; its mean margin shift is untouched and inside the default
    # tolerance, so only the retention guard can refuse it.
    for index, row in enumerate(report["margin_comparison"]["per_task"]):
        row["quant_correct"] = index in (4, 5, 6) or index == 3
    rows = router_rows_from_report(report)
    assert rows["quantized_margin_shift_fails_closed"] is False
    retention = rows["green_retention"]
    assert retention["reference_greens"] == 12
    assert (retention["retained_greens"], retention["lost_greens"], retention["gained_greens"]) == (3, 9, 1)
    assert retention["green_loss_fraction"] == pytest.approx(0.75)
    assert retention["quantized_green_loss_fails_closed"] is True
    # A caller that declares that fraction acceptable gets a measured pass.
    tolerant = router_rows_from_report(report, max_green_loss_fraction=0.75)
    assert tolerant["green_retention"]["quantized_green_loss_fails_closed"] is False


def test_router_rows_refuse_a_shift_measured_on_a_contaminated_split():
    # Held-out rows that are really dev rows must be refused, not silently used.
    report = _report(16)
    rows = router_rows_from_report(report)
    contaminated = dict(rows["calibrations"]["int8_smoothquant"])
    contaminated["precision_arm"] = "int8_smoothquant"
    from exp_e_confidence import heldout_transfer_gate

    same_tasks = [
        {"task": task, "margin": 1.0, "correct": True} for task in rows["calibrate_tasks"]
    ]
    assert heldout_transfer_gate(contaminated, same_tasks)["status"] == "heldout_contaminated"


def test_live_margin_definition_matches_the_kaggle_lane_probe():
    """A shift measured here must mean the same thing as one measured there.

    Experiment F measures the margin locally; the Kaggle QAT lane measures the
    shift that gates the same router. If the two definitions drift apart, the
    router tolerance is consuming incomparable numbers, so they are pinned to
    each other on a real (tiny) model rather than trusted to inspection.
    """
    import importlib.util

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = importlib.util.spec_from_file_location(
        "kaggle_qat_lane_margin", ROOT / "kaggle" / "run_qat_distill_lane.py"
    )
    lane = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lane)

    try:
        tokenizer = AutoTokenizer.from_pretrained(_TINY)
        model = AutoModelForCausalLM.from_pretrained(_TINY, dtype=torch.float32)
    except Exception:  # pragma: no cover - offline environment
        pytest.skip("tiny test model unavailable offline")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = ["Repair app.py so it returns 2.", "Repair lib.py so it returns 3."]

    aggregate = lane.margin_probe(model, tokenizer, prompts, max_new_tokens=8)
    local = [
        _greedy_turn(model, tokenizer, prompt, max_new_tokens=8)
        for prompt in prompts
    ]
    local_steps = [step for turn in local for step in turn["steps"]]
    assert local_steps, "the local probe must collect steps"
    assert aggregate["mean_margin"] == pytest.approx(
        sum(local_steps) / len(local_steps), rel=1e-4
    )
    assert aggregate["n_steps"] == len(local_steps)


def test_harness_block_reports_only_token_counts_it_actually_measured():
    """Token accounting must come from the run, not from a defaulted key.

    A per-task harness outcome carries no token totals (they live in the
    aggregate ``_split_metrics`` block), so reading ``outcome["total_tokens"]``
    silently yields 0 -- a field that looks measured and never is. This pins
    the harness block to counts taken from the tensors each turn really used,
    on a real (tiny) model driven through the real harness, and pins the
    phantom key out.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(_TINY)
        model = AutoModelForCausalLM.from_pretrained(_TINY, dtype=torch.float32)
    except Exception:  # pragma: no cover - offline environment
        pytest.skip("tiny test model unavailable offline")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_new_tokens = 4
    result = harness_margin_benchmark(
        small_route_tasks(1), tokenizer, max_new_tokens=max_new_tokens, max_turns=2
    )(model, "bf16")
    block = result["harness"]

    assert "total_tokens" not in block, (
        "total_tokens is not a per-task harness key; reporting it means reporting a silent 0"
    )
    assert block["prompt_tokens"] > 0, "the prompts are non-empty, so this cannot be 0"
    turns = result["margin_rows"][0]["n_turns"]
    assert turns >= 1
    # Greedy decoding emits up to max_new_tokens per turn; an early EOS may cut
    # a turn short, but the total can never exceed what was asked for nor be 0.
    assert 0 < block["generated_tokens"] <= turns * max_new_tokens
    assert block["tool_calls"] >= 0
    assert block["green_rate"] == pytest.approx(0.0)


def _json_call(name: str, **arguments) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


def test_json_tool_calls_are_translated_into_the_harness_dialect():
    """The dialect a model emits must be the dialect the harness parses.

    Qwen answers `TOOLS` with its native JSON dialect while the harness only
    parses `<tool_call>name<arg_key>..`; untranslated, every action is silently
    unexecuted. Checked against the harness's own parser rather than a
    hand-written expectation, so a change to that parser surfaces here.
    """
    from chowder.runtime_eval import parse_tool_call

    raw = _json_call("write_file", path="app.py", content="def f(): return 2")
    text, dropped, reordered = normalize_json_tool_calls(raw)
    assert parse_tool_call(text) == ("write_file", {"path": "app.py", "content": "def f(): return 2"})
    assert (dropped, reordered) == (0, False)

    # Newlines and quotes in the payload survive the round trip.
    multi = _json_call("write_file", path="app.py", content="def f():\n    return 2\n")
    body, _dropped, _reordered = normalize_json_tool_calls(multi)
    assert parse_tool_call(body)[1]["content"] == "def f():\n    return 2\n"


def test_a_batched_turn_keeps_the_call_that_can_actually_score():
    """Only a write can change what the harness grades, so a batch must not drop it.

    A model that reads and then writes in one turn is batching; the harness acts
    on one call. Keeping the first would execute the read and discard the write,
    and since green comes only from a write followed by run_tests, that choice
    alone turns a proposed fix into a recorded failure. The reordering is
    reported, not hidden.
    """
    from chowder.runtime_eval import parse_tool_call

    read = _json_call("read_file", path="app.py")
    write = _json_call("write_file", path="app.py", content="def f(): return 2")
    tests = _json_call("run_tests")

    text, dropped, reordered = normalize_json_tool_calls(read + write)
    assert parse_tool_call(text)[0] == "write_file"
    assert (dropped, reordered) == (1, True)

    # Already leading with the write: nothing is reordered.
    text, dropped, reordered = normalize_json_tool_calls(write + read)
    assert parse_tool_call(text)[0] == "write_file"
    assert (dropped, reordered) == (1, False)

    # A read-only batch stays a read; no write is ever invented.
    text, dropped, reordered = normalize_json_tool_calls(read + tests)
    assert parse_tool_call(text) == ("read_file", {"path": "app.py"})
    assert (dropped, reordered) == (1, False)


def test_unparsable_tool_calls_are_left_alone_rather_than_guessed():
    """Fail closed: a malformed call must never become an executed action."""
    from chowder.runtime_eval import parse_tool_call

    for raw in (
        "I will fix app.py now.",                                   # no call at all
        '<tool_call>{"name": "write_file", "arguments": </tool_call>',  # broken JSON
        "<tool_call>[1, 2, 3]</tool_call>",                         # not an object
        '<tool_call>{"name": "Write File!", "arguments": {"path": "a"}}</tool_call>',
        '<tool_call>{"arguments": {"path": "a"}}</tool_call>',      # no tool name
    ):
        text, dropped, reordered = normalize_json_tool_calls(raw)
        assert dropped == 0 and reordered is False, raw
        assert text == raw, f"unchanged text expected for {raw!r}"
        assert parse_tool_call(text) is None, f"nothing may be executed for {raw!r}"

    # An omitted `arguments` is not malformed -- `run_tests` takes none -- so it
    # translates to a call with no arguments rather than being discarded.
    text, dropped, reordered = normalize_json_tool_calls('<tool_call>{"name": "run_tests"}</tool_call>')
    assert parse_tool_call(text) == ("run_tests", {})
    assert (dropped, reordered) == (0, False)


def test_difficulty_scaffolds_only_the_goal_and_keeps_hard_byte_identical():
    """Difficulty may change what a task asks for, never how it is graded."""
    hard = small_route_tasks(4)
    assert [task.goal for task in hard] == [f"Repair app.py so it returns {2 + i}." for i in range(4)]
    for mode in DIFFICULTIES:
        for task in small_route_tasks(4, difficulty=mode):
            returns = int(task.expected_fix.split()[-1])
            assert task.target == "app.py"
            assert task.expected_fix == f"return {returns}"
            # The workspace always starts one revision short of the fix, so
            # difficulty can never make a task trivially green.
            assert task.initial == {"app.py": f"def f(): return {returns - 1}"}
            assert task.expected_fix not in task.initial["app.py"]
            assert task.test_count == 1 and task.test_success == "1 passed"
    guided = small_route_tasks(4, difficulty="guided")
    assert all("run_tests" in task.goal for task in guided)
    with pytest.raises(ValueError):
        small_route_tasks(4, difficulty="impossible")


def test_mixed_difficulty_keeps_both_split_halves_balanced():
    """The guided/hard mix must not line up with the calibration/held-out split.

    `router_rows_from_report` interleaves *sorted* task names, which for
    `exp_f_repair_{i}` puts every even index in one half. A one-by-one
    alternation would therefore put every guided task in the calibration half
    and every hard task in the held-out half, and the transfer test would be
    measuring the split rather than the threshold.
    """
    mixed = small_route_tasks(12, difficulty="mixed")
    guided = {"run_tests" in task.goal for task in mixed}
    assert guided == {True, False}, "mixed must carry both scaffoldings"

    by_name = {task.name: ("guided" if "run_tests" in task.goal else "hard")
               for task in small_route_tasks(12, difficulty="mixed")}
    report = {
        "ptq_config": "int8_smoothquant",
        "margin_comparison": {
            "bf16_mean_margin": 1.0,
            "quant_mean_margin": 1.0,
            "per_task": [
                {"task": name, "bf16_margin": 1.0, "quant_margin": 1.0,
                 "bf16_correct": True, "quant_correct": True}
                for name in sorted(by_name)
            ],
        },
    }
    calibrate = set(router_rows_from_report(report)["calibrate_tasks"])
    held_out = set(by_name) - calibrate
    assert calibrate and held_out
    for half in (calibrate, held_out):
        modes = [by_name[name] for name in half]
        assert modes.count("guided") == modes.count("hard"), (
            f"split half is not a mix: {sorted(half)} -> {modes}"
        )
