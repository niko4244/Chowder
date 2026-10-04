"""CPU contract tests for signed reward preprocessing and one-step training."""
from __future__ import annotations

def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"w{i}": i + 3 for i in range(80)})
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tok.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return tok


def _dataset(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)


def _one_step(dataset, tokenizer, *, chat):
    import torch
    from transformers import (
        DataCollatorForLanguageModeling,
        DataCollatorForSeq2Seq,
        LlamaConfig,
        LlamaForCausalLM,
        Trainer,
        TrainingArguments,
    )
    from chowder.backends.transformers_worker import (
        _prepare_reward_chat_dataset,
        _prepare_reward_text_dataset,
    )

    if chat:
        prepared = _prepare_reward_chat_dataset(
            dataset, tokenizer, messages_field="messages", reward_field="reward", max_length=128
        )
        base = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, label_pad_token_id=-100)
    else:
        prepared = _prepare_reward_text_dataset(
            dataset, tokenizer, text_field="text", reward_field="reward", max_length=32
        )
        base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    expected = {"input_ids", "attention_mask", "reward"}
    if chat:
        expected.add("labels")
    assert set(prepared.column_names) == expected
    assert prepared.column_names.count("reward") == 1
    assert prepared["reward"] == [1.0, -4.0]

    def collator(features):
        rewards = [float(item.pop("reward")) for item in features]
        batch = base(features)
        batch["reward"] = rewards
        return batch

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config)
    args = TrainingArguments(
        output_dir=tmp_path_for_training(), num_train_epochs=1, max_steps=1,
        per_device_train_batch_size=2, learning_rate=1e-3,
        logging_steps=1, report_to=[], use_cpu=True, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=prepared, data_collator=collator)
    result = trainer.train()
    assert result.training_loss is not None
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert trainer.state.global_step == 1


def tmp_path_for_training():
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp(prefix="chowder-reward-test-"))
    return str(path)


def test_reward_text_map_and_one_step(tmp_path):
    tok = _tokenizer()
    rows = [{"text": "w1 w2", "reward": 1.0}, {"text": "w3 w4", "reward": -4.0}]
    _one_step(_dataset(rows), tok, chat=False)


def test_reward_chat_map_and_one_step(tmp_path):
    tok = _tokenizer()
    rows = [
        {"messages": [{"role": "user", "content": "w1"}, {"role": "assistant", "content": "w2"}], "reward": 1.0},
        {"messages": [{"role": "user", "content": "w3"}, {"role": "assistant", "content": "w4"}], "reward": -4.0},
    ]
    _one_step(_dataset(rows), tok, chat=True)


def test_bounded_unlikelihood_stays_finite_for_extreme_logits():
    import torch
    from chowder.backends.transformers_worker import (
        _bounded_token_unlikelihood,
        _reward_weighted_loss,
    )

    logits = torch.tensor([[[1000.0, -1000.0], [-1000.0, 1000.0], [1000.0, -1000.0]]])
    labels = torch.tensor([[0, 1, 0]])
    loss = _bounded_token_unlikelihood(logits[:, :-1], labels[:, 1:])
    weighted = _reward_weighted_loss(logits, labels, [-1e9])
    assert torch.isfinite(loss).all()
    assert float(loss.max()) < 10.0
    assert torch.isfinite(weighted)
    assert float(weighted) < 10.0 * 4.0


def test_event_reward_rows_preserve_observation_event_and_reward():
    from chowder_batch.build_event_reward_data import event_rows_from_trace

    trace = [
        {"role": "assistant", "text": "read"},
        {"role": "tool", "tool": "read_file", "args": {"path": "missing.py"}, "observation": "ERROR: no such file: missing.py"},
        {"role": "tool", "tool": "run_tests", "args": {}, "observation": "2 passed"},
    ]
    rows = event_rows_from_trace(trace, trace_id="t")
    assert [row["event"] for row in rows] == ["nonexistent_read", "observed_green"]
    assert rows[0]["observation"].startswith("ERROR:")
    assert rows[0]["reward"] == -4.0
    assert "action" in rows[0] and "context" in rows[0]


def test_adapter_bundle_validates_parent_root_and_repair_hashes(tmp_path):
    from chowder.adapter_bundle import (
        read_adapter_bundle_manifest,
        write_adapter_bundle_manifest,
    )
    from chowder.provenance import sha256_directory

    root = tmp_path / "bundle"
    (root / "repair").mkdir(parents=True)
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"parent")
    (root / "repair" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (root / "repair" / "adapter_model.safetensors").write_bytes(b"repair")
    write_adapter_bundle_manifest(
        root,
        parent_sha256="a" * 64,
        repair_sha256=sha256_directory(root / "repair"),
    )
    assert read_adapter_bundle_manifest(root)["format"] == "chowder-adapter-bundle-v1"
    (root / "adapter_model.safetensors").write_bytes(b"tampered")
    try:
        read_adapter_bundle_manifest(root)
    except ValueError as exc:
        assert "parent adapter root" in str(exc)
    else:
        raise AssertionError("tampered parent adapter was accepted")


def test_runtime_metrics_are_hard_promotion_gates():
    from chowder.gate import evaluate_candidate
    from chowder.models import ExperimentResult, Goal, MetricTarget

    goal = Goal(
        metrics=(MetricTarget("quality", minimum=0.5),),
        gpu_hour_budget=1.0,
        minimum_promotion_gain=-1.0,
        runtime_reward_min=0.0,
        runtime_nonexistent_read_rate_max=0.0,
    )
    baseline = ExperimentResult("base", {"quality": 0.5, "runtime_reward": 1.0, "runtime_nonexistent_read_rate": 0.0}, 0.1)
    candidate = ExperimentResult("candidate", {"quality": 0.6, "runtime_reward": -1.0, "runtime_nonexistent_read_rate": 0.5}, 0.1)
    decision = evaluate_candidate(goal=goal, baseline=baseline, candidate=candidate)
    assert not decision.accepted
    assert "runtime safety gate" in decision.reason
    assert set(decision.missing_metrics) == set()


def test_runtime_benchmark_is_part_of_evaluation_protocol_when_enabled():
    from chowder.evaluators.base_text import BaseTextEvalSpec
    from chowder.evaluators.transformers_text import TransformersTextEvalSpec
    from chowder.evaluators.transformers_text import EvalSuiteSpec

    suite = (EvalSuiteSpec(name="quality", dataset="quality.jsonl"),)
    plain = TransformersTextEvalSpec("model", None, "out", suite)
    live = TransformersTextEvalSpec(
        "model", None, "out", suite,
        runtime_benchmark={"enabled": True, "max_turns": 8, "max_new_tokens": 128},
    )
    assert "runtime_benchmark" not in plain.to_dict()
    assert live.to_dict()["runtime_benchmark"]["enabled"] is True
    assert plain.digest() != live.digest()
    assert BaseTextEvalSpec("model", "out", suite).to_dict().get("runtime_benchmark") is None


def test_rrsi_first_round_requires_heldout_transfer_and_records_policy_tokens():
    from chowder.harness_evolution import (
        HarnessMetrics,
        HarnessProposal,
        select_first_round,
    )

    base_evolve = HarnessMetrics(0.33, 100.0, 0.33, 0.33, 100.0)
    candidate_evolve = HarnessMetrics(0.50, 105.0, 0.67, 0.0, 105.0)
    base_heldout = HarnessMetrics(0.25, 90.0, 0.25, 0.25, 90.0)
    candidate_heldout = HarnessMetrics(0.50, 95.0, 0.50, 0.0, 95.0)
    proposal = HarnessProposal("generic-recovery", "control_flow", ("continue after red",), candidate_evolve)
    result = select_first_round(
        base_evolve, candidate_evolve,
        evolve_incumbent=base_evolve, evolve_candidate=candidate_evolve,
        heldout_incumbent=base_heldout, heldout_candidate=candidate_heldout,
        proposal=proposal, forbidden_terms=("config_defaults.py", "retry_policy.py"),
    )
    assert result["accepted"] is True
    assert result["heldout"]["policy_tokens"] == 95.0
    regressed = select_first_round(
        base_evolve, candidate_evolve,
        evolve_incumbent=base_evolve, evolve_candidate=candidate_evolve,
        heldout_incumbent=base_heldout,
        heldout_candidate=HarnessMetrics(0.0, 95.0, 0.0, 0.0, 95.0),
        proposal=proposal, forbidden_terms=(),
    )
    assert regressed["accepted"] is False


def test_heldout_runtime_tasks_are_disjoint_from_evolve_tasks():
    from chowder.runtime_eval import HELDOUT_TASKS, TASKS

    evolve_names = {task.name for task in TASKS}
    heldout_names = {task.name for task in HELDOUT_TASKS}
    assert evolve_names.isdisjoint(heldout_names)
    assert {"config_defaults", "retry_budget", "csv_active_rows"} <= heldout_names
    assert not any(name in evolve_names for name in heldout_names)


def test_rrsi_selection_rejects_noise_cost_regressions_and_leakage():
    from chowder.harness_evolution import (
        HarnessMetrics,
        HarnessProposal,
        accept_candidate,
        annealed_edit_budget,
        leakage_free,
    )

    incumbent = HarnessMetrics(0.33, 100.0, 0.33, 0.33)
    assert annealed_edit_budget(0, 4, maximum=3) == 3
    assert annealed_edit_budget(3, 4, maximum=3) == 2
    good = HarnessMetrics(0.50, 105.0, 0.67, 0.0)
    assert accept_candidate(incumbent, good)[0]
    assert not accept_candidate(incumbent, HarnessMetrics(0.34, 100.0, 0.34, 0.0), noise_band=0.02)[0]
    assert not accept_candidate(incumbent, HarnessMetrics(0.90, 300.0, 1.0, 0.0))[0]
    proposal = HarnessProposal("answer-key", "prompt", ("version.py", "2 passed"), good)
    assert not leakage_free(proposal, ("version.py", "2 passed"))


def test_runtime_benchmark_exposes_promotion_metrics():
    from chowder_batch.runtime_benchmark import run_benchmark

    good = run_benchmark()
    bad = run_benchmark(bad_read=True)
    assert good["metrics"]["runtime_green_rate"] == 1.0
    assert good["metrics"]["runtime_nonexistent_read_rate"] == 0.0
    assert bad["metrics"]["runtime_nonexistent_read_rate"] > 0.0
    assert bad["metrics"]["runtime_reward"] < good["metrics"]["runtime_reward"]


def test_live_runtime_benchmark_scores_a_scripted_model_and_bad_reads():
    from chowder.runtime_eval import RuntimeTask, run_live_benchmark

    plan = (
        RuntimeTask("t_version", "Repair version.py.", {"version.py": "def parse_version(s):\n    return tuple(int(p) for p in s.split('.'))\n"}, "version.py", "while len(parts) < 3", 2, "2 passed"),
        RuntimeTask("t_sum", "Repair sum_text.py.", {"sum_text.py": "def total(values):\n    return sum(values) - 1\n"}, "sum_text.py", "return sum(values)", 3, "3 passed"),
        RuntimeTask("t_slug", "Repair slugify.py.", {"slugify.py": "def slugify(value):\n    return value\n"}, "slugify.py", ".strip().lower()", 2, "2 passed"),
    )
    state = {"task": 0, "turn": 0}
    tasks = [
        ("version.py", "while len(parts) < 3", "2 passed"),
        ("sum_text.py", "return sum(values)", "3 passed"),
        ("slugify.py", ".strip().lower()", "2 passed"),
    ]

    def generate(messages):
        path, fix, _ = tasks[state["task"]]
        turn = state["turn"]
        state["turn"] += 1
        if turn == 0:
            return "<tool_call>read_file<arg_key>path</arg_key><arg_value>" + path + "</arg_value></tool_call>"
        if turn == 1:
            return "<tool_call>write_file<arg_key>path</arg_key><arg_value>" + path + "</arg_value><arg_key>content</arg_key><arg_value>" + fix + "</arg_value></tool_call>"
        if turn == 2:
            return "<tool_call>run_tests</tool_call>"
        state["task"] += 1
        state["turn"] = 0
        return "The observed suite is green."

    result = run_live_benchmark(generate, max_turns=4, tasks=plan)
    assert result["metrics"]["runtime_green_rate"] == 1.0
    assert result["metrics"]["runtime_nonexistent_read_rate"] == 0.0
    assert result["metrics"]["runtime_reward"] > 0
    assert all(row["green_seen"] for row in result["tasks"])
    assert any(action["event"] == "observed_green" for row in result["tasks"] for action in row["action_rewards"])
