"""The witness gates: what a real run can record, and what the table accepts.

``chowder.placement_witnesses`` turns one fine-tune's evidence into the four
gates ``chowder_batch/moe_int8_gates.py`` cannot answer, and the runner records
them into the same table. These tests pin the rules that make that safe:

* a verdict follows the evidence, and missing evidence is "not measured" with a
  reason -- never a pass;
* a witness may write only the four witness rows, only into a table bound to the
  same artifact, and can never clear a verdict another run earned;
* ``wddm_spill`` is an operator gate and needs a basis, because no tool can see
  driver paging;
* with the runner's rows already measured, a witness file completes the table and
  a campaign may declare the placement. That is the whole point.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))

import moe_int8_gates as gates  # noqa: E402
from chowder import moe_placement_gates as mpg  # noqa: E402
from chowder import placement_witnesses as pw  # noqa: E402

# --- the evidence -------------------------------------------------------------


def test_entropy_is_the_log_of_the_support_at_uniform_load():
    import math

    assert pw.entropy([5, 5, 5, 5]) == pytest.approx(math.log(4))
    assert pw.entropy([9, 0, 0]) == pytest.approx(0.0)
    assert pw.entropy([]) == 0.0


torch = pytest.importorskip("torch")


class _LoRAB(torch.nn.Module):
    """One LoRA B matrix, named the way PEFT names it: ``lora_B.default.weight``."""

    def __init__(self) -> None:
        super().__init__()
        self.default = torch.nn.Linear(2, 2, bias=False)


class _LeafModel(torch.nn.Module):
    """A tiny model tree whose parameter names carry ``.layers.<n>.``."""

    def __init__(self, grads) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for layer in range(max(grads) + 1):
            layer_module = torch.nn.Module()
            attention = torch.nn.Module()
            attention.lora_B = _LoRAB()
            layer_module.self_attn = attention
            self.model.layers.append(layer_module)
            if grads.get(layer) is not None:
                attention.lora_B.default.weight.grad = grads[layer]


def test_gradient_gate_passes_on_finite_nonzero_leaves():
    model = _LeafModel({0: torch.ones(2, 2), 1: torch.ones(2, 2), 2: torch.ones(2, 2)})
    evidence = pw.collect_lora_grads(model)
    gate = pw.gradient_gate(evidence)
    assert evidence["layers"] == [0, 1, 2]
    assert gate["verdict"] == "pass", gate
    assert "3/3" in gate["value"]


def test_gradient_gate_fails_on_an_exactly_zero_leaf():
    model = _LeafModel({0: torch.ones(2, 2), 1: torch.ones(2, 2), 2: torch.zeros(2, 2)})
    gate = pw.gradient_gate(pw.collect_lora_grads(model))
    assert gate["verdict"] == "FAIL"
    assert "exactly zero" in gate["value"]


def test_gradient_gate_fails_on_a_non_finite_leaf():
    model = _LeafModel(
        {0: torch.ones(2, 2), 1: torch.ones(2, 2), 2: torch.full((2, 2), float("nan"))}
    )
    gate = pw.gradient_gate(pw.collect_lora_grads(model))
    assert gate["verdict"] == "FAIL"
    assert "non-finite" in gate["value"]


def test_gradient_gate_records_a_missing_gradient():
    model = _LeafModel({0: None, 1: torch.ones(2, 2), 2: torch.ones(2, 2)})
    gate = pw.gradient_gate(pw.collect_lora_grads(model))
    assert gate["verdict"] == "FAIL"
    assert "no grad" in gate["value"]


# --- smoke --------------------------------------------------------------------


def log(*losses, start=10):
    return {"entries": [{"step": start * (i + 1), "loss": loss} for i, loss in enumerate(losses)]}


def test_smoke_passes_when_the_loss_fell_over_twenty_steps():
    gate = pw.smoke_gate(log(2.4, 2.3, 2.1, 1.9, 1.8))
    assert gate["verdict"] == "pass"
    assert "mean loss 2.4" in gate["value"]


def test_smoke_fails_when_the_loss_rose():
    gate = pw.smoke_gate(log(1.8, 1.9, 2.1, 2.3, 2.4))
    assert gate["verdict"] == "FAIL"


def test_smoke_is_not_measured_below_the_preregistered_step_count():
    gate = pw.smoke_gate({"entries": [{"step": 5, "loss": 2.0}, {"step": 5, "loss": 1.9}]})
    assert gate["verdict"] == "not measured"
    assert "needs 20" in gate["value"]


def test_smoke_is_not_measured_with_one_point():
    gate = pw.smoke_gate({"entries": [{"step": 30, "loss": 1.9}]})
    assert gate["verdict"] == "not measured"
    assert "one logged loss point" in gate["value"]


def test_smoke_is_not_measured_without_a_loss():
    assert pw.smoke_gate({"entries": []})["verdict"] == "not measured"
    assert pw.smoke_gate(None)["verdict"] == "not measured"


# --- router balance -----------------------------------------------------------


def test_router_balance_passes_when_the_load_held():
    step0 = {"0": [10, 10, 10, 10], "1": [20, 20, 20, 20]}
    last = {"0": [10, 11, 10, 9], "1": [20, 19, 21, 20]}
    gate = pw.router_balance_gate({"step0": step0, "last": last})
    assert gate["verdict"] == "pass"
    assert "worst change" in gate["value"]


def test_router_balance_fails_when_one_expert_took_over():
    gate = pw.router_balance_gate(
        {"step0": {"0": [25, 25, 25, 25]}, "last": {"0": [97, 1, 1, 1]}}
    )
    assert gate["verdict"] == "FAIL"


def test_router_balance_needs_two_windows():
    gate = pw.router_balance_gate({"step0": {"0": [1, 1]}, "last": None})
    assert gate["verdict"] == "not measured"
    assert "fewer than two steps" in gate["value"]
    assert pw.router_balance_gate(None)["verdict"] == "not measured"


def test_router_recorder_counts_the_routing_decision_it_hooks():
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=8,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval()
    with pw.RouterLoadRecorder(model) as recorder:
        assert recorder.layers == 2
        with torch.no_grad():
            model(torch.randint(0, 64, (1, 8)))
        window = recorder.window()
    assert set(window) == {"0", "1"}
    for counts in window.values():
        assert len(counts) == 6
        # Every token routes to exactly 2 of 6 experts: 8 tokens, 16 slots.
        assert sum(counts) == 16
        assert pw.entropy(counts) > 0.0


class _NullCallback:
    """Stands in for ``transformers.TrainerCallback``: only the hooks matter here."""

    def __init__(self) -> None:
        pass


def test_the_callback_collects_gradients_and_router_windows_from_a_real_step():
    """One real forward/backward on a tiny MoE: the producer's own plumbing."""
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=8,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    # Two LoRA B leaves the way PEFT leaves them, so the grad gate has targets.
    for layer in model.model.layers:
        layer.self_attn.lora_B = _LoRAB()

    callback = pw.build_witness_callback(_NullCallback, model)
    callback.on_train_begin(None, None, None)
    callback.on_step_begin(None, None, None)
    out = model(torch.randint(0, 64, (1, 8)), labels=torch.randint(0, 64, (1, 8)))
    out.loss.backward()
    for layer in model.model.layers:
        layer.self_attn.lora_B.default.weight.grad = torch.ones(2, 2)
    callback.on_step_end(None, None, None)
    evidence = callback.evidence()
    callback.on_train_end(None, None, None)

    assert evidence["router_error"] is None, evidence["router_error"]
    assert set(evidence["router"]["step0"]) == {"0", "1"}
    assert all(sum(counts) == 16 for counts in evidence["router"]["step0"].values())
    assert evidence["gradient"]["leaves"] == 2

    gates = pw.derive(
        gradient=evidence["gradient"],
        router=evidence["router"],
        step_log=log(2.4, 2.2, 2.0, 1.9, 1.8),
        gpu={},
    )
    assert gates["gradient"]["verdict"] == "pass"
    assert gates["smoke"]["verdict"] == "pass"
    assert gates["router_balance"]["verdict"] == "pass"
    assert gates["wddm_spill"]["verdict"] == "not measured"


# --- the table ----------------------------------------------------------------


def runner_record(*, digest="a" * 64, peak_vram=11.2, s_per_step=1.5):
    return {
        "schema": gates.SCHEMA,
        "base_model": "G:/llm-cold/bf16/Qwen3.6-REAP160",
        "device": "cuda:0",
        "dtype": "auto (the checkpoint's)",
        "placement": mpg.EXPERT_PLACEMENT,
        "artifact": {"config_sha256": digest, "shards": 19, "shard_bytes": 1},
        "measured_at": "2026-10-20T12:00:00Z",
        "code_commit": "4aa5891",
        "steps": 3,
        "seq_len": 2048,
        "arms": {
            "int8": {
                "load_seconds": 12.0,
                "load_peak_rss_gb": 20.0,
                "total_peak_rss_gb": 21.0,
                "census": {"patched_expert_modules": 40, "remaining_expert_params": 0},
                "census_meta": {"expert_bytes_gb": 20.19, "patched_expert_modules": 40},
                "s_per_step": s_per_step,
                "step_seconds": [s_per_step] * 3,
                "peak_vram_gb": peak_vram,
                "reserved_gb": peak_vram + 0.4,
                "gates": {
                    "load_census": "measured",
                    "memory_vram": "measured",
                    "memory_vram_pass": peak_vram <= mpg.VRAM_GATE_GB,
                    "host_rss": "measured",
                    "throughput": "measured (placement step, no LoRA/optimizer)",
                    "gradient": "not measured by this runner (needs a LoRA run)",
                    "smoke": "not measured by this runner (needs a dataset)",
                    "router_balance": "not measured by this runner (needs a smoke run)",
                    "wddm_spill": "not observable from here; compare peak_vram_gb to the card",
                },
            }
        },
    }


PROSE = "# Pre-registration\n\nProse nobody may rewrite.\n\n"


def table(**kwargs):
    return PROSE + mpg.render_block(runner_record(**kwargs), previous=None)


def witness(digest="a" * 64, **overrides):
    gate = lambda value, verdict: {"value": value, "verdict": verdict, "evidence": {}}
    gates_ = {
        "gradient": gate("layers 0, 22, 44: 6/6 LoRA B leaves finite and non-zero", "pass"),
        "smoke": gate("30 steps, mean loss 2.400 -> 1.800 (-25.0%)", "pass"),
        "router_balance": gate("40 MoE layers, worst change +3.1% in layer 12", "pass"),
        "wddm_spill": gate(
            "12.1 GB peak allocated, 12.4 GB reserved of a 16.3 GB card",
            "not measured",
        ),
    }
    gates_.update(overrides)
    return {
        "schema": mpg.WITNESS_SCHEMA,
        "placement": mpg.EXPERT_PLACEMENT,
        "base_model": "G:/llm-cold/bf16/Qwen3.6-REAP160",
        "artifact": {"config_sha256": digest, "shards": 19, "shard_bytes": 1},
        "device": "cuda:0",
        "steps": 30,
        "run": "gen1-candidate-7",
        "measured_at": "2026-10-20T13:00:00Z",
        "code_commit": "4aa5891",
        "gates": gates_,
    }


def test_a_witness_updates_only_the_rows_it_owns():
    document = table()
    updated, recorded = mpg.splice_witness_rows(document, witness())
    parsed = mpg.parse_block(updated)
    assert recorded == ["gradient", "router_balance", "smoke", "wddm_spill"]
    assert parsed["verdicts"]["gradient"] == "pass"
    assert parsed["verdicts"]["smoke"] == "pass"
    # The runner's rows are untouched, number and verdict alike.
    before = mpg.parse_block(document)
    for gate in mpg.RUNNER_GATES:
        assert parsed["values"][gate] == before["values"][gate]
    assert "Prose nobody may rewrite." in updated
    assert "witness gates: `gradient`, `router_balance`, `smoke`, `wddm_spill`" in updated
    assert "gen1-candidate-7" in updated


def test_witness_value_with_vertical_bars_preserves_gradient_gate():
    """A real gradient witness says '|g|': this must not split Markdown cells."""
    entry = {"value": "19/19 leaves (smallest peak |g| 0.0075)", "verdict": "pass"}
    updated, _ = mpg.splice_witness_rows(table(), witness(gradient=entry))
    assert "&#124;g&#124;" in updated
    parsed = mpg.parse_block(updated)
    assert parsed["verdicts"]["gradient"] == "pass"
    assert parsed["values"]["gradient"] == entry["value"]


def test_a_witness_cannot_write_a_row_the_runner_owns():
    with pytest.raises(mpg.PlacementGateError, match="does not own"):
        mpg.splice_witness_rows(
            table(), witness(**{"throughput": {"value": "0.1 s", "verdict": "pass"}})
        )


def test_a_witness_from_another_artifact_is_refused():
    with pytest.raises(mpg.PlacementGateError, match="not about the same artifact"):
        mpg.splice_witness_rows(table(), witness(digest="b" * 64))


def test_a_witness_cannot_clear_a_verdict_another_run_earned():
    document = mpg.splice_witness_rows(table(), witness())[0]
    weaker = witness(**{"gradient": {"value": "not measured", "verdict": "not measured"}})
    with pytest.raises(mpg.PlacementGateError, match="cannot clear a verdict"):
        mpg.splice_witness_rows(document, weaker)


def test_a_table_without_a_block_refuses_a_witness():
    with pytest.raises(mpg.PlacementGateError, match="no gate table"):
        mpg.splice_witness_rows("prose only", witness())


def test_the_operator_gate_needs_a_basis():
    with pytest.raises(mpg.PlacementGateError, match="operator gate"):
        mpg.splice_witness_rows(
            table(), witness(wddm_spill={"value": "no spill", "verdict": "pass"})
        )
    updated, _ = mpg.splice_witness_rows(
        table(),
        witness(
            wddm_spill={
                "value": "operator pass: nvidia-smi showed no shared-memory use",
                "verdict": "pass",
                "evidence": {"basis": "nvidia-smi showed no shared-memory use across the run"},
            }
        ),
    )
    assert mpg.parse_block(updated)["verdicts"]["wddm_spill"] == "pass"


def test_completing_the_table_lets_a_campaign_declare_the_placement(tmp_path):
    """The end this whole producer exists for: runner rows + witness rows = allowed."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    digest = mpg.artifact_identity(tmp_path)["config_sha256"]
    document = table(digest=digest)
    with pytest.raises(mpg.PlacementGateError, match="not satisfied"):
        mpg.assert_declaration_allowed(document, ["campaign.json"], base_model=tmp_path)
    completed, _ = mpg.splice_witness_rows(
        document,
        witness(
            digest=digest,
            wddm_spill={
                "value": "operator pass: no shared-memory use during the run",
                "verdict": "pass",
                "evidence": {"basis": "nvidia-smi showed no shared-memory use during the run"},
            },
        ),
    )
    assert mpg.assert_declaration_allowed(completed, ["campaign.json"], base_model=tmp_path)


# --- the CLI ------------------------------------------------------------------


def test_cli_records_a_witness_file(monkeypatch, tmp_path, capsys):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    digest = mpg.artifact_identity(tmp_path)["config_sha256"]
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(table(digest=digest), encoding="utf-8")
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(witness(digest=digest)), encoding="utf-8")

    code = gates.main(["--witness", str(path), "--prereg", str(prereg)])
    printed = capsys.readouterr().out
    assert code == 0
    assert "recorded gradient, router_balance, smoke, wddm_spill" in printed
    # wddm is still "not measured", so the declaration stays refused -- and says so.
    assert "7/8 gates measured" in printed
    assert "still cannot declare" in printed
    assert mpg.parse_block(prereg.read_text(encoding="utf-8"))["verdicts"]["smoke"] == "pass"


def test_cli_witness_check_does_not_mutate_prereg(tmp_path, capsys):
    """--check is a dry run for witness import as it is for measured gates."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    digest = mpg.artifact_identity(tmp_path)["config_sha256"]
    prereg = tmp_path / "PREREG.md"
    original = table(digest=digest)
    prereg.write_text(original, encoding="utf-8")
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(witness(digest=digest)), encoding="utf-8")
    assert gates.main(["--witness", str(path), "--prereg", str(prereg), "--check"]) == 0
    assert prereg.read_text(encoding="utf-8") == original
    assert "was not written" in capsys.readouterr().out


def test_cli_operator_spill_verdict_needs_a_basis(monkeypatch, tmp_path, capsys):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(table(), encoding="utf-8")
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(witness()), encoding="utf-8")
    code = gates.main(
        ["--witness", str(path), "--prereg", str(prereg), "--spill-verdict", "pass"]
    )
    assert code == 2
    assert "--spill-basis" in capsys.readouterr().err
    assert mpg.parse_block(prereg.read_text(encoding="utf-8"))["verdicts"]["smoke"] == "not measured"


def test_cli_operator_spill_verdict_completes_the_table(tmp_path, capsys):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    digest = mpg.artifact_identity(tmp_path)["config_sha256"]
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(table(digest=digest), encoding="utf-8")
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(witness(digest=digest)), encoding="utf-8")
    code = gates.main(
        [
            "--witness",
            str(path),
            "--prereg",
            str(prereg),
            "--spill-verdict",
            "pass",
            "--spill-basis",
            "nvidia-smi showed no shared-memory use during the run",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "8/8 gates measured" in printed
    assert "may declare the placement" in printed


def test_the_router_recorder_needs_the_base_model_not_the_peft_wrapper():
    """The 2026-10-21 witness run audited the *trained* model and was refused.

    ``get_peft_model`` puts the LoRA wrapper at ``model.model``, so the text
    stack the audit looks for is one level further down and the audit -- which
    will not guess an alternate path -- raises. The producer keeps the base
    model and hands that to the recorder; the hooks reach the same
    ``mlp.gate`` modules either way.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    from chowder.moe_instrumentation import MoeArchitectureAuditError

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=8,
    )
    base = AutoModelForCausalLM.from_config(config).eval()
    wrapped = get_peft_model(
        base, LoraConfig(r=2, target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM)
    )
    with pytest.raises(MoeArchitectureAuditError, match="cannot locate model.model.layers"):
        pw.RouterLoadRecorder(wrapped)
    with pw.RouterLoadRecorder(base) as recorder:
        assert recorder.layers == 2


def test_the_recorder_audits_the_campaign_architecture_not_just_qwen3moe():
    """The shipped model is ``Qwen3_5MoeForCausalLM`` (LinearAttention + fused
    experts). It exposes the audited shape on the same ``model.model.layers``
    path, so the recorder counts the router's own indices there too."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=64,
        head_dim=8,
    )
    model = AutoModelForCausalLM.from_config(config).train()
    with pw.RouterLoadRecorder(model) as recorder:
        assert recorder.layers == 2
        out = model(
            input_ids=torch.randint(0, 64, (1, 16)),
            labels=torch.randint(0, 64, (1, 16)),
            use_cache=False,
        )
        out.loss.backward()
        window = recorder.window()
    for counts in window.values():
        assert len(counts) == 6
        assert sum(counts) == 32  # 16 tokens x top-2


def test_the_recorder_audits_the_int8_placement_it_is_meant_to_witness(tmp_path):
    """The 2026-10-21 witness run passed a clean int8 load and was refused here.

    ``prepare_int8_placement`` replaces an experts module's bf16 parameters with
    four int8 buffers, so ``gate_up_proj``/``down_proj`` are gone and the audit
    found no MoE layer at all -- on exactly the model the placement produces.
    The int8 module is a second verified shape: same router, same fused layout,
    same ``num_experts``/``act_fn``.
    """
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    from chowder.backends import moe_int8_experts as mie
    from chowder.moe_instrumentation import INT8_EXPERT_BUFFERS

    assert tuple(INT8_EXPERT_BUFFERS) == tuple(mie.INT8_EXPERT_BUFFERS)

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=8,
    )
    base = tmp_path / "tiny"
    AutoModelForCausalLM.from_config(config).save_pretrained(base)
    model, _census = mie.load_model_int8_experts(base, device="cpu")
    mie.prepare_int8_placement(model, device_type="cpu")

    with pw.RouterLoadRecorder(model) as recorder:
        assert recorder.layers == 2
        out = model(
            input_ids=torch.randint(0, 64, (1, 8)),
            labels=torch.randint(0, 64, (1, 8)),
            use_cache=False,
        )
        out.loss.backward()
        window = recorder.window()
    for counts in window.values():
        assert len(counts) == 6
        assert sum(counts) == 16  # 8 tokens x top-2


def test_an_int8_load_without_the_install_still_runs_the_bf16_experts(tmp_path):
    """Why `prepare_int8_placement` belongs in every int8 consumer.

    The loader produces the int8 buffers and drops the bf16 expert parameters,
    but the placement is not in force until the int8 forward is installed. This
    is the trap that cost the 2026-10-21 09:14 witness run a 21-minute load: the
    bf16 fused forward ran, looked for `gate_up_proj`, and raised.
    """
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    from chowder.backends import moe_int8_experts as mie

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        head_dim=8,
    )
    base = tmp_path / "tiny"
    AutoModelForCausalLM.from_config(config).save_pretrained(base)
    model, _census = mie.load_model_int8_experts(base, device="cpu")

    with pytest.raises(AttributeError, match="gate_up_proj"):
        model(input_ids=torch.randint(0, 64, (1, 8)), use_cache=False)

    mie.prepare_int8_placement(model, device_type="cpu")
    with torch.no_grad():
        out = model(input_ids=torch.randint(0, 64, (1, 8)), use_cache=False)
    assert out.logits.shape == (1, 8, 64)
