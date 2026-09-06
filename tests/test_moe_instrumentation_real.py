"""Real-hardware acceptance test for MoE instrumentation -- a genuine local
MoE checkpoint loaded with real weights, not a synthetic fake module.

Scope note (read before extending this test): the roadmap's actual target is
a local Qwen3.6-35B-A3B checkpoint (docs/MOE_DOWNSIZING.md), which does not
exist on this machine -- confirmed by an exhaustive filesystem search across
every attached drive. Per that doc's own Phase A rule ("failure to identify a
tensor/module is a hard stop, not a reason to guess a name"), this test does
not fabricate that checkpoint's presence. Instead it validates the real
instrumentation code path against OLMoE-1B-7B (/h/Models/olmoe-1b-7b), the
one genuine, local, HF-format (safetensors), architecturally-MoE checkpoint
present on this machine. This is a legitimate substitute for verifying the
*mechanism* -- confirmed by reading transformers==5.16.1's installed source,
OlmoeSparseMoeBlock/OlmoeTopKRouter/OlmoeExperts share the exact same
fused-batched-expert shape (mlp.gate.{weight,num_experts,top_k},
mlp.experts.{gate_up_proj,down_proj,num_experts,act_fn}) as
Qwen3MoeSparseMoeBlock and Qwen3_5MoeSparseMoeBlock in this same transformers
version -- but it is NOT a commissioning of the actual Qwen3.6-35B-A3B
target, which remains blocked pending that checkpoint's local availability.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_REAL_MOE_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_MOE_SMOKE") != "1",
    reason="real MoE instrumentation smoke requires CHOWDER_REAL_MOE_SMOKE=1 "
    "and the local OLMoE-1B-7B checkpoint at H:/Models/olmoe-1b-7b",
)
_OLMOE_LOCAL_PATH = Path("H:/Models/olmoe-1b-7b")


@_REAL_MOE_SMOKE
def test_real_olmoe_checkpoint_audits_and_calibrates_with_the_verified_moe_shape(tmp_path):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from chowder.moe_instrumentation import (
        DEFAULT_CALIBRATION_TEXTS,
        audit_moe_architecture,
        run_calibration,
        write_expert_importance_jsonl,
    )
    from chowder.moe_planning import ImportanceWeights, build_uniform_pruning_plan

    assert _OLMOE_LOCAL_PATH.is_dir(), f"expected a real local checkpoint at {_OLMOE_LOCAL_PATH}"

    tokenizer = AutoTokenizer.from_pretrained(str(_OLMOE_LOCAL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(_OLMOE_LOCAL_PATH),
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    audit = audit_moe_architecture(model)
    assert audit.model_type == "olmoe"
    assert audit.num_hidden_layers == 16
    assert audit.dense_layer_indices == ()  # OLMoE is uniformly sparse
    assert len(audit.moe_layers) == 16
    for structure in audit.moe_layers:
        assert structure.num_experts == 64
        assert structure.num_experts_per_tok == 8

    audit, records = run_calibration(model, tokenizer, DEFAULT_CALIBRATION_TEXTS, device="cuda")

    assert len(records) == 16 * 64
    total_selected = sum(record.selected_tokens for record in records)
    assert total_selected > 0, "no expert was ever selected -- routing hook did not fire for real"

    out_path = tmp_path / "expert_importance.jsonl"
    write_expert_importance_jsonl(
        out_path,
        audit=audit,
        records=records,
        model_source=str(_OLMOE_LOCAL_PATH),
        calibration_texts=DEFAULT_CALIBRATION_TEXTS,
        transformers_version=transformers.__version__,
    )
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 16 * 64
    provenance = json.loads(lines[0])["provenance"]
    assert provenance["model_type"] == "olmoe"
    assert provenance["moe_layer_count"] == 16

    for retention in (0.75, 0.5):
        plan = build_uniform_pruning_plan(
            records,
            retention_fraction=retention,
            minimum_survivors_per_layer=8,  # never prune below the real top_k floor
            weights=ImportanceWeights(),
        )
        assert plan.actual_retention_fraction == pytest.approx(retention, abs=0.02)
        for layer_plan in plan.layers:
            assert layer_plan.total_experts == 64
            assert len(layer_plan.keep_experts) >= 8
            # no model surgery happened -- this is a dry-run plan only
        assert model.model.layers[0].mlp.experts.gate_up_proj.shape[0] == 64
