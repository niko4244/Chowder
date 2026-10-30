"""No-weight tests of the exact expert-axis GGUF topology contract."""

from copy import deepcopy
import json

import pytest

from chowder_batch.reap_gguf_header_preflight import (
    GGUFHeaderRefusal, run_preflight, verify_header_topology,
)


def _witness():
    t = {"num_hidden_layers": 2, "num_experts": 3, "num_experts_per_tok": 2,
         "moe_intermediate_size": 8, "hidden_size": 16, "vocab_size": 25}
    cfg = {"model_type": "qwen3_5_moe", "text_config": t}
    fields = {"general.architecture": "qwen35moe", "general.file_type": 15,
              "general.quantization_version": 2}
    fields.update({"qwen35moe." + k: v for k, v in (
        ("block_count", 2), ("expert_count", 3), ("expert_used_count", 2),
        ("expert_feed_forward_length", 8), ("embedding_length", 16))})
    tensors = {"output.weight": (16, 25), "token_embd.weight": (16, 25)}
    for i in range(2):
        prefix = f"blk.{i}."
        tensors.update({prefix + "ffn_gate_inp.weight": (16, 3),
                        prefix + "ffn_down_exps.weight": (8, 16, 3),
                        prefix + "ffn_gate_exps.weight": (16, 8, 3),
                        prefix + "ffn_up_exps.weight": (16, 8, 3)})
    prune = {"keep": 3, "of": 4, "kept_experts": {"0": [0, 1, 3], "1": [0, 2, 3]}}
    return fields, tensors, cfg, prune


def test_complete_routed_inventory_passes_as_metadata_only():
    result = verify_header_topology(*_witness())
    assert result["checked_tensor_shapes"] == 10
    assert result["status"] == "GGUF_METADATA_TOPOLOGY_MATCH_ONLY"
    assert "Not a SHA-256" in result["limits"]


@pytest.mark.parametrize("file_type", (7, 15, 32))
def test_frozen_bf16_q8_q4_topology_requires_exact_gguf_stage_type(file_type):
    fields, tensors, config, prune = _witness()
    fields["general.file_type"] = file_type
    assert verify_header_topology(
        fields, tensors, config, prune, expected_file_type=file_type,
    )["file_type"] == file_type
    with pytest.raises(GGUFHeaderRefusal, match="file type"):
        verify_header_topology(
            fields, tensors, config, prune,
            expected_file_type={7: 15, 15: 32, 32: 7}[file_type],
        )


@pytest.mark.parametrize("file_type", (7, 15, 32))
def test_actual_tiny_gguf_reader_verifies_routed_moe_headers(tmp_path, file_type):
    """Real gguf-py writer/reader, not mock dictionary-only validation."""
    numpy = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    fields, tensors, config, prune = _witness()
    model = tmp_path / "candidate"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    (model / "reap-prune.json").write_text(json.dumps(prune))
    output = tmp_path / "fixture.gguf"
    writer = gguf.GGUFWriter(str(output), "qwen35moe")
    writer.add_file_type(file_type)
    writer.add_quantization_version(2)
    for key in fields:
        if key.startswith("qwen35moe."):
            writer.add_uint32(key, fields[key])
    for key, shape in tensors.items():
        writer.add_tensor(key, numpy.zeros(shape[::-1], dtype=numpy.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    receipt = run_preflight(output, model, expected_file_type=file_type)
    assert receipt["status"] == "GGUF_METADATA_TOPOLOGY_MATCH_ONLY"
    assert receipt["checked_tensor_shapes"] == 10
    assert receipt["file_type"] == file_type


@pytest.mark.parametrize("problem", ["wrong_architecture", "wrong_quant", "wrong_experts",
                                    "wrong_top_k", "wrong_layers", "lost_gate", "lost_down",
                                    "bad_axis", "out_of_range", "stale_prune", "duplicate_survivor",
                                    "wrong_vocab", "missing_selected_layer", "wrong_embedding"])
def test_forged_or_partial_gguf_metadata_refuses(problem):
    fields, tensors, cfg, prune = deepcopy(_witness())
    if problem == "wrong_architecture":
        fields["general.architecture"] = "qwen2"
    elif problem == "wrong_quant":
        fields["general.file_type"] = 7
    elif problem == "wrong_experts":
        fields["qwen35moe.expert_count"] = 4
    elif problem == "wrong_top_k":
        fields["qwen35moe.expert_used_count"] = 1
    elif problem == "wrong_layers":
        fields["qwen35moe.block_count"] = 1
    elif problem == "lost_gate":
        tensors.pop("blk.0.ffn_gate_exps.weight")
    elif problem == "lost_down":
        tensors.pop("blk.1.ffn_down_exps.weight")
    elif problem == "bad_axis":
        tensors["blk.0.ffn_up_exps.weight"] = (3, 8, 16)
    elif problem == "out_of_range":
        tensors["blk.2.ffn_up_exps.weight"] = (16, 8, 3)
    elif problem == "stale_prune":
        prune["keep"] = 2
    elif problem == "duplicate_survivor":
        prune["kept_experts"]["0"] = [0, 0, 1]
    elif problem == "wrong_vocab":
        tensors["output.weight"] = (16, 26)
    elif problem == "missing_selected_layer":
        prune["kept_experts"].pop("1")
    elif problem == "wrong_embedding":
        fields["qwen35moe.embedding_length"] = 12
    with pytest.raises(GGUFHeaderRefusal):
        verify_header_topology(fields, tensors, cfg, prune)
