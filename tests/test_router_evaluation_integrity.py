"""Receipt and actual top-k routing evidence for the CPU evaluator."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chowder.backends.router_healing_eval_worker import _routing_counts


class _RoutingModel(torch.nn.Module):
    def __init__(self, *, selected=None, top_k=2):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.top_k = top_k
        self.mlp.gate = torch.nn.Linear(1, 4, bias=False)
        with torch.no_grad():
            self.mlp.gate.weight.copy_(torch.tensor([[3.0], [2.0], [1.0], [0.0]]))
        if selected is not None:
            def return_selections(_module, _inputs, logits):
                indices = torch.tensor(selected).expand(logits.shape[0], -1)
                return logits, torch.ones_like(indices, dtype=torch.float32), indices
            self.mlp.gate.register_forward_hook(return_selections)

    def forward(self, input_ids, labels=None):
        return self.mlp.gate(input_ids.reshape(-1, 1).float())


def test_logits_only_router_counts_second_choice_as_used():
    counts = _routing_counts(torch, _RoutingModel(), [torch.ones(1, 3, dtype=torch.long)])
    assert counts == {"mlp.gate": [3, 3, 0, 0]}


def test_router_returned_selections_take_precedence_over_recomputed_logits():
    counts = _routing_counts(
        torch, _RoutingModel(selected=[2, 3]), [torch.ones(1, 3, dtype=torch.long)]
    )
    assert counts == {"mlp.gate": [0, 0, 3, 3]}


@pytest.mark.parametrize("top_k", [None, 0, 5])
def test_unsupported_routing_width_is_refused(top_k):
    with pytest.raises(RuntimeError, match="top.k"):
        _routing_counts(torch, _RoutingModel(top_k=top_k), [torch.ones(1, 3, dtype=torch.long)])


def test_out_of_range_actual_selections_are_refused():
    with pytest.raises(RuntimeError, match="indices"):
        _routing_counts(torch, _RoutingModel(selected=[2, 4]), [torch.ones(1, 3, dtype=torch.long)])


def test_real_tiny_moe_counts_all_selected_experts():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=8, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
    )).float().eval()
    counts = _routing_counts(torch, model, [torch.tensor([[1, 2, 3]])])
    assert len(counts) == 2
    assert all(sum(layer) == 3 * 2 for layer in counts.values())


@pytest.mark.parametrize("changed_pin", ["payload_manifest_sha256", "payload_tensor_sha256"])
def test_worker_rejects_wrong_receipt_before_loading_model(tmp_path, monkeypatch, changed_pin):
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM
    from chowder.backends import router_healing_eval_worker as worker
    from chowder.router_payload import RouterPayloadError, save_router_payload

    receipt = save_router_payload(
        {"mlp.gate.weight": torch.ones(2, 2)}, tmp_path / "payload",
        base_content_sha256="a" * 64, spec_digest="b" * 64, steps_completed=1,
    )
    spec = SimpleNamespace(
        device="cpu", base_model_dir="not-loaded", base_content_sha256="a" * 64,
        payload_dir=receipt["payload_dir"],
        payload_manifest_sha256=receipt["manifest_sha256"],
        payload_tensor_sha256=receipt["tensor_file_sha256"],
    )
    setattr(spec, changed_pin, "0" * 64)
    monkeypatch.setattr(worker, "resolve_base_identity", lambda _: {"content_sha256": "a" * 64})

    def forbid_model_load(*args, **kwargs):
        raise AssertionError("a bad receipt must be rejected before model loading")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbid_model_load)
    with pytest.raises(RouterPayloadError, match="receipt hash mismatch"):
        worker.evaluate(spec)
