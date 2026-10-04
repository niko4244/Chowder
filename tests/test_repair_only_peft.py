"""CPU-only PEFT fixture for the frozen-parent repair-adapter contract."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402


class TinyRepairModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.out_proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.q_proj(value))


def _config() -> LoraConfig:
    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["q_proj", "out_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def _make_live(model: torch.nn.Module, adapter: str) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name and adapter in name:
                parameter.fill_(0.125)


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def test_frozen_parent_plus_repair_adapter_saves_and_reloads(tmp_path: Path) -> None:
    """The parent stays frozen; the repair module is independently live."""
    torch.manual_seed(7)
    base = TinyRepairModel()
    parent = get_peft_model(base, _config(), adapter_name="default")
    _make_live(parent, "default")
    parent_dir = tmp_path / "parent"
    parent.save_pretrained(parent_dir, safe_serialization=True)
    # This mirrors transformers_worker.py: load the parent non-trainably, then
    # add a separate adapter and activate only that adapter for optimization.
    frozen_parent = PeftModel.from_pretrained(
        TinyRepairModel(), parent_dir, is_trainable=False
    )
    assert all(not parameter.requires_grad for parameter in frozen_parent.parameters())
    frozen_parent.add_adapter("repair", _config())
    frozen_parent.set_adapter("repair")
    for name, parameter in frozen_parent.named_parameters():
        parameter.requires_grad_("repair" in name and "lora_" in name)
    _make_live(frozen_parent, "repair")

    default_before = {
        name: parameter.detach().clone()
        for name, parameter in frozen_parent.named_parameters()
        if "default" in name and "lora_" in name
    }
    repair_before = _clone_state(frozen_parent)
    output_dir = tmp_path / "adapter"
    frozen_parent.save_pretrained(output_dir, safe_serialization=True)
    assert (output_dir / "adapter_model.safetensors").is_file()
    assert (output_dir / "repair" / "adapter_model.safetensors").is_file()

    # Saving the second adapter must not mutate the frozen default tensors.
    for name, parameter in frozen_parent.named_parameters():
        if "default" in name and "lora_" in name:
            assert torch.equal(parameter, default_before[name])

    # Reload the exact published layout: default parent at the root, repair in
    # its subdirectory, then activate both for inference.
    reloaded = PeftModel.from_pretrained(
        TinyRepairModel(), output_dir, adapter_name="default", is_trainable=False
    )
    reloaded.load_adapter(
        str(output_dir / "repair"), adapter_name="repair", is_trainable=True
    )
    reloaded.set_requires_grad(["repair"], True)
    assert any(
        parameter.requires_grad and "repair" in name
        for name, parameter in reloaded.named_parameters()
    )
    reloaded.base_model.add_weighted_adapter(
        ["default", "repair"],
        [1.0, 1.0],
        adapter_name="combined",
        combination_type="linear",
    )
    reloaded.set_adapter("combined", inference_mode=True)
    repair_state = _clone_state(reloaded)
    assert all(
        torch.equal(value, repair_before[name])
        for name, value in repair_state.items()
        if "repair" in name
    )
    assert any("default" in name for name in repair_state)
    assert all(
        not parameter.requires_grad
        for name, parameter in reloaded.named_parameters()
        if "default" in name
    )

    values = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    with torch.no_grad():
        combined = reloaded.base_model.model(values)
        parent_only_model = PeftModel.from_pretrained(
            TinyRepairModel(), parent_dir, adapter_name="default", is_trainable=True
        )
        parent_only = parent_only_model.base_model.model(values)
    assert not torch.equal(combined, parent_only)
    assert all(
        torch.equal(value, default_before[name])
        for name, value in _clone_state(reloaded).items()
        if "default" in name
    )
