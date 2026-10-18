"""The third evaluation placement: the routed experts on the host.

``resident`` and ``offload`` answer a *dense* model that does not fit on the card.
An MoE whose experts do not fit needs the sparse analogue, and training already
runs one: ``split_device_map`` loads the base with the routed experts on the host
and everything else on the device, and ``prepare_split_model`` freezes them and
patches their forward to copy one expert's slice per forward. See
``docs/quals/MOE_TRANSIENT_GRAD_PLACEMENT_PREREG_2026-10-18.md``.

What is pinned here, in order: the placement is declared while a value that is not
a placement is still refused; a request this placement cannot honour is refused
before a single weight is touched; a real evaluation load puts every expert
parameter on the host and reports the census it measured; its generations are the
resident generations token for token, with and without an adapter, since attaching
one is what re-places every other placement; and the attach leaves the split map
on the model rather than re-planning around it.

The model is a tiny randomly-initialised ``Qwen3_5MoeTextConfig`` (helpers copied
from ``test_moe_transient_grad``, deliberately not imported: a fixture those tests
may change must not couple this file). No real weights are loaded anywhere here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")

from chowder.backends.moe_transient_grad import (  # noqa: E402
    prepare_split_model,
    split_device_map,
)
from chowder.evaluators import transformers_text_worker as worker  # noqa: E402
from chowder.evaluators.placement import (  # noqa: E402
    needs_redispatch_after_adapter,
)
from chowder.evaluators.transformers_text import (  # noqa: E402
    EvalSuiteSpec,
    TransformersTextEvalSpec,
)

LAYERS, HIDDEN, MOE_INT = 4, 32, 16

#: Two short prompts, and long enough to be decoded greedily to a few tokens.
PROMPTS = ("<t1> <t2> <t3>", "<t4> <t5>")

_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the 'experts-cpu' placement copies the experts to a CUDA device",
)


def _config():
    return configuration.Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, moe_intermediate_size=MOE_INT,
        shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, full_attention_interval=2, max_position_embeddings=64,
    )


def _tiny_moe():
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_config()).float()
    for name, param in model.named_parameters():
        if ".mlp.experts." in name:
            param.requires_grad_(False)
    return model


def _tiny_tokenizer():
    """A word-level tokenizer that needs no files on disk to build."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"<t{i}>": i for i in range(30)}
    vocab["<pad>"] = 30
    vocab["<eos>"] = 31
    vocab["<unk>"] = 32
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("experts-cpu-base")
    _tiny_moe().save_pretrained(directory)
    _tiny_tokenizer().save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_adapter(tmp_path_factory) -> Path:
    """A tiny LoRA whose B matrices are non-zero, so it is a live adapter."""
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    from chowder.adapter_guard import assert_adapter_is_live

    directory = tmp_path_factory.mktemp("experts-cpu-adapter")
    model = AutoModelForCausalLM.from_config(_config())
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.02)
    peft_model.save_pretrained(directory)
    # Refuses (raises) unless at least one LoRA B is verified non-zero.
    assert_adapter_is_live(peft_model, directory)
    return directory


def _split_load(checkpoint: Path):
    """The worker's ``"experts-cpu"`` load path: the split map, then the census."""
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        dtype=torch.bfloat16,
        local_files_only=True,
        device_map=split_device_map(
            str(checkpoint), revision=None, local_files_only=True
        ),
    )
    return base, prepare_split_model(base)


def _resident_load(checkpoint: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        trust_remote_code=False,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    return model.to("cuda:0")


def _greedy_tokens(model: Any, tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer([prompt], return_tensors="pt")
    width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return generated[0, width:].tolist()


def _dataset(path: Path) -> Path:
    path.write_text(
        "".join(
            json.dumps({"prompt": prompt, "expected": prompt}) + "\n"
            for prompt in PROMPTS
        ),
        encoding="utf-8",
    )
    return path


def _spec(
    checkpoint: Path,
    output_dir: Path,
    dataset: Path,
    *,
    placement: str,
    quantization: str = "none",
    adapter_dir: str | None = None,
) -> TransformersTextEvalSpec:
    return TransformersTextEvalSpec(
        base_model=str(checkpoint),
        adapter_dir=adapter_dir,
        output_dir=str(output_dir),
        suites=(
            EvalSuiteSpec(name="tiny", dataset=str(dataset), max_new_tokens=8),
        ),
        precision="bf16",
        quantization=quantization,
        device="auto",
        placement=placement,
        offline=True,
    )


def _predictions(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "predictions-tiny.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_experts_cpu_is_a_declared_placement_and_anything_else_is_not():
    spec = _spec(
        Path("unused"),
        Path("unused"),
        Path("unused"),
        placement="experts-cpu",
    )
    assert spec.placement == "experts-cpu"

    with pytest.raises(ValueError, match="evaluation placement must be one of"):
        _spec(Path("unused"), Path("unused"), Path("unused"), placement="experts")


def test_experts_cpu_refuses_a_quantized_base_before_loading_anything(tmp_path):
    # Both paths name files that do not exist: reaching either would mean the
    # refusal came too late, after a tokenizer or a weight had been touched.
    with pytest.raises(RuntimeError, match="requires quantization 'none'"):
        worker.evaluate(
            _spec(
                tmp_path / "no-such-model",
                tmp_path / "out",
                tmp_path / "no-such-dataset.jsonl",
                placement="experts-cpu",
                quantization="4bit",
            )
        )


def test_experts_cpu_refuses_a_host_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_resolve_device", lambda torch, requested: "cpu")
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        worker.evaluate(
            _spec(
                tmp_path / "no-such-model",
                tmp_path / "out",
                tmp_path / "no-such-dataset.jsonl",
                placement="experts-cpu",
            )
        )


def test_the_placement_helpers_never_move_an_experts_cpu_base(monkeypatch):
    """``.to()`` is what would undo the split, and no helper may call it."""

    class _Recording:
        def __init__(self) -> None:
            self.moves: list[str] = []

        def to(self, device):  # noqa: ANN001 - matches torch's signature loosely
            self.moves.append(device)
            return self

    def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("no placement helper may dispatch an experts-cpu base")

    monkeypatch.setattr(worker, "dispatch_offloaded", _boom)
    base = _Recording()
    spec = SimpleNamespace(
        placement="experts-cpu", quantization="none", adapter_dir=None
    )

    assert worker.place_before_adapter(base, spec=spec, device_name="cuda:0") is base
    assert worker.placement_after_adapter(base, spec=spec, device_name="cuda:0") is base
    assert base.moves == []
    # The decision itself lives in one place; the worker only applies it.
    assert not needs_redispatch_after_adapter(
        quantization="none", placement="experts-cpu", adapter=True
    )


def test_the_attach_refuses_when_the_split_device_map_cannot_be_withheld():
    """A silent no-op there would put the experts back on the card."""

    pytest.importorskip("peft")

    class _Base:
        # Not an instance attribute: the withhold would not take effect, so the
        # attach must refuse instead of re-planning the experts onto the device.
        hf_device_map = {"model.layers.0.mlp.experts.gate_up_proj": "cpu"}

    spec = SimpleNamespace(placement="experts-cpu", adapter_dir="adapter")
    with pytest.raises(RuntimeError, match="cannot withhold the split device map"):
        worker.load_adapter(_Base(), spec)


@_CUDA_REQUIRED
def test_a_real_evaluation_places_the_experts_on_the_host_and_reports_the_census(
    tiny_checkpoint, tmp_path
):
    dataset = _dataset(tmp_path / "tiny.jsonl")
    result = worker.evaluate(
        _spec(tiny_checkpoint, tmp_path / "experts-cpu", dataset, placement="experts-cpu")
    )
    resident = worker.evaluate(
        _spec(tiny_checkpoint, tmp_path / "resident", dataset, placement="resident")
    )

    runtime = result["runtime"]
    assert runtime["placement"] == "experts-cpu"
    census = runtime["placement_census"]
    assert census["expert_params"] == census["expert_params_on_cpu"] > 0
    assert census["other_params"] == census["other_params_on_device"] > 0
    assert census["patched_expert_modules"] == LAYERS
    # The field is present, and honestly empty, for a placement that has no
    # census to report.
    assert resident["runtime"]["placement_census"] is None

    # Same greedy generations as a fully resident load of the same model.
    def _generated(output_dir: Path):
        return [
            (row["prediction"], row["generated_tokens"])
            for row in _predictions(output_dir)
        ]

    assert _generated(tmp_path / "experts-cpu") == _generated(tmp_path / "resident")


@_CUDA_REQUIRED
def test_experts_cpu_generations_match_a_resident_load_token_for_token(tiny_checkpoint):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_checkpoint), local_files_only=True)
    split, census = _split_load(tiny_checkpoint)
    resident = _resident_load(tiny_checkpoint)
    assert census["expert_params_on_cpu"] == census["expert_params"] > 0

    for prompt in PROMPTS:
        assert _greedy_tokens(split, tokenizer, prompt) == _greedy_tokens(
            resident, tokenizer, prompt
        ), prompt


@_CUDA_REQUIRED
def test_an_attached_adapter_leaves_the_experts_on_the_host(
    tiny_checkpoint, tiny_adapter
):
    """The attach is the case that re-places every other placement."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_checkpoint), local_files_only=True)
    base, census = _split_load(tiny_checkpoint)
    spec = SimpleNamespace(
        placement="experts-cpu", adapter_dir=str(tiny_adapter), quantization="none"
    )
    model = worker.load_adapter(base, spec)

    experts = [
        (name, param)
        for name, param in model.named_parameters()
        if ".mlp.experts." in name
    ]
    assert len(experts) == census["expert_params"] > 0
    assert {param.device.type for _, param in experts} == {"cpu"}
    # It is a real attach, on the device, not a no-op...
    adapter_params = [param for name, param in model.named_parameters() if "lora_" in name]
    assert adapter_params and {param.device.type for param in adapter_params} == {"cuda"}
    # ...and the split map the load installed is still the model's own.
    assert "cpu" in set(base.hf_device_map.values())

    # The whole point: this arm still generates what a resident arm generates.
    resident_adapter = worker.load_adapter(_resident_load(tiny_checkpoint), spec)
    for prompt in PROMPTS:
        assert _greedy_tokens(model, tokenizer, prompt) == _greedy_tokens(
            resident_adapter, tokenizer, prompt
        ), prompt
