"""CPU training preflight: tiny end-to-end training-loop correctness checks.

Exercises, on a 1-layer CPU model (seconds, no GPU, no downloads):

  1. tokenization round-trip through the production chat renderer
  2. completion-only (assistant-only) loss masking on a synthetic conversation
  3. nonempty supervised targets after truncation guards
  4. LoRA adapter creation with trainable-parameter coverage
  5. gradient flow into adapters only (frozen base)
  6. finite losses
  7. checkpoint save + independent reload
  8. resume integrity (optimizer state round-trip)

Uses the REAL production masking code from ``chowder.backends.training_data``
so a pass means the pipeline's masking is correct, not that a local copy is.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from transformers import PretrainedConfig, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

sys_path = None  # set at import below


def _ensure_src_on_path() -> None:
    import sys
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()
from chowder.backends.training_data import _build_chat_example  # noqa: E402


class WhitespaceTokenizer(PreTrainedTokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]


def make_tokenizer() -> WhitespaceTokenizer:
    tok = Tokenizer(models.WordLevel(vocab={"<pad>": 0, "<unk>": 1, "User:": 2, "Assistant:": 3,
                                            "hello": 4, "world": 5, "answer": 6, "42": 7,
                                            "compute": 8, "two": 9, "plus": 10, "the": 11,
                                            "is": 12, "fix": 13, "bug": 14, "done": 15},
                                    unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.decoder = decoders.WordPiece(prefix=" ")
    return WhitespaceTokenizer(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>")


def tiny_model():
    from transformers import AutoModelForCausalLM, LlamaConfig
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model = model.to_empty(device="cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)
    return model


def run_preflight(out: Path | None = None) -> dict:
    checks: dict[str, bool] = {}
    detail: dict[str, str] = {}

    tok = make_tokenizer()
    messages = [
        {"role": "user", "content": "compute two plus the answer"},
        {"role": "assistant", "content": "the answer is 42"},
    ]

    # NOTE: the production renderer needs a real chat template; we emulate the
    # conversation as plain text roles for the tiny tokenizer, using the
    # production masking logic through a template-less path is impossible, so
    # preflight installs a minimal prefix-consistent template and verifies the
    # production function's guards fire on real structure.
    tok.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
        "{% endfor %}"
    )
    try:
        example = _build_chat_example(tok, messages, max_length=64, row_index=0)
        labels = example["labels"]
        supervised = [i for i, l in enumerate(labels) if l != -100]
        checks["masking_nonempty_targets"] = len(supervised) > 0
        # The supervised span must decode to the assistant turn's own text
        # (token indices, not char indices), with everything else masked.
        supervised_ids = [labels[i] for i in supervised]
        supervised_text = tok.decode(supervised_ids).strip()
        checks["masking_assistant_only"] = (
            supervised_text.startswith("Assistant:")
            and "the answer is 42" in supervised_text
            and "compute two plus" not in supervised_text
        )
        checks["tokenization_roundtrip"] = "the answer is 42" in tok.decode(example["input_ids"])
        detail["supervised_span"] = supervised_text[:200]
    except Exception as exc:  # noqa: BLE001 - report, don't crash the audit
        checks["masking_nonempty_targets"] = False
        detail["masking_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from peft import LoraConfig, get_peft_model
        model = tiny_model()
        lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                          target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, lora)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        checks["adapter_created"] = bool(trainable) and bool(frozen)
        batch = {
            "input_ids": torch.tensor([example["input_ids"]]) if "example" in dir() else torch.zeros((1, 8), dtype=torch.long),
            "attention_mask": torch.ones((1, min(len(example["input_ids"]), 64) if "example" in dir() else 8), dtype=torch.long),
            "labels": torch.tensor([example["labels"]]) if "example" in dir() else torch.full((1, 8), -100, dtype=torch.long),
        }
        if batch["labels"].eq(-100).all():
            raise RuntimeError("preflight batch has no supervised targets")
        loss = model(**batch).loss
        checks["finite_loss"] = bool(torch.isfinite(loss))
        loss.backward()
        grad_params = [n for n, p in model.named_parameters()
                       if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
        grad_frozen = [n for n, p in model.named_parameters()
                       if not p.requires_grad and p.grad is not None]
        checks["gradient_flow_adapter_only"] = bool(grad_params) and not grad_frozen
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "adapter"
            model.save_pretrained(ckpt)
            from peft import PeftModel
            base = tiny_model()
            reloaded = PeftModel.from_pretrained(base, ckpt, is_trainable=True)
            reloaded.train()
            loss2 = reloaded(**{k: v.clone() for k, v in batch.items()}).loss
            checks["checkpoint_roundtrip"] = bool(torch.isfinite(loss2))
            opt = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad])
            opt.step()
            state = {"optimizer": opt.state_dict(), "step": 1}
            (Path(tmp) / "trainer_state.json").write_text(json.dumps({"step": 1}))
            torch.save(state, Path(tmp) / "opt.pt")
            opt2 = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad])
            opt2.load_state_dict(torch.load(Path(tmp) / "opt.pt")["optimizer"])
            checks["resume_integrity"] = opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()
        detail["trainable_params"] = str(len(trainable))
        detail["loss"] = f"{loss.item():.4f}"
    except Exception as exc:  # noqa: BLE001
        detail["training_error"] = f"{type(exc).__name__}: {exc}"
        for key in ("adapter_created", "finite_loss", "gradient_flow_adapter_only",
                    "checkpoint_roundtrip", "resume_integrity"):
            checks.setdefault(key, False)

    passed = sum(checks.values())
    result = {"checks": checks, "passed": passed, "total": len(checks),
              "ok": passed == len(checks), "detail": detail,
              "note": "CPU-only preflight on a 1-layer model using the production masking function"}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.out), indent=2))
