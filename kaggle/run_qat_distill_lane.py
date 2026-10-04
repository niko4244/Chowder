"""Kaggle notebook script: QAT accuracy recovery + teacher-trajectory distillation.

Run this in a private Kaggle GPU notebook AFTER ``bootstrap_environment.py``
style pinning. Unlike the local Windows lane, this environment *may* install
``modelopt[hf]`` (isolated notebook venv, transformers >=4.57,<5.15 there is
acceptable); the Chowder package itself is installed from the pinned commit
with the ``ptq,train`` extras.

Provenance-first contract, matching the rest of this directory:

* The Chowder install is pinned to a full 40-char commit and cross-checked
  against pip's ``direct_url.json``; a mismatch aborts before any GPU work.
* The batch-010 dataset is validated before use: every row must be an
  evolve-split, green-verified trajectory generated under a state-aware
  harness with a real ``messages`` transcript. Anything else aborts — a
  synthetic or hand-filled dataset is never tolerated.
* Every reported number is measured in-process: margins come from greedy
  generations with ``output_scores=True``, losses from actual backward steps.
  No throughput or speedup claim is emitted.
* The margin-shift verdict mirrors ``exp_e_confidence.margin_shift_fails_closed``
  (keep the two in sync): serving a quantized arm with an unmeasured or
  over-tolerance mean-margin shift fails routing closed on the local lane.
* No Kaggle Secrets are read; tokens never reach the report.

Typical flow: install -> fingerprint -> validate batch-010 -> BF16 margin
probe -> QAT (mtq INT8 + frozen scales + LoRA) -> quantized margin probe ->
shift report (may fail closed) -> optional KD vs teacher -> SFT on teacher
trajectories -> adapter + report written atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ALLOWED_TRAJECTORY_HARNESSES = {"state_aware", "state_aware+recovery"}


def install_chowder(commit_sha: str) -> None:
    """Install chowder-ai[ptq,train] from the exact pinned commit."""
    if len(commit_sha) != 40 or any(c not in "0123456789abcdef" for c in commit_sha.lower()):
        raise ValueError(
            "--commit must be a full 40-character git commit sha, not a branch name or short hash: "
            f"{commit_sha!r}"
        )
    spec = f"chowder-ai[ptq,train] @ git+https://github.com/niko4244/Chowder.git@{commit_sha}"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", spec], check=True)


def installed_chowder_commit() -> str | None:
    """Best-effort read of the resolved VCS commit from package metadata."""
    import importlib.metadata

    try:
        dist = importlib.metadata.distribution("chowder-ai")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        return None
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info") or {}
    return vcs_info.get("commit_id")


def verify_install(commit_sha: str) -> dict:
    installed = installed_chowder_commit()
    if installed != commit_sha:
        raise RuntimeError(
            f"installed chowder commit {installed!r} does not match the requested {commit_sha!r}; "
            "refusing to continue on a cached-wheel or pin mismatch"
        )
    import modelopt

    return {"installed_commit": installed, "nvidia_modelopt_version": modelopt.__version__}


def fingerprint(out_dir: Path) -> dict:
    """Record the environment evidence the mission requires for every run."""
    import torch
    import transformers

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:
        info["peft"] = None
    (out_dir / "environment.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def load_teacher_rows(batch010_path: str | Path) -> list[dict]:
    """Strictly validate and load batch-010 trajectories; never synthesize."""
    path = Path(batch010_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} contains no trajectories; refusing to train on an empty dataset")
    checked: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not a JSON object")
        if row.get("split") != "evolve" or row.get("green_verified") is not True:
            raise ValueError(f"row {index} is not a green-verified evolve trajectory")
        if row.get("harness") not in ALLOWED_TRAJECTORY_HARNESSES:
            raise ValueError(f"row {index} was not generated under a selected state-aware harness")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {index} is missing its transcript")
        if messages[-1].get("role") != "assistant" or not str(messages[-1].get("content", "")).strip():
            raise ValueError(f"row {index} does not end in a nonempty assistant report")
        if any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not str(message.get("content", "")).strip()
            for message in messages
        ):
            raise ValueError(f"row {index} has a malformed message")
        checked.append(row)
    return checked


def _assistant_label_spans(messages: list[dict], tokenizer) -> tuple[list[int], list[bool]]:
    """Tokenize the templated transcript message by message.

    Returns (input_ids, labels) where labels are ``-100`` outside assistant
    content. Fails closed if per-message tokenization does not exactly
    reconstruct the full transcript (prefix-tokenization drift would silently
    mislabel assistant tokens).
    """
    input_ids: list[int] = []
    labels: list[int] = []
    for index, message in enumerate(messages):
        prefix = tokenizer.apply_chat_template(
            messages[: index + 1], tokenize=False, add_generation_prompt=False
        )
        previous = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=False
        ) if index else ""
        if not prefix.startswith(previous):
            raise ValueError("chat template is not prefix-consistent; refusing to mislabel tokens")
        segment = tokenizer(prefix[len(previous):], add_special_tokens=False)["input_ids"]
        input_ids.extend(segment)
        labels.extend(segment if message["role"] == "assistant" else [-100] * len(segment))
    return input_ids, labels


def sft_batches(rows: list[dict], tokenizer, max_len: int = 1024):
    for row in rows:
        input_ids, labels = _assistant_label_spans(row["messages"], tokenizer)
        if len(input_ids) > max_len:
            continue
        if all(label == -100 for label in labels):
            raise ValueError(f"trajectory {row.get('id', '?')} produced no supervised tokens")
        yield {
            "input_ids": input_ids,
            "labels": labels,
            "row_sha256": hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }


def margin_probe(model, tokenizer, prompts: list[str], *, max_new_tokens: int = 32) -> dict:
    """Mean selected-token margin over greedy generations, measured in-process."""
    import torch

    if not prompts:
        raise ValueError("margin probe requires at least one prompt")
    step_margins: list[float] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        for scores in output.scores:
            probs = torch.softmax(scores[0].float(), dim=-1)
            top2 = torch.topk(probs, 2)
            if not torch.isfinite(top2.values).all() or top2.values[1] <= 0:
                continue
            step_margins.append(float(torch.log(top2.values[0]) - torch.log(top2.values[1])))
    if not step_margins:
        raise RuntimeError("margin probe collected no usable steps")
    return {
        "mean_margin": sum(step_margins) / len(step_margins),
        "n_steps": len(step_margins),
        "n_prompts": len(prompts),
    }


def qat_recover(model, calib_texts: list[str], qat_texts: list[str], tokenizer, *, epochs: int, lr: float) -> list[dict]:
    """Quantize (INT8 SmoothQuant), freeze scales, then LoRA-train the recovery steps."""
    import peft
    import torch
    import modelopt.torch.quantization as mtq

    if not calib_texts or not qat_texts:
        raise ValueError("QAT requires real calibration and training texts")

    def calib_loop(m):
        for text in calib_texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(m.device)
            m(**encoded)

    mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop=calib_loop)
    frozen = 0
    for module in model.modules():
        if type(module).__name__ == "TensorQuantizer" and isinstance(getattr(module, "amax", None), torch.nn.Parameter):
            module.amax.requires_grad_(False)
            frozen += 1
    if not frozen:
        raise RuntimeError("no calibrated amax parameters found to freeze; calibration failed silently")

    lora = peft.LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora)
    model.print_trainable_parameters()
    base_trainable = [
        name for name, param in model.base_model.model.named_parameters()
        if param.requires_grad and "lora" not in name
    ]
    if base_trainable:
        raise RuntimeError(f"non-LoRA parameters became trainable: {base_trainable[:3]}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    for epoch in range(epochs):
        for text in qat_texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            loss = model(**encoded).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite QAT loss at epoch {epoch}; aborting before contamination")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append({"epoch": epoch, "loss": float(loss.detach())})
    return history


def build_shift_report(bf16_margin: float | None, quant_margin: float | None, tolerance: float) -> dict:
    """Mirror of exp_e_confidence.margin_shift_fails_closed (keep in sync)."""
    import math

    if bf16_margin is None or quant_margin is None or not math.isfinite(tolerance) or tolerance < 0:
        shift = None if (bf16_margin is None or quant_margin is None) else quant_margin - bf16_margin
        return {"margin_shift": shift, "quantized_margin_shift_fails_closed": True}
    shift = quant_margin - bf16_margin
    return {
        "margin_shift": shift,
        "quantized_margin_shift_fails_closed": abs(shift) > tolerance,
    }


def kd_attach(student, teacher_path: str | Path, tokenizer, texts: list[str], *, steps: int, lr: float) -> list[dict]:
    """Attach a frozen teacher through modelopt's kd_loss mode and measure losses."""
    import torch
    from transformers import AutoModelForCausalLM
    import modelopt.torch.distill as mtd

    teacher = AutoModelForCausalLM.from_pretrained(str(teacher_path), dtype="bfloat16").to(student.device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    kd_model = mtd.convert(
        student,
        [("kd_loss", mtd.KDLossConfig(teacher_model=teacher, criterion=mtd.LogitsDistillationLoss(temperature=2.0)))],
    )
    optimizer = torch.optim.AdamW([p for p in kd_model.parameters() if p.requires_grad], lr=lr)
    history = []
    for step, text in enumerate(list(texts)[:steps]):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(kd_model.device)
        loss = kd_model(**encoded).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite KD loss at step {step}; aborting")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach())})
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="full 40-char chowder commit this lane is pinned to")
    parser.add_argument("--batch010", required=True, help="green-verified batch-010 JSONL from the local lane")
    parser.add_argument("--base-model", required=True, help="HF path of the small (Spark) base model")
    parser.add_argument("--out-dir", default="/kaggle/working")
    parser.add_argument("--kd-teacher-path", help="optional HF teacher path to attach through modelopt KD")
    parser.add_argument("--margin-tolerance", type=float, default=0.2)
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin-prompts", default="2+2=?\nName the capital of France.\n")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    install_chowder(args.commit)
    environment = {**fingerprint(out_dir), **verify_install(args.commit)}

    rows = load_teacher_rows(args.batch010)
    by_task = sorted({row.get("task_name", "?") for row in rows})
    print(f"batch-010 validated: {len(rows)} trajectories over tasks {by_task}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = [line for line in args.margin_prompts.splitlines() if line.strip()]

    def load_base():
        return AutoModelForCausalLM.from_pretrained(args.base_model, dtype="bfloat16").cuda()

    bf16_model = load_base()
    bf16_margin = margin_probe(bf16_model, tokenizer, prompts)
    del bf16_model
    torch.cuda.empty_cache()

    qat_model = load_base()
    calib_texts = [row["messages"][-1]["content"] for row in rows]
    qat_history = qat_recover(
        qat_model, calib_texts, calib_texts, tokenizer, epochs=args.qat_epochs, lr=args.lr
    )
    quant_margin = margin_probe(qat_model, tokenizer, prompts)
    shift = build_shift_report(bf16_margin["mean_margin"], quant_margin["mean_margin"], args.margin_tolerance)
    del qat_model
    torch.cuda.empty_cache()

    sft_model = load_base()
    import peft

    sft_model = peft.get_peft_model(
        sft_model,
        peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"),
    )
    if args.kd_teacher_path:
        kd_history = kd_attach(
            sft_model, args.kd_teacher_path, tokenizer, calib_texts, steps=8, lr=args.lr
        )
    else:
        kd_history = []
    sft_history = []
    optimizer = torch.optim.AdamW([p for p in sft_model.parameters() if p.requires_grad], lr=args.lr)
    for epoch in range(args.sft_epochs):
        for batch in sft_batches(rows, tokenizer):
            tensors = {k: torch.tensor([v]) for k, v in batch.items() if k != "row_sha256"}
            loss = sft_model(input_ids=tensors["input_ids"], labels=tensors["labels"]).loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite SFT loss; aborting before adapter save")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sft_history.append({"epoch": epoch, "loss": float(loss.detach())})
    adapter_dir = out_dir / "spark_lora_adapter"
    sft_model.save_pretrained(str(adapter_dir))

    report = {
        "environment": environment,
        "batch010": {"path": str(args.batch010), "n_rows": len(rows), "tasks": by_task},
        "margin_probe": {"bf16": bf16_margin, "quantized": quant_margin, "tolerance": args.margin_tolerance, **shift},
        "routing_implication": (
            "quantized arm fails closed: recalibrate per-precision via "
            "exp_e_confidence.calibrate_margin_threshold_per_precision before serving"
            if shift["quantized_margin_shift_fails_closed"]
            else "shift within declared tolerance; per-precision calibration still required before serving"
        ),
        "qat_history": qat_history,
        "kd_history": kd_history,
        "sft_history": sft_history,
        "adapter_dir": str(adapter_dir),
    }
    report_path = out_dir / "qat_distill_report.json"
    with report_path.open("x", encoding="utf-8") as stream:  # atomic create; never overwrite
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
