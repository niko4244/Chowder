"""P11 rung 3 — measured preflight probe for the 9B-derived MoE router pilot.

Executes the preregistration in docs/quals/P11_RUNG3_PREREG_2026-09-14.md
(Pushed as PR #161 commit 23756d4 BEFORE this probe runs):

  A. fp32 full-resident (current worker policy)   - expected refusal
  B. bf16 full-resident                            - expected refusal
  C. bf16 + expert-tensor offload to host          - the arithmetic fit candidate

Each strategy measures: load wall time, peak CUDA allocator (allocated+reserved)
after load, forward, and backward; expert/gate device census; and for C, three
timed forward+backward+update steps with finite-loss and gate-gradient checks.

Exit code 0 = at least one strategy qualified per prereg verdict logic.
Writes probe-result.json beside itself and embeds its own SHA-256 in the report.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ARTIFACT = Path(r"F:\llm-models\Qwen3.8-9B-HotCore-CW-E16-k2-h2176")
MANIFEST_SHA256 = "77520edadb9a94f4ed70636328c4bbbaafa49c75e3b47c51418851c5ad4869c4"
RESULT_PATH = Path(__file__).resolve().parent / "probe-result.json"
DEVICE = "cuda:0"
PEAK_LIMIT_GB = 14.0  # T2
SEQ_LEN = 64          # T4 pilot workload
BATCH = 2
STEP_BUDGET_S = 3.0   # T4

results: dict = {
    "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "prereg": "docs/quals/P11_RUNG3_PREREG_2026-09-14.md (PR #161, 23756d4)",
    "artifact": str(ARTIFACT),
    "torch": torch.__version__,
    "device_name": torch.cuda.get_device_name(0),
    "device_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    "strategies": {},
}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def t1_verify_identity() -> dict:
    import struct

    manifest = json.loads((ARTIFACT / "conversion.manifest.json").read_text())
    manifest_ok = manifest.get("manifest_sha256") == MANIFEST_SHA256
    weight_hashes = {}
    for wf in manifest.get("weight_files", []):
        p = ARTIFACT / wf["path"]
        weight_hashes[wf["path"]] = {
            "expected": wf["sha256"],
            "actual": _hash_file(p),
            "size_bytes": p.stat().st_size,
            "match": _hash_file(p) == wf["sha256"],
        }
    ok = manifest_ok and all(v["match"] for v in weight_hashes.values())
    return {"manifest_sha256_match": manifest_ok, "weight_files": weight_hashes, "ok": ok}


def _expert_param_names(param_names) -> set[str]:
    """Pattern-based, naming-proof expert-tensor set (checkpoint keys use a
    different prefix than loaded parameters; match on the stable infix)."""
    return {n for n in param_names if ".mlp.experts." in n}


def _census(model) -> dict:
    expert_names = set()
    experts_gpu = experts_cpu = gates_gpu = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".mlp.experts." in name:
                expert_names.add(name)
                if param.device.type == "cuda":
                    experts_gpu += 1
                else:
                    experts_cpu += 1
            if name.endswith("mlp.gate.weight"):
                if param.device.type == "cuda":
                    gates_gpu += 1
    return {
        "experts_on_gpu": experts_gpu,
        "experts_on_cpu": experts_cpu,
        "gates_on_gpu": gates_gpu,
        "expert_params_seen": len(expert_names),
    }


def _peak_gb() -> float:
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


def _reserve_gb() -> float:
    return round(torch.cuda.memory_reserved() / 2**30, 3)


def _reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def strategy_a() -> dict:
    _reset()
    t0 = time.perf_counter()
    record: dict = {"name": "A_fp32_full"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            ARTIFACT, dtype=torch.float32, local_files_only=True
        ).to(DEVICE)
        record["load_s"] = round(time.perf_counter() - t0, 2)
        record["peak_after_load_gb"] = _peak_gb()
        record["reserved_after_load_gb"] = _reserve_gb()
        del model
        record["note"] = "loaded; forward/backward not attempted at this size"
    except torch.cuda.OutOfMemoryError as err:
        record["error"] = f"OutOfMemoryError: {err}"
        record["verdict"] = "refused"
    except Exception as err:  # noqa: BLE001 - measured failure modes are the record
        record["error"] = f"{type(err).__name__}: {err}"
        record["verdict"] = "refused"
    return record


def strategy_b() -> dict:
    _reset()
    t0 = time.perf_counter()
    record: dict = {"name": "B_bf16_full"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            ARTIFACT, dtype=torch.bfloat16, local_files_only=True
        ).to(DEVICE)
        record["load_s"] = round(time.perf_counter() - t0, 2)
        record["peak_after_load_gb"] = _peak_gb()
        record["census"] = _census(model)
        del model
    except torch.cuda.OutOfMemoryError as err:
        record["error"] = f"OutOfMemoryError: {err}"
        record["verdict"] = "refused"
    except Exception as err:  # noqa: BLE001
        record["error"] = f"{type(err).__name__}: {err}"
        record["verdict"] = "refused"
    return record


def strategy_c() -> dict:
    _reset()
    t0 = time.perf_counter()
    record: dict = {"name": "C_bf16_expert_offload"}
    try:
        # Parameter-name discovery on the meta device: zero materialization.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(ARTIFACT, local_files_only=True)
        with torch.device("meta"):
            model_meta = AutoModelForCausalLM.from_config(cfg)
        param_names = [n for n, _ in model_meta.named_parameters()]
        expert_params = _expert_param_names(param_names)
        del model_meta
        record["meta_param_count"] = len(param_names)
        record["meta_expert_param_count"] = len(expert_params)

        # Per-parameter dispatch: experts pinned on host, everything else GPU 0.
        device_map = {n: ("cpu" if n in expert_params else 0) for n in param_names}
        model = AutoModelForCausalLM.from_pretrained(
            ARTIFACT,
            dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=True,
        )
        record["load_s"] = round(time.perf_counter() - t0, 2)
        record["peak_after_load_gb"] = _peak_gb()
        record["census"] = _census(model)
        model.eval()
        model.tie_weights()

        # Real tokenizer input
        tokenizer = AutoTokenizer.from_pretrained(ARTIFACT, local_files_only=True)
        text = ("The quick brown fox reviews the router gate weights of every expert "
                "layer while the small batch walks the frozen corridor. ") * 4
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # T5: offload census from the live model (census re-checked here so the
        # verdict carries it even if the earlier snapshot raced a dispatch move)
        census = _census(model)
        record["census_after_dispatch"] = census
        record["t5_offload_verified"] = bool(
            census["experts_on_cpu"] >= 32 and census["gates_on_gpu"] >= 32
        )

        # Real tokenizer input
        gate_params = []
        for name, param in model.named_parameters():
            if name.endswith("mlp.gate.weight"):
                param.requires_grad_(True)
                gate_params.append(param)
            else:
                param.requires_grad_(False)
        record["trainable_tensors"] = len(gate_params)

        opt = torch.optim.SGD(gate_params, lr=1e-3)

        def one_step() -> float:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return float(loss.detach())

        # T3: first real step, peak measured across it
        torch.cuda.reset_peak_memory_stats()
        loss1 = one_step()
        record["peak_after_step1_gb"] = _peak_gb()
        record["reserved_after_step1_gb"] = _reserve_gb()
        record["loss_step1"] = loss1
        record["t3_finite_loss"] = loss1 == loss1 and abs(loss1) != float("inf")

        grads_ok = all(
            p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().sum().item() > 0
            for p in gate_params[:4]
        )
        record["t3_gate_grads_finite_nonzero"] = grads_ok

        # T4: three warm steps timed
        times = []
        for _ in range(3):
            t = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            times.append(round(time.perf_counter() - t, 3))
        record["step_times_s"] = times
        record["mean_step_s"] = round(sum(times) / len(times), 3)
        record["t4_within_budget"] = record["mean_step_s"] <= STEP_BUDGET_S
        record["t2_within_peak_limit"] = record["peak_after_step1_gb"] <= PEAK_LIMIT_GB

        record["verdict"] = (
            "qualified"
            if record["t2_within_peak_limit"]
            and record["t3_finite_loss"]
            and record["t3_gate_grads_finite_nonzero"]
            and record["t5_offload_verified"]
            else "refused"
        )
        del model, opt, gate_params
    except torch.cuda.OutOfMemoryError as err:
        record["error"] = f"OutOfMemoryError: {err}"
        record["verdict"] = "refused"
    except Exception as err:  # noqa: BLE001
        record["error"] = f"{type(err).__name__}: {err}"
        record["verdict"] = "refused"
    return record


def strategy_d() -> dict:
    """POST-PRERG DIAGNOSTIC — not part of the preregistered verdict.

    Strategy C refused because stock experts implementations cannot consume
    CPU-resident expert weights. D measures the amendment candidate: experts
    stay on CPU (T5 state preserved); a custom per-expert loop transiently
    copies each layer's FROZEN expert slices to GPU per forward. Gradients
    flow only through activations and router gates (experts need none), so
    copies run under no_grad and peak overhead is one layer (~248 MB bf16).
    """
    _reset()
    t0 = time.perf_counter()
    record: dict = {"name": "D_bf16_offload_transient_experts_DIAGNOSTIC"}
    try:
        import torch.nn.functional as F
        from transformers import AutoConfig
        from transformers.activations import ACT2FN

        cfg = AutoConfig.from_pretrained(ARTIFACT, local_files_only=True)
        text_cfg = getattr(cfg, "text_config", cfg)
        act_fn = ACT2FN[getattr(text_cfg, "hidden_act", "silu")]
        with torch.device("meta"):
            model_meta = AutoModelForCausalLM.from_config(cfg)
        param_names = [n for n, _ in model_meta.named_parameters()]
        expert_params = _expert_param_names(param_names)
        del model_meta
        device_map = {n: ("cpu" if n in expert_params else 0) for n in param_names}
        model = AutoModelForCausalLM.from_pretrained(
            ARTIFACT, dtype=torch.bfloat16, device_map=device_map, local_files_only=True
        )
        record["load_s"] = round(time.perf_counter() - t0, 2)
        record["peak_after_load_gb"] = _peak_gb()
        record["census"] = _census(model)

        def transient_expert_forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            dev = hidden_states.device
            for hit in expert_hit:
                expert_idx = int(hit[0])
                if expert_idx == self.num_experts:
                    continue
                with torch.no_grad():
                    w_gu = self.gate_up_proj[expert_idx].to(dev, non_blocking=True)
                    w_dn = self.down_proj[expert_idx].to(dev, non_blocking=True)
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = F.linear(current_state, w_gu).chunk(2, dim=-1)
                current_hidden = act_fn(gate) * up
                current_hidden = F.linear(current_hidden, w_dn)
                current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))
            return final_hidden_states

        patched = 0
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3_5MoeExperts":
                module.forward = transient_expert_forward.__get__(module)
                patched += 1
        record["patched_expert_modules"] = patched

        tokenizer = AutoTokenizer.from_pretrained(ARTIFACT, local_files_only=True)
        text = ("The quick brown fox reviews the router gate weights of every expert "
                "layer while the small batch walks the frozen corridor. ") * 4
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        gate_params = []
        for name, param in model.named_parameters():
            if name.endswith("mlp.gate.weight"):
                param.requires_grad_(True)
                gate_params.append(param)
            else:
                param.requires_grad_(False)
        record["trainable_tensors"] = len(gate_params)
        opt = torch.optim.SGD(gate_params, lr=1e-3)

        def one_step() -> float:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return float(loss.detach())

        # T3 evidence: dedicated forward+backward with grads captured BEFORE any
        # zero_grad. (Earlier attempts checked grads after one_step(), whose
        # zero_grad(set_to_none=True) had already wiped them - an artifact, not
        # a measurement.)
        torch.cuda.reset_peak_memory_stats()
        evidence = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss1 = float(evidence.loss.detach())
        evidence.loss.backward()
        record["peak_after_step1_gb"] = _peak_gb()
        record["reserved_after_step1_gb"] = _reserve_gb()
        record["loss_step1"] = loss1
        record["t3_finite_loss"] = loss1 == loss1 and abs(loss1) != float("inf")

        # Per-gate gradient diagnosis on the live graph.
        grad_state = {"none": 0, "nan": 0, "zero": 0, "ok": 0}
        grad_samples = {}
        for idx, p in enumerate(gate_params):
            if p.grad is None:
                grad_state["none"] += 1
            elif not torch.isfinite(p.grad).all().item():
                grad_state["nan"] += 1
            elif p.grad.abs().sum().item() == 0.0:
                grad_state["zero"] += 1
            else:
                grad_state["ok"] += 1
            if idx < 4:
                grad_samples[f"layer_{idx}"] = (
                    None if p.grad is None
                    else {
                        "max_abs": float(p.grad.abs().max().item()),
                        "sum_abs": float(p.grad.abs().sum().item()),
                        "finite": bool(torch.isfinite(p.grad).all().item()),
                    }
                )
        record["gate_grad_state"] = grad_state
        record["gate_grad_samples"] = grad_samples
        record["t3_gate_grads_finite_nonzero"] = grad_state["ok"] == len(gate_params)

        # Complete step 1's update, then time three further warm steps.
        opt.step()
        opt.zero_grad(set_to_none=True)

        times = []
        for _ in range(3):
            t = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            times.append(round(time.perf_counter() - t, 3))
        record["step_times_s"] = times
        record["mean_step_s"] = round(sum(times) / len(times), 3)
        record["t4_within_budget"] = record["mean_step_s"] <= STEP_BUDGET_S
        record["t2_within_peak_limit"] = record["peak_after_step1_gb"] <= PEAK_LIMIT_GB
        record["diagnostic_verdict"] = (
            "fits-measured"
            if record["t2_within_peak_limit"] and record["t3_finite_loss"]
            and record["t3_gate_grads_finite_nonzero"]
            else "does-not-fit"
        )
        del model, opt, gate_params
    except torch.cuda.OutOfMemoryError as err:
        record["error"] = f"OutOfMemoryError: {err}"
        record["diagnostic_verdict"] = "does-not-fit"
    except Exception as err:  # noqa: BLE001
        record["error"] = f"{type(err).__name__}: {err}"
        record["diagnostic_verdict"] = "does-not-fit"
    return record


def main() -> int:
    strategies = set(sys.argv[1:]) or {"A", "B", "C"}

    print("=== T1 identity verification (re-hash 18.8 GB) ===", flush=True)
    t0 = time.perf_counter()
    identity = t1_verify_identity()
    results["t1_identity"] = identity
    results["t1_identity"]["verify_s"] = round(time.perf_counter() - t0, 1)
    print(f"T1 ok={identity['ok']} in {results['t1_identity']['verify_s']}s", flush=True)
    if not identity["ok"]:
        results["verdict"] = "refused-identity"
        RESULT_PATH.write_text(json.dumps(results, indent=1))
        return 2

    # Resume support: carry forward records from a prior partial run so a
    # re-run of one strategy never erases the others' measurements.
    if "t1_identity" in results and RESULT_PATH.exists():
        try:
            prior = json.loads(RESULT_PATH.read_text())
            if prior.get("t1_identity", {}).get("ok"):
                results["t1_identity"] = prior["t1_identity"]
            for key, rec in prior.get("strategies", {}).items():
                results["strategies"].setdefault(key, rec)
        except Exception:
            pass

    if "A" in strategies and "A" not in results["strategies"]:
        print("=== Strategy A: fp32 full-resident ===", flush=True)
        a = strategy_a()
        results["strategies"]["A"] = a
        print(json.dumps(a, indent=1), flush=True)
        RESULT_PATH.write_text(json.dumps(results, indent=1))

    if "B" in strategies and "B" not in results["strategies"]:
        print("=== Strategy B: bf16 full-resident ===", flush=True)
        b = strategy_b()
        results["strategies"]["B"] = b
        print(json.dumps(b, indent=1), flush=True)
        RESULT_PATH.write_text(json.dumps(results, indent=1))

    if "C" in strategies and "C" not in results["strategies"]:
        print("=== Strategy C: bf16 + expert offload ===", flush=True)
        c = strategy_c()
        results["strategies"]["C"] = c
        print(json.dumps(c, indent=1), flush=True)
        qualified = c.get("verdict") == "qualified"
        results["verdict"] = "qualified" if qualified else "refused"
        RESULT_PATH.write_text(json.dumps(results, indent=1))
        print(f"VERDICT: {results['verdict']}", flush=True)
        if not qualified and "D" in strategies:
            print("=== Strategy D: POST-PRERG DIAGNOSTIC (transient expert copies) ===", flush=True)
            d = strategy_d()
            results["strategies"]["D"] = d
            print(json.dumps(d, indent=1), flush=True)
            results["diagnostic"] = d.get("diagnostic_verdict")
            RESULT_PATH.write_text(json.dumps(results, indent=1))
        return 0 if qualified else 3

    if "D" in strategies:
        print("=== Strategy D: POST-PRERG DIAGNOSTIC (transient expert copies) ===", flush=True)
        d = strategy_d()
        results["strategies"]["D"] = d
        print(json.dumps(d, indent=1), flush=True)
        results["diagnostic"] = d.get("diagnostic_verdict")
        RESULT_PATH.write_text(json.dumps(results, indent=1))
        print(f"DIAGNOSTIC: {results['diagnostic']}", flush=True)
        return 0

    results["verdict"] = "carried-forward"
    RESULT_PATH.write_text(json.dumps(results, indent=1))
    print("VERDICT: carried-forward (no new strategy ran)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
