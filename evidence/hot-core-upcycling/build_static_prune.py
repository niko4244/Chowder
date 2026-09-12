"""Build and verify the static-prune checkpoint -- the recommended artifact.

Prediction, stated before the build: the pruned model computes exactly the dense
function restricted to the kept channels, which is what the mask emulation already
measured under the SAME corpus-wide ranking. So it should score eval A 9.7968 and
eval B 12.9605, plus the small systematic drift seen twice before between a mask on
the dense model and a real converted checkpoint (+1.6% and +2.5%), attributed to
the two tensor layouts quantising differently under nf4.

Note down_proj rows go from 12,288 to 3,440 elements and 3,440 is not a multiple of
64, so nf4's 64-element blocks no longer align with row boundaries. That is a
concrete reason to expect drift here and is why the prediction carries a band
rather than a point.
"""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

DENSE = r"F:\llm-models\Qwen3.8-9B-abliterated-25-bf16"
RANK = Path(r"F:\llm-models\_a4b\rank-9b-corpuswide-v1.json")
OUT_DIR = Path(r"F:\llm-models\Qwen3.8-9B-Pruned-CW-3440")
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")
REPORT = Path(r"F:\llm-models\_a4b\static-prune-checkpoint.json")
KEEP, EVAL_MAXLEN = 3440, 384
PRED = {"eval_a": 9.7968, "eval_b": 12.9605}
DENSE_PPL = {"eval_a": 5.2707, "eval_b": 4.3044}
MOE_CW = {"eval_a": 13.4224, "eval_b": 15.2936}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.channel_importance import ChannelRanking
    from chowder.static_prune import plan_static_prune, prune_checkpoint

    ranking = ChannelRanking.read(RANK)
    plan = plan_static_prune(DENSE, ranking, keep_channels=KEEP)
    log(f"ranking {ranking.digest()[:12]} ({ranking.calibration.get('split')})")
    log(f"plan: keep {plan.keep_channels}/{plan.intermediate_size} "
        f"(f={plan.keep_channels/plan.intermediate_size:.4f})")
    log(f"  FFN {plan.dense_ffn_params/1e9:.3f}B -> {plan.pruned_ffn_params/1e9:.3f}B "
        f"(removed {(plan.dense_ffn_params-plan.pruned_ffn_params)/1e9:.3f}B)")
    log(f"  total {(plan.always_on_params+plan.dense_ffn_params)/1e9:.3f}B -> "
        f"{plan.total_params/1e9:.3f}B, and total == ACTIVE (nothing conditional)")

    if (OUT_DIR / "conversion.provenance.json").is_file():
        prov = json.loads((OUT_DIR/"conversion.provenance.json").read_text(encoding="utf-8"))
        log("reusing existing prune")
    else:
        log(f"pruning -> {OUT_DIR}")
        t0 = time.time()
        prov = prune_checkpoint(DENSE, OUT_DIR, ranking, keep_channels=KEEP)
        log(f"  pruned in {time.time()-t0:.0f}s, "
            f"{prov['storage']['actual_output_bytes']/1e9:.2f} GB "
            f"(source {prov['storage']['source_bytes']/1e9:.2f} GB)")

    rows = [json.loads(l)["prompt"] for l in
            CALIB.read_text(encoding="utf-8").splitlines() if l.strip()]
    evals = {"eval_a": [rows[i] for i in range(1,64,2)],
             "eval_b": [rows[i] for i in range(632,696)]}

    tok = AutoTokenizer.from_pretrained(OUT_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        OUT_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, trust_remote_code=False)
    model.eval(); model.config.use_cache = False
    log(f"loaded; {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated")
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    log(f"loaded config intermediate_size = {cfg.intermediate_size} (expect {KEEP})")
    assert cfg.intermediate_size == KEEP

    measured = {}
    for key, prompts in evals.items():
        total = n = 0.0
        bad = False
        with torch.no_grad():
            for text in prompts:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=EVAL_MAXLEN)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                if enc["input_ids"].shape[1] < 8: continue
                loss = model(**enc, labels=enc["input_ids"]).loss.float().item()
                if not math.isfinite(loss):
                    bad = True
                    break
                total += loss*(enc["input_ids"].shape[1]-1); n += enc["input_ids"].shape[1]-1
        measured[key] = float("inf") if bad else math.exp(total/n)
        d = measured[key]-PRED[key]
        log(f"  {key}: {measured[key]:.4f} ({measured[key]/DENSE_PPL[key]:.3f}x dense)  "
            f"predicted {PRED[key]:.4f}  drift {d:+.4f} ({100*d/PRED[key]:+.1f}%)")

    log("")
    log(f"{'':<30}{'eval A':>11}{'eval B':>11}{'geo-mean x dense':>19}")
    def gm(d): return math.sqrt((d['eval_a']/DENSE_PPL['eval_a'])*(d['eval_b']/DENSE_PPL['eval_b']))
    for lab, d in (("dense parent", DENSE_PPL), ("hot-core MoE init (CW)", MOE_CW),
                   ("STATIC PRUNE (CW)", measured)):
        log(f"  {lab:<28}{d['eval_a']:>11.4f}{d['eval_b']:>11.4f}{gm(d):>18.3f}x")
    log("")
    log(f"static prune is {100*(1-gm(measured)/gm(MOE_CW)):.0f}% better than the MoE "
        f"at equal active compute, and stores {plan.total_params/1e9:.3f}B vs 9.410B.")

    REPORT.write_text(json.dumps({
        "source": DENSE, "output": str(OUT_DIR), "keep_channels": KEEP,
        "ranking_digest": ranking.digest(), "plan": plan.to_dict(),
        "predicted": PRED, "measured": measured,
        "drift": {k: measured[k]-PRED[k] for k in measured},
        "dense_ppl": DENSE_PPL, "moe_corpuswide_init": MOE_CW,
        "geomean_x_dense": {"static_prune": gm(measured), "moe_init": gm(MOE_CW)},
        "output_manifest_sha256": prov["output_manifest_sha256"],
    }, indent=2)+"\n", encoding="utf-8")
    log(f"wrote {REPORT}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
