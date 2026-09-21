# Phase 6: Weight-Preserving Dense → MoE Conversion Plan

**Status: plan only. Nothing has been converted. No conversion code exists
yet (this document defines it). Parent A — the first conversion input — is
still downloading at its pinned revision. Nothing here may be read as "a
sparse model exists".**

Mission context (docs/QWEN38_SPARSE_PROGRAM.md, Phase 6): the first
dense→MoE experiment must answer *"does a weight-preserving conversion
work at all?"* — not *"how small can we make it?"* This plan takes that
literally and defines a conversion whose output is **numerically exact**
at initialization, then measures everything that happens after.

---

## 1. The verified substrate (read from installed transformers 5.16.1 source)

Every name below was read from the installed library source this session —
not from memory, blog posts, or another model family's conventions.

Dense parent (A/B/C/D, all confirmed dense `qwen3_5` in the program doc's
manifest):

- `Qwen3_5DecoderLayer.mlp = Qwen3_5MLP(config, config.intermediate_size)`
  with `gate_proj`, `up_proj` (`hidden→17408`) and `down_proj`
  (`17408→hidden`), all bias-free, activation `silu` — a SwiGLU block:
  `down(act(gate(x)) * up(x))`.
- 64 text decoder layers, `hidden_size` 5120, plus a nested
  `Qwen3_5VisionModel` (333 tensors) and `mtp.*` head (15 tensors).
  Both are **out of scope** for conversion (Phases 17/18: preserve and
  hash-verify, never touch).

Target sparse architecture (`transformers/models/qwen3_5_moe/`):

- `Qwen3_5MoeSparseMoeBlock`:
  - `gate = Qwen3_5MoeTopKRouter` — `weight: (num_experts, hidden_size)`,
    softmax over all experts, top-k select, then
    **renormalize**: `router_top_value /= router_top_value.sum(-1, keepdim=True)`.
  - `experts = Qwen3_5MoeExperts` — **fused 3D parameters**:
    `gate_up_proj: (E, 2·moe_int, hidden)`, `down_proj: (E, hidden, moe_int)`;
    per-expert compute is the same SwiGLU (`act(gate)·up → down`).
  - `shared_expert = Qwen3_5MoeMLP(config, shared_expert_intermediate_size)`
    (default 512) gated by `shared_expert_gate: Linear(hidden, 1, bias=False)`
    through `sigmoid`, **added** to the routed output.
- Text config `Qwen3_5MoeTextConfig` (`model_type qwen3_5_moe_text`),
  defaults `num_experts=256`, `num_experts_per_tok=8`,
  `moe_intermediate_size=512`, `router_aux_loss_coef=0.001`.
- `hidden_act` is `silu` on **both** sides — the same activation function,
  which is what makes the exactness argument below valid at all.

## 2. Core mechanism: partition-conversion, and why it is output-exact

The mission says *partition existing FFN weights into experts* — not copy
them E times (that is upcycling, E× the storage for E× the compute) and not
initialize new experts (that is not weight-preserving). Partitioning splits
the dense FFN's **intermediate dimension** into E contiguous channel
groups. Because SwiGLU is a per-channel computation — each intermediate
channel `i` contributes
`down[i, :] · (act(gate[i, :]) · up[i, :])` independently, and the output
is their sum — routing a token to the group that contains all of its
top-k channels and summing the selected groups reproduces the dense output
**exactly**, bit-for-bit, before any training.

The three construction rules that make exactness hold:

1. **Partition by channel identity.** Channel group `j` owns dense rows
   `gate_proj[j·moe_int : (j+1)·moe_int]`, the same slice of `up_proj`, and
   the corresponding columns of `down_proj`. No mixing, no averaging, no
   re-initialization. The dense weight matrix is exactly recoverable by
   concatenating the groups back — the conversion is invertible by
   construction, which the exactness harness verifies as a second property.
2. **Router = the partition map.** `TopKRouter.weight` is set so that for
   every token, top-1 (or top-k) selection returns exactly the groups that
   own that token's channels. With per-channel scoring scores
   `s = ||W_gate_cols||₂² + ||W_up_cols||₂² + ||W_down_rows||₂²` evaluated
   **per position in the intermediate dimension** and max-pooled within
   each group, the pre-softmax logit of group `j` is `s_j` and the top-k
   renormalized weights become group shares of the selected mass — the
   routed sum is the dense sum (renormalization is what keeps the weights
   summing to exactly the same total). The exactness harness verifies this
   numerically rather than trusting the argument.
3. **Shared expert starts as an exact no-op.** `shared_expert` weights are
   zero-initialized and `shared_expert_gate` bias-free weight is
   zero-initialized: `sigmoid(0) · 0 = 0`, so the added path contributes
   exactly nothing at init and becomes trainable during Phase 8 healing.
   (Recorded explicitly: this is the one place the converted model has
   parameters the dense parent did not have — 2·(H→512 + 512→H) + H
   weights per layer, ~15.8 MB total across 64 layers in bf16 — and they
   are *zero*, not random.)

What the converted model does differently at init: **FLOPs, not outputs.**
At top-1, per-token MLP compute drops to 1/E of dense plus the (zero)
shared expert; at top-8 it is 8/E. Storage *grows* by the router
(E·5120 per layer) and the zero shared expert — ~14.6 MB per layer set,
~0.93 GB total, on top of the 51.77 GiB parent. This plan deliberately
does **not** reduce active parameters below the dense floor — measured
by Phase 11 accounting (`src/chowder/parameter_accounting.py`, PR #123)
at **9.78B active/token** (attention 7.24B + embeddings 2.54B + norms),
10.21B with the MTP head — correcting this plan's original ~10.55B
hand estimate; that is what the accounting module is for. Attacking the
floor is Phases 9–10's work, after the machinery is proven.

## 3. What this plan explicitly is NOT

- **Not expert-copy upcycling.** E independent copies of the dense FFN
  would be E× storage and E× init compute for zero capability gain. Any
  future expert-expansion stage (Phase 9) splits or grows from the
  partition map, recorded in provenance — it does not silently re-invent
  weights.
- **Not distillation.** The absolute lineage rule applies: the converted
  model's lineage is the pinned parent's weights plus a router and zeros.
  Teacher signal may inform *later repair* (Teacher Fabric), never this
  transformation.
- **Not pruning.** No magnitude pruning, no expert dropping here. Phase 7
  instrumentation and Phase 9 sparsification operate on the converted
  model through the existing evidence machinery.
- **Not a Memory Fabric experiment.** Conversion runs through the plain
  Transformers stack (Phase 15: architecture surgery ≠ training executor;
  Phase 16: no untested composition with Memory Fabric).

## 4. Expert-count ladder (first experiment geometry)

All rungs keep `E · moe_int = 17408` so **per-token MLP FLOPs at top-1 are
identical across rungs** and only routing granularity changes:

| Rung | E | top-k | moe_int | Init output | Storage delta |
|---|---|---|---|---|---|
| P6-A | 8 | 1 | 2176 | bit-exact | +0.93 GB |
| P6-B | 16 | 1 | 1088 | bit-exact | +0.96 GB |
| P6-C | 32 | 1 | 544 | bit-exact | +1.04 GB |

### Errata from implementation (stages 1–2, measured — do not trust the table's last column above)

Implementation (stages 1–2 below) proved two claims in this plan wrong,
and the honest corrections change the init geometry:

1. **Top-1 rungs are NOT init-output-preserving (erratum to §2 rule 2).**
   A dense FFN output is the sum over ALL intermediate channels, so any
   top-k < E drops the unselected groups' contributions. Init exactness
   with the stock `TopKRouter` requires **top-k = E** (zero router →
   exactly uniform 1/E selection). Per-channel token-conditional routing
   (selecting the groups owning a token's largest channels) requires a
   custom router — deferred to the healing phase, not assumed.
2. **Only the linear leg may be pre-scaled (silu is not positively
   homogeneous).** Scaling gate rows by E would scale `silu(E·a) ≠
   E·silu(a)` — a different function. The implemented exact scheme:
   gate/up slices fused **verbatim**, down columns scaled ×E by exact
   exponent shift (bf16/f16) or value multiply (f32/f64). Then
   `Σ_e (1/E)·down_e·(silu(gate_e·x)·(up_e·x)) = Σ_j down[:,j]·silu(gate_j·x)·(up_j·x)`
   — the dense sum, exactly, channel partition being disjoint.

**Measured init forward deviation** (tiny fixture, E=8, documented gate
parameter per §5.2 — float association, since the MoE sums E partial
reductions where the dense path does one):

| dtype | max_abs | max_rel | router uniform | dense recovery |
|---|---|---|---|---|
| float32 | 1.341e-07 (~1 ulp) | 2.6e-04 | exact (0.0) | bitwise |
| bfloat16 | 1.953e-03 (~½ ulp) | 1.1e+00 | exact (0.0) | bitwise |

The init-exact claim therefore holds **up to float-association noise on
the forward**, with weights recovering from the converted checkpoint to
the dense source **bit for bit** (the stronger, must-hold property).

(The real Qwen3.5-MoE geometry — 256 experts × 512, top-8 — is the
far end of the same axis and stays out of scope until the machinery is
proven; jumping there first would confound routing discovery with
conversion correctness.)

Uniform conversion of all 64 text decoder layers matches the target
architecture's shape (`Qwen3_5MoeDecoderLayer` carries a SparseMoeBlock in
every layer). Mixed dense/MoE layer patterns are deferred until evidence
demands them; if that happens, the config-level mechanism to use is
`mlp_only_layers`-style bookkeeping **only if** the installed
`Qwen3_5Moe` classes actually support it — hard stop on assumed support,
per the audit rule.

## 5. Tool plan (new modules, existing conventions)

Each module follows the repo's established discipline: frozen dataclasses
with `__post_init__`, honesty docstrings, hard-stop errors instead of
guesses (`MoeArchitectureAuditError` pattern), and no fabricated evidence.

1. **`src/chowder/dense_to_moe.py`** — the conversion core.
   - `PartitionScheme`: seeded (recorded, reproducible) channel→group map;
     carries its own sha256; default is contiguous, shuffled variants only
     with an explicit flag (contiguous keeps the dense matrix recoverable
     by simple concatenation).
   - `build_moe_text_config(dense_config, E, top_k)`: emits
     `Qwen3_5MoeTextConfig`-compatible JSON with every field carried over
     and `moe_intermediate_size = intermediate_size // E`; refuses silently
     mismatched fields (hard stop).
   - `convert_checkpoint(source_dir, out_dir, scheme)`: streams shard by
     shard via safetensors (never loads 55 GB into RAM), rewrites only
     `layers.*.mlp.{gate_proj,up_proj,down_proj}` into fused 3D
     `experts.{gate_up_proj,down_proj}` + router + zero shared expert;
     copies everything else byte-identical; writes MTP/vision file hashes
     before and after (Phase 17/18 evidence); emits provenance (parent
     pin, scheme digest, per-file hashes) and a **full-mode
     `local_model_manifest`** of the output directory.
   - CLI: `chowder moe dense-to-moe --source <dir> --experts 8 --out <dir>
     [--profile-only]` (profile-only plans and budgets without writing).
2. **`src/chowder/conversion_exactness.py`** — the proof harness.
   Builds a **tiny random composite** `qwen3_5` (2 text layers + a tiny
   real vision tower, matching the real checkpoints' key layout —
   `model.language_model.layers.*` / `mtp.*` / `model.visual.*`), converts
   it, loads both through the real transformers classes, and **measures**
   forward equality on fixed inputs plus dense recovery compared directly
   against the dense source checkpoint. Must-holds (recovery bitwise,
   router exactly uniform at zero logits) raise `ConversionExactnessError`
   — hard stop, never a tolerance widened to make a test pass. Forward
   deviation is *reported*, and gated by the documented association bound
   recorded in the errata above (a fixture test enforces the gate).
3. **Validation ladder** (each stage persisted in the registry):
   1. tiny-random conversion in CI (exactness + invertibility) — **DONE**
      (`tests/test_conversion_exactness.py`, gated on CHOWDER_REAL_ML_SMOKE;
      measured numbers in the errata table above),
   2. real dense checkpoint → conversion machinery on real weights →
      exactness — **DONE at layer-0 scope**: `plan_conversion` runs on the
      real cached parent A and the gated stage-2 test exercises the real
      fusion code path on real layer-0 bytes (multi-shard-capable since
      parent A layer 15 straddles shards — 63/64 co-locate),
   3. parent A profile-only dry run (budgets, no writes) — not started,
   4. parent A real conversion → full manifest → `audit_moe_architecture`
      on the loaded result → loading smoke on the RTX 5060 Ti — not started
      (~52 GiB output; a real disk-acquisition decision).
4. **Reuse, not re-invention**: `moe_instrumentation.audit_moe_architecture`
   validates the converted model's shape before it is trusted;
   `MoeCalibrationRecorder` + `chowder moe expert-importance` instrument
   healing; `local_model_manifest` signs inputs and outputs; the existing
   hard regression gate judges every healing run.

## 6. Router healing (mission Phase 8, bounded)

After conversion, only `gate.weight` and `shared_expert_gate` train for a
bounded budget; expert weights stay frozen so the router problem is
isolated from the expert-weight problem. Uniform-partition init makes the
router *already balanced* at step 0 (every group owns 1/E of channels);
healing's actual job is letting token-conditional routing emerge. Telemetry:
routing entropy, expert load distribution, dead/overloaded expert counts
(via `MoeCalibrationRecorder`), plus the shared-expert gate magnitude (its
growth measures how much dense behavior the shared path is absorbing).
Success = hard regression gate pass on the protected evaluation suite —
which requires the suite content to exist first (Phase 4 dependency,
recorded as such).

## 7. Honest risks and open questions

- **Bit-exactness across code paths is claimed by math and verified by
  test — the verification is the deliverable**, not the math. First thing
  `conversion_exactness.py` must establish empirically: is bf16 CPU
  forward bit-identical between the dense and converted implementations?
  If not, the observed deviation is documented and gated, never normalized
  away.
- **Streaming rewrite of 18 shards on Windows** (fsync semantics, disk
  headroom for a temp shard): budget ~52 GiB free *in addition to* the
  source; F:/G: placement is a real constraint, and the profile-only mode
  exists to catch it before any bytes move.
- **Init-exact ≠ post-training-equal.** Everything this plan guarantees is
  at step 0. Whether healing preserves capability is an empirical question
  the gate answers; no claim is made here.
- **A3B/A4B is not reached by this plan.** The dense floor stands at a
  measured 9.78B active/token after conversion (Phase 11 accounting).
  Attacking attention/DeltaNet/channels is Phase 10, with its own
  hypothesis and regression suite per mechanism. If the honest frontier lands at A7B or A9B, that is a result,
  not a failure (the program doc's Pareto rule).
- **MTP/multimodal interplay**: conversion leaves them byte-identical and
  hash-verified, but the *loaded* model's MTP path has never executed on
  this machine — the loading smoke test must include one MTP forward,
  or say it did not.

## 8. Milestone mapping

Satisfied by this document once merged: *first dense→MoE transformation
plan generated*. Deliberately **not** claimed: the transformation itself
(needs parent A verified on disk + the validation ladder), router healing
(needs the protected suite for gating), and everything Phase 9+.
