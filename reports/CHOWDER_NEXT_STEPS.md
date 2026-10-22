# Chowder prioritized engineering backlog

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

| Rank | Owner-sized task | Acceptance / prerequisite | Status |
|---|---|---|---|
| P0 | Qualify PR #269 recovery hardening | CI green; simulate hard kill before and after each journal/record boundary; verify repeated entry, total spend and cross-process ownership | IN PROGRESS; no auto-resume |
| P0 | Require signed/verified safe-resume plan before attempting a killed campaign | Exact artifact hashes, immutable prior attempt set, recomputed charges, single-writer lease, no re-spend | BLOCKED by incomplete recovery design |
| P0 | Qualify #268 current int8 retention-default build | Re-run 3-step gate with isolated real RTX 5060 Ti, <=14.5 GiB, no WDDM spill, complete log/revision | BLOCKED by hardware access and approval |
| P0 | Settle recipe feasibility | 1000 * measured step time <= 2.5h or explicitly pre-register a distinct budget/recipe and preserve previous failed target | UNMET on current host |
| P1 | Complete int8 placement witnesses | 20+ real optimizer steps with >=1% loss drop, finite nonzero LoRA B grads, router entropy +/-20%, adapter reload | NOT CERTIFIED |
| P1 | Finish REAP128/REAP160/dense/random prereg gate matrix | Exact paired quality, Q4 speed, residency, termination/control; frozen fallbacks only | UNVERIFIED |
| P1 | Produce actual Gen-0 freeze and model package | Full safetensors hashes, tokenizer/config, routing, quant, legal status, replayable loader, immutable judge record | BLOCKED |
| P1 | Resolve data licence N2 and contaminated-source screening | Audited per-row manifest/hashes, Fable-free control, explicit permission decision | OPEN |
| P2 | Review old PRs #226/#210 and clean model-mix docs | Rebase/resolve tested deltas, do not merge blindly; doc correction on audit branch | PARTIAL |
| P2 | Re-run CLI/TUI, cancellation, recovery and package CI matrix | Full passed test log with exact OS/python/source revision | NOT RUN IN THIS AUDIT |

**Compute policy:** Synthetic and non-destructive CPU checks can run without a GPU reservation. Expensive CUDA experiments and Gen-1 training need explicit authorization and a prospective budget that meets the frozen rules. Promotion always needs explicit human approval.
