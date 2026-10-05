"""Intervention families: what kind of change a candidate proposes, and how
much the system is allowed to trust it.

A family answers, in one place, the questions the autonomous loop otherwise
rediscovers per candidate: which model failure the intervention addresses,
which parameters may vary, which architectures it is valid for, what evidence
must exist before it runs, what compute it needs, what it risks, and -- via
:data:`Maturity` -- whether the normal growth loop may propose it at all.

Maturity is the isolation boundary. Production mechanisms are the only ones a
default campaign may propose; qualified-experimental mechanisms need the
campaign to say so, explicitly, before launch; research mechanisms need a
research campaign; rejected mechanisms are not proposed again unless a new
hypothesis and an explicit reopen say otherwise. Nothing here promotes a
mechanism's maturity -- measured evidence does, through the evidence store and
an operator; this module only refuses to let code drift past the label.

A family also names the in-repo artifacts that implement and measure it
(:attr:`InterventionFamily.implementation`). The registry is kept honest
from both ends by tests: a declared artifact must exist where it says it
does, and a family that ships no mechanism is not an experiment. The
mechanisms rescued from the 2026-10-03 main-clone fold entered here by
file-level extraction, one experiment at a time, each carrying the tests
and measured records its maturity label cites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "Maturity",
    "InterventionFamily",
    "InterventionFamilyRefusal",
    "family_registry",
    "family_from_id",
    "families_for_campaign",
    "assert_maturity_permits",
]


class Maturity(str, Enum):
    """How much the system trusts an intervention class.

    The label is a *gate*, not a boast: it decides who may propose the
    family, never whether a specific candidate of it will work.
    """

    #: The default autonomous loop may propose it inside a normal campaign.
    PRODUCTION = "production"
    #: The campaign must explicitly permit experimental interventions.
    QUALIFIED_EXPERIMENTAL = "qualified-experimental"
    #: Only a research campaign (explicitly labeled as such) may propose it.
    RESEARCH = "research"
    #: Not proposed again without a new hypothesis and an explicit reopen.
    REJECTED = "rejected"


class InterventionFamilyRefusal(RuntimeError):
    """A family cannot be used the way the caller asked."""


@dataclass(frozen=True)
class InterventionFamily:
    """One class of intervention, with its contract attached."""

    #: Dotted id, ``<group>.<mechanism>`` (training.replay-balanced).
    family_id: str
    name: str
    #: The model-failure class this family intends to address.
    target_failure_class: str
    #: Parameters a candidate may vary, each with a type and a range.
    parameters: Mapping[str, Mapping[str, Any]]
    #: Architecture families the intervention is valid for; empty = any.
    valid_architectures: tuple[str, ...] = ()
    #: What evidence must exist before a candidate of this family may run.
    evidence_required: tuple[str, ...] = ()
    #: Coarse compute class (local-cpu, single-gpu, multi-gpu).
    compute_class: str = "single-gpu"
    #: Risks this family can introduce, checked against the retention gate.
    risks: tuple[str, ...] = ()
    #: Evaluation dimensions that must be measured on it.
    eval_dimensions: tuple[str, ...] = ()
    maturity: Maturity = Maturity.RESEARCH
    #: The experiment artifacts that produced the current maturity label.
    basis: tuple[str, ...] = ()
    #: In-repo artifacts that implement and measure this family: modules,
    #: experiment drivers, tests, evidence records -- repo-relative paths.
    #: A declared artifact that does not exist is a broken claim, and the
    #: registry drift guard in the tests refuses one.
    implementation: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if "." not in self.family_id:
            raise InterventionFamilyRefusal(
                f"family id {self.family_id!r} must be '<group>.<mechanism>'"
            )
        for parameter_name, spec in self.parameters.items():
            if "type" not in spec or "range" not in spec:
                raise InterventionFamilyRefusal(
                    f"family {self.family_id!r} parameter {parameter_name!r} "
                    "must declare a type and a range: an undeclared parameter "
                    "range is an unbounded search"
                )
        if self.maturity is Maturity.REJECTED and not self.basis:
            raise InterventionFamilyRefusal(
                f"family {self.family_id!r} is rejected without a recorded "
                "basis: a rejection is a measured claim, not a taste"
            )
        for artifact in self.implementation:
            parts = artifact.replace("\\", "/").split("/")
            if artifact.startswith(("/", "\\")) or ".." in parts:
                raise InterventionFamilyRefusal(
                    f"family {self.family_id!r} declares implementation "
                    f"artifact {artifact!r}: implementation paths are "
                    "repo-relative, so a claim cannot point outside the repo"
                )


#: The registry, as of this branch. Maturity labels cite their basis; a label
#: without a basis is a claim this codebase has not earned. Families from the
#: rescued experimental work (PR #203's mechanisms) are RESEARCH -- they have
#: artifacts, tests and measured records, not production qualification -- and
#: the one whose own measurements rejected it is registered REJECTED with that
#: measurement as its basis. ``implementation`` names the artifacts behind
#: every label; the drift guard in
#: tests/test_growth_interventions_evidence_hypotheses.py refuses a family
#: whose implementation is missing or undeclared.
_REGISTRY: tuple[InterventionFamily, ...] = (
    InterventionFamily(
        family_id="training.sft-curriculum",
        name="SFT / curriculum refinement",
        target_failure_class="target-capability-weakness",
        parameters={
            "learning_rate": {"type": "float", "range": [1e-5, 1e-3]},
            "scheduler": {"type": "enum", "range": ["cosine", "linear", "constant"]},
            "warmup_steps": {"type": "int", "range": [0, 200]},
            "lora_rank": {"type": "int", "range": [4, 128]},
            "lora_alpha": {"type": "int", "range": [8, 256]},
            "target_modules": {"type": "enum-list", "range": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
            "batch_size": {"type": "int", "range": [1, 32]},
            "gradient_accumulation": {"type": "int", "range": [1, 16]},
        },
        valid_architectures=(),
        evidence_required=("baseline-profile", "declared-curriculum"),
        compute_class="single-gpu",
        risks=("target-overfit", "replay-dilution"),
        eval_dimensions=("target-capability", "retained-capabilities", "termination"),
        maturity=Maturity.PRODUCTION,
        basis=(
            "qualified LoRA post-training path (backends/transformers_peft.py)",
            "growth campaign generation loop (#191-#198)",
        ),
        implementation=(
            "src/chowder/backends/transformers_peft.py",
            "src/chowder/backends/training_data.py",
            "src/chowder/backends/transformers_worker.py",
        ),
        notes="The production path every campaign before 0.5 used.",
    ),
    InterventionFamily(
        family_id="training.replay-balanced",
        name="Replay-balanced training",
        target_failure_class="retention-loss",
        parameters={
            "replay_rate": {"type": "float", "range": [0.0, 0.5]},
        },
        valid_architectures=(),
        evidence_required=("baseline-profile", "replay-dataset-provenance"),
        compute_class="single-gpu",
        risks=("replay-dilution", "target-underfit"),
        eval_dimensions=("target-capability", "retained-capabilities"),
        maturity=Maturity.PRODUCTION,
        basis=("screening lane replay arms (docs/KAGGLE_PROVIDER_ACCEPTANCE.md Run 4)",),
        implementation=(
            "src/chowder/backends/transformers_peft.py",
            "src/chowder/backends/transformers_worker.py",
            "src/chowder/backends/training_data.py",
        ),
        notes=(
            "Run 4 measured a replay-decay intervention as harmful on every "
            "seed -- that falsified ONE parameter point, not the family."
        ),
    ),
    InterventionFamily(
        family_id="training.adapter-continuation",
        name="Adapter continuation (parent-adapter)",
        target_failure_class="insufficient-compute-from-parent",
        parameters={
            "learning_rate": {"type": "float", "range": [1e-5, 1e-3]},
        },
        valid_architectures=(),
        evidence_required=("parent-adapter-digest",),
        compute_class="single-gpu",
        risks=("compounding-adapter-drift",),
        eval_dimensions=("target-capability", "retained-capabilities"),
        maturity=Maturity.PRODUCTION,
        basis=("growth loop continuation campaigns (#190-#198)",),
        implementation=(
            "src/chowder/backends/transformers_peft.py",
            "src/chowder/checkpoint_discovery.py",
            "src/chowder/adapter_guard.py",
        ),
    ),
    InterventionFamily(
        family_id="architecture.conditional-ffn",
        name="Conditional FFN compute",
        target_failure_class="inference-efficiency",
        parameters={
            "active_experts": {"type": "int", "range": [1, 8]},
        },
        valid_architectures=("moe",),
        evidence_required=("dense-vs-moe-equivalence-audit", "router-health-check"),
        compute_class="single-gpu",
        risks=("capability-collapse-under-sparsity", "router-degeneration"),
        eval_dimensions=("target-capability", "retained-capabilities", "latency", "vram"),
        maturity=Maturity.RESEARCH,
        basis=("fold main-clone rescue: conditional_compute.py + conditional_profile.py",),
        implementation=(
            "src/chowder/conditional_compute.py",
            "src/chowder/conditional_profile.py",
            "chowder_batch/exp_c_profile.py",
            "tests/test_conditional_compute.py",
            "tests/test_conditional_profile.py",
            "docs/EXPERIMENT_C_CONDITIONAL_COMPUTE.md",
        ),
        notes="Experimental mechanisms from the rescued work; tests only, no production qualification.",
    ),
    InterventionFamily(
        family_id="architecture.hybrid-lm",
        name="Hybrid attention / state-space blocks",
        target_failure_class="long-context-efficiency",
        parameters={
            "hybrid_ratio": {"type": "float", "range": [0.0, 1.0]},
        },
        valid_architectures=("dense",),
        evidence_required=("conversion-exactness-audit",),
        compute_class="single-gpu",
        risks=("capability-regression",),
        eval_dimensions=("target-capability", "retained-capabilities", "latency"),
        maturity=Maturity.RESEARCH,
        basis=("fold main-clone rescue: experimental_hybrid_lm.py",),
        implementation=(
            "src/chowder/experimental_hybrid_lm.py",
            "chowder_batch/exp_d_hybrid_lm.py",
            "examples/experiment_d/configs/",
            "tests/test_experimental_hybrid_lm.py",
            "tests/test_exp_d_hybrid_lm.py",
            "docs/EXPERIMENT_D_LOW_ACTIVE_HYBRID_LM.md",
        ),
    ),
    InterventionFamily(
        family_id="compression.low-rank-vocab",
        name="Low-rank vocabulary / head",
        target_failure_class="vram-footprint",
        parameters={
            "rank": {"type": "int", "range": [16, 2048]},
        },
        valid_architectures=("dense",),
        evidence_required=("spectrum-analysis", "post-conversion-eval"),
        compute_class="single-gpu",
        risks=("unacceptable-degradation",),
        eval_dimensions=("perplexity", "generation-quality", "termination"),
        maturity=Maturity.RESEARCH,
        basis=("fold main-clone rescue: low_rank_vocab.py + low_rank_checkpoint.py",),
        implementation=(
            "src/chowder/low_rank_vocab.py",
            "src/chowder/low_rank_checkpoint.py",
            "chowder_batch/low_rank_convert.py",
            "chowder_batch/low_rank_real_eval.py",
            "chowder_batch/low_rank_inventory.py",
            "chowder_batch/low_rank_probes.py",
            "chowder_batch/low_rank_recovery_pilot.py",
            "chowder_batch/low_rank_teacher_cache.py",
            "tests/test_low_rank_vocab.py",
            "docs/LOW_RANK_VOCAB_EXPERIMENT.md",
        ),
        notes=(
            "Prior measured evidence exists in the fold's records: a flat "
            "spectrum made one compression run degrade unacceptably. Record "
            "the specific outcome in the evidence store before proposing again."
        ),
    ),
    InterventionFamily(
        family_id="compression.ptq",
        name="Post-training quantization variants",
        target_failure_class="vram-footprint",
        parameters={
            "quantization": {"type": "enum", "range": ["int8-weight-only", "smoothquant", "nf4"]},
        },
        valid_architectures=("dense",),
        evidence_required=("repair-set-behavior-eval",),
        compute_class="single-gpu",
        risks=("repair-capability-loss", "termination-loss"),
        eval_dimensions=("repair-success", "termination", "perplexity", "latency"),
        maturity=Maturity.RESEARCH,
        basis=("fold main-clone rescue + kaggle QAT lane tests",),
        implementation=(
            "chowder_batch/exp_f_ptq_margin.py",
            "kaggle/run_qat_distill_lane.py",
            "evidence/exp_f_ptq_margin_qwen25_1p5b_int8sq_20260925.json",
            "evidence/exp_f_ptq_margin_qwen25_1p5b_int8sq_guided20_20260926.json",
            "evidence/exp_f_ptq_margin_qwen25_1p5b_int8wo_guided20_20260926.json",
            "tests/test_exp_f_ptq_margin.py",
            "tests/test_kaggle_qat_lane.py",
        ),
        notes=(
            "Margin statistics alone never qualify a quantization: the "
            "measured repair/generation surface decides. The shipped exp_f "
            "records are the worked example -- margins moved +0.0076 while "
            "accuracy fell 0.3 -> 0.0."
        ),
    ),
    InterventionFamily(
        family_id="inference.retrieval",
        name="Retrieval augmentation",
        target_failure_class="factual-weakness",
        parameters={
            "method": {"type": "enum-list", "range": ["bm25", "dense", "sparse_learned"]},
            "top_k": {"type": "int", "range": [1, 16]},
        },
        valid_architectures=(),
        evidence_required=("retrieval-corpus-provenance",),
        compute_class="local-cpu",
        risks=("retrieval-contamination", "corpus-scale-misjudgement"),
        eval_dimensions=("factual-accuracy", "fabrication-rate", "latency"),
        maturity=Maturity.RESEARCH,
        basis=(
            "fold main-clone rescue: exp_e_corpus.py (BM25 / dense / learned-sparse retrievers)",
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md phase 3: factual +0.50 with a 1.00 citation rate; the learned sparse layer was not competitive at this corpus size (198 KB vs 21 KB)",
        ),
        implementation=(
            "chowder_batch/exp_e_corpus.py",
            "chowder_batch/exp_e_pipeline.py",
            "chowder_batch/exp_e_run.py",
            "chowder_batch/exp_e_tasks.py",
            "tests/test_exp_e_pipeline.py",
            "tests/test_exp_e_run.py",
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md",
        ),
        notes=(
            "The one lever that measured quality-positive. Its evidence is 18 "
            "dev tasks and one small corpus: qualified for a research campaign, "
            "not for a default one."
        ),
    ),
    InterventionFamily(
        family_id="inference.speculative",
        name="Prompt-lookup n-gram speculative decoding",
        target_failure_class="latency",
        parameters={
            "ngram": {"type": "int", "range": [1, 5]},
            "max_draft": {"type": "int", "range": [1, 16]},
        },
        valid_architectures=(),
        evidence_required=("acceptance-rate-measurement", "output-equivalence-check"),
        compute_class="local-cpu",
        risks=("output-quality-drift", "overhead-on-non-copy-work"),
        eval_dimensions=("latency", "output-equivalence", "termination"),
        maturity=Maturity.RESEARCH,
        basis=(
            "fold main-clone rescue: exp_e_speculative.py + exp_e_spec_llamacpp.py",
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md phase 2: up to 2.8x tok/s with identical outputs on copy-shaped prompts, ~4% overhead elsewhere; every draft token teacher-verified rather than accepted blind",
        ),
        implementation=(
            "chowder_batch/exp_e_speculative.py",
            "chowder_batch/exp_e_spec_llamacpp.py",
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md",
        ),
        notes=(
            "The measured speedup is 4 prompts wide and the mechanism never "
            "claims equivalence it did not verify: outputs are compared, and "
            "the teacher argmax verifies every draft position."
        ),
    ),
    InterventionFamily(
        family_id="inference.confidence-routing",
        name="Confidence-gated routing to a larger model",
        target_failure_class="confidence-calibration",
        parameters={
            "signal": {"type": "enum", "range": ["logprob-margin", "self-review"]},
            "margin_shift_tolerance": {"type": "float", "range": [0.0, 1.0]},
        },
        valid_architectures=(),
        evidence_required=("calibration-table", "heldout-transfer-gate"),
        compute_class="single-gpu",
        risks=("confident-and-wrong", "quality-loss-versus-large-control"),
        eval_dimensions=("accuracy", "large-invocation-rate", "calibration"),
        maturity=Maturity.REJECTED,
        basis=(
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md phase 4: routed small 0.55 / routed large 0.57 against an always-large control at 0.72, with the confident-and-wrong cell 3 of 14 -- self-review confidence is not correctness",
            "fold main-clone rescue: exp_e_confidence.py -- logprob-margin extraction, margin calibration, and the fail-closed shift/green-retention guards the reopen would have to satisfy",
        ),
        implementation=(
            "chowder_batch/exp_e_confidence.py",
            "tests/test_exp_e_confidence.py",
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md",
        ),
        notes=(
            "Registered REJECTED, not merely unqualified: the rescued work "
            "measured the dangerous cell as populated. It returns only through "
            "an explicit reopen naming a NEW confidence signal, and the shipped "
            "guards (margin-shift and green-retention fail-closed) are the bar "
            "that signal must clear."
        ),
    ),
    InterventionFamily(
        family_id="runtime.harness-repair",
        name="Runtime harness repair mechanisms",
        target_failure_class="agent-runtime-failure",
        parameters={
            "mechanism": {"type": "enum-list", "range": ["state_aware", "recovery"]},
            "max_turns": {"type": "int", "range": [1, 8]},
        },
        valid_architectures=(),
        evidence_required=("runtime-trace-benchmark", "heldout-task-split"),
        compute_class="single-gpu",
        risks=("premature-completion", "nonexistent-read", "task-family-overfit"),
        eval_dimensions=("runtime-reward", "runtime-green-rate", "runtime-nonexistent-read-rate", "runtime-execution-cost"),
        maturity=Maturity.RESEARCH,
        basis=(
            "fold main-clone rescue: runtime_eval.py (batch-009 controlled harness experiment)",
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md phase 5: harness verification is the signal that makes repair escalation trustworthy; LLM self-review is not",
        ),
        implementation=(
            "src/chowder/runtime_eval.py",
            "chowder_batch/run_runtime_harness_compare.py",
            "chowder_batch/batch009_harness_experiment.py",
            "chowder_batch/runtime_benchmark.py",
            "chowder_batch/runtime_trace_reward.py",
            "chowder_batch/build_event_reward_data.py",
            "chowder_batch/build_batch008_event_data.py",
            "chowder_batch/batch008_event_reward_train.jsonl",
            "chowder_batch/batch004_runtime_trace.jsonl",
            "chowder_batch/batch005_runtime_trace.jsonl",
            "chowder_batch/batch006_runtime_trace.jsonl",
            "chowder_batch/runtime_loop_trace.jsonl",
            "tests/test_runtime_harness_mechanisms.py",
            "tests/test_batch010_contract.py",
        ),
        notes=(
            "The model stays frozen; the harness is the object under change. "
            "Measured at this scale the repair tasks exceeded both models' "
            "tool-use ability (0/4 repaired), so these mechanisms are verified "
            "by named metrics and trace tests, not yet by a capability win."
        ),
    ),
    InterventionFamily(
        family_id="runtime.harness-evolution",
        name="Regularized harness selection",
        target_failure_class="agent-runtime-failure",
        parameters={
            "noise_band": {"type": "float", "range": [0.0, 1.0]},
            "beta0": {"type": "float", "range": [0.0, 1.0]},
            "beta1": {"type": "float", "range": [0.0, 2.0]},
            "window": {"type": "int", "range": [1, 10]},
        },
        valid_architectures=(),
        evidence_required=("runtime-trace-benchmark", "heldout-task-split", "cost-attribution"),
        compute_class="local-cpu",
        risks=("noise-chasing", "cost-creep", "heldout-leak"),
        eval_dimensions=("runtime-reward", "runtime-green-rate", "runtime-nonexistent-read-rate", "runtime-execution-cost"),
        maturity=Maturity.RESEARCH,
        basis=(
            "fold main-clone rescue: harness_evolution.py (RRSI-style regularized selection)",
        ),
        implementation=(
            "src/chowder/harness_evolution.py",
            "chowder_batch/batch009_harness_experiment.py",
            "tests/test_runtime_harness_mechanisms.py",
        ),
        notes=(
            "Acceptance is a gate, not a score: a gain inside the noise band "
            "or unaffordable in cost is refused before it becomes a proposal."
        ),
    ),
    InterventionFamily(
        family_id="training.teacher-distillation",
        name="Teacher-generated distillation data",
        target_failure_class="target-capability-weakness",
        parameters={
            "max_tokens": {"type": "int", "range": [160, 1024]},
        },
        valid_architectures=(),
        evidence_required=("teacher-provenance", "restricted-execution-verification"),
        compute_class="single-gpu",
        risks=("teacher-errors-become-labels", "abstention-collapse"),
        eval_dimensions=("target-capability", "retained-capabilities", "fabrication-rate"),
        maturity=Maturity.RESEARCH,
        basis=(
            "fold main-clone rescue: exp_b_teacher_data.py (text / tool-decision / abstention rows, restricted-python verification)",
            "docs/EXPERIMENT_E_PREDICTIVE_INFERENCE.md conclusions: the teacher is still the best reasoner and the verifier of last resort, so its rows are the distillation source -- and its errors are the risk",
        ),
        implementation=(
            "chowder_batch/exp_b_teacher_data.py",
            "chowder_batch/exp_b_restricted_python.py",
            "chowder_batch/exp_b_granite_baseline.py",
            "chowder_batch/exp_b_toolchain_check.py",
            "tests/test_exp_b_teacher_data.py",
            "docs/EXPERIMENT_B_GRANITE_DISTILLATION.md",
        ),
        notes=(
            "Rows are only admitted through a verification surface: code rows "
            "must execute, tool rows are graded against the expected decision, "
            "and a teacher error is recorded as a failure instead of a label."
        ),
    ),
)


def family_registry() -> tuple[InterventionFamily, ...]:
    """Every declared family, in registry order."""
    return tuple((*_REGISTRY, *_EXTRA_FAMILIES))


def family_from_id(family_id: str) -> InterventionFamily:
    """Fail closed on an unknown family id."""
    for family in (*_REGISTRY, *_EXTRA_FAMILIES):
        if family.family_id == family_id:
            return family
    raise InterventionFamilyRefusal(
        f"no intervention family {family_id!r} is registered; an unregistered "
        "family has no declared parameters, risks or maturity -- refusing "
        "rather than inventing one"
    )


def assert_maturity_permits(
    maturity: Maturity,
    *,
    campaign_policy: Mapping[str, Any],
    family_id: str | None = None,
) -> None:
    """The isolation gate: who may propose what, in one check.

    ``campaign_policy`` is the campaign declaration's intervention policy
    (``{"experimental_interventions": bool, "research_campaign": bool,
    "reopen": {family_id: hypothesis_id}}``). Absent keys are refusals, not
    defaults: silence never grants permission below PRODUCTION.
    """
    if maturity is Maturity.PRODUCTION:
        return
    if maturity is Maturity.QUALIFIED_EXPERIMENTAL:
        if campaign_policy.get("experimental_interventions") is True:
            return
        raise InterventionFamilyRefusal(
            "an experimental intervention family may only be proposed by a "
            "campaign whose policy explicitly permits experimental "
            "interventions (experimental_interventions: true)"
        )
    if maturity is Maturity.RESEARCH:
        if campaign_policy.get("research_campaign") is True and (
            campaign_policy.get("experimental_interventions") is True
        ):
            return
        raise InterventionFamilyRefusal(
            "a research intervention family may only be proposed by a "
            "campaign that is explicitly a research campaign and permits "
            "experimental interventions"
        )
    # REJECTED: only an explicit reopen, keyed by family id and naming the
    # NEW hypothesis that justifies revisiting the measured rejection, can
    # revive one. A reopen without a hypothesis id is the rediscovery the
    # evidence store exists to prevent.
    reopens = campaign_policy.get("reopen")
    hypothesis_id = reopens.get(family_id) if isinstance(reopens, Mapping) else None
    if family_id and isinstance(hypothesis_id, str) and hypothesis_id.strip():
        return
    raise InterventionFamilyRefusal(
        "a rejected intervention family is not proposed again unless the "
        "campaign explicitly reopens it with a new hypothesis (reopen: "
        "{family_id: hypothesis_id})"
    )


def families_for_campaign(campaign_policy: Mapping[str, Any]) -> tuple[InterventionFamily, ...]:
    """The families this campaign's policy may propose, in registry order.

    A REJECTED family appears only when the policy explicitly reopens it, and
    then only with a new hypothesis id attached -- the reopen mapping's value
    is the hypothesis that justifies revisiting the rejection.
    """
    permitted: list[InterventionFamily] = []
    for family in family_registry():
        try:
            assert_maturity_permits(
                family.maturity,
                campaign_policy=campaign_policy,
                family_id=family.family_id,
            )
        except InterventionFamilyRefusal:
            continue
        permitted.append(family)
    return tuple(permitted)


_EXTRA_FAMILIES: list[InterventionFamily] = []


def register_family(family: InterventionFamily) -> None:
    """Add an operator/research-declared family to the registry.

    The shipped registry covers the families this branch knows; new ones
    (including REJECTED verdicts from measured evidence) enter through here,
    and the frozen dataclass's own checks keep an undocumented label out.
    """
    if any(existing.family_id == family.family_id for existing in (*_REGISTRY, *_EXTRA_FAMILIES)):
        raise InterventionFamilyRefusal(
            f"family {family.family_id!r} is already registered"
        )
    _EXTRA_FAMILIES.append(family)
