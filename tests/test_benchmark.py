"""Task 8: score the dev set (expected clean, Task 6 already proved the
machinery) and the frozen hidden set (Task 7) with the same benchmark
run, then check the hidden-set classifications against the predictions
pre-registered in docs/HIDDEN_SET_FREEZE.md *before* this file existed.

A classification mismatch here would be a finding to write up (which
case, why, what it reveals about the rules' real coverage), not a test to
quietly loosen. As it happens, every H1-H8 prediction was verified
manually while drafting docs/HIDDEN_SET_FREEZE.md (see that file's own
disclosure of the contamination risk this implies) -- this test makes
that check durable rather than a one-off manual run.
"""
from typing import Any, Sequence

from chowder.benchmark import BenchmarkCase, run_benchmark
from chowder.hypothesis_generation import RuleBasedGenerator
from chowder.incident import SignatureKind, compute_fingerprint
from chowder.investigation import InvestigationStatus, RemediationOutcome, config_patch_digest
from chowder.replay import ReplayGroundTruth

from fixtures_incidents import (
    ALL_DEV_FIXTURES,
    DPO_BACKWARD_PASS_OOM_NEAR_LIMIT,
    DPO_LOGITS_FP32_UPCAST_OOM,
    DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH,
    GATED_DELTA_RULE_CUBLAS_FAILURE,
    HF_HUB_TRANSIENT_DROP,
    KAGGLE_DATASET_FLATTENED_PATH,
    PEFT_KBIT_PREP_OOM,
    QWEN3_5_ARCH_NOT_RECOGNIZED,
    QWEN3_5_CONV1D_NO_ENGINE,
    STALE_EXECUTOR_ARTIFACT_MISSING_FLAG,
    WRONG_ACCELERATOR_PROVISIONED,
)
from fixtures_incidents_hidden import (
    ALL_HIDDEN_FIXTURES,
    HIDDEN_ATTENTION_OOM,
    HIDDEN_CHECKPOINT_CHECKSUM_MISMATCH,
    HIDDEN_DISK_FULL_MID_DOWNLOAD,
    HIDDEN_HF_503_SERVICE_UNAVAILABLE,
    HIDDEN_LLAMA_CUDNN_EXECUTION_FAILED_T4X2,
    HIDDEN_QWEN3_5_ARCH_NOT_RECOGNIZED_A10,
    HIDDEN_QWEN3_5_ATTRIBUTEERROR_VERSION_GAP,
    HIDDEN_WRONG_MACHINE_SHAPE_V100_REQUEST_P100,
)

Kind = SignatureKind
Outcome = RemediationOutcome

# A (config_patch, outcome) pair, not a {patch: outcome} dict -- a dict
# can't be a dict key (unhashable), so ground truth per case is built from
# a small list of pairs and digested below.
_PatchOutcome = tuple[dict, Outcome]


def _gt(capture, patch_outcomes: Sequence[_PatchOutcome]) -> ReplayGroundTruth:
    return ReplayGroundTruth(
        fingerprint_sha256=compute_fingerprint(capture).fingerprint_sha256,
        outcomes={config_patch_digest(patch): outcome for patch, outcome in patch_outcomes},
    )


def _cases(specs: Sequence[tuple[Any, SignatureKind, Sequence[_PatchOutcome]]]) -> tuple[BenchmarkCase, ...]:
    return tuple(
        BenchmarkCase(capture=capture, ground_truth=_gt(capture, patch_outcomes), expected_signature_kind=kind)
        for capture, kind, patch_outcomes in specs
    )


# Every dev fixture, its known-correct signature_kind (already proven
# exhaustively in test_incident.py), and ground truth for every candidate
# RuleBasedGenerator could propose for it -- the same modeling used in
# tests/test_dev_fixture_run.py (Task 6), duplicated deliberately rather
# than imported: each test file stays self-contained, matching this
# project's existing convention (test_probes.py, test_closeout.py, etc.
# each define their own local fixtures/expectations rather than reaching
# into another test module).
_DEV_SPECS: tuple[tuple[Any, SignatureKind, Sequence[_PatchOutcome]], ...] = (
    (HF_HUB_TRANSIENT_DROP, Kind.NETWORK_TRANSIENT, [({"resume_download": True}, Outcome.RESOLVED)]),
    (
        PEFT_KBIT_PREP_OOM,
        Kind.CUDA_OOM,
        [
            ({"allocator_conf": "expandable_segments:True"}, Outcome.RESOLVED),
            ({"max_length": 1024}, Outcome.DID_NOT_RESOLVE),
        ],
    ),
    (
        KAGGLE_DATASET_FLATTENED_PATH,
        Kind.ARTIFACT_NOT_FOUND,
        [({"dataset_path_resolution": "search_by_filename"}, Outcome.RESOLVED)],
    ),
    (
        QWEN3_5_ARCH_NOT_RECOGNIZED,
        Kind.DEPENDENCY_INCOMPATIBLE,
        [({"transformers_version": "5.10.2"}, Outcome.RESOLVED)],
    ),
    (
        QWEN3_5_CONV1D_NO_ENGINE,
        Kind.CUDA_KERNEL_UNAVAILABLE,
        [({"cudnn_enabled": False}, Outcome.RESOLVED)],
    ),
    (
        GATED_DELTA_RULE_CUBLAS_FAILURE,
        Kind.CUDA_EXECUTION_FAILED,
        [({"transformers_version": "5.10.2"}, Outcome.RESOLVED)],
    ),
    (
        WRONG_ACCELERATOR_PROVISIONED,
        Kind.HARDWARE_INCOMPATIBLE,
        [({"kernel_metadata.machine_shape": "NvidiaTeslaT4"}, Outcome.RESOLVED)],
    ),
    (
        STALE_EXECUTOR_ARTIFACT_MISSING_FLAG,
        Kind.CONFIG_INVALID,
        [({"resync_kernel_dataset": True}, Outcome.RESOLVED)],
    ),
    (
        DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH,
        Kind.CUDA_DEVICE_MISMATCH,
        [({"device_map": "balanced_low_0"}, Outcome.DID_NOT_RESOLVE)],
    ),
    (
        DPO_LOGITS_FP32_UPCAST_OOM,
        Kind.CUDA_OOM,
        [
            ({"allocator_conf": "expandable_segments:True"}, Outcome.DID_NOT_RESOLVE),
            ({"max_length": 1024}, Outcome.RESOLVED),
        ],
    ),
    (
        # Confirmed real evidence: allocator_conf was already active the
        # whole time and did not prevent this incident; the max-length cap
        # got past an unrelated forward-pass OOM but never closed this one's
        # backward-pass gap. Neither known remediation resolves it -- same
        # honesty discipline as DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH.
        DPO_BACKWARD_PASS_OOM_NEAR_LIMIT,
        Kind.CUDA_OOM,
        [
            ({"allocator_conf": "expandable_segments:True"}, Outcome.DID_NOT_RESOLVE),
            ({"max_length": 1024}, Outcome.DID_NOT_RESOLVE),
        ],
    ),
)
assert {s[0].incident_id for s in _DEV_SPECS} == {c.incident_id for c in ALL_DEV_FIXTURES}


# The 8 hidden fixtures with the predictions pre-registered in
# docs/HIDDEN_SET_FREEZE.md, plus ground truth for whatever
# RuleBasedGenerator's frozen table can propose for each -- H3 and H8 get
# no outcomes at all because the frozen generator has zero candidates for
# UNKNOWN or ARTIFACT_CORRUPTED, so nothing is ever looked up for them.
_HIDDEN_SPECS: tuple[tuple[Any, SignatureKind, Sequence[_PatchOutcome]], ...] = (
    (
        HIDDEN_QWEN3_5_ARCH_NOT_RECOGNIZED_A10,
        Kind.DEPENDENCY_INCOMPATIBLE,
        [({"transformers_version": "5.10.2"}, Outcome.RESOLVED)],
    ),
    (
        HIDDEN_LLAMA_CUDNN_EXECUTION_FAILED_T4X2,
        Kind.CUDA_EXECUTION_FAILED,
        [({"transformers_version": "5.10.2"}, Outcome.RESOLVED)],
    ),
    (HIDDEN_DISK_FULL_MID_DOWNLOAD, Kind.UNKNOWN, []),
    (
        HIDDEN_HF_503_SERVICE_UNAVAILABLE,
        Kind.NETWORK_TRANSIENT,
        [({"resume_download": True}, Outcome.RESOLVED)],
    ),
    (
        HIDDEN_QWEN3_5_ATTRIBUTEERROR_VERSION_GAP,
        Kind.DEPENDENCY_INCOMPATIBLE,
        [({"transformers_version": "5.10.2"}, Outcome.RESOLVED)],
    ),
    (
        HIDDEN_WRONG_MACHINE_SHAPE_V100_REQUEST_P100,
        Kind.HARDWARE_INCOMPATIBLE,
        [({"kernel_metadata.machine_shape": "NvidiaTeslaT4"}, Outcome.RESOLVED)],
    ),
    (
        HIDDEN_ATTENTION_OOM,
        Kind.CUDA_OOM,
        [
            ({"allocator_conf": "expandable_segments:True"}, Outcome.DID_NOT_RESOLVE),
            ({"max_length": 1024}, Outcome.RESOLVED),
        ],
    ),
    (HIDDEN_CHECKPOINT_CHECKSUM_MISMATCH, Kind.ARTIFACT_CORRUPTED, []),
)
assert {s[0].incident_id for s in _HIDDEN_SPECS} == {c.incident_id for c in ALL_HIDDEN_FIXTURES}


_DEV_HONESTLY_ABANDONED = {
    DPO_TRAINER_DEVICE_MAP_AUTO_MISMATCH.incident_id,
    DPO_BACKWARD_PASS_OOM_NEAR_LIMIT.incident_id,
}


def test_dev_set_scores_clean():
    """Task 6 already proved the dev set resolves (or is correctly
    abandoned); this just confirms the scorer agrees and produces a
    complete, well-formed CaseScore per case."""
    report = run_benchmark(list(_cases(_DEV_SPECS)), [], RuleBasedGenerator())
    assert len(report.dev_scores) == 11
    for score in report.dev_scores:
        assert score.correct_classification, score.incident_id
        if score.incident_id in _DEV_HONESTLY_ABANDONED:
            assert score.final_status is InvestigationStatus.ABANDONED
            assert not score.recovery_success
        else:
            assert score.final_status is InvestigationStatus.RESOLVED
            assert score.recovery_success
            assert score.reproducible is True


def test_hidden_set_classifications_match_pre_registered_predictions():
    """Field-for-field check against docs/HIDDEN_SET_FREEZE.md's H1-H8
    table, per the plan's Task 8 test spec -- a mismatch would be reported
    as a finding, not silently absorbed."""
    report = run_benchmark([], list(_cases(_HIDDEN_SPECS)), RuleBasedGenerator())
    assert len(report.hidden_scores) == 8
    for score in report.hidden_scores:
        assert score.actual_signature_kind == score.expected_signature_kind, (
            f"{score.incident_id}: predicted {score.expected_signature_kind}, "
            f"classifier said {score.actual_signature_kind}"
        )
        assert score.correct_classification


def test_hidden_set_cases_with_no_frozen_candidate_are_honestly_abandoned():
    """H3 (UNKNOWN) and H8 (ARTIFACT_CORRUPTED) have zero candidates in the
    frozen generator table -- the benchmark must not paper over that with
    a fabricated resolution. Both should end ABANDONED with zero attempted
    trials, not silently skipped or miscounted as resolved."""
    report = run_benchmark([], list(_cases(_HIDDEN_SPECS)), RuleBasedGenerator())
    by_id = {s.incident_id: s for s in report.hidden_scores}
    for capture in (HIDDEN_DISK_FULL_MID_DOWNLOAD, HIDDEN_CHECKPOINT_CHECKSUM_MISMATCH):
        score = by_id[capture.incident_id]
        assert score.final_status is InvestigationStatus.ABANDONED
        assert not score.recovery_success
        assert score.trials_to_resolution is None
        assert score.resolving_config_patch is None


def test_report_never_exposes_an_aggregate_hidden_set_score():
    """The plan is explicit: no aggregate hidden-set pass rate, because 8
    cases can't support a generalization claim. BenchmarkReport should
    only ever expose the per-case tuples, never a combined pass-rate
    field."""
    report = run_benchmark([], list(_cases(_HIDDEN_SPECS)), RuleBasedGenerator())
    field_names = {f for f in vars(report) if not f.startswith("_")}
    assert field_names == {"dev_scores", "hidden_scores"}


def test_false_blame_flags_the_hyperparameter_fix_for_an_infrastructure_incident():
    """Honest, not comfortable: DPO_LOGITS_FP32_UPCAST_OOM's real resolving
    patch (max_length) is namespaced model_data, even though its root
    cause (CUDA_OOM) is infrastructure -- the structured false-blame check
    must flag this rather than being tuned to look clean on this project's
    own dev set."""
    report = run_benchmark(list(_cases(_DEV_SPECS)), [], RuleBasedGenerator())
    by_id = {s.incident_id: s for s in report.dev_scores}
    upcast_score = by_id[DPO_LOGITS_FP32_UPCAST_OOM.incident_id]
    assert upcast_score.avoided_false_blame is False

    # Contrast: the kbit-prep OOM resolves via an infrastructure-namespaced
    # patch (allocator_conf) -- no false blame there.
    kbit_score = by_id[PEFT_KBIT_PREP_OOM.incident_id]
    assert kbit_score.avoided_false_blame is True


def test_render_table_produces_readable_output_without_crashing():
    report = run_benchmark(list(_cases(_DEV_SPECS)), list(_cases(_HIDDEN_SPECS)), RuleBasedGenerator())
    table = report.render_table()
    assert "dev" in table
    assert "hidden" in table
    assert HIDDEN_DISK_FULL_MID_DOWNLOAD.incident_id in table
