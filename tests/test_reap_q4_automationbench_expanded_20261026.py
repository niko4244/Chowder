"""No GPU: fail-closed exploratory context vs original frozen geometry tests."""

from pathlib import Path
from unittest.mock import patch

import pytest

from chowder_batch import reap_q4_automationbench_expanded_20261026 as ex


def test_separate_profile_only_context_modified_and_original_model_still_registered():
    server, official = ex._commands(Path("F:/private/fresh"), count=6)
    assert server[:8] == [str(ex.watchdog.SERVER), "--fit", "on", "-np", "1",
                           "-c", "32768", "--jinja"]
    assert "--tasks" in official and "--toolset" in official
    assert official[official.index("--toolset") + 1] == "api"
    assert "--max-concurrent" in official
    assert official[official.index("--max-concurrent") + 1] == "1"
    assert ex.reap_serve.FLAGS == ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
    assert ex.PROFILE.startswith("NONCOMPARABLE")


def test_full_600_keeps_all_source_tasks():
    _, args = ex._commands(Path("F:/private/fresh"), count=600)
    assert "--tasks" not in args
    assert args[args.index("--num-examples") + 1] == "600"


def test_one_real_official_short_task_exactly_pinned():
    task = "sales.apply_project_label"
    server, evaluator = ex._commands(Path("F:/private/fresh"), count=1, task_name=task)
    assert server[1:8] == ex.FLAGS
    assert evaluator[evaluator.index("--num-examples") + 1] == "1"
    assert evaluator[evaluator.index("--tasks") + 1] == task


def test_unverified_single_task_rejected_before_gpu():
    with pytest.raises(ValueError, match="source-pinned"):
        ex._commands(Path("F:/private/fresh"), count=1, task_name="not-official")


def test_repeated_invalid_model_tool_json_is_counted_without_modification(tmp_path):
    p = tmp_path / "native.log"
    p.write_text("Failed to parse tool call arguments as JSON\n" * 4)
    assert ex._repeated_tool_json_errors(p) == 4
    assert ex._repeated_tool_json_errors(p) < ex.MALFORMED_TOOL_ABORT_COUNT
    with p.open("a", encoding="utf8") as log:
        log.write("Failed to parse tool call arguments as JSON\n")
    assert ex._repeated_tool_json_errors(p) >= ex.MALFORMED_TOOL_ABORT_COUNT


@pytest.mark.parametrize("count", [0, 1, 2, 7, 599, 601, True])
def test_unregistered_task_counts_fail_closed(count):
    with pytest.raises(ValueError):
        ex._preflight(Path("F:/private/fresh"), count=count)


def test_source_bad_or_absent_context_geometry_refused(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("n_ctx = 32768\nn_ctx_seq = 4096\nn_seq_max = 1\n")
    with pytest.raises(ValueError, match="per-sequence"):
        ex._verify_expanded_geometry(log)
    log.write_text("n_ctx = 32768\nn_ctx_seq = 32768\nn_seq_max = 4\n")
    with pytest.raises(ValueError, match="concurrency"):
        ex._verify_expanded_geometry(log)


def test_exact_context_geometry_native_log_accepted(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("llama_context: n_ctx = 32768\n"
                   "llama_context: n_ctx_seq = 32768\n"
                   "llama_context: n_seq_max = 1\n")
    result = ex._verify_expanded_geometry(log)
    assert result["observed_context_per_seq"] == 32768
    assert result["observed_sequences"] == 1
    assert len(result["native_log_sha256"]) == 64


def test_conservative_extra_gpu_fails_before_source_hash_or_process_start(tmp_path):
    with (patch.object(ex.base, "_output_check"),
          patch.object(ex.base, "_fresh_admission", return_value=(
              {"status": "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"},
              {"gpu_free_mib": 12350})),
          patch.object(ex.watchdog, "free_vram_mib", return_value=12350)):
        result = ex._preflight(tmp_path / "fresh", count=6)
        assert result["status"] == "REFUSED_RESOURCE_ADMISSION"
        assert result["requires_extra_gpu_mib"] == 512


def test_missing_original_admission_refuses_extra_context(tmp_path):
    with (patch.object(ex.base, "_output_check"),
          patch.object(ex.base, "_fresh_admission", return_value=(
              {"status": "REFUSED", "reasons": ["commit below source-bound floor"]},
              {"gpu_free_mib": 14000})),
          patch.object(ex.watchdog, "free_vram_mib", return_value=14000)):
        result = ex._preflight(tmp_path / "fresh", count=6)
        assert result["status"] == "REFUSED_RESOURCE_ADMISSION"


def test_no_gguf_hash_or_gpu_spawn_on_failed_preflight(tmp_path):
    with (patch.object(ex, "_preflight", return_value={"status": "REFUSED_RESOURCE_ADMISSION"}),
          patch.object(ex.base, "_source_check") as src,
          patch.object(ex.watchdog, "sha") as model_hash,
          patch.object(ex.subprocess, "Popen") as start):
        result = ex._run(tmp_path / "new", count=6)
        assert result["status"] == "REFUSED_RESOURCE_ADMISSION"
        src.assert_not_called()
        model_hash.assert_not_called()
        start.assert_not_called()
