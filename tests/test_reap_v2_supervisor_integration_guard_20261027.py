"""CPU-only adversarial tests for original REAP frozen-wave/Job native binding."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from chowder_batch import reap_v2_frozen_wave_20261027 as wave
from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as supervisor
from chowder_batch import reap_v2_supervisor_integration_guard_20261027 as guard


MODEL = Path("G:/reap128-v2-q4_k_m-20261025.gguf")
SERVER = Path("F:/llama.cpp/bin/llama-server.exe")


def native_prefix(
    *, fitted=True, cpu=None, layers=41, ctx=16384, slots=4, extra="", model=MODEL
):
    source = (
        f"loading model '{model}'\n"
        f"llama_context: n_ctx = {ctx}\n"
        f"llama_context: n_seq_max = {slots}\n"
        f"llama_context: n_ctx_seq = {ctx // slots}\n"
        "llama_model_loader: n_expert = 128\n"
        "llama_model_loader: n_expert_used = 8\n"
        f"llama_model_load_tensors: offloaded {layers}/41 layers to GPU\n"
        "llama_kv_cache_init: CUDA0 KV buffer size = 320.00 MiB\n"
    )
    if fitted:
        source += "llama_params_fit: successfully fit on GPU\n"
    if cpu is not None:
        source += f"llama_context: n_cpu_moe = {cpu}\n"
    return (source + extra).encode("utf-8")


def sample():
    return {
        "physical_available_bytes": 8 * 2**30,
        "commit_headroom_bytes": 10 * 2**30,
        "disk_free_bytes": 10 * 2**30,
        "gpu": {
            "uuid": "GPU-12345678",
            "total_mib": 65536,
            "used_mib": 46000,
            "free_mib": 19300,
            "reserved_mib": 236,
            "utilization_pct": 30,
        },
        "root_owner_valid": True,
        "job_processes_valid": True,
        "job_process_count": 3,
        "foreign_ports": [],
        "foreign_processes": [],
        "owned_listener": True,
        "native_started": True,
        "native_residency": supervisor.parse_native_residency(
            native_prefix(fitted=False).decode("utf-8")
        ),
        "model_log_bytes": 12000,
        "elapsed_seconds": 120,
        "startup_seconds": 120,
    }


def test_red_old_supervisor_accepts_unfitted_log_before_any_native_fit():
    forged = supervisor.parse_native_residency(
        native_prefix(fitted=False).decode("utf-8")
    )
    assert forged["cpu_expert_offload"] == 0
    assert supervisor.validate_sample(sample()) == []  # real severe acceptance bug
    assert (
        "NATIVE_SUCCESSFUL_FIT_UNPROVEN"
        in guard.assess_native_startup(
            native_prefix(fitted=False), expected_model=MODEL
        )["reasons"]
    )


@pytest.mark.parametrize(
    "data,reason",
    [
        (native_prefix(fitted=False), "NATIVE_SUCCESSFUL_FIT_UNPROVEN"),
        (native_prefix(cpu=2), "NATIVE_CPU_EXPERT_OFFLOAD"),
        (native_prefix(layers=40), "NATIVE_LAYER_OFFLOAD"),
        (native_prefix(ctx=32768), "NATIVE_FROZEN_CONTEXT"),
        (native_prefix(slots=1), "NATIVE_FROZEN_SLOTS"),
        (native_prefix(model=Path("F:/foreign.gguf")), "NATIVE_MODEL_IDENTITY"),
        (
            native_prefix(extra="blk.7.ffn_exps.weight overridden to CPU\n"),
            "NATIVE_CPU_EXPERT_OFFLOAD",
        ),
        (native_prefix(extra="(3 overflowing)\n"), "NATIVE_CPU_EXPERT_OFFLOAD"),
        (
            native_prefix(extra="llama_context: n_ctx = 32768\n"),
            "NATIVE_CONTEXT_AMBIGUOUS",
        ),
        (b"", "NATIVE_STARTUP_SOURCE_INVALID"),
        (b"\xff\xfe", "NATIVE_STARTUP_SOURCE_INVALID"),
    ],
)
def test_frozen_native_source_fail_closed(data, reason):
    assert reason in guard.assess_native_startup(data, expected_model=MODEL)["reasons"]


def test_native_happy_path_is_observation_only_not_model_selection():
    verdict = guard.assess_native_startup(native_prefix(), expected_model=MODEL)
    assert verdict["status"] == "SOURCE_FIT_WITNESSED_ONLY"
    assert verdict["reasons"] == []
    assert verdict["residency"]["slots"] == 4
    assert verdict["residency"]["cpu_expert_offload"] == 0
    assert verdict["gen0"] == {
        "status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
    real_style = native_prefix().replace(
        b"llama_params_fit: successfully fit",
        b"common_fit_params: successfully fit params",
    )
    real_style += b"".join(
        f"0.10 I slot load_model: id {n} | new slot, n_ctx = 4096\n".encode()
        for n in range(4)
    )
    assert (
        guard.assess_native_startup(real_style, expected_model=MODEL)["reasons"] == []
    )


def test_authentic_original_windows_q4_1319_full_log_prefix_read_only():
    path = Path(
        "F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152/"
        "reap128_v2_q4/server.log"
    )
    if not path.is_file():
        pytest.skip("original F-drive Windows native evidence absent on this machine")
    with path.open("rb") as file:
        prefix = file.read(guard.MAX_SOURCE_BYTES)
    verdict = guard.assess_native_startup(prefix, expected_model=MODEL)
    assert verdict["status"] == "SOURCE_FIT_WITNESSED_ONLY", verdict
    assert verdict["reasons"] == [], verdict
    assert verdict["residency"]["slots"] == 4
    assert verdict["residency"]["context_per_slot"] == 4096
    assert verdict["residency"]["kv_total_mib"] == 320.0
    assert verdict["residency"]["cpu_expert_offload"] == 0


def test_cli_imported_dataclass_must_be_rebound_from_exact_original(monkeypatch):
    import dataclasses

    local = wave.study_for("reap128_v2_q4")
    CLIStudy = dataclasses.make_dataclass(
        "Study", [(f.name, f.type) for f in dataclasses.fields(local)], frozen=True
    )
    injected = CLIStudy(*dataclasses.astuple(local))
    assert injected != local
    assert guard.rebind_wave_study(injected) is local
    with pytest.raises(guard.GuardRefused):
        guard.rebind_wave_study(CLIStudy(local.arm, local.suite, 0, local.model_sha256))


def test_live_owner_identity_requires_job_member_birth_and_frozen_argv(
    monkeypatch, tmp_path
):
    controller = wave.study_for("reap128_v2_q4")
    command = wave.command(
        controller, model=MODEL, output=tmp_path / "out", server=SERVER, port=8790
    )
    root = {"pid": 1001, "birth": 1729000000.0, "argv": command}
    server = {
        "pid": 1002,
        "birth": 1729000001.0,
        "argv": [str(SERVER), *wave.FLAGS, "--port", "8790", "-m", str(MODEL)],
        "parent": 1001,
    }
    evaluator = {
        "pid": 1003,
        "birth": 1729000002.0,
        "argv": [
            sys.executable,
            str(wave.REPO / "chowder_batch" / "reap_eval.py"),
            *command[command.index("--") + 1 :],
        ],
        "parent": 1001,
    }
    owner = {
        "active": True,
        "job_present": True,
        "job_limit_kill_on_close": True,
        "root_pid": 1001,
        "root_birth": 1729000000.0,
        "members": {1001: root, 1002: server, 1003: evaluator},
        "owned_listener": {"pid": 1002, "port": 8790},
        "other_listeners": [],
    }
    good = guard.assess_bound_owner(
        owner,
        study=controller,
        model=MODEL,
        server=SERVER,
        output=tmp_path / "out",
        port=8790,
    )
    assert good["status"] == "OWNER_BOUND_OBSERVATION_ONLY"
    assert good["member_count"] == 3
    for tamper, expected in [
        ({"root_birth": 1729000050.0}, "ROOT_CREATE_TIME_MISMATCH"),
        ({"job_limit_kill_on_close": False}, "JOB_KILL_ON_CLOSE_UNPROVEN"),
        ({"owned_listener": {"pid": 1, "port": 8790}}, "OWNED_LISTENER_MISMATCH"),
        ({"other_listeners": [{"pid": 55032, "port": 8890}]}, "FOREIGN_LISTENER"),
        (
            {
                "members": {
                    1001: root,
                    1002: {**server, "argv": ["cmd", "/c", "foreign.exe"]},
                }
            },
            "NATIVE_SERVER_COMMAND_MISMATCH",
        ),
        (
            {"members": {1001: root, 1002: {**server, "birth": 1728999999.0}}},
            "CHILD_CREATED_BEFORE_ROOT",
        ),
        (
            {
                "members": {
                    1001: root,
                    1002: server,
                    1003: {**evaluator, "argv": ["cmd.exe", "/c", "whoami"]},
                }
            },
            "FOREIGN_OWNED_CHILD_COMMAND",
        ),
    ]:
        assert (
            expected
            in guard.assess_bound_owner(
                {**owner, **tamper},
                study=controller,
                model=MODEL,
                server=SERVER,
                output=tmp_path / "out",
                port=8790,
            )["reasons"]
        )


def test_integrated_sample_refuses_unproven_fit_despite_old_supervisor_pass(tmp_path):
    proof = guard.assess_runtime_integration(
        sample(), startup=native_prefix(fitted=False), expected_model=MODEL
    )
    assert proof["status"] == "REFUSED"
    assert "NATIVE_SUCCESSFUL_FIT_UNPROVEN" in proof["reasons"]
    assert (
        guard.assess_runtime_integration(
            sample(), startup=native_prefix(), expected_model=MODEL
        )["status"]
        == "RUNTIME_INTEGRATION_OBSERVED_ONLY"
    )


def test_receipt_cross_binding_undischarged_postclose_proof_refuses(tmp_path):
    study = wave.study_for("reap128_v2_q4")
    source_sha = "c" * 64
    expected = "e" * 64
    command = wave.command(
        study, model=MODEL, output=tmp_path / "out", server=SERVER, port=8790
    )
    initial = {
        "schema": supervisor.SCHEMA,
        "arm": study.arm,
        "status": "PRELAUNCH_SOURCE_AND_CAPACITY_VERIFIED",
        "model_sha256": study.model_sha256,
        "command": command,
        "native_server_sha256": expected,
        "original_offline_source_proof": {
            "sha256": source_sha,
            "source_offline_verified": True,
        },
        "gen0": guard.GEN0_BLOCKED,
    }
    final = {
        "schema": supervisor.SCHEMA,
        "arm": study.arm,
        "status": "CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY",
        "model_sha256": study.model_sha256,
        "child_pid": 1001,
        "child_create_time_unix": 1729000000.0,
        "original_offline_source_proof": {
            "sha256": source_sha,
            "source_offline_verified": True,
        },
        "owned_job_handle_closed_with_kill_on_close": True,
        "post_close_descendant_extinction_independently_witnessed": False,
        "exit_code": 0, "breaches": [],
        "pre_cleanup_live_owned_descendant_count": 0,
        "pre_cleanup_natural_owned_tree_extinction_witnessed": True,
        "all_private_job_lifetime_processes_source_witnessed": True,
        "pre_cleanup_private_job_lifetime_accounting": {
            "total_processes": 3, "active_processes": 0,
            "total_page_fault_count": 0,
        },
        "original_live_job_owned_role_identities": {
            "root": {"pid": 1001, "birth": 1729000000.0},
            "server": {"pid": 1002, "birth": 1729000001.0},
            "evaluator": {"pid": 1003, "birth": 1729000002.0},
        },
        "fully_witnessed_native_fit": True,
        "original_live_job_root_server_evaluator_lineage_witnessed": True,
        "original_native_fit_source_sha256": "f" * 64,
        "observations": [{
            "original_live_owned_job_lineage_status": "OWNER_BOUND_ACTIVE_OBSERVED_ONLY",
            "original_native_fit_source_status": "NATIVE_SOURCE_FIT_VERIFIED_RUNTIME_ONLY",
        }],
        "gen0": guard.GEN0_BLOCKED,
    }
    assert (
        "OWNED_TREE_EXTINCTION_UNVERIFIED"
        in guard.assess_bound_receipts(
            initial,
            final,
            study=study,
            model=MODEL,
            server=SERVER,
            output=tmp_path / "out",
            port=8790,
        )["reasons"]
    )
    good = guard.assess_bound_receipts(
        initial,
        {**final, "post_close_descendant_extinction_independently_witnessed": True},
        study=study,
        model=MODEL,
        server=SERVER,
        output=tmp_path / "out",
        port=8790,
    )
    assert good["status"] == "RECEIPTS_CONSISTENT_NOT_SCIENTIFIC_APPROVAL"
    assert good["gen0"]["release_approved"] is False
    for key, falsified in (
        ("all_private_job_lifetime_processes_source_witnessed", False),
        ("pre_cleanup_natural_owned_tree_extinction_witnessed", False),
        ("original_live_job_root_server_evaluator_lineage_witnessed", False),
        ("fully_witnessed_native_fit", False),
        ("pre_cleanup_live_owned_descendant_count", 1),
        ("exit_code", 84),
        ("breaches", ["ALREADY_ABORTED"]),
        ("observations", []),
        ("pre_cleanup_private_job_lifetime_accounting", {
            "total_processes": 1, "active_processes": 0,
            "total_page_fault_count": 0,
        }),
    ):
        bad_result = guard.assess_bound_receipts(
            initial,
            {**final, "post_close_descendant_extinction_independently_witnessed": True,
             key: falsified},
            study=study, model=MODEL, server=SERVER,
            output=tmp_path / "out", port=8790,
        )
        assert bad_result["status"] == "REFUSED", (key, bad_result)
    bad = guard.assess_bound_receipts(
        initial,
        {**final, "original_offline_source_proof": {"sha256": "0" * 64}},
        study=study,
        model=MODEL,
        server=SERVER,
        output=tmp_path / "out",
        port=8790,
    )
    assert "ORIGINAL_OFFLINE_SOURCE_MISMATCH" in bad["reasons"]


def test_benign_windows_owned_pid_creation_and_natural_extinction():
    """Harmless Python sleep child only; never kill or signal other processes."""
    if sys.platform != "win32":
        pytest.skip("Native Windows only")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(0.25)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        import psutil

        born = psutil.Process(process.pid).create_time()
        observed = guard.observe_descendant_extinction({process.pid: born})
        assert "ORIGINAL_CHILD_STILL_ALIVE" in observed["reasons"]
        assert process.wait(timeout=5) == 0
        after = guard.observe_descendant_extinction({process.pid: born})
        assert after["status"] == "OWNED_DESCENDANTS_EXTINCT_OBSERVED_ONLY"
        assert after["model_selection_approved"] is False
    finally:
        process.wait(timeout=5)
