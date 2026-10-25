import io
import json
import os
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from chowder_batch import reap_eval, reap_serve  # noqa: E402

FIT_LOG = """\
llama_params_fit_impl: projected to use 20310 MiB of device memory vs. 15200 MiB of free device memory
llama_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce device memory by 6134 MiB
llama_params_fit_impl: filling dense-only layers back-to-front:
llama_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 5060 Ti): 41 layers (19 overflowing), 12000 MiB used, 3100 MiB free
llama_params_fit: successfully fit params to free device memory
load_tensors: offloaded 41/41 layers to GPU
srv  log_server_r: unrelated line
"""

FIT_NONE = """\
llama_params_fit_impl: projected to use 12000 MiB of device memory vs. 15200 MiB of free device memory
llama_params_fit_impl:   - CUDA0 (NVIDIA GeForce RTX 5060 Ti): 41 layers (0 overflowing), 12000 MiB used, 3100 MiB free
load_tensors: offloaded 41/41 layers to GPU
"""

FIT_CPU_MOE = "llama_params_fit_impl: set n_cpu_moe = 17\nload_tensors: offloaded 24/41 layers to GPU\n"

FIT_OVERRIDE = (
    "tensor blk.30.ffn_up_exps.weight buffer type overridden to CPU\n"
    "tensor blk.30.ffn_down_exps.weight buffer type overridden to CPU\n"
    "tensor blk.30.ffn_gate_exps.weight buffer type overridden to CPU\n"
    "tensor blk.31.ffn_up_exps.weight buffer type overridden to CPU\n"
    "tensor blk.31.ffn_down_exps.weight buffer type overridden to CPU\n"
    "tensor blk.31.ffn_gate_exps.weight buffer type overridden to CPU\n"
)


def test_parse_overflowing_layers():
    lines, n = reap_serve.parse_offload(FIT_LOG)
    assert n == 19
    assert any("19 overflowing" in l for l in lines)
    assert not any("unrelated" in l for l in lines)


def test_parse_zero_overflow():
    _, n = reap_serve.parse_offload(FIT_NONE)
    assert n == 0


def test_parse_n_cpu_moe():
    _, n = reap_serve.parse_offload(FIT_CPU_MOE)
    assert n == 17


def test_parse_override_lines_count_layers():
    _, n = reap_serve.parse_offload(FIT_OVERRIDE)
    assert n == 2


def test_parse_unparseable_is_null():
    lines, n = reap_serve.parse_offload("load_tensors: offloaded 20/41 layers to GPU\nhello\n")
    assert n is None
    assert lines == ["load_tensors: offloaded 20/41 layers to GPU"]
    assert reap_serve.parse_offload("") == ([], None)


class FakeProc:
    terminated = False
    pid = 4242

    def __init__(self, cmd, stdout=None, stderr=None, **kw):
        self.cmd = cmd
        self.env = kw.get("env")
        stdout.write(FIT_LOG.encode())
        stdout.flush()
        FakeProc.last = self

    def poll(self):
        return None

    def terminate(self):
        FakeProc.terminated = True

    def kill(self):
        FakeProc.terminated = True

    def wait(self, timeout=None):
        return 0


class FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b'{"status":"ok"}'


def _setup(monkeypatch, eval_rc=0, health_fail=False):
    calls = []
    FakeProc.terminated = False
    monkeypatch.setattr(reap_serve, "_require_owned_listener", lambda proc, port: None)

    class FakeEvalProc:
        def __init__(self, cmd):
            calls.append(cmd)

        def poll(self):
            return eval_rc

        def wait(self, timeout=None):
            return eval_rc

    def fake_popen(cmd, **kw):
        if any("reap_eval.py" in str(part) for part in cmd):
            return FakeEvalProc(cmd)
        return FakeProc(cmd, **kw)

    monkeypatch.setattr(reap_serve.subprocess, "Popen", fake_popen)

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class R:
            returncode = eval_rc
            stdout = "14321\n"
        return R()

    monkeypatch.setattr(reap_serve.subprocess, "run", fake_run)

    def fake_urlopen(url, timeout=None):
        if health_fail:
            raise OSError("down")
        return FakeResp()

    monkeypatch.setattr(reap_serve.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(reap_serve.time, "sleep", lambda s: None)
    return calls


def test_serve_json_and_eval_run(tmp_path, monkeypatch):
    calls = _setup(monkeypatch)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"abc")
    rc = reap_serve.main(["--gguf", str(gguf), "--arm", "dense", "--port", "8790",
                          "--out", str(tmp_path / "eval"), "--", "--port", "8790", "--arm", "dense",
                          "--out", str(tmp_path / "eval")])
    assert rc == 0
    cmd = FakeProc.last.cmd
    assert cmd[1:] == ["--fit", "on", "-np", "4", "-c", "16384", "--jinja", "--port", "8790", "-m", str(gguf)] or \
        cmd[1:] == ["-m", str(gguf), "--fit", "on", "-np", "4", "-c", "16384", "--jinja", "--port", "8790"]
    sj = json.loads((tmp_path / "eval" / "dense" / "serve.json").read_text())
    assert sj["gguf_sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sj["experts_offloaded"] == 19
    assert sj["vram_used_mib"] == 14321
    assert sj["offload_log_lines"]
    assert {"gguf", "flags", "started", "ready_seconds"} <= set(sj)
    assert (tmp_path / "eval" / "dense" / "server.log").exists()
    eval_cmd = [c for c in calls if any("reap_eval.py" in str(x) for x in c)][0]
    assert eval_cmd[-6:] == ["--port", "8790", "--arm", "dense", "--out", str(tmp_path / "eval")]
    assert FakeProc.terminated
    assert FakeProc.last.env["LLAMA_ARG_LOG_VERBOSITY"] == "4" and sj["log_verbosity"] == "4"


def test_server_terminated_when_eval_fails(tmp_path, monkeypatch):
    _setup(monkeypatch, eval_rc=3)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"abc")
    rc = reap_serve.main(["--gguf", str(gguf), "--arm", "a", "--out", str(tmp_path), "--"])
    assert rc == 3
    assert FakeProc.terminated


def test_server_terminated_when_health_times_out(tmp_path, monkeypatch):
    _setup(monkeypatch, health_fail=True)
    ticks = iter(range(0, 10000, 400))
    monkeypatch.setattr(reap_serve.time, "time", lambda: next(ticks))
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"abc")
    try:
        reap_serve.main(["--gguf", str(gguf), "--arm", "a", "--out", str(tmp_path), "--"])
        raised = False
    except TimeoutError:
        raised = True
    assert raised and FakeProc.terminated


def test_eval_keeps_predicted_per_second(monkeypatch):
    payload = {"choices": [{"message": {"content": "#### 4"}, "finish_reason": "stop"}],
               "usage": {"completion_tokens": 5}, "timings": {"predicted_per_second": 42.5}}

    class Resp(FakeResp):
        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(reap_eval.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    out = reap_eval._chat(8790, "q", 16)
    row = reap_eval._score("gsm8k", {"id": "x", "expected": "4"}, out)
    assert row["predicted_per_second"] == 42.5
    payload.pop("timings")
    row = reap_eval._score("gsm8k", {"id": "x", "expected": "4"}, reap_eval._chat(8790, "q", 16))
    assert row["predicted_per_second"] is None


FULL_FIT = (
    "0.00.269 I common_fit_params: successfully fit params to free device memory\n"
    "0.00.376 I load_tensors: offloaded 41/41 layers to GPU\n"
)


def test_clean_full_fit_is_a_measured_zero():
    lines, n = reap_serve.parse_offload(FULL_FIT)
    assert n == 0 and len(lines) == 2


def test_full_fit_with_expert_override_counts_layers():
    log = FULL_FIT + "tensor blk.7.ffn_up_exps.weight (q4_K) buffer type overridden to CPU\n"
    assert reap_serve.parse_offload(log)[1] == 1


def test_all_layers_without_fit_line_stays_unknown():
    assert reap_serve.parse_offload("load_tensors: offloaded 41/41 layers to GPU\n")[1] is None


@pytest.mark.parametrize("eval_args", [
    ["--port", "8767", "--arm", "reap160", "--out", "{out}"],
    ["--port", "8790", "--arm", "dense", "--out", "{out}"],
    ["--port", "8790", "--arm", "reap160", "--out", "{other}"],
    ["--port", "8790", "--port", "8767", "--arm", "reap160", "--out", "{out}"],
])
def test_serve_refuses_mismatched_or_ambiguous_eval_target_before_launch(tmp_path, monkeypatch, eval_args):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"small")
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    args = [str(tmp_path / "evaluation") if token == "{out}" else
            str(tmp_path / "different") if token == "{other}" else token for token in eval_args]
    with pytest.raises(ValueError, match="eval"):
        reap_serve.main(["--gguf", str(gguf), "--arm", "reap160", "--port", "8790",
                         "--out", str(tmp_path / "evaluation"), "--", *args])


def test_serve_refuses_foreign_listener_before_launch(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"small")
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(RuntimeError, match="port.*in use"):
            reap_serve.main(["--gguf", str(gguf), "--arm", "reap160", "--port", str(port),
                             "--out", str(tmp_path / "evaluation"), "--",
                             "--port", str(port), "--arm", "reap160", "--out", str(tmp_path / "evaluation")])


def _serve_args(gguf, out, arm="reap128_v2", port=8790):
    return ["--gguf", str(gguf), "--arm", arm, "--port", str(port),
            "--out", str(out), "--", "--port", str(port), "--arm", arm, "--out", str(out)]


def test_resume_preserves_original_serve_and_log(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    assert reap_serve.main(_serve_args(gguf, out)) == 0
    arm_dir = out / "reap128_v2"
    original_log = (arm_dir / "server.log").read_bytes()
    original_serve = (arm_dir / "serve.json").read_bytes()
    assert reap_serve.main(_serve_args(gguf, out)) == 0
    assert (arm_dir / "server.log").read_bytes() == original_log
    assert (arm_dir / "serve.json").read_bytes() == original_serve
    assert len(list(arm_dir.glob("server.resume-*.log"))) == 1
    assert len(list(arm_dir.glob("serve.resume-*.json"))) == 1
    assert not (arm_dir / ".serve.lock").exists()


def test_refuses_existing_different_model_without_mutation(tmp_path, monkeypatch):
    _setup(monkeypatch)
    first, second = tmp_path / "a.gguf", tmp_path / "b.gguf"
    first.write_bytes(b"model A")
    second.write_bytes(b"model B")
    out = tmp_path / "eval"
    reap_serve.main(_serve_args(first, out))
    arm_dir = out / "reap128_v2"
    previous = {p.name: p.read_bytes() for p in arm_dir.iterdir()}
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="model|GGUF|identity"):
        reap_serve.main(_serve_args(second, out))
    assert {p.name: p.read_bytes() for p in arm_dir.iterdir()} == previous


@pytest.mark.parametrize("legacy", ["server.log", "serve.json", "gsm8k.jsonl", "summary.json"])
def test_refuses_unclaimed_existing_output(tmp_path, monkeypatch, legacy):
    gguf = tmp_path / "a.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    arm_dir = out / "reap128_v2"
    arm_dir.mkdir(parents=True)
    (arm_dir / legacy).write_bytes(b"historical evidence")
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="existing|identity|incomplete|unclaimed"):
        reap_serve.main(_serve_args(gguf, out))
    assert (arm_dir / legacy).read_bytes() == b"historical evidence"


def test_refuses_historical_results_without_input_witness(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "a.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    reap_serve.main(_serve_args(gguf, out))
    arm_dir = out / "reap128_v2"
    (arm_dir / "gsm8k.jsonl").write_bytes(b'historical answers\n')
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="witness|unclaimed"):
        reap_serve.main(_serve_args(gguf, out))
    assert (arm_dir / "gsm8k.jsonl").read_bytes() == b'historical answers\n'


def test_refuses_invalid_arm_path_and_concurrent_lock(tmp_path, monkeypatch):
    gguf = tmp_path / "a.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="arm"):
        reap_serve.main(_serve_args(gguf, out, "../archive"))
    arm_dir = out / "reap128_v2"
    arm_dir.mkdir(parents=True)
    (arm_dir / ".serve.lock").write_text("other owner")
    with pytest.raises((RuntimeError, ValueError), match="lock|session|existing"):
        reap_serve.main(_serve_args(gguf, out))
    assert (arm_dir / ".serve.lock").read_text() == "other owner"


def test_resume_accepts_matching_witnessed_partial_results(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    reap_serve.main(_serve_args(gguf, out))
    arm_dir = out / "reap128_v2"
    served = json.loads((arm_dir / "serve.json").read_text())
    witness = {"schema": reap_eval.IDENTITY_SCHEMA, "suite": "gsm8k",
               "model": {key: served[key] for key in ("gguf", "gguf_sha256", "flags")}}
    (arm_dir / "gsm8k.inputs.json").write_text(json.dumps(witness))
    (arm_dir / "gsm8k.jsonl").write_text('{"id":"gsm8k-0","correct":true}\n')
    existing = {p.name: p.read_bytes() for p in arm_dir.iterdir()}
    assert reap_serve.main(_serve_args(gguf, out)) == 0
    assert all((arm_dir / name).read_bytes() == payload for name, payload in existing.items())


def test_resume_refuses_changed_partial_results_before_launch(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    reap_serve.main(_serve_args(gguf, out))
    arm_dir = out / "reap128_v2"
    served = json.loads((arm_dir / "serve.json").read_text())
    (arm_dir / "gsm8k.inputs.json").write_text(json.dumps({
        "schema": reap_eval.IDENTITY_SCHEMA, "suite": "gsm8k",
        "model": {key: served[key] for key in ("gguf", "gguf_sha256", "flags")},
    }))
    (arm_dir / "gsm8k.jsonl").write_text('{"id":"gsm8k-0"}\n')
    (arm_dir / "summary.json").write_text(json.dumps({
        "gsm8k": {"file_sha256": "0" * 64, "n": 1},
    }))
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="summary"):
        reap_serve.main(_serve_args(gguf, out))
    assert not (arm_dir / ".serve.lock").exists()


def test_evaluator_is_stopped_if_launched_server_exits(monkeypatch):
    class GoneServer:
        def poll(self):
            return 7

    class PendingEval:
        stopped = False

        def poll(self):
            return 1 if self.stopped else None

        def wait(self, timeout=None):
            if not self.stopped:
                raise reap_serve.subprocess.TimeoutExpired("eval", timeout)
            return 1

        def terminate(self):
            self.stopped = True

    evaluation = PendingEval()
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: evaluation)
    with pytest.raises(RuntimeError, match="server exited during evaluation"):
        reap_serve._run_eval(GoneServer(), ["fake-evaluator"])
    assert evaluation.stopped


def test_model_changed_while_hashing_is_rejected_before_launch(tmp_path, monkeypatch):
    gguf = tmp_path / "mutable.gguf"
    gguf.write_bytes(b"model A")
    real_sha = reap_serve._sha256

    def changed_sha(path):
        digest = real_sha(path)
        path.write_bytes(b"model B updated")
        return digest

    monkeypatch.setattr(reap_serve, "_sha256", changed_sha)
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    out = tmp_path / "eval"
    with pytest.raises(RuntimeError, match="changed while hashing"):
        reap_serve.main(_serve_args(gguf, out))
    assert not (out / "reap128_v2" / ".serve.lock").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows listener owner proof")
def test_windows_listener_pid_refuses_foreign_and_accepts_own_process(monkeypatch):
    class FakeCheck:
        returncode = 0
        stdout = "67890\n"

    class Owner:
        pid = 4242

    with monkeypatch.context() as patch:
        patch.setattr(reap_serve.subprocess, "run", lambda *a, **kw: FakeCheck())
        with pytest.raises(RuntimeError, match="not owned"):
            reap_serve._require_owned_listener(Owner(), 8790)

    # A real, temporary localhost socket verifies the standard Windows PID lookup.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        class CurrentProcess:
            pid = os.getpid()
        reap_serve._require_owned_listener(CurrentProcess(), sock.getsockname()[1])


def test_refuse_forged_offload_from_original_log_on_resume(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    assert reap_serve.main(_serve_args(gguf, out)) == 0
    arm = out / "reap128_v2"
    receipt = json.loads((arm / "serve.json").read_text())
    assert receipt["experts_offloaded"] == 19
    receipt["experts_offloaded"] = 0
    (arm / "serve.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="offload|log|receipt|evidence"):
        reap_serve.main(_serve_args(gguf, out))


def test_q4_without_same_session_200_rows_stays_unknown(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    assert reap_serve.main(_serve_args(gguf, out, arm="reap128_v2_q4")) == 0
    receipt = json.loads((out / "reap128_v2_q4" / "serve.json").read_text())
    assert receipt["experts_offloaded"] is None
    assert receipt["measured_experts_offloaded"] == 19
    assert "speed_session" not in receipt


def test_q4_nonfrozen_eval_args_do_not_certify_speed(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    args = _serve_args(gguf, out, arm="dense_q4") + ["--suites", "gsm8k", "--limit", "199"]
    assert reap_serve.main(args) == 0
    receipt = json.loads((out / "dense_q4" / "serve.json").read_text())
    assert receipt["experts_offloaded"] is None


def test_tampered_startup_log_refuses_compatible_resume(tmp_path, monkeypatch):
    _setup(monkeypatch)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    assert reap_serve.main(_serve_args(gguf, out)) == 0
    arm = out / "reap128_v2"
    with (arm / "server.log").open("ab") as log:
        log.write(b"\nextra generation logging is allowed\n")
    assert reap_serve.main(_serve_args(gguf, out)) == 0
    original = (arm / "server.log").read_bytes()
    (arm / "server.log").write_bytes(b"X" + original[1:])
    monkeypatch.setattr(reap_serve.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="log|receipt|evidence"):
        reap_serve.main(_serve_args(gguf, out))


def _mock_frozen_q4_scores(monkeypatch, *, missing_last=False, bad_item=False,
                           missing_witness=False, tampered_startup=False):
    items = [{"id": f"gsm8k-{i}", "prompt": f"question {i}", "expected": "1"}
             for i in range(1319)]
    monkeypatch.setattr(reap_eval, "_load", lambda suite: items)

    def write_scored_rows(server, command):
        words = command
        out = Path(words[words.index("--out") + 1])
        arm = words[words.index("--arm") + 1]
        root = out / arm
        if not missing_witness:
            reap_eval._bind_evaluation_inputs(root, "gsm8k", items)
        count = 199 if missing_last else 200
        rows = [dict(id=f"gsm8k-{i}", correct=True, terminated=True,
                     trigram_ratio=1.0, completion_tokens=10, predicted_per_second=70.0,
                     seconds=1.0, text="answer") for i in range(count)]
        if bad_item:
            rows[0]["id"] = "gsm8k-201"
        result_file = root / "gsm8k.jsonl"
        result_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        saved = reap_eval._summarize_rows(rows, result_file)
        (root / "summary.json").write_text(json.dumps({"gsm8k": saved}), encoding="utf-8")
        if tampered_startup:
            path = root / "server.log"
            raw = path.read_bytes()
            path.write_bytes(b"X" + raw[1:])
        return 0

    monkeypatch.setattr(reap_serve, "_run_eval", write_scored_rows)


@pytest.mark.skipif(os.name != "nt", reason="same-session speed attestation requires Windows listener PID")
def test_new_q4_same_session_first_200_receipt_certifies_source_and_fit(tmp_path, monkeypatch):
    _setup(monkeypatch)
    _mock_frozen_q4_scores(monkeypatch)
    gguf = tmp_path / "q4.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    args = _serve_args(gguf, out, arm="reap128_v2_q4") + ["--suites", "gsm8k", "--limit", "200"]
    assert reap_serve.main(args) == 0
    arm = out / "reap128_v2_q4"
    recorded = json.loads((arm / "serve.json").read_text())
    assert recorded["experts_offloaded"] == recorded["measured_experts_offloaded"] == 19
    proof = recorded["speed_session"]
    assert proof["schema"] == "chowder.reap-q4-same-session-speed.v1"
    assert proof["n"] == 200
    assert proof["rows_sha256"] == reap_serve._sha256(arm / "gsm8k.jsonl")
    assert proof["source_items_sha256"] == reap_eval._digest(reap_eval._load("gsm8k"))
    assert reap_serve._admit_output(arm, gguf=gguf.resolve(), digest=reap_serve._sha256(gguf))
    assert not (arm / ".serve.lock").exists()


@pytest.mark.parametrize("attack", ["missing_last", "bad_item", "missing_witness", "tampered_startup"])
@pytest.mark.skipif(os.name != "nt", reason="same-session speed attestation requires Windows listener PID")
def test_q4_refuses_false_same_session_200_proof(tmp_path, monkeypatch, attack):
    _setup(monkeypatch)
    _mock_frozen_q4_scores(monkeypatch, **{attack: True})
    gguf = tmp_path / "q4.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    args = _serve_args(gguf, out, arm="reap128_v2_q4") + ["--suites", "gsm8k", "--limit", "200"]
    with pytest.raises(ValueError, match="(speed|source|summary|identity|log|witness|result|row)"):
        reap_serve.main(args)
    recorded = json.loads((out / "reap128_v2_q4" / "serve.json").read_text())
    assert recorded["experts_offloaded"] is None
    assert "speed_session" not in recorded
    assert not (out / "reap128_v2_q4" / ".serve.lock").exists()


@pytest.mark.skipif(os.name != "nt", reason="same-session speed attestation requires Windows listener PID")
def test_q4_certified_speed_refuses_later_row_tampering(tmp_path, monkeypatch):
    _setup(monkeypatch)
    _mock_frozen_q4_scores(monkeypatch)
    gguf = tmp_path / "q4.gguf"
    gguf.write_bytes(b"model A")
    out = tmp_path / "eval"
    args = _serve_args(gguf, out, arm="dense_q4") + ["--suites", "gsm8k", "--limit", "200"]
    assert reap_serve.main(args) == 0
    arm = out / "dense_q4"
    file = arm / "gsm8k.jsonl"
    file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="summary|speed|evidence"):
        reap_serve._admit_output(arm, gguf=gguf.resolve(), digest=reap_serve._sha256(gguf))


def test_q4_missing_verified_fit_stays_unknown_even_with_frozen_args(tmp_path):
    serving = {"measured_experts_offloaded": None}
    assert reap_serve._certify_q4_speed_session(
        tmp_path, eval_args=["--suites", "gsm8k", "--limit", "200"],
        receipt=serving, session_id="test", port=8790, server_pid=4242,
    ) is None
