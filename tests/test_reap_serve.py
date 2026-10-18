import io
import json
import sys
from pathlib import Path

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

    def __init__(self, cmd, stdout=None, stderr=None, **kw):
        self.cmd = cmd
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
    monkeypatch.setattr(reap_serve.subprocess, "Popen", FakeProc)

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
