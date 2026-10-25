"""Real pinned quantizer tests for Q4_K_M mini-group byte preservation."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from gguf import GGUFReader, GGMLQuantizationType

from chowder_batch import reap_gguf_chunked as core
from chowder_batch import reap_gguf_q4_chunked as q4
from test_reap_gguf_chunked import _small_bf16


def test_q4_merge_refuses_incomplete_provenance(tmp_path):
    model = tmp_path / "bf16.gguf"
    _small_bf16(model)
    with pytest.raises(ValueError, match="fresh output and complete chunk"):
        q4.merge_q4(model, [], tmp_path / "absent.gguf", {})


def test_q4_refuses_a_running_q8_or_other_quantizer(monkeypatch):
    import psutil

    fake_owner = SimpleNamespace(info={
        "pid": os.getpid() + 1, "name": "python.exe",
        "cmdline": ["python", "-m", "chowder_batch.reap_gguf_chunked", "--execute"],
    })
    monkeypatch.setattr(psutil, "process_iter", lambda fields: [fake_owner])
    with pytest.raises(ValueError, match="concurrent Q8/Q4"):
        q4._require_no_competing_quantizer()
    fake_owner.info = {"pid": os.getpid() + 2, "name": "llama-quantize.exe",
                       "cmdline": []}
    with pytest.raises(ValueError, match="concurrent Q8/Q4"):
        q4._require_no_competing_quantizer()
    fake_owner.info = {"pid": os.getpid() + 3, "name": "llama-server.exe",
                       "cmdline": ["ollama", "serve"]}
    q4._require_no_competing_quantizer()


@pytest.mark.skipif(
    os.name != "nt" or not Path("C:/llama.cpp/bin/llama-quantize.exe").is_file(),
    reason="real registered pinned Windows llama-quantize is unavailable",
)
def test_q4_minis_match_pinned_monolith_raw_tensors(tmp_path, monkeypatch):
    from chowder_batch import reap_gguf_supervised as supervisor

    try:
        q4._require_no_competing_quantizer()
    except ValueError:
        pytest.skip("another registered Q8/Q4 constructor currently owns the host")
    source = tmp_path / "source.gguf"
    _small_bf16(source)
    quantizer = Path("C:/llama.cpp/bin/llama-quantize.exe")
    groups = core.groups_for(GGUFReader(str(source), "r"), max_mib=8)
    assert len(groups) >= 2

    def run_small(command, **kwargs):
        p = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        )
        assert p.returncode == 0, p.stdout + p.stderr
        return p.returncode

    monkeypatch.setattr(supervisor, "_owned_stage", run_small)
    parts, receipts = [], {}
    for i, names in enumerate(groups):
        mini = tmp_path / f"mini-{i}.gguf"
        out = tmp_path / f"part-{i}.gguf"
        original = core.create_mini(source, mini, names)
        assert core.hash_file(mini) == original["source_mini_sha256"]
        quant = core.quantize_mini(
            mini, out, quantizer=quantizer, mode="Q4_K_M",
            memory_mib=1, log=io.StringIO(),
        )
        receipts.update(quant["quantized_tensors"])
        parts.append(out)
    assembled = tmp_path / "combined.gguf"
    result = q4.merge_q4(source, parts, assembled, receipts)
    assert result["tensors"] == 6
    assert result["mode"] == "Q4_K_M"
    assert GGUFReader(str(assembled), "r").get_field("general.file_type").contents() == 15
    canonical = tmp_path / "monolithic.gguf"
    run_small([str(quantizer), str(source), str(canonical), "Q4_K_M", "2"])
    expected = {t.name: t for t in GGUFReader(str(canonical), "r").tensors}
    actual = {t.name: t for t in GGUFReader(str(assembled), "r").tensors}
    assert set(actual) == set(expected)
    assert any(t.tensor_type == GGMLQuantizationType.Q4_K for t in actual.values())
    for name, tensor in actual.items():
        assert tensor.tensor_type == expected[name].tensor_type
        assert list(tensor.shape) == list(expected[name].shape)
        assert core.hash_tensor_data(tensor.data) == core.hash_tensor_data(expected[name].data)
