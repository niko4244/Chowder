"""No model loads: isolated synthetic GGUF stage publication and guard tests."""

from __future__ import annotations

import pytest

from chowder_batch import reap_gguf_supervised as supervise


def _spec(tmp_path):
    source = tmp_path / "candidate"
    source.mkdir()
    bf16, q8, q4 = (tmp_path / label for label in
                    ("new-original.gguf", "new-q8.gguf", "new-q4.gguf"))
    return {
        "candidate": str(source),
        "conversion_command": ["python", "convert_hf_to_gguf.py", str(source),
                               "--no-mtp", "--outtype", "bf16", "--outfile", str(bf16)],
        "q8_quality_command": ["llama-quantize.exe", str(bf16), str(q8), "Q8_0", "2"],
        "q4_fit_command": ["llama-quantize.exe", str(bf16), str(q4), "Q4_K_M", "2"],
        "capacity": {
            "bf16": {"intended": str(bf16)},
            "q8_0": {"intended": str(q8)},
            "q4_k_m": {"intended": str(q4)},
        },
    }


def _stub_command(command, **kwargs):
    if "--outfile" in command:
        destination = command[command.index("--outfile") + 1]
    else:
        destination = command[2]
    from pathlib import Path
    Path(destination).write_bytes(b"GGUF" + (3).to_bytes(4, "little") + b"x" * 1048576)
    return 0


def test_gguf_three_stage_publication_is_atomic_and_source_bf16_based(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    commands = []
    monkeypatch.setattr(supervise, "resource_problems", lambda *args, **kwargs: [])

    def record(command, **kwargs):
        commands.append(command)
        return _stub_command(command, **kwargs)

    monkeypatch.setattr(supervise, "_owned_stage", record)
    log = tmp_path / "job.log"
    assert supervise.build(specification=spec, log_path=log, execute=False)["status"] == (
        "ADMITTED_NOT_EXECUTED"
    )
    assert not log.exists()
    result = supervise.build(specification=spec, log_path=log, execute=True)
    assert result["status"] == "GGUF_CPU_BUILDS_COMPLETED_NOT_QUALITY_QUALIFIED"
    assert len(result["phases"]) == 3
    assert commands[0][commands[0].index("--outtype") + 1] == "bf16"
    assert commands[1][-2] == "Q8_0" and commands[2][-2] == "Q4_K_M"
    assert commands[1][1] == spec["capacity"]["bf16"]["intended"]
    assert commands[2][1] == spec["capacity"]["bf16"]["intended"]
    for phase in result["phases"]:
        assert __import__("pathlib").Path(phase["output"]).is_file()
        assert len(phase["sha256"]) == 64
    with pytest.raises(ValueError, match="new"):
        supervise.build(specification=spec, log_path=log, execute=True)


def test_gguf_supervisor_rejects_admission_and_never_launches(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    monkeypatch.setattr(supervise, "resource_problems",
                        lambda *args, **kwargs: ["unowned physical RAM below floor"])
    monkeypatch.setattr(supervise, "_owned_stage",
                        lambda *args, **kwargs: pytest.fail("must never launch"))
    with pytest.raises(ValueError, match="admission refused"):
        supervise.build(specification=spec, log_path=tmp_path / "log", execute=True)
