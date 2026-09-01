from chowder.hardware import _parse_nvidia_smi, detect_hardware


def test_parse_nvidia_smi_multiple_gpus():
    parsed = _parse_nvidia_smi(
        "NVIDIA RTX 5060 Ti, 16380, 00000000:01:00.0\n"
        "NVIDIA RTX 2060, 6144, 00000000:02:00.0\n"
    )
    assert len(parsed) == 2
    assert parsed[0].vendor == "nvidia"
    assert parsed[0].name == "NVIDIA RTX 5060 Ti"
    assert 15.9 < parsed[0].memory_gb < 16.1
    assert parsed[1].bus_id == "00000000:02:00.0"


def test_parse_nvidia_smi_ignores_bad_rows():
    parsed = _parse_nvidia_smi("garbage\nGPU, not-a-number, bus\n")
    assert parsed == ()


def test_detect_hardware_returns_storage_and_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr("chowder.hardware._detect_nvidia", lambda: ())
    snapshot = detect_hardware(tmp_path)
    assert snapshot.cpu_count >= 1
    assert snapshot.storage_total_gb > 0
    assert snapshot.storage_free_gb >= 0
    assert snapshot.accelerators == ()
