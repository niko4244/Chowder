"""CPU-only input preflight tests for the real MoE placement witness runner.

These cases must refuse before torch, tokenizer or checkpoint loading.
"""

import pytest

from chowder_batch import moe_int8_witness_run as witness


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--seq", "1"),
        ("--steps", "0"),
        ("--examples", "0"),
        ("--lora-r", "0"),
        ("--lr", "nan"),
        ("--lr", "0"),
    ],
)
def test_bad_witness_args_refuse_before_model_load(tmp_path, capsys, flag, value):
    calibration = tmp_path / "calib.jsonl"
    calibration.write_text('{"text":"valid example"}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        witness.main([
            "--base", str(tmp_path / "unused-model"),
            "--calib", str(calibration),
            "--out", str(tmp_path / "witness.json"),
            flag, value,
        ])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "payload",
    ["", "{not valid json}\n", '{"messages":[]}\n', '{"text":"   "}\n'],
)
def test_unusable_calibration_refuses_before_model_load(tmp_path, capsys, payload):
    calibration = tmp_path / "calib.jsonl"
    calibration.write_text(payload, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        witness.main([
            "--base", str(tmp_path / "unused-model"),
            "--calib", str(calibration),
            "--out", str(tmp_path / "witness.json"),
        ])
    assert exc.value.code == 2
    assert "calibration" in capsys.readouterr().err
