"""The header projection: arithmetic that must reconcile with what was measured.

The projection exists to make the prereg's pre-measurement claim checkable on a
box whose card is busy. That is only worth something if the derived terms are
right, so these tests pin the two that are not estimates: the int8 grid and the
resident non-expert parameters have to reproduce the loader's own census exactly
on a real (if tiny) checkpoint. The rest -- the floor, the refusals, the CLI --
is pinned around that.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))

import moe_activation_probe as probe  # noqa: E402
import moe_int8_gates as gates  # noqa: E402
from chowder import moe_placement_gates as mpg  # noqa: E402
from chowder import moe_vram_projection as mvp  # noqa: E402
from chowder.backends import moe_int8_experts as mie  # noqa: E402


def fake_checkpoint(monkeypatch, tmp_path, header):
    """A directory with one empty shard whose header is ``header``."""
    base = tmp_path / "checkpoint"
    base.mkdir(exist_ok=True)
    (base / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(mvp, "header", lambda path: dict(header))
    return base


def fused_header(prefix="model.layers.0", experts=4, inter=6, hidden=3):
    return {
        f"{prefix}.mlp.experts.gate_up_proj": {"dtype": "BF16", "shape": [experts, 2 * inter, hidden]},
        f"{prefix}.mlp.experts.down_proj": {"dtype": "BF16", "shape": [experts, hidden, inter]},
        f"{prefix}.mlp.gate.weight": {"dtype": "BF16", "shape": [experts, hidden]},
    }


def per_expert_header(prefix="model.layers.0", experts=4, inter=6, hidden=3):
    header = {
        f"{prefix}.mlp.experts.{expert}.{leaf}.weight": {"dtype": "BF16", "shape": shape}
        for expert in range(experts)
        for leaf, shape in (
            ("gate_proj", [inter, hidden]),
            ("up_proj", [inter, hidden]),
            ("down_proj", [hidden, inter]),
        )
    }
    header[f"{prefix}.mlp.gate.weight"] = {"dtype": "BF16", "shape": [experts, hidden]}
    return header


# --- the two derived terms ----------------------------------------------------


def test_the_header_projection_reproduces_the_loaders_own_census():
    """Not a tolerance: the loader allocates 1 byte per expert parameter and one
    fp32 scale per row, so anything but equality means the projection is sizing a
    different placement than the one that was measured."""
    from chowder.backends import moe_int8_experts as loader_module

    base = Path(gates._self_check_workspace())
    model, census = loader_module.load_model_int8_experts(base, device="cpu")
    dtype, _source = mie.resolve_non_expert_dtype(base)
    inventory = mvp.inventory(base, dtype=str(dtype).replace("torch.", ""))
    projection = mvp.project(inventory, seq_len=8, vocab_size=64)

    assert projection["host_expert_bytes"] == census["expert_bytes"]
    assert projection["host_non_expert_bytes"] == census["non_expert_bytes"]
    assert inventory["expert_layers"] == census["expert_modules"]
    assert projection["host_expert_int8_bytes"] + projection["host_expert_scale_bytes"] == census["expert_bytes"]
    del model


def test_fused_and_per_expert_layouts_size_the_same_grid(monkeypatch, tmp_path):
    """A REAP export's fused key and transformers' own per-expert keys describe one
    layer of the same experts, so they must cost the same int8 bytes and scales."""
    fused = mvp.inventory(
        fake_checkpoint(monkeypatch, tmp_path, fused_header()), dtype="bfloat16"
    )
    legacy = mvp.inventory(
        fake_checkpoint(monkeypatch, tmp_path, per_expert_header()), dtype="bfloat16"
    )
    assert fused["expert_int8_bytes"] == legacy["expert_int8_bytes"] == 4 * (2 * 6 * 3 + 3 * 6)
    assert fused["expert_bytes"] == legacy["expert_bytes"]
    # One fp32 scale per weight row: experts x out_features, for gate_up and down.
    assert fused["expert_scale_bytes"] == legacy["expert_scale_bytes"] == (4 * 2 * 6 + 4 * 3) * 4
    assert fused["expert_layers"] == legacy["expert_layers"] == 1


def test_a_non_expert_tensor_is_costed_at_the_resolved_dtype(monkeypatch, tmp_path):
    """The first census's defect in miniature: the same shards cost half the bytes
    once the dtype rule resolves to bf16 instead of upcasting to fp32."""
    base = fake_checkpoint(monkeypatch, tmp_path, fused_header())
    bf16 = mvp.inventory(base, dtype="bfloat16")
    fp32 = mvp.inventory(base, dtype="float32")
    assert fp32["non_expert_bytes"] == 2 * bf16["non_expert_bytes"]
    assert bf16["expert_bytes"] == fp32["expert_bytes"]


def test_the_block_scratch_is_capped_by_the_matrix_it_fills():
    """A small matrix cannot be charged for a row block it has no rows to fill."""
    capped = mvp.dequant_block_bytes(4, itemsize=2, rows_available=6)
    assert capped == 6 * 4 * (1 + 4 + 2) + 6 * 4
    unbounded = mvp.dequant_block_bytes(4, itemsize=2)
    assert unbounded > capped


def test_the_step_figure_is_a_floor_not_a_peak(monkeypatch, tmp_path):
    base = fake_checkpoint(monkeypatch, tmp_path, fused_header())
    projection = mvp.project(mvp.inventory(base, dtype="bfloat16"), seq_len=16, vocab_size=32)
    assert projection["activations_bytes"] is None
    assert projection["step_floor_bytes"] == (
        projection["host_non_expert_bytes"]
        + projection["layer_dequant_bytes"]
        + projection["dequant_block_bytes"]
    )
    assert projection["step_floor_with_gate_loss_bytes"] > projection["step_floor_bytes"]


def test_a_measured_activation_term_turns_the_floor_into_a_total(monkeypatch, tmp_path):
    """With the term supplied the projection can print a total; without it the row
    must say ``unsized`` rather than quietly standing in a zero."""
    base = fake_checkpoint(monkeypatch, tmp_path, fused_header())
    inventory = mvp.inventory(base, dtype="bfloat16")
    without = mvp.project(inventory, seq_len=16, vocab_size=32)
    assert without["step_total_bytes"] is None

    activation = int(0.5 * mvp.GIB)
    with_term = mvp.project(inventory, seq_len=16, vocab_size=32, activation_bytes=activation)
    assert with_term["step_total_bytes"] == (
        with_term["step_floor_bytes"] + activation + with_term["gate_loss_bytes"]
    )
    text = mvp.render(with_term, {"source": "nothing"})
    assert "step total" in text
    assert "unsized" not in text


def test_the_rendered_projection_names_the_rows_it_will_not_size(monkeypatch, tmp_path):
    base = fake_checkpoint(monkeypatch, tmp_path, fused_header())
    projection = mvp.project(mvp.inventory(base, dtype="bfloat16"), seq_len=16, vocab_size=32)
    text = mvp.render(projection, {"source": "nothing", "int8_plus_scales": None})
    assert "unsized on purpose" in text
    assert "not measured" in text


# --- the activation probe's extrapolation ------------------------------------


def test_the_probe_fits_a_line_through_two_sequence_lengths():
    """The seq-2048 figure is an extrapolation from two measured points, so the fit
    itself has to be right: bytes = fixed + slope * seq, slope per token."""
    points = [
        {"layer_kind": "linear_attention", "seq_len": 128, "saved_bytes": 152_594_432, "saved_gib": 0.1421},
        {"layer_kind": "linear_attention", "seq_len": 512, "saved_bytes": 397_613_568, "saved_gib": 0.3703},
    ]
    fitted = probe.fit(points)
    assert fitted["measured_at"] == [128, 512]
    assert fitted["bytes_per_token"] == pytest.approx(638_071, rel=1e-4)
    predicted = {point["seq_len"]: point["predicted_gib"] for point in fitted["predicted_at"]}
    assert predicted[512] == pytest.approx(0.3703, abs=1e-3)
    # The published seq-2048 figure for one linear-attention layer.
    at_2048 = (fitted["fixed_bytes"] + fitted["bytes_per_token"] * 2048) / mvp.GIB
    assert at_2048 == pytest.approx(1.283, abs=1e-3)


def test_the_probe_refuses_a_directory_that_is_not_there(tmp_path):
    assert probe.main(["--base-model", str(tmp_path / "missing")]) == 2


# --- when the two cannot both be right ---------------------------------------


def test_a_census_that_does_not_match_is_a_disagreement():
    projection = {
        "host_expert_bytes": int(18.82 * mvp.GIB),
        "host_non_expert_bytes": int(4.55 * mvp.GIB),
        "host_resident_bytes": int(23.37 * mvp.GIB),
        "step_floor_bytes": int(5.54 * mvp.GIB),
    }
    agreeing = {"int8_plus_scales": 18.82, "non_expert": 4.55, "host_peak": 25.56, "step_vram": 11.0}
    assert mvp.disagreements(projection, agreeing) == []
    assert mvp.disagreements(projection, {"int8_plus_scales": 20.5})


def test_a_measured_peak_below_the_projected_floor_is_a_disagreement():
    projection = {
        "host_expert_bytes": 0,
        "host_non_expert_bytes": 0,
        "host_resident_bytes": 0,
        "step_floor_bytes": int(5.54 * mvp.GIB),
    }
    assert mvp.disagreements(projection, {"step_vram": 5.6}) == []
    problems = mvp.disagreements(projection, {"step_vram": 4.0})
    assert problems and "below the projected floor" in problems[0]


# --- the runner mode ---------------------------------------------------------


def run_project_only(*extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "chowder_batch" / "moe_int8_gates.py"), "--project-only", *extra],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "CHOWDER_NO_GPU": "1", "PYTHONPATH": str(ROOT / "src")},
        timeout=600,
    )


def test_project_only_prints_the_projection_and_writes_no_table():
    completed = run_project_only("--self-check")
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "projection from shard headers only" in completed.stdout
    assert "step VRAM floor" in completed.stdout
    match = re.search(r"artifact\s+(\S+)", completed.stdout)
    assert match, completed.stdout
    assert not (Path(match.group(1)) / "PREREG-self-check.md").exists()


def test_project_only_refuses_a_recorded_census_it_cannot_reproduce(tmp_path):
    """A toy checkpoint cannot have held 20 GB of int8, so a table that says it did
    is a disagreement, and the run says so instead of printing both numbers as if
    they described one placement."""
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(mpg.render_block(gates_record()), encoding="utf-8")
    completed = run_project_only("--self-check", "--prereg", str(prereg))
    assert completed.returncode == gates.PROJECTION_DISAGREES_EXIT, completed.stdout[-2000:]
    assert "DISAGREES" in completed.stderr


def gates_record():
    """A gate record with numbers from the placement, not from a toy."""
    return {
        "schema": gates.SCHEMA,
        "base_model": "G:/llm-cold/bf16/Qwen3.6-REAP160",
        "device": "cpu",
        "dtype": "auto (the checkpoint's)",
        "artifact": {"config_sha256": "a" * 64, "shards": 26, "shard_bytes": 1},
        "measured_at": "2026-10-20T15:31:24Z",
        "code_commit": "1a4b535",
        "steps": 0,
        "seq_len": 2048,
        "arms": {
            "int8": {
                "arm": "int8",
                "load_peak_rss_gb": 25.56,
                "total_peak_rss_gb": 25.56,
                "census": {"patched_expert_modules": 40, "remaining_expert_params": 0},
                "census_meta": {"expert_bytes_gb": 18.82, "patched_expert_modules": 40},
                "s_per_step": None,
                "step_seconds": [],
                "peak_vram_gb": None,
                "reserved_gb": None,
                "gates": {
                    "load_census": "measured",
                    "memory_vram": "not measured (CPU arm)",
                    "memory_vram_pass": None,
                    "host_rss": "measured",
                    "throughput": "not measured (--steps 0: census only)",
                },
            }
        },
    }
