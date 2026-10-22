"""The header projection: arithmetic that must reconcile with what was measured.

The projection exists to make the prereg's pre-measurement claim checkable on a
box whose card is busy. That is only worth something if the derived terms are
right, so these tests pin the two that are not estimates: the int8 grid and the
resident non-expert parameters have to reproduce the loader's own census exactly
on a real (if tiny) checkpoint. The rest -- the floor, the refusals, the CLI --
is pinned around that.
"""

import json
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

# ``chowder.backends.moe_int8_experts`` imports safetensors, and ``torch`` is in
# the ``train`` extra: importing either at module scope would stop this file
# collecting in the light CI job, which is where the build check below has to
# run. The tests that need them import them, and say so.


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
    pytest.importorskip("torch")
    from chowder.backends import moe_int8_experts as loader_module

    base = Path(gates._self_check_workspace())
    model, census = loader_module.load_model_int8_experts(base, device="cpu")
    dtype, _source = loader_module.resolve_non_expert_dtype(base)
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


#: The artifact's own terms, as the census and the header projection measured
#: them on 2026-10-20 (GiB in the comments; the integers are what the two runs
#: wrote). The CI gate below is built from these plus the probe's fitted lines,
#: because CI has neither the 43 GB checkpoint nor a card.
ARTIFACT_INVENTORY = {
    "dtype": "bfloat16",
    "non_expert_bytes": 4_880_983_296,  # 4.55
    "expert_int8_bytes": 20_132_659_200,  # 18.75
    "expert_scale_bytes": 78_643_200,  # 0.07, one fp32 per weight row
    "expert_bytes": 20_211_302_400,  # 18.82, the census' own number
    "layer_dequant_bytes": 1_006_632_960,  # 0.94
    "dequant_block_bytes": 58_785_792,  # 0.06
    "shards": 26,
    "tensors": 1026,
    "source_dtypes": ["BF16"],
    "expert_layers": 40,
    "expert_matrices_per_layer": [2],
    "layer_expert_numel_max": 503_316_480,
    "non_expert_numel": 2_440_491_648,
    "tower_bytes": 893_142_496,
}

#: The probe's fitted lines: (fixed bytes, bytes per token) per layer kind.
ARTIFACT_FITS = {
    "checkpointed": {"linear_attention": (8_192, 20_492.0), "full_attention": (8_192, 20_492.0)},
    "eager": {
        "linear_attention": (70_921_387, 638_070.7),
        "full_attention": (61_503_488, 146_578.0),
    },
}


def test_a_change_that_moves_the_projected_step_fails_here_and_not_on_the_card():
    """The build check for the gate: the artifact's terms, the probe's lines, the
    projection and the verdict, with no weights and no device.

    It pins the total (10.04 GiB) rather than only asserting "under 14.5", because
    the expression this replaced was 13.12 GiB and also under the gate: a drift
    that quietly spends a gigabyte would pass a threshold and still change the
    answer. An intended move updates this number and the prereg together. On a box
    that has the artifact the same question is `--project-only`, which exits 6
    when the projection does not fit.
    """
    counts = {"linear_attention": 30, "full_attention": 10}
    retained = probe_record(
        {kind: line(*ARTIFACT_FITS["checkpointed"][kind]) for kind in counts},
        checkpointed=True,
        counts=counts,
    )
    eager = probe_record(
        {kind: line(*ARTIFACT_FITS["eager"][kind]) for kind in counts},
        checkpointed=False,
        counts=counts,
    )
    term = mvp.activation_term(retained, eager, seq_len=2048)
    projection = mvp.project(
        ARTIFACT_INVENTORY,
        seq_len=2048,
        vocab_size=248_320,
        activation_bytes=term["total_bytes"],
        activation_mode="checkpointed",
    )
    verdict = mvp.gate_verdict(projection, gate_gb=mpg.VRAM_GATE_GB)

    assert projection["step_total_bytes"] == 10_783_082_309, (
        "the projected step moved: update this number and the prereg's figures together, "
        "or find out what changed"
    )
    assert verdict["step_total_gib"] == pytest.approx(10.04, abs=0.01)
    assert verdict["margin_gib"] == pytest.approx(4.46, abs=0.02)
    assert verdict["reachable"] is True


def test_the_scoring_term_is_one_chunk_not_two_copies_of_the_sequence():
    """The old expression materialised the whole sequence twice in fp32; a chunked
    cross-entropy holds one chunk. At the run's seq 2048 and vocab 248320 that is
    the difference between a third of the gate and a seventh of it."""
    seq, vocab = 2048, 248320
    elements = seq * vocab
    old = elements * 2 + 2 * elements * 4          # bf16 logits + float() + square()
    new = mvp.gate_loss_bytes(seq_len=seq, vocab_size=vocab, itemsize=2)
    assert new / mvp.GIB == pytest.approx(1.66, abs=0.05)
    assert new < 0.4 * old
    # The logits stay: a campaign's own step produces them too.
    assert new > elements * 2


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


def test_the_probe_reports_an_unfitted_kind_as_unknown_not_zero():
    """One measured point is not a line: the model total must say unknown rather
    than add up to a zero that reads as "this kind costs nothing"."""
    fits = {"linear_attention": {"fixed_bytes": 1, "bytes_per_token": 2}}
    assert probe.fit([{"seq_len": 512, "saved_bytes": 10}]) == {}
    totals = probe.model_totals(fits, {"linear_attention": 30, "full_attention": 10}, ("linear_attention", "full_attention"), [512])
    assert totals == {"512": None}

    both = {**fits, "full_attention": {"fixed_bytes": 1, "bytes_per_token": 1}}
    totals = probe.model_totals(
        both, {"linear_attention": 30, "full_attention": 10}, ("linear_attention", "full_attention"), [512]
    )
    # 30 x (1 + 2*512) + 10 x (1 + 512) bytes: a number, not a placeholder.
    assert totals["512"] == pytest.approx(35_880 / mvp.GIB, abs=5e-3)


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
    # The runner resolves the artifact's config through transformers and its
    # header inventory through safetensors; the light CI job has neither.
    pytest.importorskip("transformers")
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


# --- the checkpointed step: the step the gate is defined on ------------------


def line(fixed, slope):
    """One fitted retention line, in the shape the probe writes."""
    return {"fixed_bytes": fixed, "bytes_per_token": slope, "measured_at": [128, 512], "predicted_at": []}


def probe_record(fits, *, checkpointed, counts=None, base_model="G:/llm-cold/bf16/Qwen3.6-REAP160"):
    return {
        "schema": probe.SCHEMA,
        "base_model": base_model,
        "layers_per_kind": counts or {"linear_attention": 30, "full_attention": 10},
        "gradient_checkpointing": bool(checkpointed),
        "fits": fits,
    }


def test_the_checkpointed_step_retains_the_inputs_and_reruns_one_layer():
    """With recompute on, the device holds the between-layer inputs plus -- at the
    peak -- one layer's whole forward as backward re-runs it. Summing the eager
    line over thirty layers would describe a step no campaign performs."""
    retained = probe_record(
        {"linear_attention": line(1_000, 10), "full_attention": line(500, 1)}, checkpointed=True
    )
    eager = probe_record(
        {"linear_attention": line(1_000, 1_000), "full_attention": line(500, 1)}, checkpointed=False
    )
    term = mvp.activation_term(retained, eager, seq_len=100)
    assert term["retained_bytes"] == (1_000 + 10 * 100) * 30 + (500 + 100) * 10
    assert term["recompute_bytes"] == 1_000 + 1_000 * 100, "the largest kind, one layer"
    assert term["total_bytes"] == term["retained_bytes"] + term["recompute_bytes"]
    assert term["mode"] == "checkpointed"


def test_the_preregs_checkpointed_terms_follow_from_the_two_measured_lines():
    """The prereg's 13.12 GiB, recomposed from the probe's own lines rather than
    hand-summed: 40 layers x 0.039 GiB of retained layer inputs, plus the one
    linear-attention layer being re-run (its eager line, 1.28 GiB). The lines are
    the probe's as recorded on 2026-10-20; re-measuring them updates this test with
    the prereg, which is the point of computing it instead of typing it."""
    counts = {"linear_attention": 30, "full_attention": 10}
    retained = probe_record(
        {kind: line(8_192, 20_492) for kind in counts}, checkpointed=True, counts=counts
    )
    eager = probe_record(
        {
            "linear_attention": line(70_921_387, 638_070.7),
            "full_attention": line(61_503_488, 146_578.0),
        },
        checkpointed=False,
        counts=counts,
    )
    term = mvp.activation_term(retained, eager, seq_len=2048)
    assert term["retained_bytes"] / mvp.GIB == pytest.approx(1.56, abs=0.01), "40 layers of inputs"
    assert term["recompute_bytes"] / mvp.GIB == pytest.approx(1.28, abs=0.01), "the layer in re-run"
    assert term["total_bytes"] / mvp.GIB == pytest.approx(2.85, abs=0.01)

    # Retained is the checkpointed line, not the eager one: 41.86 GiB of eager
    # retention would make the step 3.6x the gate instead of inside it.
    eager_all = probe_record(eager["fits"], checkpointed=True, counts=counts)
    assert mvp.activation_term(eager_all, eager, seq_len=2048)["retained_bytes"] / mvp.GIB == pytest.approx(
        41.86, abs=0.1
    )


def test_swapped_probe_records_are_refused_rather_than_quietly_trusted():
    """Either swap produces a plausible number -- forty layers of eager retention,
    or a peak with no layer being re-run -- so the roles are checked."""
    fits = {"linear_attention": line(1, 1), "full_attention": line(1, 1)}
    with pytest.raises(ValueError, match="must be the checkpointed run"):
        mvp.activation_term(probe_record(fits, checkpointed=False), probe_record(fits, checkpointed=True), seq_len=512)
    with pytest.raises(ValueError, match="must be the eager run"):
        mvp.activation_term(probe_record(fits, checkpointed=True), probe_record(fits, checkpointed=True), seq_len=512)
    with pytest.raises(ValueError, match="different base models"):
        mvp.activation_term(
            probe_record(fits, checkpointed=True),
            probe_record(fits, checkpointed=False, base_model="G:/elsewhere"),
            seq_len=512,
        )


def test_a_record_that_does_not_say_which_step_it_measured_is_refused():
    """A record from before the mode was written down, or one whose own points
    disagree, cannot stand for either step -- and the filename is not evidence."""
    fits = {"linear_attention": line(1, 1), "full_attention": line(1, 1)}
    unmarked = {
        key: value
        for key, value in probe_record(fits, checkpointed=True).items()
        if key != "gradient_checkpointing"
    }
    with pytest.raises(ValueError, match="does not say which step"):
        mvp.activation_term(unmarked, probe_record(fits, checkpointed=False), seq_len=512)

    mixed = {**unmarked, "points": [{"gradient_checkpointing": True}, {"gradient_checkpointing": False}]}
    assert mvp._probe_mode(mixed) is None
    assert mvp._probe_mode({**unmarked, "points": [{"gradient_checkpointing": True}]}) is True


def test_a_kind_with_no_line_is_refused_rather_than_summed_away():
    """A total over fewer layers reads as "that kind costs nothing", which is the
    failure the probe itself refuses."""
    retained = probe_record({"linear_attention": line(1, 1)}, checkpointed=True)
    eager = probe_record({"linear_attention": line(1, 1)}, checkpointed=False)
    with pytest.raises(ValueError, match="full_attention"):
        mvp.activation_term(retained, eager, seq_len=512)


def test_the_gate_verdict_states_reachability_and_its_margin():
    reachable = mvp.gate_verdict({"step_total_bytes": int(13.12 * mvp.GIB)}, gate_gb=14.5)
    assert reachable["reachable"] is True
    assert reachable["step_total_gib"] == pytest.approx(13.12, abs=0.01)
    assert reachable["margin_gib"] == pytest.approx(1.38, abs=0.02)

    over = mvp.gate_verdict({"step_total_bytes": int(15.9 * mvp.GIB)}, gate_gb=14.5)
    assert over["reachable"] is False
    assert over["margin_gib"] < 0

    # A floor under the gate is not a fit: no activation term, no answer.
    assert mvp.gate_verdict({"step_total_bytes": None}, gate_gb=14.5)["reachable"] is None


def test_the_rendered_step_says_which_step_it_is(monkeypatch, tmp_path):
    base = fake_checkpoint(monkeypatch, tmp_path, fused_header())
    inventory = mvp.inventory(base, dtype="bfloat16")
    checkpointed = mvp.project(
        inventory, seq_len=16, vocab_size=32, activation_bytes=int(mvp.GIB), activation_mode="checkpointed"
    )
    text = mvp.render(checkpointed, {"source": "nothing"}, verdict=mvp.gate_verdict(checkpointed, gate_gb=14.5))
    assert "layer inputs + one layer re-run" in text
    assert "reachable" in text

    eager = mvp.project(inventory, seq_len=16, vocab_size=32, activation_bytes=int(mvp.GIB))
    eager_text = mvp.render(eager, {"source": "nothing"})
    assert "every layer, retained" in eager_text
    assert "NO campaign runs" in eager_text


def test_project_only_answers_the_gate_from_the_two_probe_records(tmp_path):
    counts = {"linear_attention": 30, "full_attention": 10}
    fits = {"linear_attention": line(1_000, 1), "full_attention": line(1_000, 1)}
    retained = tmp_path / "retained.json"
    eager = tmp_path / "eager.json"
    retained.write_text(json.dumps(probe_record(fits, checkpointed=True, counts=counts)), encoding="utf-8")
    eager.write_text(json.dumps(probe_record(fits, checkpointed=False, counts=counts)), encoding="utf-8")
    out = tmp_path / "projection.json"
    completed = run_project_only(
        "--self-check",
        "--activation-probe",
        str(retained),
        "--activation-probe-eager",
        str(eager),
        "--out",
        str(out),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"]["reachable"] is True
    assert payload["projection"]["activation_mode"] == "checkpointed"
    assert payload["activation_breakdown"]["total_bytes"] == payload["projection"]["activations_bytes"]
    assert "layer inputs + one layer re-run" in completed.stdout

    # The same wiring, the other answer: a step that does not fit exits 6.
    blown = tmp_path / "retained-over.json"
    blown.write_text(
        json.dumps(
            probe_record(
                {"linear_attention": line(int(40 * mvp.GIB), 1), "full_attention": line(int(40 * mvp.GIB), 1)},
                checkpointed=True,
                counts=counts,
            )
        ),
        encoding="utf-8",
    )
    over = run_project_only("--self-check", "--activation-probe", str(blown), "--activation-probe-eager", str(eager))
    assert over.returncode == gates.PROJECTED_OVER_GATE_EXIT, over.stdout[-2000:]
    assert "NOT reachable" in over.stdout
    assert "PROJECTED OVER GATE" in over.stderr


def test_project_only_refuses_a_half_probe_pair(tmp_path):
    fits = {"linear_attention": line(1, 1), "full_attention": line(1, 1)}
    one = tmp_path / "one.json"
    one.write_text(json.dumps(probe_record(fits, checkpointed=True)), encoding="utf-8")
    completed = run_project_only("--self-check", "--activation-probe", str(one))
    assert completed.returncode == 2
    assert "are a pair" in completed.stderr
