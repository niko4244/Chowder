"""The int8 placement's gate table: writing it, reading it, and refusing stale numbers.

``chowder_batch/moe_int8_gates.py`` measures the placement and
``chowder.moe_placement_gates`` owns the table. These tests pin the contract
that makes the table worth reading:

* a measurement is spliced into the prereg without touching its prose, and a row
  the run could not answer keeps the recorded number instead of erasing it;
* a recorded gate that gets worse fails the run, a better one does not;
* a table measured on another artifact is never overwritten;
* a campaign cannot declare the placement while the table is unmeasured, failed,
  or stale.

The measurement itself is not re-run here: ``run`` is replaced with a record, so
these test the table and the rule, not the GPU.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))

import moe_int8_gates as gates  # noqa: E402
from chowder import moe_placement_gates as mpg  # noqa: E402
from chowder.backends.moe_transient_grad import INT8_PLACEMENT  # noqa: E402
from chowder.growth.campaign_prepare import (  # noqa: E402
    CampaignPrepareRefusal,
    placement_gate_refusal,
)

PLACEHOLDER = f"""# Pre-registration — the placement

Prose an operator wrote, which no run may rewrite.

{mpg.BLOCK_BEGIN}

| gate | value | verdict |
|---|---|---|
| `load_census` | not measured | not measured |
| `memory_vram` | not measured | not measured |
| `host_rss` | not measured | not measured |
| `throughput` | not measured | not measured |
| `gradient` | not measured by this runner | not measured |
| `smoke` | not measured by this runner | not measured |
| `router_balance` | not measured by this runner | not measured |
| `wddm_spill` | not observable from here | not measured |

- artifact: none yet
- measured at: never

{mpg.BLOCK_END}

Trailing prose that must survive too.
"""


def record(
    *,
    digest="a" * 64,
    peak_vram=11.2,
    s_per_step=1.5,
    expert_bytes=20.19,
    rss=21.0,
    steps=3,
    remaining_expert_params=0,
    patched=40,
    gradient="not measured by this runner (needs a LoRA run)",
):
    return {
        "schema": gates.SCHEMA,
        "base_model": "G:/llm-cold/bf16/Qwen3.6-REAP160",
        "device": "cuda:0",
        "dtype": "auto (the checkpoint's)",
        "placement": INT8_PLACEMENT,
        "artifact": {"config_sha256": digest, "shards": 19, "shard_bytes": 1},
        "measured_at": "2026-10-20T12:00:00Z",
        "code_commit": "92ed738",
        "steps": steps,
        "seq_len": 2048,
        "arms": {
            "int8": {
                "arm": "int8",
                "load_seconds": 12.0,
                "load_peak_rss_gb": rss - 0.5,
                "total_peak_rss_gb": rss,
                "census": {
                    "patched_expert_modules": patched,
                    "remaining_expert_params": remaining_expert_params,
                },
                "census_meta": {
                    "expert_bytes_gb": expert_bytes,
                    "patched_expert_modules": patched,
                },
                "s_per_step": s_per_step if steps else None,
                "step_seconds": [s_per_step] * steps if steps else [],
                "peak_vram_gb": peak_vram,
                "reserved_gb": peak_vram + 0.5 if peak_vram is not None else None,
                "gates": {
                    "load_census": "measured",
                    "memory_vram": "measured" if peak_vram is not None else "not measured (CPU arm)",
                    "memory_vram_pass": None if peak_vram is None else peak_vram <= gates.VRAM_GATE_GB,
                    "host_rss": "measured",
                    "throughput": (
                        "not measured (--steps 0: census only)" if not steps else "measured (placement step, no LoRA/optimizer)"
                    ),
                    "gradient": gradient,
                    "smoke": "not measured by this runner (needs a dataset)",
                    "router_balance": "not measured by this runner (needs a smoke run)",
                    "wddm_spill": "not observable from here; compare peak_vram_gb to the card",
                },
            },
            "bulk": {"s_per_step": 4.0},
        },
    }


def clause(text, **kwargs):
    return mpg.splice_block(text, mpg.render_block(record(**kwargs)))


def prewritten(tmp_path, *, text):
    path = tmp_path / "PREREG.md"
    path.write_text(text, encoding="utf-8")
    return path


# --- the block itself ---------------------------------------------------------


def test_rendered_block_reads_back_with_its_numbers():
    parsed = mpg.parse_block(mpg.render_block(record()))
    assert parsed["verdicts"]["memory_vram"] == "pass"
    assert parsed["verdicts"]["load_census"] == "measured"
    assert parsed["numbers"]["memory_vram"] == pytest.approx(11.2)
    assert parsed["numbers"]["throughput"] == pytest.approx(1.5)
    assert parsed["numbers"]["load_census"] == pytest.approx(20.19)
    assert parsed["config_sha256"] == "a" * 64


def test_splicing_a_measurement_leaves_the_prose_alone():
    measured = clause(PLACEHOLDER)
    assert "Prose an operator wrote, which no run may rewrite." in measured
    assert measured.rstrip().endswith("Trailing prose that must survive too.")
    assert measured.count(mpg.BLOCK_BEGIN) == 1
    assert "11.2 GB peak" in measured
    assert mpg.parse_block(measured)["verdicts"]["memory_vram"] == "pass"


def test_a_row_the_run_could_not_answer_keeps_its_recorded_number():
    """`--steps 0` answers no throughput; the table must not lose the measured one."""
    first = clause(PLACEHOLDER)
    census_only = mpg.render_block(record(steps=0), previous=mpg.parse_block(first))
    parsed = mpg.parse_block(mpg.splice_block(first, census_only))
    assert parsed["verdicts"]["throughput"] == "measured (placement step)"
    assert parsed["numbers"]["throughput"] == pytest.approx(1.5)
    assert "carried forward" in parsed["block"]
    assert "throughput" in parsed["carried"]


# --- regressions --------------------------------------------------------------


def test_pass_to_fail_is_a_regression():
    previous = mpg.parse_block(mpg.render_block(record()))
    problems = mpg.regressions(previous, record(peak_vram=gates.VRAM_GATE_GB + 1.0))
    assert any("memory_vram: was pass" in problem for problem in problems)


def test_a_slower_step_is_a_regression_and_a_faster_one_is_not():
    previous = mpg.parse_block(mpg.render_block(record()))
    assert any(
        "throughput:" in problem for problem in mpg.regressions(previous, record(s_per_step=2.5))
    )
    assert not mpg.regressions(previous, record(s_per_step=1.2))


def test_a_step_inside_the_timer_spread_is_not_a_regression():
    """A gate that fires on timer jitter is a gate an operator learns to ignore."""
    previous = mpg.parse_block(mpg.render_block(record()))
    noisy = record(s_per_step=1.7)
    noisy["arms"]["int8"]["step_seconds"] = [1.9, 1.6]  # 0.3 s of spread
    assert not mpg.regressions(previous, noisy)
    steady = record(s_per_step=1.7)
    steady["arms"]["int8"]["step_seconds"] = [1.7, 1.7]
    assert any("throughput:" in problem for problem in mpg.regressions(previous, steady))
    assert "spread 0.3 s" in mpg.render_block(noisy)


def test_a_larger_expert_grid_is_a_regression():
    previous = mpg.parse_block(mpg.render_block(record()))
    problems = mpg.regressions(previous, record(expert_bytes=24.0))
    assert any("load_census:" in problem for problem in problems)


def test_a_different_artifact_is_not_comparable():
    previous = mpg.parse_block(mpg.render_block(record()))
    problems = mpg.regressions(previous, record(digest="b" * 64))
    assert any(problem.startswith("artifact:") for problem in problems)


def test_a_census_only_run_is_not_a_regression():
    previous = mpg.parse_block(mpg.render_block(record()))
    assert mpg.regressions(previous, record(steps=0, peak_vram=None)) == []


# --- the CLI writes the table -------------------------------------------------


@pytest.fixture()
def no_measurement(monkeypatch):
    """The measurement is this suite's `record`; nothing here touches a GPU."""
    def fake_run(base, **kwargs):
        return record(steps=kwargs["steps"], peak_vram=None if kwargs["steps"] == 0 else 11.2)

    monkeypatch.setattr(gates, "run", fake_run)


def test_cli_writes_the_measurement_into_the_prereg(monkeypatch, tmp_path, no_measurement):
    monkeypatch.setenv("CHOWDER_NO_GPU", "1")
    prereg = prewritten(tmp_path, text=PLACEHOLDER)
    out = tmp_path / "gates.json"
    code = gates.main(
        ["--base-model", str(tmp_path), "--prereg", str(prereg), "--out", str(out), "--steps", "3"]
    )
    assert code == 0
    assert json.loads(out.read_text())["arms"]["int8"]["peak_vram_gb"] == 11.2
    document = prereg.read_text(encoding="utf-8")
    assert "Prose an operator wrote, which no run may rewrite." in document
    assert "11.2 GB peak (gate <= 14.5 GB)" in document
    assert mpg.parse_block(document)["verdicts"]["memory_vram"] == "pass"


def test_self_check_never_writes_the_placement_prereg(monkeypatch, tmp_path, no_measurement):
    """A synthetic model's numbers must not become the placement's table."""
    monkeypatch.delenv("CHOWDER_MOE_GATE_TABLE", raising=False)
    monkeypatch.setattr(gates, "_self_check_workspace", lambda: tmp_path)
    repo_prereg = mpg.resolve_prereg_path()
    before = repo_prereg.read_text(encoding="utf-8") if repo_prereg.is_file() else None
    assert gates.main(["--self-check"]) == 0
    assert (tmp_path / "PREREG-self-check.md").is_file()
    after = repo_prereg.read_text(encoding="utf-8") if repo_prereg.is_file() else None
    assert after == before


def test_a_census_only_run_keeps_the_measured_throughput(
    monkeypatch, tmp_path, no_measurement, capsys
):
    """`--steps 0` answers no throughput: the table keeps the number it has."""
    monkeypatch.setenv("CHOWDER_NO_GPU", "1")
    prereg = prewritten(tmp_path, text=clause(PLACEHOLDER))
    code = gates.main(["--base-model", str(tmp_path), "--prereg", str(prereg), "--steps", "0"])
    assert code == 0
    parsed = mpg.parse_block(prereg.read_text(encoding="utf-8"))
    assert parsed["numbers"]["throughput"] == pytest.approx(1.5)
    assert parsed["verdicts"]["throughput"] == "measured (placement step)"
    # Both rows are unanswered here: no throughput, and no device for VRAM.
    assert (
        "carried forward from the previous measurement: memory_vram, throughput"
        in capsys.readouterr().out
    )


def test_cli_fails_when_a_recorded_gate_regresses(monkeypatch, tmp_path):
    monkeypatch.setenv("CHOWDER_NO_GPU", "1")
    prereg = prewritten(tmp_path, text=clause(PLACEHOLDER))
    monkeypatch.setattr(gates, "run", lambda base, **kwargs: record(peak_vram=gates.VRAM_GATE_GB + 2.0))
    code = gates.main(["--base-model", str(tmp_path), "--prereg", str(prereg), "--steps", "3"])
    assert code == 1
    # The worse numbers are the ones on record: a failing run must not leave the
    # passing ones standing as if they were current.
    parsed = mpg.parse_block(prereg.read_text(encoding="utf-8"))
    assert parsed["verdicts"]["memory_vram"] == "FAIL"


def test_cli_refuses_to_overwrite_a_table_from_another_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("CHOWDER_NO_GPU", "1")
    prereg = prewritten(tmp_path, text=clause(PLACEHOLDER))
    before = prereg.read_text(encoding="utf-8")
    monkeypatch.setattr(gates, "run", lambda base, **kwargs: record(digest="c" * 64))
    code = gates.main(["--base-model", str(tmp_path), "--prereg", str(prereg), "--steps", "3"])
    assert code == gates.ARTIFACT_MISMATCH_EXIT
    assert prereg.read_text(encoding="utf-8") == before


def test_a_new_baseline_replaces_the_table_it_could_not_compare(monkeypatch, tmp_path):
    monkeypatch.setenv("CHOWDER_NO_GPU", "1")
    prereg = prewritten(tmp_path, text=clause(PLACEHOLDER))
    monkeypatch.setattr(gates, "run", lambda base, **kwargs: record(digest="c" * 64, peak_vram=12.0))
    code = gates.main(
        ["--base-model", str(tmp_path), "--prereg", str(prereg), "--steps", "3", "--allow-new-baseline"]
    )
    assert code == 0
    assert mpg.parse_block(prereg.read_text(encoding="utf-8"))["config_sha256"] == "c" * 64


def test_check_compares_without_writing(monkeypatch, tmp_path, no_measurement):
    monkeypatch.setenv("CHOWDER_NO_GPU", "1")
    prereg = prewritten(tmp_path, text=clause(PLACEHOLDER))
    before = prereg.read_text(encoding="utf-8")
    assert gates.main(["--base-model", str(tmp_path), "--prereg", str(prereg), "--check"]) == 0
    assert prereg.read_text(encoding="utf-8") == before
    monkeypatch.setattr(gates, "run", lambda base, **kwargs: record(s_per_step=9.0))
    assert gates.main(["--base-model", str(tmp_path), "--prereg", str(prereg), "--check"]) == 1
    assert prereg.read_text(encoding="utf-8") == before


# --- the declaration rule -----------------------------------------------------


def fully_measured(**kwargs):
    return mpg.splice_block(
        PLACEHOLDER,
        mpg.render_block(
            record(
                gradient="measured (finite, non-zero on every LoRA B leaf)",
                **kwargs,
            )
        ),
    )


def test_a_declaration_is_refused_while_a_gate_is_unmeasured():
    with pytest.raises(mpg.PlacementGateError, match="not satisfied"):
        mpg.assert_declaration_allowed(PLACEHOLDER, ["docs/quals/campaign.json"])


def test_a_declaration_is_refused_when_the_table_has_no_block():
    with pytest.raises(mpg.PlacementGateError, match="no gate table"):
        mpg.assert_declaration_allowed("prose only", ["docs/quals/campaign.json"])


def test_a_fully_measured_table_on_this_artifact_allows_it(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    digest = mpg.artifact_identity(tmp_path)["config_sha256"]
    text = fully_measured(digest=digest)
    text = text.replace(
        "| `smoke` | not measured by this runner (needs a dataset) | not measured |",
        "| `smoke` | 20 steps lowered the loss; adapter loaded | measured |",
    ).replace(
        "| `router_balance` | not measured by this runner (needs a smoke run) | not measured |",
        "| `router_balance` | entropy within 8% of step 0 | measured |",
    ).replace(
        "| `wddm_spill` | not observable from here; compare peak_vram_gb to the card | not measured |",
        "| `wddm_spill` | 0 B shared | measured |",
    )
    assert mpg.assert_declaration_allowed(text, ["campaign.json"], base_model=tmp_path)


def test_a_table_measured_on_another_artifact_does_not_allow_it(tmp_path):
    text = fully_measured(digest="d" * 64)
    text = text.replace("| not measured |", "| measured |")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(mpg.PlacementGateError, match="different artifact"):
        mpg.assert_declaration_allowed(text, ["campaign.json"], base_model=tmp_path)


def test_a_failed_gate_does_not_allow_it():
    text = mpg.splice_block(PLACEHOLDER, mpg.render_block(record(peak_vram=gates.VRAM_GATE_GB + 3)))
    text = text.replace("| not measured |", "| measured |")
    with pytest.raises(mpg.PlacementGateError, match="failed=\\['memory_vram'\\]"):
        mpg.assert_declaration_allowed(text, ["campaign.json"])


def test_campaign_prepare_refuses_the_placement_while_the_table_is_unmeasured(monkeypatch, tmp_path):
    prereg = prewritten(tmp_path, text=PLACEHOLDER)
    monkeypatch.setenv("CHOWDER_MOE_GATE_TABLE", str(prereg))
    with pytest.raises(CampaignPrepareRefusal, match="PREPARE_SCHEMA.*not satisfied"):
        placement_gate_refusal(
            expert_placement=INT8_PLACEMENT, where="campaign.json", base_model=tmp_path
        )


def test_campaign_prepare_leaves_other_placements_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("CHOWDER_MOE_GATE_TABLE", str(tmp_path / "absent.md"))
    placement_gate_refusal(expert_placement="cpu-transient-grad", where="campaign.json", base_model=tmp_path)
    placement_gate_refusal(expert_placement="resident", where="campaign.json", base_model=tmp_path)


def test_campaign_prepare_refuses_a_missing_table(monkeypatch, tmp_path):
    monkeypatch.setenv("CHOWDER_MOE_GATE_TABLE", str(tmp_path / "absent.md"))
    with pytest.raises(CampaignPrepareRefusal, match="pre-registration"):
        placement_gate_refusal(
            expert_placement=INT8_PLACEMENT, where="campaign.json", base_model=tmp_path
        )
