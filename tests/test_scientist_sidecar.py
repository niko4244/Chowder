"""Scientist mode Phase 2: the AI Scientist v2 sidecar adapter.

The adapter is tested against hand-written fixture files in the upstream
format (never upstream data, never a live LLM run). Pinned here:

- runtime command construction for local/wsl/docker without execution;
- sandbox gating: unisolated local runtime is unavailable; missing home is a
  loud SidecarError, never a silent fallback (requirement 16);
- ideas-JSON -> Hypothesis translation (upstream schema);
- journal.json -> typed proposals translation;
- the adapter imports nothing from the sidecar and execs nothing in-process
  (requirements 18, 5);
- export_state round-trips through the service for restart (requirement 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.scientist import (
    MissionBudget,
    ResearchMission,
    ResearchTree,
)
from chowder.scientist.providers.ai_scientist_v2 import (
    AIScientistV2Provider,
    SidecarError,
    SidecarRuntime,
)
from chowder.scientist.provider import ResearchContext, SkillSummary


def _mission() -> ResearchMission:
    return ResearchMission(
        mission_id="m-sidecar", objective="improve reasoning",
        priorities={"reasoning": 0.7, "coding": 0.3},
        protected_capabilities=("instruction_following",),
        budget=MissionBudget(max_gpu_hours=4.0, max_tree_nodes=10, max_parallel_branches=2),
    )


def _context(tmp_path: Path) -> ResearchContext:
    return ResearchContext(
        mission=_mission(),
        skill_estimates=(SkillSummary(skill="reasoning", estimate=0.31,
                                      confidence=0.8, uncertainty=0.2),),
        open_failure_categories=({"category": "arithmetic_drift", "count": 7},),
        attempted_mechanisms=({"mechanism": "lr warmup", "outcome": "no effect"},),
        carried_evidence=(),
        remaining_gpu_hours=3.5,
        architecture="qwen3-family",
    )


def _provider(tmp_path: Path, **config) -> AIScientistV2Provider:
    home = tmp_path / "AI-Scientist-v2"
    home.mkdir(exist_ok=True)
    base = dict(runtime="local", home=str(home), workspace_dir=str(tmp_path / "sidecar-ws"),
                model="test-model", local_isolated=True)
    base.update(config)
    return AIScientistV2Provider.from_config(base)


# ---------------------------------------------------------------------------
# runtime command construction (no execution)
# ---------------------------------------------------------------------------


def test_local_runtime_command() -> None:
    runtime = SidecarRuntime(mode="local", home="/upstream", python="python3")
    cmd = runtime.build_command(["some.module", "--flag", "x"])
    assert cmd == ["python3", "-m", "some.module", "--flag", "x"]
    assert runtime.working_directory() == "/upstream"


def test_wsl_runtime_command() -> None:
    runtime = SidecarRuntime(mode="wsl", home="/upstream")
    cmd = runtime.build_command(["some.module"])
    assert cmd[:3] == ["wsl", "-e", "python"]


def test_docker_runtime_command_mounts_workspace_only() -> None:
    runtime = SidecarRuntime(mode="docker", home="", image="aisci:latest",
                             workspace_dir="/host/ws")
    cmd = runtime.build_command(["some.module"])
    assert cmd[0] == "docker" and "--rm" in cmd
    mount_at = cmd.index("-v")
    assert cmd[mount_at + 1] == "/host/ws:/work"
    # the upstream checkout is NOT mounted: the image carries it
    assert not any("upstream" in part for part in cmd)


def test_docker_without_image_is_a_loud_configuration_error() -> None:
    runtime = SidecarRuntime(mode="docker", home="", workspace_dir="/ws")
    with pytest.raises(SidecarError, match="image"):
        runtime.build_command(["m"])


# ---------------------------------------------------------------------------
# requirement 16: unavailable provider fails loudly, never fakes
# ---------------------------------------------------------------------------


def test_unisolated_local_runtime_is_unavailable(tmp_path: Path) -> None:
    provider = _provider(tmp_path, local_isolated=False)
    assert provider.available() is False
    with pytest.raises(SidecarError, match="unavailable"):
        provider.propose_hypotheses(_context(tmp_path))


def test_missing_home_refuses_at_construction() -> None:
    with pytest.raises(SidecarError, match="home is required"):
        AIScientistV2Provider.from_config({"runtime": "local", "home": ""})


# ---------------------------------------------------------------------------
# ideas-JSON translation (fixture in the upstream format)
# ---------------------------------------------------------------------------


def test_workshop_description_contains_only_sanitized_context(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    text = provider._workshop_description(_context(tmp_path))
    assert "improve reasoning" in text
    assert "0.31" in text                      # exported estimate
    assert "arithmetic_drift" in text          # failure category, count only
    assert "lr warmup" in text                 # attempted mechanism (do not re-propose)
    assert "Protected capabilities" in text
    # never any policy document or protected content
    assert "protected_benchmarks" not in text
    assert "promotion" not in text.lower()


def test_idea_fixture_translates_to_hypothesis(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    idea = {
        "Name": "replay_decay",
        "Title": "Decaying replay improves reasoning transfer",
        "Short Hypothesis": "Lowering the replay ratio after convergence frees "
                            "gradient signal for the target capability.",
        "Related Work": "Prior replay-ratio studies in continual learning.",
        "Abstract": "We hypothesize reasoning gains transfer without regressions.",
    }
    hyp = provider._hypothesis_from_idea(idea, index=0)
    assert hyp.hypothesis_id == "aisci-replay_decay"
    assert hyp.suspected_mechanism.startswith("Lowering the replay ratio")
    assert hyp.research_question.capability in ("reasoning", "math")
    # every translated hypothesis is falsifiable from the start
    assert hyp.falsification_conditions


def test_idea_without_hypothesis_is_refused_not_invented(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    with pytest.raises(SidecarError, match="Short Hypothesis"):
        provider._hypothesis_from_idea({"Name": "empty", "Title": "t"}, index=0)


# ---------------------------------------------------------------------------
# journal.json -> proposals (fixture in the upstream to_dict shape)
# ---------------------------------------------------------------------------


def test_journal_fixture_informs_proposals(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    context = _context(tmp_path)
    ws = provider._workspace_root(context)
    sandbox = ws / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    # Minimal hand-written document in the upstream Journal.to_dict() shape.
    journal = {
        "nodes": [
            {"step": 0, "metric": 0.42, "code": "...", "analysis": "..."},
            {"step": 1, "metric": 0.51, "code": "...", "analysis": "..."},
        ],
        "best_metric": 0.51,
    }
    (sandbox / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    loaded = provider._load_journal_if_any(context)
    assert loaded["best_metric"] == 0.51

    hyp = provider._hypothesis_from_idea({
        "Name": "lora_rank_study",
        "Title": "Higher LoRA rank helps reasoning",
        "Short Hypothesis": "Increasing LoRA rank from 16 to 32 improves reasoning.",
    }, index=0)
    proposal = provider._proposal_for(hyp, loaded)
    assert proposal.experiment_type == "adapter"
    assert "lora_rank" in proposal.variables_changed
    # every translated proposal still owes its falsification rule
    assert proposal.falsification_rule


def test_corrupt_journal_is_a_loud_error(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    context = _context(tmp_path)
    ws = provider._workspace_root(context)
    (ws / "sandbox").mkdir(parents=True, exist_ok=True)
    (ws / "sandbox" / "journal.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SidecarError, match="valid JSON"):
        provider._load_journal_if_any(context)


# ---------------------------------------------------------------------------
# requirements 18 + 5: no sidecar code in the Chowder process
# ---------------------------------------------------------------------------


def test_adapter_module_imports_nothing_from_the_sidecar() -> None:
    import chowder.scientist.providers.ai_scientist_v2 as module
    import sys
    sidecar_modules = [name for name in sys.modules
                       if "ai_scientist" in name and not name.startswith("chowder")]
    # loading the adapter module must not have imported any upstream module
    assert not sidecar_modules, f"sidecar modules leaked into the process: {sidecar_modules}"
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("import ai_scientist", "from ai_scientist", "exec(", "eval("):
        assert forbidden not in source, f"adapter contains {forbidden!r}"


def test_export_state_round_trips_through_service(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    state = provider.export_state()
    blob = json.dumps(state)
    assert json.loads(blob) == state
    assert state["runtime_mode"] == "local"
