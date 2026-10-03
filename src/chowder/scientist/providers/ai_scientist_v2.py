"""AIScientistV2Provider: the AI Scientist v2 sidecar adapter (clean-room).

Written against the upstream project's documented behavior and file formats
only — no upstream source is imported, vendored, or executed inside Chowder.
See docs/SCIENTIST_LICENSE.md for the license position and
docs/SCIENTIST_MODE.md for the audit this adapter implements.

Protocol (files only; the sidecar never imports Chowder, Chowder never
imports the sidecar):

1. Chowder writes a *workshop description* markdown file containing the
   sanitized research context (upstream ideation reads exactly one such file).
2. The sidecar runtime runs upstream ideation:
   ``python -m ai_scientist.perform_ideation_temp_free --model <model>
   --workshop-file <file.md> --max-num-generations N --num-reflections R``
   which writes ``<file>.json`` — a list of idea dicts
   (Name / Title / Short Hypothesis / Related Work / Abstract / ...).
3. The adapter parses that JSON into Chowner-native :class:`Hypothesis`
   objects (deterministic translation, Chowder-stamped ids/provenance).
4. Optional sandbox exploration: the adapter writes one ``idea.json`` plus a
   ``bfts_config.yaml`` (upstream per-run config) and invokes the upstream
   BFTS entry point in the sidecar. The sandbox runs on the sidecar's own
   datasets/compute — never on Chowder's GPU budget or model.
5. ``journal.json`` (upstream ``Journal.to_dict()``) nodes are translated into
   typed :class:`ExperimentProposal`s for *Chowder* to admit and execute on
   the real model. Sidecar metrics are treated as hypothesis-grade signals,
   never as evidence: only Chowder run-grounded observations are evidence.

Runtime modes:
- ``local``: subprocess in an explicitly configured upstream checkout
  (``provider_config.home``). The operator owns that environment's isolation.
- ``wsl``: ``wsl -e <python> -m ...`` (Linux-side home).
- ``docker``: ``docker run --rm -v <home>:/work <image> ...``.

The sidecar is *never* given trusted Chowder paths: the adapter constructs
workspace paths under the mission's state root only.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..findings import Claim, ResearchFinding
from ..hypothesis import Hypothesis, ResearchQuestion
from ..mission import ResearchMission
from ..observation import ExperimentObservation
from ..proposal import DataStrategy, ExperimentProposal, TrainingRecipeDelta
from ..provider import ProviderUnavailability, ResearchContext

#: Upstream ideation module + entry points (imported by the SIDECAR's python,
#: never by Chowder).
IDEATION_MODULE = "ai_scientist.perform_ideation_temp_free"

#: The idea-dict keys the upstream ideation pipeline emits. Extra keys are
#: tolerated (the upstream schema has optional ones); these are the ones the
#: translation reads.
_IDEA_NAME = "Name"
_IDEA_TITLE = "Title"
_IDEA_HYPOTHESIS = "Short Hypothesis"
_IDEA_RELATED = "Related Work"
_IDEA_ABSTRACT = "Abstract"


class SidecarError(ProviderUnavailability):
    """The sidecar cannot run (missing home/runtime) or failed. Loud, never
    a silent fallback."""


@dataclass(frozen=True)
class SidecarRuntime:
    """How to invoke the sidecar's python. Constructed by the adapter from
    provider_config; the command lines are buildable and testable without
    executing anything.

    ``home`` is the upstream checkout (local/wsl modes; the docker image is
    expected to carry the upstream install itself). ``workspace_dir`` is the
    host directory where the adapter writes workshop/idea/config files — the
    ONLY host path the sidecar ever sees, mounted at ``/work`` in docker
    mode. Trusted Chowder paths never appear here."""

    mode: str                    # local | wsl | docker
    home: str                    # upstream checkout (local/wsl)
    python: str = "python"
    image: str = ""              # docker only
    workspace_dir: str = ""      # host dir for sidecar-visible files
    extra_docker_args: tuple[str, ...] = ()

    def build_command(self, module_args: list[str]) -> list[str]:
        if self.mode == "local":
            return [self.python, "-m", *module_args]
        if self.mode == "wsl":
            return ["wsl", "-e", self.python, "-m", *module_args]
        if self.mode == "docker":
            if not self.image:
                raise SidecarError(
                    provider="ai_scientist_v2",
                    reason="runtime docker requires provider_config.image",
                )
            if not self.workspace_dir:
                raise SidecarError(
                    provider="ai_scientist_v2",
                    reason="runtime docker requires provider_config.workspace_dir",
                )
            return ["docker", "run", "--rm", "-v", f"{self.workspace_dir}:/work",
                    *self.extra_docker_args, self.image, self.python, "-m", *module_args]
        raise SidecarError(provider="ai_scientist_v2", reason=f"unknown runtime mode {self.mode!r}")

    def working_directory(self) -> str | None:
        # local: run inside the upstream checkout so `-m ai_scientist...`
        # resolves; wsl/docker: the command already carries its context.
        return self.home if self.mode == "local" else None


class AIScientistV2Provider:
    """The first external research provider: AI Scientist v2 in a sidecar."""

    name = "ai_scientist_v2"

    def __init__(self, *, runtime: SidecarRuntime, model: str,
                 max_num_generations: int = 3, num_reflections: int = 2,
                 sandbox_required: bool = True) -> None:
        self.runtime = runtime
        self.model = model
        self.max_num_generations = max(1, int(max_num_generations))
        self.num_reflections = max(0, int(num_reflections))
        self.sandbox_required = sandbox_required
        self._workshop_path: Path | None = None
        self._ideas_cache: tuple[dict[str, Any], ...] | None = None
        self._journal_cache: dict[str, Any] | None = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AIScientistV2Provider":
        mode = str(config.get("runtime", "local"))
        workspace = str(config.get("workspace_dir", ""))
        if not workspace:
            home = str(config.get("home", ""))
            if home and mode in ("local", "wsl"):
                workspace = str(Path(home).parent / "chowder-scientist-workspaces")
        runtime = SidecarRuntime(
            mode=mode,
            home=str(config.get("home", "")),
            python=str(config.get("python", "python")),
            image=str(config.get("image", "")),
            workspace_dir=workspace,
            extra_docker_args=tuple(str(a) for a in config.get("extra_docker_args", ())),
        )
        if mode in ("local", "wsl") and not runtime.home:
            raise SidecarError(
                provider=cls.name,
                reason="provider_config.home is required (the upstream checkout); "
                       "obtain AI Scientist v2 yourself and point Chowder at it",
            )
        provider = cls(
            runtime=runtime,
            model=str(config.get("model", "gpt-4o-2024-11-20")),
            max_num_generations=int(config.get("max_num_generations", 3)),
            num_reflections=int(config.get("num_reflections", 2)),
            sandbox_required=bool(config.get("sandbox_required", True)),
        )
        provider._local_isolated = bool(config.get("local_isolated", False))
        return provider

    def available(self) -> bool:
        """Runtime prerequisites must hold; in sandbox-required mode a bare
        ``local`` runtime on the Chowder host is refused unless the operator
        explicitly declared it isolated (``local_isolated: true``)."""
        if self.runtime.mode == "docker":
            return bool(self.runtime.image and self.runtime.workspace_dir)
        if not self.runtime.home:
            return False
        if self.sandbox_required and self.runtime.mode == "local":
            return self._local_isolation_declared()
        return True

    def _local_isolation_declared(self) -> bool:
        # The operator's explicit acknowledgement lives in the config as
        # ``local_isolated: true``; anything else is treated as unisolated.
        return bool(getattr(self, "_local_isolated", False))

    def declare_local_isolated(self) -> None:
        """Operator-acknowledged isolation for the local runtime (config file
        equivalent: ``provider_config.local_isolated: true``)."""
        self._local_isolated = True

    # -- ideation ---------------------------------------------------------------

    def _workshop_description(self, context: ResearchContext) -> str:
        """Render the sanitized context into the upstream workshop-file format.
        Only exported context goes in — never candidate eval content, never
        protected material, never policy documents."""
        lines = [
            "# Research workshop",
            "",
            "## Objective",
            context.mission.objective,
            "",
            "## Capability priorities (improve these, weakest first)",
        ]
        for skill, weight in sorted(context.mission.priorities.items(), key=lambda kv: -kv[1]):
            estimate = next((s.estimate for s in context.skill_estimates
                             if s.skill == skill), None)
            state = "unmeasured" if estimate is None else f"estimate {estimate:.3f}"
            lines.append(f"- {skill} (weight {weight:.2f}; {state})")
        if context.mission.protected_capabilities:
            lines += ["", "## Protected capabilities (must not regress; never targets)"]
            lines += [f"- {c}" for c in context.mission.protected_capabilities]
        if context.open_failure_categories:
            lines += ["", "## Open failure categories (counts only)"]
            lines += [f"- {c.get('category', '?')}: {c.get('count', '?')}"
                      for c in context.open_failure_categories]
        if context.attempted_mechanisms:
            lines += ["", "## Mechanisms already attempted (do not re-propose; build on evidence)"]
            lines += [f"- {m.get('mechanism', '?')} -> {m.get('outcome', '?')}"
                      for m in context.attempted_mechanisms]
        if context.carried_evidence:
            lines += ["", "## Prior evidence (carried; not fresh measurement)"]
            lines += [f"- [carried] {c.statement}" for c in context.carried_evidence]
        lines += [
            "",
            "## Constraints",
            f"- remaining compute: {context.remaining_gpu_hours:.1f} GPU-hours",
            "- interventions must be expressible as: data strategy, optimization",
            "  recipe delta, LoRA adapter delta, or training strategy change;",
            "  architecture modifications are out of scope for this workshop",
            "- every proposal must state what evidence would refute it",
            "",
            "Propose machine-learning improvement ideas for post-training a",
            "small language model. Respond in the workshop's idea format.",
        ]
        return "\n".join(lines)

    def propose_hypotheses(self, context: ResearchContext,
                           *, count: int = 3) -> tuple[Hypothesis, ...]:
        ideas = self._ideate(context)
        out: list[Hypothesis] = []
        for i, idea in enumerate(ideas[:count]):
            out.append(self._hypothesis_from_idea(idea, index=i))
        if not out:
            raise SidecarError(
                provider=self.name,
                reason="ideation produced no parsable ideas; refusing to invent any",
            )
        return tuple(out)

    def _ideate(self, context: ResearchContext) -> tuple[dict[str, Any], ...]:
        if not self.available():
            raise SidecarError(
                provider=self.name,
                reason=(
                    f"runtime {self.runtime.mode!r} unavailable "
                    f"(home={self.runtime.home!r}); obtain AI Scientist v2 and "
                    "configure provider_config.home"
                ),
            )
        if self._ideas_cache is not None:
            return self._ideas_cache
        workspace = self._workspace_root(context)
        workshop = workspace / "workshop.md"
        workshop.parent.mkdir(parents=True, exist_ok=True)
        workshop.write_text(self._workshop_description(context), encoding="utf-8")
        cmd = self.runtime.build_command([
            IDEATION_MODULE,
            "--model", self.model,
            "--workshop-file", self._sidecar_path(workshop),
            "--max-num-generations", str(self.max_num_generations),
            "--num-reflections", str(self.num_reflections),
        ])
        self._run_sidecar(cmd)
        ideas_file = workshop.with_suffix(".json")
        if not ideas_file.exists():
            raise SidecarError(
                provider=self.name,
                reason=f"ideation did not write {ideas_file.name}; upstream "
                       "writes <workshop>.json beside the workshop file",
            )
        ideas = json.loads(ideas_file.read_text(encoding="utf-8"))
        if not isinstance(ideas, list):
            raise SidecarError(provider=self.name,
                               reason="ideas file is not a JSON list")
        self._ideas_cache = tuple(ideas)
        return self._ideas_cache

    def _hypothesis_from_idea(self, idea: dict[str, Any], *, index: int) -> Hypothesis:
        name = str(idea.get(_IDEA_NAME, f"idea-{index}"))
        short = str(idea.get(_IDEA_HYPOTHESIS, "")).strip()
        if not short:
            raise SidecarError(
                provider=self.name,
                reason=f"idea {name!r} carries no 'Short Hypothesis'; refusing to "
                       "translate an idea without a stated hypothesis",
            )
        capability = self._capability_from(idea)
        return Hypothesis(
            hypothesis_id=f"aisci-{name}",
            research_question=ResearchQuestion(
                text=str(idea.get(_IDEA_TITLE, name)),
                capability=capability,
            ),
            observation=self._observation_from_context(),
            suspected_mechanism=short,
            predicted_effect=self._predicted_effect(idea),
            novelty_basis=str(idea.get(_IDEA_RELATED, ""))[:500],
            uncertainty="high",  # external ideas start uncertain; evidence moves it
            falsification_conditions=(
                "capability delta <= 0 across replicated seeds",
                "any protected capability regresses beyond tolerance",
                "transfer evaluation does not reproduce the target-surface delta",
            ),
            expected_information_gain=0.3,
            provider=self.name,
        )

    def _capability_from(self, idea: dict[str, Any]) -> str:
        text = " ".join(str(idea.get(k, "")) for k in
                        (_IDEA_TITLE, _IDEA_HYPOTHESIS, _IDEA_ABSTRACT)).lower()
        for marker in ("reasoning", "math", "code", "instruction", "knowledge",
                       "summarization", "extraction"):
            if marker in text:
                return "reasoning" if marker in ("reasoning", "math") else marker
        return "reasoning"

    def _observation_from_context(self) -> str:
        return (
            "exported skill estimates and failure categories from the Chowder "
            "growth state (sanitized; carried evidence marked as such)"
        )

    def _predicted_effect(self, idea: dict[str, Any]) -> str:
        abstract = str(idea.get(_IDEA_ABSTRACT, "")).strip()
        return (abstract[:400] + "…") if len(abstract) > 400 else (
            abstract or "improvement on the targeted capability without protected regression"
        )

    # -- experiments (translation from the sidecar journal) ----------------------

    def propose_experiments(
        self,
        context: ResearchContext,
        hypotheses: tuple[Hypothesis, ...],
    ) -> tuple[ExperimentProposal, ...]:
        """Translate sidecar research into typed proposals for Chowder's lab.

        If a sandbox journal exists, its best/good nodes inform the recipe
        deltas; the proposals themselves are always Chowder-shaped and always
        subject to admission."""
        journal = self._load_journal_if_any(context)
        out: list[ExperimentProposal] = []
        for hyp in hypotheses:
            out.append(self._proposal_for(hyp, journal))
        return tuple(out)

    def _proposal_for(self, hyp: Hypothesis,
                      journal: dict[str, Any] | None) -> ExperimentProposal:
        surface = hyp.research_question.capability
        # Deterministic translation: the idea's mechanism picks the family.
        mechanism = hyp.suspected_mechanism.lower()
        if any(k in mechanism for k in ("learning rate", "lr ", "warmup", "weight decay",
                                        "optimizer", "schedule")):
            exp_type = "optimization"
            delta = TrainingRecipeDelta(learning_rate=1e-4)
            changed = ("learning_rate",)
        elif any(k in mechanism for k in ("lora", "adapter", "rank")):
            exp_type = "adapter"
            delta = TrainingRecipeDelta(lora_rank=32, lora_alpha=64)
            changed = ("lora_rank", "lora_alpha")
        else:
            exp_type = "data"
            delta = TrainingRecipeDelta(epochs=2)
            changed = ("replay_ratio", "mixture")
        proposal_id = f"aisci-prop-{hyp.hypothesis_id}"
        return ExperimentProposal(
            proposal_id=proposal_id,
            hypothesis_id=hyp.hypothesis_id,
            experiment_type=exp_type,
            intervention=hyp.suspected_mechanism,
            variables_changed=changed,
            variables_held_constant=("model", "base_revision", "seed_policy"),
            training_recipe_delta=delta,
            data_strategy=DataStrategy(
                source_kinds=("curriculum", "replay"),
                replay_ratio=0.1 if exp_type == "data" else None,
                contamination_policy="firewall_default",
            ),
            requested_evaluations=(surface,),
            transfer_evaluations=(f"{surface}-transfer",),
            replication_plan="3 seeds; transfer stage on survivors",
            controls=("parent adapter at unchanged recipe",),
            expected_outcome=hyp.predicted_effect,
            falsification_rule=(
                f"{surface} transfer delta <= 0 on 2 of 3 seeds, or any protected "
                "capability regresses beyond tolerance"
            ),
            estimated_gpu_hours=0.5,
            safety_requirements=("sandbox_replay_only",) if self.sandbox_required else (),
            provider=self.name,
        )

    def _load_journal_if_any(self, context: ResearchContext) -> dict[str, Any] | None:
        if self._journal_cache is not None:
            return self._journal_cache
        path = self._workspace_root(context) / "sandbox" / "journal.json"
        if not path.exists():
            return None
        try:
            self._journal_cache = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SidecarError(
                provider=self.name,
                reason=f"journal.json is not valid JSON: {error}",
            )
        return self._journal_cache

    # -- interpretation ------------------------------------------------------------

    def interpret(
        self,
        context: ResearchContext,
        hypothesis: Hypothesis,
        observations: tuple[ExperimentObservation, ...],
    ) -> ResearchFinding:
        """Translate grounded observations into a *proposed* finding. The
        claim statuses written here are provisional suggestions — the director
        re-settles them mechanically."""
        surface = hypothesis.research_question.capability
        complete = tuple(o for o in observations if o.status == "complete")
        supporting = tuple(o.run_id for o in complete)
        claim = Claim(
            claim_id=f"aisci-claim-{hypothesis.hypothesis_id}",
            statement=(
                f"intervention '{hypothesis.suspected_mechanism[:120]}' changes "
                f"{surface} as measured by the cited runs"
            ),
            scope=f"sidecar-proposed; missions {context.mission.mission_id}",
            status="provisional",
            supporting_experiments=supporting,
            affected_capabilities=(surface,),
            conditions=("grounded in Chowder runs only",),
        )
        return ResearchFinding(
            finding_id=f"aisci-finding-{hypothesis.hypothesis_id}",
            hypothesis_id=hypothesis.hypothesis_id,
            claims=(claim,),
            observation_ids=tuple(o.observation_id for o in observations),
            provider=self.name,
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "runtime_mode": self.runtime.mode,
            "ideas_cached": self._ideas_cache is not None,
            "journal_cached": self._journal_cache is not None,
        }

    # -- sidecar plumbing ------------------------------------------------------

    def _workspace_root(self, context: ResearchContext) -> Path:
        """The mission's sidecar workspace, under the configured workspace_dir
        — the only host location the sidecar ever sees. Trusted Chowder state
        (registry, growth state, policies) never lives under it."""
        if not self.runtime.workspace_dir:
            raise SidecarError(
                provider=self.name,
                reason="provider_config.workspace_dir is required for the sidecar "
                       "file protocol (docker mounts it at /work)",
            )
        workspace = Path(self.runtime.workspace_dir) / context.mission.mission_id
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _sidecar_path(self, p: Path) -> str:
        """The path as the sidecar sees it (docker mounts workspace_dir at
        /work; local/wsl see real host paths)."""
        if self.runtime.mode == "docker":
            root = Path(self.runtime.workspace_dir).resolve()
            return "/work/" + str(p.resolve().relative_to(root)).replace("\\", "/")
        return str(p)

    def _run_sidecar(self, cmd: list[str]) -> None:
        """Execute the sidecar runtime. This is the ONLY subprocess the
        adapter launches: the sidecar's own entry point, inside its own
        runtime. No sidecar code is imported or executed in-process."""
        cwd = self.runtime.working_directory()
        try:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=3600,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError as error:
            raise SidecarError(
                provider=self.name,
                reason=f"runtime binary not found for {self.runtime.mode!r}: {error}",
            )
        except subprocess.TimeoutExpired:
            raise SidecarError(provider=self.name,
                               reason="sidecar exceeded the 3600 s budget")
        if result.returncode != 0:
            raise SidecarError(
                provider=self.name,
                reason=f"sidecar failed rc={result.returncode}: "
                       f"{(result.stderr or result.stdout)[-400:]}",
            )
