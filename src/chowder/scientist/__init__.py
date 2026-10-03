"""chowder.scientist: the native research control layer.

The scientist proposes; Chowder admits; Chowder executes; Chowder measures;
the scientist interprets; Chowder verifies. See docs/SCIENTIST_MODE.md.
"""

from .findings import Claim, ClaimEvidence, ReplicationPolicy, ResearchFinding
from .hypothesis import Hypothesis, ResearchQuestion
from .lab_bridge import CompiledExperiment, CompilationRefusal, ExperimentCompiler
from .mission import MissionBudget, ResearchMission
from .observation import ExperimentObservation, Measurement
from .proposal import (
    DataStrategy,
    ExperimentConstraint,
    ExperimentProposal,
    TrainingRecipeDelta,
)
from .provider import (
    CarriedEvidence,
    ProviderUnavailability,
    ResearchContext,
    ScientistProvider,
    SkillSummary,
)
from .research_decision import ResearchDecision
from .research_director import DirectorRefusal, ResearchDirector
from .research_memory import ResearchMemory
from .research_service import ModelResearchService, ResearchServiceError
from .research_tree import ResearchBranch, ResearchNode, ResearchTree

__all__ = [
    "CarriedEvidence",
    "Claim",
    "ClaimEvidence",
    "CompiledExperiment",
    "CompilationRefusal",
    "DataStrategy",
    "DirectorRefusal",
    "ExperimentCompiler",
    "ExperimentConstraint",
    "ExperimentObservation",
    "ExperimentProposal",
    "Hypothesis",
    "Measurement",
    "MissionBudget",
    "ModelResearchService",
    "ProviderUnavailability",
    "ReplicationPolicy",
    "ResearchContext",
    "ResearchDecision",
    "ResearchDirector",
    "ResearchFinding",
    "ResearchMemory",
    "ResearchMission",
    "ResearchQuestion",
    "ResearchServiceError",
    "ResearchTree",
    "SkillSummary",
    "ScientistProvider",
    "TrainingRecipeDelta",
]
