"""chowder.scientist: the native research control layer.

The scientist proposes; Chowder admits; Chowder executes; Chowder measures;
the scientist interprets; Chowder verifies. See docs/SCIENTIST_MODE.md.
"""

from .compute import (
    ComputeProvider,
    ExperimentClass,
    ExperimentRequest,
    ExperimentScheduler,
    KaggleProvider,
    LocalCudaProvider,
    ProviderQuota,
    RunPodProvider,
    SchedulerRefusal,
    Submission,
    provider_from_config,
)
from .screening_halving import (
    HalvingRoundOutcome,
    ScreeningHalving,
    ScreeningHalvingOutcome,
    settle_screening_round,
)
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
    "ComputeProvider",
    "ExperimentClass",
    "ExperimentRequest",
    "ExperimentScheduler",
    "KaggleProvider",
    "LocalCudaProvider",
    "CompilationRefusal",
    "DataStrategy",
    "DirectorRefusal",
    "ExperimentCompiler",
    "ExperimentConstraint",
    "ExperimentObservation",
    "ExperimentProposal",
    "Hypothesis",
    "HalvingRoundOutcome",
    "Measurement",
    "MissionBudget",
    "ModelResearchService",
    "ProviderUnavailability",
    "ProviderQuota",
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
    "RunPodProvider",
    "SchedulerRefusal",
    "provider_from_config",
    "ScreeningHalving",
    "ScreeningHalvingOutcome",
    "settle_screening_round",
    "SkillSummary",
    "Submission",
    "ScientistProvider",
    "TrainingRecipeDelta",
]
