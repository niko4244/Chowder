"""Chowder: autonomous evidence-gated post-training orchestration."""

from .engine import EvolutionEngine
from .models import Goal, MetricTarget, Experiment, ExperimentResult

__all__ = [
    "EvolutionEngine",
    "Goal",
    "MetricTarget",
    "Experiment",
    "ExperimentResult",
]

__version__ = "0.1.0"
