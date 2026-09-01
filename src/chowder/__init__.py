"""Chowder: autonomous evidence-gated post-training orchestration."""

from .engine import EvolutionEngine
from .models import Experiment, ExperimentResult, Goal, MetricTarget, OptimizationDirection

__all__ = [
    "EvolutionEngine",
    "Goal",
    "MetricTarget",
    "OptimizationDirection",
    "Experiment",
    "ExperimentResult",
]

__version__ = "0.1.0"
