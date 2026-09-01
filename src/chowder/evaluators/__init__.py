"""Independent evaluation backends for Chowder."""

from .transformers_text import TransformersTextEvaluator, TransformersTextEvalSpec

__all__ = ["TransformersTextEvaluator", "TransformersTextEvalSpec"]
