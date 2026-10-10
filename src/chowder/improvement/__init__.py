"""Safety boundaries for Chowder's autonomous improvement layer."""

from .constitution import (
    Constitution,
    ConstitutionViolation,
    ObjectiveIdentity,
    ProtectedSurface,
)

__all__ = [
    "Constitution",
    "ConstitutionViolation",
    "ObjectiveIdentity",
    "ProtectedSurface",
]
