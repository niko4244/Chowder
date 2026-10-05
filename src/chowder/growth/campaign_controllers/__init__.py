"""The campaign controllers, one module per decision the runner defers.

``campaign_runner`` composes these; it owns sequencing and the two injection
seams, and owns no decision of its own. Import a controller directly when that is
the decision you mean -- ``from chowder.growth.campaign_controllers.promotion
import _adjudicate`` says something ``from campaign_runner import`` does not.
"""

from __future__ import annotations

from . import (  # noqa: F401
    contracts,
    declared,
    evaluation,
    training,
    planning,
    certification,
    promotion,
    readiness,
)

__all__ = [
    "contracts",
    "declared",
    "evaluation",
    "training",
    "planning",
    "certification",
    "promotion",
    "readiness",
]
