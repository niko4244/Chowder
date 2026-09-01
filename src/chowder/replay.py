from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .investigation import RemediationOutcome, config_patch_digest


@dataclass(frozen=True)
class ReplayGroundTruth:
    """The known outcome of every config_patch a benchmark case's
    ``ReplayExecutor`` is allowed to be asked about.

    Keyed by ``config_patch_digest`` (not the raw patch), so lookup is
    exact and order-independent. An unlisted patch is a hard error at
    lookup time -- a benchmark case must specify ground truth for
    everything a hypothesis generator might plausibly propose, not
    silently default to failure, which would let an incomplete fixture
    quietly pass by coincidence rather than by actually covering the case.
    """

    fingerprint_sha256: str
    outcomes: Mapping[str, RemediationOutcome]

    def outcome_for(self, config_patch: Mapping[str, Any]) -> RemediationOutcome:
        digest = config_patch_digest(config_patch)
        if digest not in self.outcomes:
            raise KeyError(
                f"no ground truth recorded for this config_patch against "
                f"fingerprint {self.fingerprint_sha256[:12]}...; "
                "add it to this ReplayGroundTruth before proposing it"
            )
        return self.outcomes[digest]


@dataclass(frozen=True)
class ReplayExecutor:
    """Test-support executor: looks up a pre-recorded outcome instead of
    actually training anything.

    Real incidents cost 15 minutes to several hours each on real hardware
    -- a benchmark that has to reproduce them live to test whether Chowder
    handles them would be prohibitively expensive to iterate on. Not
    fixture-specific despite only being used with fixtures today: lives in
    ``src/chowder/`` rather than ``tests/`` because both the walking
    skeleton and the full benchmark need it, and a real package import is
    cleaner than reaching across the src/tests boundary.
    """

    ground_truth: ReplayGroundTruth

    def run(self, config_patch: Mapping[str, Any]) -> RemediationOutcome:
        return self.ground_truth.outcome_for(config_patch)
