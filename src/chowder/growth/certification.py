"""Certification of a measured generation against the frozen mini-slice protocol.

The campaign runner and the frozen gen2 judge must answer the same question --
*may this candidate be promoted on the evidence in this run root?* -- and they
must never answer it differently. So the mechanism lives here, in production,
and each caller supplies its own policy (the required benchmark set, the
regression tolerance, the trusted ancestor). The runner applies it **before**
the lineage record is written; the judge applies it again at audit time over the
same artifacts. Neither owns a second implementation of the verification.

Three properties are enforced, all fail-closed:

* **A measurement is evidence only if its bytes exist.** A protected row must
  name its raw artifact and declare that artifact's sha256; the digest is
  recomputed over the real bytes, and the per-sample values must be exactly
  ``n_samples`` values whose mean *is* the row's score. A row naming a file that
  is not there, carrying no digest, or reporting a score its own samples
  contradict is refused with a named reason.
* **An arm is what it says it is.** A row's ``measurement_origin`` must match the
  arm's role, its ``generation_version`` must be the generation the arm claims,
  and the report's ``model_identity`` must name the bytes the arm measured: the
  digest the campaign selected for the candidate, the frozen parent adapter for
  the parent arm, the frozen base model for the trusted-ancestor arm. A missing
  identity is UNDECIDED evidence; a mismatched one is a hard failure.
* **Branch protection is judged against the trusted ancestor**, not only against
  the immediate parent: a candidate that merely matches an already-regressed
  parent cannot pass, and an unresolved ancestor cannot become the protection
  baseline by default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from chowder.evals.result import MEASURED_PARENT, MEASURED_THIS_GENERATION, BenchmarkRun, EvalReport
from chowder.provenance import sha256_file

from .training_binding import directory_digest

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

#: The frozen mini-slice protocol (prereg section 3), as the *values* gen2
#: declares. A caller always supplies a protocol explicitly -- the frozen judge
#: from its own constants, the campaign runner from the manifest -- so no
#: default here can become a threshold nobody declared. A slice that does not
#: match every field of the supplied protocol is not the declared measurement.
PROTECTED_N_SAMPLES = 16
PROTECTED_SAMPLE_INDICES = tuple(range(16))
PROTECTED_SEED = 1234
PROTECTED_SHUFFLE = False
PROTECTED_DECODING: Mapping[str, Any] = {
    "temperature": 0.0,
    "do_sample": False,
    "max_new_tokens": 512,
}
PROTECTED_PROMPT_POLICY = "chat_template"


@dataclass(frozen=True)
class ProtocolSpec:
    """The declared mini-slice protocol a measured slice must match.

    ``n_samples`` plus :meth:`sample_indices` express the frozen "first N items
    in dataset order": the row's own ``sample_indices`` must be ``0..N-1``.
    """

    n_samples: int
    seed: int
    decoding: Mapping[str, Any]
    prompt_policy: str
    shuffle: bool = False

    def sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n_samples))

    @classmethod
    def from_mapping(
        cls, document: Mapping[str, Any], *, source: str = "<memory>"
    ) -> "ProtocolSpec":
        """Read the declared protocol fields, refusing anything malformed."""
        n_samples = document.get("n_samples")
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"{source}: protection.n_samples must be a positive integer")
        seed = document.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{source}: protection.seed must be an integer")
        shuffle = document.get("shuffle")
        if not isinstance(shuffle, bool):
            raise ValueError(f"{source}: protection.shuffle must be a bool")
        decoding = document.get("decoding")
        if not isinstance(decoding, Mapping) or not decoding:
            raise ValueError(f"{source}: protection.decoding must be a non-empty object")
        prompt_policy = document.get("prompt_policy")
        if not isinstance(prompt_policy, str) or not prompt_policy:
            raise ValueError(f"{source}: protection.prompt_policy must be a non-empty string")
        return cls(
            n_samples=int(n_samples),
            seed=int(seed),
            decoding=dict(decoding),
            prompt_policy=prompt_policy,
            shuffle=shuffle,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "decoding": dict(self.decoding),
            "prompt_policy": self.prompt_policy,
        }


#: The gen2 mini-slice protocol, as the frozen prereg declares it. A caller may
#: pass its own spec; this one exists so the judge and the shipped campaign
#: manifest can be checked against the same declared values.
GEN2_PROTOCOL = ProtocolSpec(
    n_samples=PROTECTED_N_SAMPLES,
    seed=PROTECTED_SEED,
    decoding=dict(PROTECTED_DECODING),
    prompt_policy=PROTECTED_PROMPT_POLICY,
    shuffle=PROTECTED_SHUFFLE,
)

#: The shape a declared artifact digest must have before it is recomputed, and
#: the tolerance for "the aggregate *is* the mean of the per-sample values".
ARTIFACT_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SAMPLE_MEAN_TOLERANCE = 1e-6

#: Named reasons. A refusal says which verification failed, not merely that
#: something did.
MEASUREMENT_ARTIFACT_MISSING = "MEASUREMENT_ARTIFACT_MISSING"
MEASUREMENT_ARTIFACT_ESCAPES_RUN_ROOT = "MEASUREMENT_ARTIFACT_ESCAPES_RUN_ROOT"
MEASUREMENT_DIGEST_ABSENT = "MEASUREMENT_DIGEST_ABSENT"
MEASUREMENT_DIGEST_MISMATCH = "MEASUREMENT_DIGEST_MISMATCH"
MEASUREMENT_SAMPLES_INCONSISTENT = "MEASUREMENT_SAMPLES_INCONSISTENT"
ARM_GENERATION_MISMATCH = "ARM_GENERATION_MISMATCH"
ARM_ADAPTER_DIGEST_MISSING = "ARM_ADAPTER_DIGEST_MISSING"
ARM_ADAPTER_DIGEST_MISMATCH = "ARM_ADAPTER_DIGEST_MISMATCH"
ARM_BASE_DIGEST_MISSING = "ARM_BASE_DIGEST_MISSING"
ARM_BASE_DIGEST_MISMATCH = "ARM_BASE_DIGEST_MISMATCH"
ARM_UNREADABLE = "ARM_UNREADABLE"


class ArmError(RuntimeError):
    """One measured generation's evidence is not present or not parseable."""


@dataclass(frozen=True)
class MeasuredArm:
    """One generation's durable evaluation report, with the role it must play.

    ``origin`` is the provenance every row of the arm must declare;
    ``generation`` is the generation the report must be about (empty means the
    role does not pin one, which only a caller that has no frozen label may do).
    """

    label: str
    origin: str
    generation: str
    report: EvalReport
    path: Path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_origin: str,
        expected_generation: str,
        label: str,
    ) -> "MeasuredArm":
        if not path.is_file():
            raise ArmError(f"{label} artifact {path.name} is missing")
        try:
            report = EvalReport.load(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ArmError(f"{label} artifact {path.name} is unreadable: {error}") from error
        return cls(
            label=label,
            origin=expected_origin,
            generation=expected_generation,
            report=report,
            path=path,
        )

    def run_for(self, qualified_id: str) -> BenchmarkRun | None:
        """The single usable run for a benchmark, or None when absent/refused."""
        matches = [
            run for run in self.report.runs if run.benchmark_qualified_id == qualified_id
        ]
        if len(matches) != 1:
            return None
        run = matches[0]
        if run.measurement_origin != self.origin:
            return None
        if self.generation and run.generation_version != self.generation:
            return None
        return run

    def duplicate_ids(self) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        for run in self.report.runs:
            counts[run.benchmark_qualified_id] = counts.get(run.benchmark_qualified_id, 0) + 1
        return tuple(sorted(qid for qid, count in counts.items() if count > 1))

    def identity_problems(
        self,
        *,
        adapter_digest: str = "",
        base_digest: str = "",
        require_generation: bool = True,
    ) -> tuple[str, ...]:
        """Every way this arm fails to name the bytes its role says it measured.

        The digest binding is what makes ``candidate_evaluation.json`` and
        ``chosen_candidate.json`` one truth instead of two: the report must name
        the artifact digest the campaign selected, not merely a generation label
        that could describe any adapter.
        """
        problems: list[str] = []
        if require_generation and self.generation:
            if self.report.generation_version != self.generation:
                problems.append(
                    f"{ARM_GENERATION_MISMATCH}: the {self.label} report is for "
                    f"generation {self.report.generation_version!r}, expected "
                    f"{self.generation!r}"
                )
        identity = self.report.model_identity or {}
        if not isinstance(identity, Mapping):
            return tuple(
                problems
                + [f"{ARM_ADAPTER_DIGEST_MISSING}: the {self.label} report's "
                   "model_identity is not an object, so it names no measured bytes"]
            )
        if adapter_digest:
            declared = str(identity.get("adapter_digest", ""))
            if not declared:
                problems.append(
                    f"{ARM_ADAPTER_DIGEST_MISSING}: the {self.label} report names no "
                    "adapter_digest, so it is not bound to the artifact it measured"
                )
            elif declared != adapter_digest:
                problems.append(
                    f"{ARM_ADAPTER_DIGEST_MISMATCH}: the {self.label} report measured "
                    f"adapter {declared}, the campaign selected {adapter_digest}"
                )
        if base_digest:
            declared_base = str(identity.get("base_model_digest", ""))
            if not declared_base:
                problems.append(
                    f"{ARM_BASE_DIGEST_MISSING}: the {self.label} report names no "
                    "base_model_digest, so it is not bound to the base model it measured"
                )
            elif declared_base != base_digest:
                problems.append(
                    f"{ARM_BASE_DIGEST_MISMATCH}: the {self.label} report measured base "
                    f"{declared_base}, the campaign declared {base_digest}"
                )
        return tuple(problems)


def digest_of(path: Path) -> str:
    """The repo's canonical digest: directory tree, or single-file sha256."""
    if path.is_dir():
        digest, _entries = directory_digest(path)
        return digest
    return sha256_file(path)


def measurement_artifact(run_root: Path, ref: str) -> tuple[Path | None, str]:
    """Resolve a row's ``raw_artifact_ref``, or say why it cannot be verified."""
    path = Path(ref)
    if not path.is_absolute():
        if ".." in path.parts:
            return None, (
                f"{MEASUREMENT_ARTIFACT_ESCAPES_RUN_ROOT}: {ref!r} points outside the "
                "run root"
            )
        path = run_root / path
    if not path.exists():
        return None, (
            f"{MEASUREMENT_ARTIFACT_MISSING}: raw_artifact_ref {ref!r} does not exist, "
            "so the measurement cannot be verified against its own bytes"
        )
    return path, ""


def evidence_problems(run: BenchmarkRun, *, run_root: Path, metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """The measurement-verification half of the protocol check."""
    problems: list[str] = []
    if not run.raw_artifact_ref:
        problems.append("raw_artifact_ref missing (no underlying evidence named)")
    else:
        artifact, reason = measurement_artifact(run_root, run.raw_artifact_ref)
        if artifact is None:
            problems.append(reason)
        else:
            declared = metadata.get("artifact_sha256")
            if not isinstance(declared, str) or not ARTIFACT_SHA256_PATTERN.fullmatch(declared):
                problems.append(
                    f"{MEASUREMENT_DIGEST_ABSENT}: artifact_sha256 {declared!r} is not a "
                    "64-character sha256, so the row is not bound to the bytes it names"
                )
            else:
                try:
                    actual = digest_of(artifact)
                except OSError as error:  # unreadable is unverifiable
                    problems.append(
                        f"{MEASUREMENT_ARTIFACT_MISSING}: raw artifact "
                        f"{run.raw_artifact_ref!r} cannot be hashed ({error})"
                    )
                else:
                    if actual != declared:
                        problems.append(
                            f"{MEASUREMENT_DIGEST_MISMATCH}: raw artifact "
                            f"{run.raw_artifact_ref!r} hashes to {actual}, the row "
                            f"declares {declared}"
                        )
    samples = tuple(run.per_sample_scores)
    if len(samples) != run.n_samples:
        problems.append(
            f"{MEASUREMENT_SAMPLES_INCONSISTENT}: per_sample_scores carries "
            f"{len(samples)} values for n_samples={run.n_samples}"
        )
    elif samples and isinstance(run.score, (int, float)) and not isinstance(run.score, bool):
        mean = sum(float(sample) for sample in samples) / len(samples)
        if abs(mean - float(run.score)) > SAMPLE_MEAN_TOLERANCE:
            problems.append(
                f"{MEASUREMENT_SAMPLES_INCONSISTENT}: score {run.score} is not the mean "
                f"of per_sample_scores ({mean:.6f})"
            )
    return tuple(problems)


def protocol_problems(
    run: BenchmarkRun, *, run_root: Path, protocol: "ProtocolSpec"
) -> tuple[str, ...]:
    """Every way a protected slice fails to match the declared protocol.

    Beyond the declared protocol fields, this verifies that the row *is* the
    measurement it claims (see :func:`evidence_problems`): a row that names no
    existing artifact, declares no digest, or reports a score its own samples do
    not support is refused rather than certified.
    """
    metadata = run.metadata or {}
    problems: list[str] = []
    if run.n_samples != protocol.n_samples:
        problems.append(f"n_samples={run.n_samples} (expected {protocol.n_samples})")
    indices = metadata.get("sample_indices")
    if not isinstance(indices, list) or tuple(indices) != protocol.sample_indices():
        problems.append(
            f"sample_indices={indices!r} (expected 0..{protocol.n_samples - 1})"
        )
    if metadata.get("seed") != protocol.seed:
        problems.append(f"seed={metadata.get('seed')!r} (expected {protocol.seed})")
    if metadata.get("shuffle") is not protocol.shuffle:
        problems.append(f"shuffle={metadata.get('shuffle')!r} (expected {protocol.shuffle})")
    decoding = metadata.get("decoding")
    if not isinstance(decoding, Mapping):
        problems.append("decoding metadata missing")
    else:
        for key, expected in protocol.decoding.items():
            if decoding.get(key) != expected:
                problems.append(f"decoding.{key}={decoding.get(key)!r} (expected {expected!r})")
    if metadata.get("prompt_policy") != protocol.prompt_policy:
        problems.append(
            f"prompt_policy={metadata.get('prompt_policy')!r} "
            f"(expected {protocol.prompt_policy!r})"
        )
    if not isinstance(run.score, (int, float)) or isinstance(run.score, bool):
        problems.append("score is not a measured number")
    problems.extend(evidence_problems(run, run_root=run_root, metadata=metadata))
    return tuple(problems)


@dataclass(frozen=True)
class SliceCheck:
    """One required slice in one arm, with the reason it is or is not usable."""

    status: str
    run: BenchmarkRun | None
    detail: str


def slice_status(
    arm: MeasuredArm | None,
    qualified_id: str,
    *,
    label: str,
    run_root: Path,
    protocol: ProtocolSpec,
) -> SliceCheck:
    """Locate one required slice in one arm, with its provenance and protocol."""
    if arm is None:
        return SliceCheck(UNKNOWN, None, f"{label} arm unavailable")
    if qualified_id in arm.duplicate_ids():
        return SliceCheck(FAIL, None, f"{label} arm carries {qualified_id} more than once")
    run = arm.run_for(qualified_id)
    if run is None:
        present = any(run.benchmark_qualified_id == qualified_id for run in arm.report.runs)
        if present:
            return SliceCheck(
                FAIL,
                None,
                f"{label} arm carries {qualified_id} with provenance "
                f"{arm.origin} / generation {arm.generation or 'any'} refused",
            )
        return SliceCheck(UNKNOWN, None, f"{label} arm has no {qualified_id} measurement")
    problems = protocol_problems(run, run_root=run_root, protocol=protocol)
    if problems:
        return SliceCheck(FAIL, run, f"{label} protocol mismatch: " + "; ".join(problems))
    return SliceCheck(
        PASS, run, f"{label} {qualified_id} = {run.score:.4f} ({run.n_samples} items)"
    )


@dataclass(frozen=True)
class CertificationRow:
    """One certification requirement and how it was decided."""

    requirement: str
    status: str
    detail: str


@dataclass(frozen=True)
class Certification:
    """The mechanical answer to "does the measured evidence protect the branch?".

    ``status`` is PASS only when every requirement was decided and held, FAIL when
    any requirement was breached, and UNKNOWN when required evidence is missing or
    unusable. Both non-PASS states refuse promotion: a run may not promote on
    evidence that could not be certified.
    """

    status: str
    rows: tuple[CertificationRow, ...] = ()
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "rows": [
                {"requirement": row.requirement, "status": row.status, "detail": row.detail}
                for row in self.rows
            ],
        }


@dataclass(frozen=True)
class ProtectionPolicy:
    """What the caller requires; the mechanism below never invents a threshold."""

    required_protected: Sequence[str]
    slice_regression_max: float
    trusted_ancestor_version: str
    protocol: ProtocolSpec = GEN2_PROTOCOL
    candidate_version: str = ""
    parent_version: str = ""


def _combine(rows: Sequence[CertificationRow]) -> Certification:
    statuses = {row.status for row in rows}
    if FAIL in statuses:
        status = FAIL
    elif UNKNOWN in statuses:
        status = UNKNOWN
    else:
        status = PASS
    reasons = tuple(
        f"{row.requirement}: {row.detail}" for row in rows if row.status != PASS
    )
    return Certification(status=status, rows=tuple(rows), reasons=reasons)


def certify_protection(
    *,
    policy: ProtectionPolicy,
    run_root: Path,
    candidate: MeasuredArm | None,
    parent: MeasuredArm | None,
    ancestor: MeasuredArm | None,
    identity_problems: Sequence[str] = (),
) -> Certification:
    """Adjudicate protected-slice coverage and branch protection.

    The candidate side is the gate: every required slice must exist once,
    candidate-measured, protocol-exact and bound to real bytes. The parent and
    the trusted ancestor are compared slice by slice, and a candidate that only
    matches an already-regressed parent still fails against the ancestor. An
    unavailable arm is UNKNOWN -- never a pass, and never a fallback to another
    arm's numbers.
    """
    rows: list[CertificationRow] = []

    for problem in identity_problems:
        rows.append(
            CertificationRow(
                requirement="evaluation evidence names the bytes it measured",
                status=FAIL if "MISMATCH" in problem else UNKNOWN,
                detail=problem,
            )
        )

    required = tuple(policy.required_protected)
    if not required:
        rows.append(
            CertificationRow(
                requirement="a required protected set is declared",
                status=UNKNOWN,
                detail="the campaign declares no protected benchmark set to certify against",
            )
        )

    if candidate is None:
        rows.append(
            CertificationRow(
                requirement="candidate measured evidence",
                status=UNKNOWN,
                detail="candidate arm unavailable",
            )
        )
        return _combine(rows)

    for qualified_id in required:
        check = slice_status(
            candidate,
            qualified_id,
            label="candidate",
            run_root=run_root,
            protocol=policy.protocol,
        )
        rows.append(
            CertificationRow(
                requirement=f"candidate {qualified_id} measured + protocol-exact",
                status=check.status,
                detail=check.detail,
            )
        )
        if check.status != PASS or check.run is None:
            continue
        if parent is None:
            rows.append(
                CertificationRow(
                    requirement=f"{qualified_id} immediate-parent protection",
                    status=UNKNOWN,
                    detail="parent arm unavailable",
                )
            )
            continue
        parent_check = slice_status(
            parent,
            qualified_id,
            label="parent",
            run_root=run_root,
            protocol=policy.protocol,
        )
        if parent_check.status != PASS or parent_check.run is None:
            rows.append(
                CertificationRow(
                    requirement=f"{qualified_id} immediate-parent protection",
                    status=parent_check.status,
                    detail=parent_check.detail,
                )
            )
            continue
        delta = check.run.score - parent_check.run.score
        rows.append(
            CertificationRow(
                requirement=(
                    f"{qualified_id} candidate-vs-parent regression <= "
                    f"{policy.slice_regression_max}"
                ),
                status=PASS if delta >= -policy.slice_regression_max else FAIL,
                detail=(
                    f"candidate {check.run.score:.4f} vs parent "
                    f"{parent_check.run.score:.4f} (delta {delta:+.4f})"
                ),
            )
        )

    if ancestor is None:
        rows.append(
            CertificationRow(
                requirement=(
                    f"trusted-ancestor protection (vs {policy.trusted_ancestor_version})"
                ),
                status=UNKNOWN,
                detail=(
                    "the trusted ancestor arm is missing, so branch protection "
                    "cannot be established"
                ),
            )
        )
        return _combine(rows)

    ancestor_status = UNKNOWN
    ancestor_details: list[str] = []
    broken = False
    for qualified_id in required:
        candidate_check = slice_status(
            candidate,
            qualified_id,
            label="candidate",
            run_root=run_root,
            protocol=policy.protocol,
        )
        ancestor_check = slice_status(
            ancestor,
            qualified_id,
            label="ancestor",
            run_root=run_root,
            protocol=policy.protocol,
        )
        if candidate_check.status != PASS or ancestor_check.status != PASS:
            ancestor_details.append(f"{qualified_id}: {candidate_check.detail}; {ancestor_check.detail}")
            continue
        assert candidate_check.run is not None and ancestor_check.run is not None
        delta = candidate_check.run.score - ancestor_check.run.score
        ancestor_details.append(
            f"{qualified_id}: candidate {candidate_check.run.score:.4f} vs "
            f"{policy.trusted_ancestor_version} {ancestor_check.run.score:.4f} "
            f"(delta {delta:+.4f})"
        )
        if delta < -policy.slice_regression_max:
            broken = True
        ancestor_status = PASS
    if broken:
        ancestor_status = FAIL
    elif ancestor_status != PASS:
        ancestor_status = UNKNOWN
    rows.append(
        CertificationRow(
            requirement=(
                f"trusted-ancestor protection (vs {policy.trusted_ancestor_version})"
            ),
            status=ancestor_status,
            detail="; ".join(ancestor_details),
        )
    )
    return _combine(rows)


#: The role each arm plays, and the provenance its rows must declare.
ARM_ROLES: Mapping[str, tuple[str, str]] = {
    "candidate": (MEASURED_THIS_GENERATION, "candidate_version"),
    "parent": (MEASURED_PARENT, "parent_version"),
    "ancestor": (MEASURED_PARENT, "trusted_ancestor_version"),
}


def certify_run_root(
    *,
    policy: ProtectionPolicy,
    run_root: Path,
    arms: Mapping[str, Path | None],
    expected_adapter_digests: Mapping[str, str] | None = None,
    expected_base_digests: Mapping[str, str] | None = None,
) -> Certification:
    """Load the run root's arms and certify them, without inventing evidence.

    A declared arm whose file is not there -- or is unreadable, or is for the
    wrong generation -- is UNKNOWN evidence rather than an error: the caller
    learns that promotion cannot be certified from this root, which is the honest
    answer when the run wrote no such arm. Digests are compared only where the
    caller declares an expectation; a role for which nothing was selected simply
    carries no digest requirement.
    """
    adapter_digests = dict(expected_adapter_digests or {})
    base_digests = dict(expected_base_digests or {})
    loaded: dict[str, MeasuredArm | None] = {}
    unavailable: list[str] = []
    identity_problems: list[str] = []
    for role, (origin, generation_field) in ARM_ROLES.items():
        path = arms.get(role)
        generation = str(getattr(policy, generation_field, "") or "")
        if path is None:
            unavailable.append(f"{role} arm was not declared")
            loaded[role] = None
            continue
        try:
            arm = MeasuredArm.load(
                path,
                expected_origin=origin,
                expected_generation=generation,
                label=role,
            )
        except ArmError as error:
            unavailable.append(f"{role} arm unavailable ({ARM_UNREADABLE}: {error})")
            loaded[role] = None
            continue
        loaded[role] = arm
        identity_problems.extend(
            arm.identity_problems(
                adapter_digest=adapter_digests.get(role, ""),
                base_digest=base_digests.get(role, ""),
            )
        )
    certification = certify_protection(
        policy=policy,
        run_root=run_root,
        candidate=loaded.get("candidate"),
        parent=loaded.get("parent"),
        ancestor=loaded.get("ancestor"),
        identity_problems=identity_problems,
    )
    if not unavailable:
        return certification
    rows = list(certification.rows) + [
        CertificationRow(
            requirement="every measured arm is present and readable",
            status=UNKNOWN,
            detail="; ".join(unavailable),
        )
    ]
    return _combine(rows)
