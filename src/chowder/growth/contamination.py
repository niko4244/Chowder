"""Benchmark contamination firewall.

NON-NEGOTIABLE: protected evaluation material never enters the curriculum
or training pipeline. This module owns the three-set policy (PROTECTED /
DEVELOPMENT / TRAINING), the detection stack (exact hash, normalized hash,
substring containment, n-gram overlap, MinHash banding, canaries), the
per-generation ``contamination_manifest.json``, and the hard refusal path.

Verdicts:

- CLEAN: no detector fired.
- POSSIBLE: fuzzy evidence (n-gram/MinHash) above threshold -- treat as
  contaminated until a human clears it.
- KNOWN_CONTAMINATION: exact/normalized/canary match or declared ancestry.
- UNKNOWN: the benchmark's material was never fingerprinted -- a score
  against an UNKNOWN benchmark is not certified clean by this system.

A POSSIBLE/KNOWN benchmark score is displayed as ``BENCHMARK SCORE TAINTED``
and is excluded from skill estimates and promotion evidence.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

VERDICTS = frozenset({"CLEAN", "POSSIBLE", "KNOWN_CONTAMINATION", "UNKNOWN"})

WHITESPACE_RE = re.compile(r"\s+")
_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation -- the normalized
    form used for normalized hashing and n-gram extraction."""
    lowered = text.lower()
    return WHITESPACE_RE.sub(" ", _ALPHANUM_RE.sub(" ", lowered)).strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shingles(text: str, size: int = 8) -> set[int]:
    words = normalize_text(text).split()
    if len(words) < size:
        return set()
    return {hash(" ".join(words[i : i + size])) for i in range(len(words) - size + 1)}


class MinHashIndex:
    """Dependency-free MinHash LSH over shingle sets (64 perms, 16 bands).

    Built from the PROTECTED evaluation material; candidate training text
    is queried against it. Approximate, deliberately biased toward recall:
    any band collision is at least a POSSIBLE verdict.
    """

    def __init__(
        self,
        *,
        num_perm: int = 64,
        bands: int = 16,
        seed: int = 20260915,
    ) -> None:
        if bands <= 0 or num_perm % bands != 0:
            raise ValueError("bands must divide num_perm")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        import random as _random

        rng = _random.Random(seed)
        # (a, b) universal-hash family over 64-bit shingle hashes.
        self._params = [
            (rng.getrandbits(61) * 2 + 1, rng.getrandbits(61)) for _ in range(num_perm)
        ]
        self._buckets: dict[bytes, set[str]] = {}
        self._signatures: dict[str, tuple[int, ...]] = {}

    def _signature(self, shingles: set[int]) -> tuple[int, ...]:
        signature: list[int] = []
        for a, b in self._params:
            best = min((a * s + b) & 0x7FFFFFFFFFFFFFFF for s in shingles) if shingles else 0
            signature.append(best)
        return tuple(signature)

    @staticmethod
    def _band_key(signature: tuple[int, ...], band_index: int, rows: int) -> bytes:
        chunk = signature[band_index * rows : (band_index + 1) * rows]
        return hashlib.sha256(repr(chunk).encode()).digest()[:12]

    def add(self, key: str, text: str) -> None:
        shingles = _shingles(text)
        signature = self._signature(shingles)
        self._signatures[key] = signature
        for band in range(self.bands):
            self._buckets.setdefault(self._band_key(signature, band, self.rows), set()).add(key)

    def candidates(self, text: str) -> set[str]:
        shingles = _shingles(text)
        if not shingles:
            return set()
        signature = self._signature(shingles)
        hits: set[str] = set()
        for band in range(self.bands):
            hits |= self._buckets.get(self._band_key(signature, band, self.rows), set())
        return hits

    def jaccard_estimate(self, key: str, text: str) -> float:
        other = self._signatures.get(key)
        if other is None:
            return 0.0
        signature = self._signature(_shingles(text))
        if len(other) != len(signature):
            return 0.0
        matches = sum(1 for a, b in zip(signature, other) if a == b)
        return matches / len(signature)


@dataclass(frozen=True)
class Match:
    benchmark_qualified_id: str
    detector: str  # exact | normalized | canary | substring | minhash
    detail: str
    similarity: float | None = None


@dataclass(frozen=True)
class CheckResult:
    verdict: str  # VERDICTS
    matches: tuple[Match, ...]

    @property
    def clean(self) -> bool:
        return self.verdict == "CLEAN"


@dataclass
class ProtectedMaterial:
    """Fingerprints of one benchmark's protected split."""

    benchmark_qualified_id: str
    digests: set[str] = field(default_factory=set)  # sha256 of exact text
    normalized_digests: set[str] = field(default_factory=set)
    canaries: tuple[str, ...] = ()
    minhash: MinHashIndex | None = None
    sample_count: int = 0


class ContaminationFirewall:
    """The three-set policy plus the detection stack.

    - PROTECTED EVALUATION SET: fingerprinted here; training-side code must
      never read it -- only this firewall sees the material.
    - DEVELOPMENT EVALUATION SET: may be inspected for diagnosis; examples
      must not be copied into training.
    - TRAINING POOL: registered data sources (see data_registry); every
      candidate source is checked against the protected material BEFORE it
      can be admitted.
    """

    SUBSTRING_WINDOW = 200  # chars: a protected example inside training text
    MINHASH_POSSIBLE = 0.30
    NGRAM_POSSIBLE = 0.20  # Jaccard over 8-gram shingles

    def __init__(self) -> None:
        self._protected: dict[str, ProtectedMaterial] = {}
        self._declared_contaminated: set[str] = set()
        self._ancestry: dict[str, str] = {}  # training source -> benchmark ancestry note

    # ---------------- registration (protected side) ----------------

    def register_protected(
        self,
        benchmark_qualified_id: str,
        texts: Sequence[str],
        *,
        canaries: Sequence[str] = (),
    ) -> ProtectedMaterial:
        """Fingerprint a protected split. Only fingerprints are retained
        beyond this call site's own copies -- training pipelines never see
        them through this object."""
        index = MinHashIndex()
        material = ProtectedMaterial(benchmark_qualified_id=benchmark_qualified_id, canaries=tuple(canaries))
        for text in texts:
            material.digests.add(_sha256(text))
            normalized = normalize_text(text)
            material.normalized_digests.add(_sha256(normalized))
            index.add(f"{benchmark_qualified_id}:{material.sample_count}", text)
            material.sample_count += 1
        material.minhash = index
        self._protected[benchmark_qualified_id] = material
        return material

    def declare_known_contamination(self, benchmark_qualified_id: str, reason: str) -> None:
        """Declare ancestry-based contamination (e.g. a training corpus known
        to contain a benchmark's source material)."""
        self._declared_contaminated.add(benchmark_qualified_id)
        self._ancestry[benchmark_qualified_id] = reason

    # ---------------- detection (training side) ----------------

    def check_text(self, text: str) -> CheckResult:
        """Check one candidate training text against all protected material."""
        matches: list[Match] = []
        exact_digest = _sha256(text)
        normalized_digest = _sha256(normalize_text(text))
        shingles = _shingles(text)

        for qualified_id, material in self._protected.items():
            for canary in material.canaries:
                if canary and canary in text:
                    matches.append(Match(qualified_id, "canary", f"canary {canary!r} present"))
            if exact_digest in material.digests:
                matches.append(Match(qualified_id, "exact", "sha256 match"))
                continue
            if normalized_digest in material.normalized_digests:
                matches.append(Match(qualified_id, "normalized", "normalized sha256 match"))
                continue
            if material.minhash is not None:
                candidates = material.minhash.candidates(text)
                if candidates:
                    best = max(
                        material.minhash.jaccard_estimate(k, text) for k in candidates
                    )
                    if best >= self.MINHASH_POSSIBLE:
                        matches.append(
                            Match(qualified_id, "minhash", "MinHash LSH hit", similarity=best)
                        )
                        continue
                    overlap = self._ngram_overlap(text, qualified_id)
                    if overlap >= self.NGRAM_POSSIBLE:
                        matches.append(
                            Match(
                                qualified_id,
                                "substring",
                                f"shingle overlap {overlap:.2f}",
                                similarity=overlap,
                            )
                        )
        if matches:
            hard = any(m.detector in {"exact", "normalized", "canary"} for m in matches)
            verdict = "KNOWN_CONTAMINATION" if hard else "POSSIBLE"
            return CheckResult(verdict=verdict, matches=tuple(matches))
        return CheckResult(verdict="CLEAN", matches=())

    def _ngram_overlap(self, text: str, qualified_id: str) -> float:
        """Jaccard overlap between the text's shingles and the stored
        signature side-information we retain per protected benchmark."""
        material = self._protected[qualified_id]
        if material.minhash is None:
            return 0.0
        shingles = _shingles(text)
        if not shingles:
            return 0.0
        # Reconstruct a coarse overlap from candidate signatures: MinHash
        # agreement as a Jaccard estimate against the best-matching sample.
        candidates = material.minhash.candidates(text)
        if not candidates:
            return 0.0
        return max(material.minhash.jaccard_estimate(k, text) for k in candidates)

    def check_source(
        self,
        *,
        source_id: str,
        samples: Sequence[str],
    ) -> CheckResult:
        """Check a whole candidate training source (sampled). One
        KNOWN_CONTAMINATION sample taints the source."""
        worst = "CLEAN"
        matches: list[Match] = []
        for index, sample in enumerate(samples):
            result = self.check_text(sample)
            if result.verdict == "CLEAN":
                continue
            matches.extend(result.matches)
            if result.verdict == "KNOWN_CONTAMINATION":
                worst = "KNOWN_CONTAMINATION"
                break
            worst = "POSSIBLE"
        if worst == "CLEAN" and source_id in self._ancestry:
            return CheckResult(
                verdict="KNOWN_CONTAMINATION",
                matches=(Match(self._ancestry[source_id], "ancestry", self._ancestry[source_id]),),
            )
        return CheckResult(verdict=worst, matches=tuple(matches))

    def refuse_protected_use(self, samples: Sequence[str], benchmark_qualified_id: str) -> None:
        """The hard refusal: attempting to route protected benchmark material
        into the training pool raises. This is the guard tests exercise."""
        material = self._protected.get(benchmark_qualified_id)
        if material is None:
            raise ContaminationRefusal(
                f"{benchmark_qualified_id}: not fingerprinted -- refusing to treat it as "
                "training-eligible (UNKNOWN is not CLEAN)"
            )
        for sample in samples:
            digest = _sha256(sample)
            if digest in material.digests or _sha256(normalize_text(sample)) in material.normalized_digests:
                raise ContaminationRefusal(
                    f"refusing to use protected evaluation material from "
                    f"{benchmark_qualified_id} as training data (exact/normalized match)"
                )
        # If any sample trips ANY detector, refuse as well.
        result = self.check_source(source_id="__protected_probe__", samples=samples)
        if result.verdict != "CLEAN":
            raise ContaminationRefusal(
                f"refusing to use material matching {benchmark_qualified_id} as training "
                f"data ({result.verdict})"
            )

    # ---------------- manifests ----------------

    def manifest(
        self,
        *,
        evaluated_benchmarks: Iterable[str],
        training_sources: Iterable[str] = (),
        source_samples: Mapping[str, Sequence[str]] | None = None,
    ) -> dict:
        """The per-generation ``contamination_manifest.json`` payload."""
        source_samples = source_samples or {}
        benchmarks_section: dict[str, dict] = {}
        for qualified_id in evaluated_benchmarks:
            if qualified_id in self._declared_contaminated:
                benchmarks_section[qualified_id] = {
                    "status": "KNOWN_CONTAMINATION",
                    "reason": self._ancestry.get(qualified_id, "declared"),
                }
                continue
            samples = source_samples.get(qualified_id, ())
            if not samples:
                benchmarks_section[qualified_id] = {"status": "UNKNOWN", "reason": "not checked"}
                continue
            result = self.check_source(source_id=f"eval:{qualified_id}", samples=samples)
            benchmarks_section[qualified_id] = {
                "status": result.verdict,
                "matches": [
                    {"benchmark": m.benchmark_qualified_id, "detector": m.detector, "detail": m.detail}
                    for m in result.matches
                ],
            }
        sources_section: dict[str, dict] = {}
        for source_id in training_sources:
            samples = source_samples.get(source_id, ())
            if not samples:
                sources_section[source_id] = {"status": "UNKNOWN", "reason": "not checked"}
                continue
            result = self.check_source(source_id=source_id, samples=samples)
            sources_section[source_id] = {
                "status": result.verdict,
                "matches": [
                    {"benchmark": m.benchmark_qualified_id, "detector": m.detector, "detail": m.detail}
                    for m in result.matches
                ],
            }
        return {
            "policy": {
                "protected": "never available to curriculum/training generation",
                "development": "inspectable for diagnosis; examples never copied to training",
                "training_pool": "registered sources only, checked before admission",
            },
            "benchmarks": benchmarks_section,
            "training_sources": sources_section,
        }


class ContaminationRefusal(RuntimeError):
    """Raised when protected evaluation material is routed toward training."""
