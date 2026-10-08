"""Declared inputs as a Kaggle dataset: the bytes the attempt will verify.

:class:`~chowder.growth.kaggle_compute.KaggleComputeBackend` dispatches a
kernel that re-hashes every declared input at its declared kernel path;
something has to put those bytes where the kernel can mount them. This module
is that something: it stages the declared inputs (one flat file per declared
name), uploads them as a Kaggle dataset, and returns the dataset reference
plus the candidate kernel paths the declared inputs may appear at. Kaggle
mounts a dataset at ``/kaggle/input/<slug>/`` and, when that name is already
taken by another dataset, at ``/kaggle/input/<owner>/<slug>/`` -- the CLI's
own documented rule -- so the kernel resolves the declared bytes by digest
across the candidates rather than assuming one
(``chowder.growth.kaggle_kernel.run_kernel_job``).

The dataset is content-addressed: its slug carries a digest of the declared
input set, so re-publishing the same declaration targets the same dataset and
a changed input set can never silently reuse the previous one. Publishing is
``datasets create`` first, then ``datasets version`` when the dataset already
exists; whichever call fails last fails the publish with *both* outputs
attached. The publisher deliberately does not parse ``datasets files`` or
``datasets status`` output: the guarantee that the kernel mounted the
declared bytes is the kernel's own re-hash, compared by the backend, and a
listing parser would be a weaker claim dressed in the same vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .compute_backend import (
    ComputeBackendRefusal,
    DeclaredInput,
    verify_declared_inputs,
)
from .kaggle_cli_transport import KAGGLE_CLI_NAME, Runner, default_runner

__all__ = [
    "KAGGLE_INPUT_SCHEMA",
    "KAGGLE_INPUT_EMPTY",
    "KAGGLE_INPUT_BASENAMES_COLLIDE",
    "KAGGLE_INPUT_BYTES_CHANGED",
    "KAGGLE_INPUT_CREATED",
    "KAGGLE_INPUT_VERSIONED",
    "INPUT_DATASET_FILE",
    "KaggleInputError",
    "KaggleInputDataset",
    "KaggleInputPublisher",
    "declared_inputs_digest",
]

KAGGLE_INPUT_SCHEMA = "KAGGLE_INPUT_SCHEMA"
KAGGLE_INPUT_EMPTY = "KAGGLE_INPUT_EMPTY"
KAGGLE_INPUT_BASENAMES_COLLIDE = "KAGGLE_INPUT_BASENAMES_COLLIDE"
KAGGLE_INPUT_BYTES_CHANGED = "KAGGLE_INPUT_BYTES_CHANGED"

#: What the create/version call did, recorded so a run can name it.
KAGGLE_INPUT_CREATED = "created"
KAGGLE_INPUT_VERSIONED = "versioned"

#: The metadata file every Kaggle dataset folder must carry.
INPUT_DATASET_FILE = "dataset-metadata.json"

#: Kaggle's documented slug rule: 3-50 chars of [a-z0-9-], starting and
#: ending alphanumeric.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$")
#: Kaggle's documented title rule (create): 6-50 characters.
_TITLE_MIN, _TITLE_MAX = 6, 50
#: The mount root every Kaggle kernel sees datasets under.
_MOUNT_ROOT = "/kaggle/input"


class KaggleInputError(RuntimeError):
    """The declared inputs could not be published as a dataset.

    Raised, never returned: an upload that did not happen cannot be recorded
    as a dataset reference, because a reference that does not exist would make
    every later mount refusal a mystery.
    """


def declared_inputs_digest(inputs: Sequence[DeclaredInput]) -> str:
    """The content identity of one declared input set, order-independent."""
    payload = [
        {"name": entry.name, "sha256": entry.sha256, "bytes": entry.bytes}
        for entry in sorted(inputs, key=lambda entry: entry.name)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class KaggleInputDataset:
    """One published dataset version, and where its files may be mounted."""

    reference: str
    slug: str
    content_digest: str
    operation: str
    #: Declared input name -> file name inside the dataset.
    files: Mapping[str, str]
    #: Declared input name -> candidate kernel paths, in resolution order.
    input_paths: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "slug": self.slug,
            "content_digest": self.content_digest,
            "operation": self.operation,
            "files": dict(self.files),
            "input_paths": {
                name: list(paths) for name, paths in self.input_paths.items()
            },
        }


@dataclass
class KaggleInputPublisher:
    """The official ``kaggle`` CLI behind the declared-input upload."""

    owner: str
    runner: Runner = default_runner
    command: tuple[str, ...] = (KAGGLE_CLI_NAME,)
    staging_root: Path | None = None
    #: The dataset's declared license. Defaults to Kaggle's ``unknown``: the
    #: publisher will not stamp a license the campaign never chose, and the
    #: declared material carries its own provenance.
    license_name: str = "unknown"

    def __post_init__(self) -> None:
        if not str(self.owner).strip():
            raise KaggleInputError(
                "owner must be the Kaggle username the CLI is authenticated as"
            )
        if not self.command:
            raise KaggleInputError("command must name the kaggle executable")
        if not str(self.license_name).strip():
            raise KaggleInputError("license_name must be a declared license name")

    # ------------------------------------------------------------------
    # the upload
    # ------------------------------------------------------------------

    def publish(
        self,
        inputs: Sequence[DeclaredInput],
        *,
        dataset: str,
        title: str,
        version_message: str = "",
    ) -> KaggleInputDataset:
        """Upload one declared input set as a dataset; return its mount surface.

        ``dataset`` is the base slug the campaign declares; the published slug
        appends the declared-input digest, so the dataset identity *is* the
        content identity.
        """
        entries = tuple(inputs)
        if not entries:
            raise ComputeBackendRefusal(
                KAGGLE_INPUT_EMPTY,
                "there is nothing to publish: a dataset with no declared input "
                "would mount an empty surface an attempt cannot verify",
            )
        owner = str(self.owner).strip()
        digest = declared_inputs_digest(entries)
        slug = f"{str(dataset).strip()}-{digest[:12]}"
        if not _SLUG_RE.match(slug):
            raise ComputeBackendRefusal(
                KAGGLE_INPUT_SCHEMA,
                f"derived dataset slug {slug!r} is not a valid Kaggle dataset "
                "id (3-50 chars of [a-z0-9-], starting and ending "
                "alphanumeric); choose a base slug the digest suffix can "
                "complete",
            )
        title_text = str(title).strip()
        if not _TITLE_MIN <= len(title_text) <= _TITLE_MAX:
            raise ComputeBackendRefusal(
                KAGGLE_INPUT_SCHEMA,
                f"dataset title must be {_TITLE_MIN}-{_TITLE_MAX} characters "
                f"(Kaggle's rule), got {len(title_text)}",
            )

        # R2's pre-push half, applied to the upload: the bytes about to be
        # shipped are the bytes the campaign declared.
        verify_declared_inputs(entries)

        files: dict[str, str] = {}
        input_paths: dict[str, tuple[str, ...]] = {}
        taken: dict[str, str] = {}
        for entry in entries:
            filename = Path(entry.path).name
            if not filename:
                raise ComputeBackendRefusal(
                    KAGGLE_INPUT_SCHEMA,
                    f"declared input {entry.name!r} has no file name to publish "
                    f"under: {entry.path!r}",
                )
            existing = taken.get(filename)
            if existing is not None:
                raise ComputeBackendRefusal(
                    KAGGLE_INPUT_BASENAMES_COLLIDE,
                    f"declared inputs {existing!r} and {entry.name!r} both name "
                    f"the file {filename!r}; a flat dataset cannot carry both, "
                    "and renaming one would make the mounted path a lie",
                )
            taken[filename] = entry.name
            files[entry.name] = filename
            input_paths[entry.name] = (
                f"{_MOUNT_ROOT}/{slug}/{filename}",
                f"{_MOUNT_ROOT}/{owner}/{slug}/{filename}",
            )

        staging = self._stage(entries, files, slug, owner, title_text)
        operation = self._upload(staging, slug, digest, version_message)
        return KaggleInputDataset(
            reference=f"{owner}/{slug}",
            slug=slug,
            content_digest=digest,
            operation=operation,
            files=files,
            input_paths=input_paths,
        )

    # ------------------------------------------------------------------
    # staging and the CLI
    # ------------------------------------------------------------------

    def _stage(
        self,
        entries: Sequence[DeclaredInput],
        files: Mapping[str, str],
        slug: str,
        owner: str,
        title: str,
    ) -> Path:
        base = (
            Path(self.staging_root)
            if self.staging_root is not None
            else Path(tempfile.mkdtemp(prefix="chowder-kaggle-inputs-"))
        )
        staging = base / slug
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for entry in entries:
            _copy_verified(entry, staging / files[entry.name])
        (staging / INPUT_DATASET_FILE).write_text(
            json.dumps(
                {
                    "title": title,
                    "id": f"{owner}/{slug}",
                    "licenses": [{"name": str(self.license_name).strip()}],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return staging

    def _upload(
        self, staging: Path, slug: str, digest: str, version_message: str
    ) -> str:
        try:
            self._kaggle("datasets", "create", "-p", str(staging), "-q")
            return KAGGLE_INPUT_CREATED
        except KaggleInputError as create_error:
            # The content-addressed slug already exists: version it with the
            # same bytes rather than refusing a re-run of the same declaration.
            message = str(version_message).strip() or (
                f"chowder declared inputs {digest[:12]}"
            )
            try:
                self._kaggle(
                    "datasets", "version", "-p", str(staging), "-m", message, "-q"
                )
            except KaggleInputError as version_error:
                raise KaggleInputError(
                    f"the dataset {slug!r} could neither be created nor "
                    f"versioned; create: {create_error}; version: {version_error}"
                ) from version_error
            return KAGGLE_INPUT_VERSIONED

    def _kaggle(self, *args: str) -> str:
        argv = [*self.command, *args]
        try:
            result = self.runner(argv)
        except FileNotFoundError as exc:
            raise KaggleInputError(
                f"the kaggle CLI was not found ({exc}); install the official "
                "`kaggle` package and authenticate it before publishing inputs"
            ) from exc
        except Exception as exc:
            raise KaggleInputError(
                f"`{' '.join(argv)}` could not run: {exc}"
            ) from exc
        if result.returncode != 0:
            raise KaggleInputError(
                f"`{' '.join(argv)}` exited {result.returncode}: "
                f"{(result.stderr or result.stdout or '').strip()[-2000:]}"
            )
        return result.stdout or ""


def _copy_verified(entry: DeclaredInput, target: Path) -> None:
    """Copy one declared input, hashing the bytes that actually land on disk.

    ``verify_declared_inputs`` hashed the source a moment ago; this hashes the
    copy, so a file that moved in between refuses instead of being uploaded as
    bytes no declaration ever bound.
    """
    digest = hashlib.sha256()
    size = 0
    with Path(entry.path).open("rb") as source, target.open("wb") as sink:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            sink.write(chunk)
    sha256 = digest.hexdigest()
    if sha256 != entry.sha256 or size != entry.bytes:
        raise ComputeBackendRefusal(
            KAGGLE_INPUT_BYTES_CHANGED,
            f"declared input {entry.name!r} changed between declaration and "
            f"upload: copied {sha256} ({size} bytes), declared "
            f"{entry.sha256} ({entry.bytes} bytes)",
        )
