import hashlib
from dataclasses import replace

from chowder.incident import compute_fingerprint
from chowder.investigation import RemediationOutcome, RemediationRecord, RemediationRegistry
from chowder.probes import (
    ArtifactIntegrityProbe,
    HardwareCompatibilityProbe,
    InstalledPackageProbe,
    KnownWorkingConfigProbe,
    ProbeContext,
)

from fixtures_incidents import PEFT_KBIT_PREP_OOM, WRONG_ACCELERATOR_PROVISIONED


def _context(capture, registry=None) -> ProbeContext:
    return ProbeContext(
        capture=capture,
        fingerprint=compute_fingerprint(capture),
        registry=registry or RemediationRegistry(),
    )


def test_installed_package_probe_reports_captured_packages():
    context = _context(PEFT_KBIT_PREP_OOM)
    result = InstalledPackageProbe().run(context)
    assert result.observation["installed_packages"] == dict(
        PEFT_KBIT_PREP_OOM.environment.installed_packages
    )


def test_hardware_compatibility_probe_flags_real_sm60_incident():
    """Grounded against the real P100/T4x2 incident: the installed PyTorch
    build's floor is sm_70, and this fixture's environment carries the
    unintended sm_60 P100 that caused the actual failure."""
    context = _context(WRONG_ACCELERATOR_PROVISIONED)
    result = HardwareCompatibilityProbe().run(context)
    assert result.observation["detected_sm"] == 60
    assert result.observation["compatible"] is False


def test_hardware_compatibility_probe_passes_supported_hardware():
    context = _context(PEFT_KBIT_PREP_OOM)  # sm_75 Tesla T4
    result = HardwareCompatibilityProbe().run(context)
    assert result.observation["detected_sm"] == 75
    assert result.observation["compatible"] is True


def test_known_working_config_probe_surfaces_resolved_history_only():
    capture = PEFT_KBIT_PREP_OOM
    fingerprint = compute_fingerprint(capture)
    resolved = RemediationRecord(
        remediation_id="resolved-fix",
        fingerprint_sha256=fingerprint.fingerprint_sha256,
        signature_kind=fingerprint.signature_kind,
        description="worked",
        config_patch={"allocator_conf": "expandable_segments:True"},
        outcome=RemediationOutcome.RESOLVED,
        attempts_used=1,
        gpu_hours_spent=0.1,
    )
    failed = RemediationRecord(
        remediation_id="failed-fix",
        fingerprint_sha256=fingerprint.fingerprint_sha256,
        signature_kind=fingerprint.signature_kind,
        description="did not work",
        config_patch={"batch_size": 1},
        outcome=RemediationOutcome.DID_NOT_RESOLVE,
        attempts_used=1,
        gpu_hours_spent=0.1,
    )
    registry = RemediationRegistry(records=(resolved, failed))
    result = KnownWorkingConfigProbe().run(_context(capture, registry))
    assert result.observation["resolved_remediation_ids"] == ("resolved-fix",)
    assert result.observation["resolved_config_patches"] == (
        {"allocator_conf": "expandable_segments:True"},
    )


def test_known_working_config_probe_empty_when_no_history():
    result = KnownWorkingConfigProbe().run(_context(PEFT_KBIT_PREP_OOM))
    assert result.observation["resolved_remediation_ids"] == ()


def test_artifact_integrity_probe_reports_match(tmp_path):
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"real checkpoint bytes")
    expected = hashlib.sha256(b"real checkpoint bytes").hexdigest()

    # partial_artifact_ref is frozen on FailureCapture; build a variant with
    # it set rather than mutating the shared fixture.
    capture = replace(PEFT_KBIT_PREP_OOM, partial_artifact_ref=str(artifact))
    context = _context(capture)

    result = ArtifactIntegrityProbe(expected_sha256=expected).run(context)
    assert result.observation["checked"] is True
    assert result.observation["matches"] is True


def test_artifact_integrity_probe_reports_corruption(tmp_path):
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"corrupted bytes, not what was downloaded")
    expected = hashlib.sha256(b"real checkpoint bytes").hexdigest()

    capture = replace(PEFT_KBIT_PREP_OOM, partial_artifact_ref=str(artifact))
    context = _context(capture)

    result = ArtifactIntegrityProbe(expected_sha256=expected).run(context)
    assert result.observation["checked"] is True
    assert result.observation["matches"] is False


def test_artifact_integrity_probe_handles_missing_reference():
    result = ArtifactIntegrityProbe(expected_sha256="does-not-matter").run(
        _context(PEFT_KBIT_PREP_OOM)  # partial_artifact_ref is None on this fixture
    )
    assert result.observation["checked"] is False
