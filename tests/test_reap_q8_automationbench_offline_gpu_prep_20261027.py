"""CPU-only source transformation tests: never execute a model or Kaggle job."""

import ast
import hashlib
import json

import pytest

from chowder_batch import reap_q8_automationbench_kaggle_20261026 as original
from chowder_batch import reap_q8_automationbench_offline_gpu_prep_20261027 as prep


def source():
    config = {"profile": "six", "native_flags": list(original.PROFILES["six"]["flags"])}
    return original.KERNEL_TEMPLATE.replace("__LOCKED_CONFIG_LITERAL__", repr(json.dumps(config)))


def installer():
    raw = prep.PREFLIGHT_SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prep.PREFLIGHT_SHA
    return raw.decode("utf-8")


@pytest.mark.parametrize("profile", ["six", "guard2"])
def test_patches_only_offline_bootstrap_and_preserves_original_official_protocol(profile):
    before = source()
    after = prep.patch_kernel(before, installer(), profile=profile)
    ast.parse(after)
    assert "OFFLINE_PACKAGES, offline_proof = bootstrap_offline(archive)" in after
    assert "namespace['install_and_smoke'](wheels, manifest, archive, OFFLINE_TARGET)" in after
    assert "len(list(wheels.glob('*.whl'))) != 108" in after
    assert "'PYTHONPATH':str(OFFLINE_PACKAGES)+os.pathsep+str(source)" in after
    assert "shutil.rmtree(owned)" in after
    assert "--no-index" in after and "--require-hashes" in after
    for unchanged in (
        "'--toolset','api','--domains','all'",
        "'--max-concurrent','1','--max-steps',str(CFG['max_steps'])",
        "'--no-ensure-complete','--export-json',str(result)",
        "R['native_argv'] = [str(executable),*CFG['native_flags']",
        "R['status'] = 'RAW_Q8_OFFICIAL_PUBLIC_TASKS_REQUIRE_INDEPENDENT_ASSERTION_AUDIT'",
        "if evaluator.returncode: raise RuntimeError('OFFICIAL evaluator failed; NO score')",
    ):
        assert unchanged in before and unchanged in after
    assert after.count("def bootstrap_offline(archive):") == 1


def test_rejects_modified_cpu_bootstrap_source():
    with pytest.raises(ValueError, match="source drift"):
        prep.patch_kernel(source(), installer() + "\n# modified", profile="six")


def test_rejects_duplicate_or_missing_original_source_anchor():
    with pytest.raises(ValueError, match="missing or ambiguous"):
        prep.patch_kernel(source().replace("def snapshot():\n", ""), installer(), profile="six")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        prep.patch_kernel(source() + "\ndef snapshot():\n    pass\n", installer(), profile="six")


def test_rejects_unregistered_profile():
    with pytest.raises(ValueError, match="unsupported arm"):
        prep.patch_kernel(source(), installer(), profile="full600")
