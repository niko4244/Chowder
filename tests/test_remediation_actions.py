import pytest

from chowder.remediation_actions import (
    ConfigRemediationAdapter,
    RemediationActionKind,
    UnsupportedRemediationAction,
    action_from_config_patch,
    require_capability,
)


def test_dependency_pin_is_not_misrepresented_as_training_parameter():
    action = action_from_config_patch({"transformers_version": "5.10.2"})
    assert action.kind is RemediationActionKind.PIN_DEPENDENCY
    assert action.parameters == {"package": "transformers", "version": "5.10.2"}


def test_kaggle_machine_shape_requires_hardware_reprovision_capability():
    action = action_from_config_patch(
        {"kernel_metadata.machine_shape": "NvidiaTeslaT4"}
    )
    assert action.kind is RemediationActionKind.REPROVISION_HARDWARE
    with pytest.raises(UnsupportedRemediationAction, match="does not support"):
        require_capability(ConfigRemediationAdapter(), action)


def test_hf_retry_is_a_download_action_not_a_generic_config_edit():
    action = action_from_config_patch({"resume_download": True})
    assert action.kind is RemediationActionKind.RETRY_DOWNLOAD
    with pytest.raises(UnsupportedRemediationAction):
        ConfigRemediationAdapter().apply(action, resolved_config={"backend": {}})


def test_allocator_fix_compiles_to_environment_action():
    action = action_from_config_patch(
        {"allocator_conf": "expandable_segments:True"}
    )
    assert action.kind is RemediationActionKind.SET_ENVIRONMENT
    updated = ConfigRemediationAdapter().apply(
        action,
        resolved_config={"backend": {"runtime": {}}},
    )
    assert (
        updated["backend"]["runtime"]["environment"]["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True"
    )


def test_max_length_is_applied_to_backend_not_environment():
    action = action_from_config_patch({"max_length": 1024})
    assert action.kind is RemediationActionKind.SET_TRAINING_PARAMETER
    updated = ConfigRemediationAdapter().apply(
        action,
        resolved_config={"backend": {"training": {"batch_size": 1}}},
    )
    assert updated["backend"]["max_length"] == 1024
    assert updated["backend"]["training"]["batch_size"] == 1


def test_device_map_is_a_runtime_option():
    action = action_from_config_patch({"device_map": "balanced_low_0"})
    updated = ConfigRemediationAdapter().apply(
        action,
        resolved_config={"backend": {}},
    )
    assert updated["backend"]["runtime"]["device_map"] == "balanced_low_0"


def test_unknown_patch_fails_closed():
    with pytest.raises(UnsupportedRemediationAction, match="no typed remediation action"):
        action_from_config_patch({"totally_unknown_knob": True})


def test_multi_action_patch_fails_closed_instead_of_partially_applying():
    with pytest.raises(UnsupportedRemediationAction, match="exactly one typed action"):
        action_from_config_patch({"max_length": 1024, "resume_download": True})


def test_adapter_does_not_mutate_source_config():
    config = {"backend": {"training": {"batch_size": 2}}}
    action = action_from_config_patch({"batch_size": 1})
    updated = ConfigRemediationAdapter().apply(action, resolved_config=config)
    assert config == {"backend": {"training": {"batch_size": 2}}}
    assert updated["backend"]["training"]["batch_size"] == 1
