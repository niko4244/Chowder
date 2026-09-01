from chowder.calibration import (
    _median_gbps,
    calibrate_hardware,
    calibrate_host_memory,
    calibrate_storage,
)


def test_median_gbps_uses_median_duration():
    value = _median_gbps(1024**3, [2.0, 1.0, 3.0])
    assert value == 0.5


def test_storage_calibration_uses_temp_file_and_cleans_up(tmp_path):
    result = calibrate_storage(tmp_path, sample_mib=1, passes=1)
    assert result.read_gbps > 0
    assert result.write_gbps > 0
    assert not list(tmp_path.glob('.chowder-cal-*'))


def test_host_memory_calibration_reports_effective_copy_bandwidth():
    result = calibrate_host_memory(sample_mib=1, passes=2)
    assert result.copy_gbps > 0


def test_combined_calibration_can_skip_cuda(tmp_path):
    result = calibrate_hardware(tmp_path, sample_mib=1, passes=1, include_cuda=False)
    assert result.storage is not None
    assert result.host_memory is not None
    assert result.cuda is None
