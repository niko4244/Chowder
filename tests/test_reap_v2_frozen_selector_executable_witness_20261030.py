"""CPU-only adversarial tests: no model, F: access or production approvals."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import pytest

from chowder_batch import reap_gates, reap_select, reap_serve, reap_v2_source_identity
from chowder_batch import reap_v2_frozen_gate_evidence_bridge_20261027 as bridge
from chowder_batch import reap_v2_frozen_selector_executable_witness_20261030 as witness


def test_original_frozen_selector_source_and_live_functions_are_authenticated():
    report = witness.verify_frozen_selector_executable()
    assert report["original_selector_unchanged"] is True
    assert {key: item["verified_functions"] for key, item in report["modules"].items()} == {
        "reap_select": 12, "reap_gates": 14, "reap_v2_source_identity": 2,
        "reap_serve": 18,
    }
    for name, expected in witness.FROZEN_SOURCE_SHA256.items():
        source = Path(witness.__file__).parent / f"{name}.py"
        assert hashlib.sha256(witness._git_lf_source(source.read_bytes())).hexdigest() == expected
    assert report["gen0_status"] == "BLOCKED"
    assert report["numeric_runtime"]["verified_statistical_functions"] == 29
    assert report["json_runtime"]["verified_functions_and_methods"] == 22
    assert report["chosen"] is None and report["release_approved"] is False


@pytest.mark.parametrize("module,attribute", [
    (reap_select, "evaluate"),
    (reap_select, "select"),
    (reap_select, "_input_digests"),
    (reap_select, "_status"),
    (reap_gates, "compute"),
    (reap_gates, "_gate0"),
    (reap_gates, "_speed"),
    (reap_serve, "parse_offload"),
    (reap_v2_source_identity, "frozen_v2_source_identity"),
])
def test_executing_function_swap_refused(monkeypatch, module, attribute):
    monkeypatch.setattr(module, attribute, lambda *_args, **_kwargs: "PASS")
    with pytest.raises(witness.SelectorExecutableRefused, match="executing function"):
        witness.verify_frozen_selector_executable()


@pytest.mark.parametrize("module,attribute,value", [
    (reap_gates, "G0_TOL", 1.0),
    (reap_gates, "G1_THRESH", -1.0),
    (reap_select, "MARGIN", -0.99),
    (reap_select, "FROZEN_SERVE_FLAGS", ["--fit", "off"]),
    (reap_v2_source_identity, "FROZEN_V2_BASE_ROWS", 1),
    (reap_v2_source_identity, "MAX_ADMITTED_BYTES", 1 << 30),
    (reap_serve, "FLAGS", ["--fit", "off"]),
])
def test_runtime_policy_constant_mutation_refused(monkeypatch, module, attribute, value):
    monkeypatch.setattr(module, attribute, value)
    with pytest.raises(witness.SelectorExecutableRefused, match="frozen|source"):
        witness.verify_frozen_selector_executable()


def test_in_place_model_reference_mutation_refused(monkeypatch):
    monkeypatch.setitem(reap_select.FROZEN_REFERENCE_MODEL_SHA256,
                        "dense_v2", "0" * 64)
    with pytest.raises(witness.SelectorExecutableRefused, match="FROZEN_REFERENCE_MODEL_SHA256"):
        witness.verify_frozen_selector_executable()


def test_in_place_canonical_source_mutation_refused(monkeypatch):
    monkeypatch.setitem(reap_gates.FROZEN_SOURCE_SHA256, "gsm8k", "0" * 64)
    with pytest.raises(witness.SelectorExecutableRefused, match="FROZEN_SOURCE_SHA256"):
        witness.verify_frozen_selector_executable()


@pytest.mark.parametrize("module,attribute,replacement", [
    (statistics, "stdev", lambda *_args: 0.0),
    (statistics, "fmean", lambda *_args: 1.0),
    (statistics, "_ss", lambda *_args: (float, 0, 0, 50)),
    (math, "sqrt", lambda *_args: 1.0),
    (statistics, "fsum", lambda *_args: 0.0),
    (json, "loads", lambda *_args, **_kwargs: {}),
])
def test_ambient_numeric_and_json_dependency_hijack_refused(
    monkeypatch, module, attribute, replacement,
):
    monkeypatch.setattr(module, attribute, replacement)
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="statistical|dependency|JSON function"):
        witness.verify_frozen_selector_executable()


def test_ambient_stdlib_module_alias_swap_refused(monkeypatch):
    monkeypatch.setattr(reap_gates, "statistics", SimpleNamespace(
        stdev=lambda *_args: 0.0, fmean=lambda *_args: 1.0,
    ))
    with pytest.raises(witness.SelectorExecutableRefused, match="module binding changed"):
        witness.verify_frozen_selector_executable()


def test_native_offload_regex_mutation_refused(monkeypatch):
    import re

    monkeypatch.setattr(reap_serve, "_ALL_LAYERS", re.compile(".*"))
    with pytest.raises(witness.SelectorExecutableRefused, match="offload regex changed"):
        witness.verify_frozen_selector_executable()


def test_json_decoder_function_code_hijack_refused(monkeypatch):
    replacement = (lambda *_args, **_kwargs: {"forged": "accepted"}).__code__
    monkeypatch.setattr(json.loads, "__code__", replacement)
    assert json.loads("not-json") == {"forged": "accepted"}
    with pytest.raises(witness.SelectorExecutableRefused, match="JSON function changed"):
        witness.verify_frozen_selector_executable()
    blocked = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert blocked["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert blocked["selection"]["status"] == "BLOCKED"
    assert blocked["chosen"] is None and blocked["release_approved"] is False


def test_json_decoder_class_method_hijack_refused(monkeypatch):
    replacement = (lambda *_args, **_kwargs: {"forged": "accepted"}).__code__
    monkeypatch.setattr(json.decoder.JSONDecoder.decode, "__code__", replacement)
    with pytest.raises(witness.SelectorExecutableRefused, match="JSON method changed"):
        witness.verify_frozen_selector_executable()


def test_json_decoder_in_place_kwdefault_hijack_refused(monkeypatch):
    monkeypatch.setitem(json.decoder.JSONDecoder.__init__.__kwdefaults__,
                        "parse_int", lambda text: 200 if text == "0" else int(text))
    assert json.loads('{"n":0}', object_pairs_hook=dict) == {"n": 200}
    with pytest.raises(witness.SelectorExecutableRefused, match="JSON method changed"):
        witness.verify_frozen_selector_executable()
    result = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert result["selection"]["status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


def test_json_default_native_scanner_replacement_refused(monkeypatch):
    replacement = json.decoder.JSONDecoder()
    replacement.parse_int = lambda *_args: 200
    replacement.scan_once = json.scanner.make_scanner(replacement)
    monkeypatch.setattr(json._default_decoder, "scan_once", replacement.scan_once)
    assert json.loads('{"n":0}') == {"n": 200}
    with pytest.raises(witness.SelectorExecutableRefused, match="decoder binding changed"):
        witness.verify_frozen_selector_executable()


def test_actual_gate1_false_pass_from_stdev_hijack_is_refused(monkeypatch):
    """A changed imported dependency can reverse a real frozen gate result."""
    base = {f"gsm8k-{i}": {"correct": True} for i in range(50)}
    arm = {f"gsm8k-{i}": {"correct": i != 0} for i in range(50)}
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "FAIL"
    monkeypatch.setattr(statistics, "stdev", lambda *_args: 0.0)
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "PASS"
    with pytest.raises(witness.SelectorExecutableRefused, match="statistics.stdev"):
        witness.verify_frozen_selector_executable()
    blocked = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert blocked["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert blocked["selection"]["status"] == "BLOCKED"
    assert blocked["chosen"] is None and blocked["release_approved"] is False


def test_statistical_function_default_mutation_refused(monkeypatch):
    monkeypatch.setattr(statistics.stdev, "__defaults__", (0,))
    with pytest.raises(witness.SelectorExecutableRefused, match="statistics.stdev"):
        witness.verify_frozen_selector_executable()


@pytest.mark.parametrize("attribute,replacement", [
    ("compute", lambda *_args, **_kwargs: {"gate1": "PASS"}),
    ("_load", lambda *_args, **_kwargs: {}),
    ("lower_bound", lambda *_args: 1.0),
])
def test_imported_selector_alias_swap_refused(monkeypatch, attribute, replacement):
    monkeypatch.setattr(reap_select, attribute, replacement)
    with pytest.raises(witness.SelectorExecutableRefused, match="binding changed"):
        witness.verify_frozen_selector_executable()


def test_frozen_default_mutation_refused(monkeypatch):
    monkeypatch.setattr(reap_gates._lb_gate, "__defaults__", (False,))
    with pytest.raises(witness.SelectorExecutableRefused, match="defaults differ"):
        witness.verify_frozen_selector_executable()


def test_loaded_module_path_swap_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(reap_select, "__file__", str(tmp_path / "reap_select.py"))
    with pytest.raises(witness.SelectorExecutableRefused, match="loaded module"):
        witness.verify_frozen_selector_executable()


def test_frozen_source_sha_drift_refused(monkeypatch):
    monkeypatch.setitem(witness.FROZEN_SOURCE_SHA256, "reap_select", "0" * 64)
    with pytest.raises(witness.SelectorExecutableRefused, match="source SHA drift"):
        witness.verify_frozen_selector_executable()


def test_git_line_endings_normalize_only_crlf():
    assert witness._git_lf_source(b"a\nb\n") == b"a\nb\n"
    assert witness._git_lf_source(b"a\r\nb\r\n") == b"a\nb\n"
    with pytest.raises(witness.SelectorExecutableRefused, match="line endings"):
        witness._git_lf_source(b"a\rb\n")


def test_bridge_independently_pins_witness_implementation(monkeypatch):
    bridge._verify_frozen_selector_source()
    monkeypatch.setattr(bridge, "FROZEN_SELECTOR_WITNESS_SHA256", "0" * 64)
    with pytest.raises(bridge.FrozenEvidenceError, match="witness source drift"):
        bridge._verify_frozen_selector_source()


@pytest.mark.parametrize("attribute", ["reap_select", "reap_gates"])
def test_bridge_cannot_replace_original_selector_module_binding(monkeypatch, attribute):
    monkeypatch.setattr(bridge, attribute, SimpleNamespace(
        select=lambda *_args: {"status": "SELECTED", "chosen": "reap128_v2"},
    ))
    with pytest.raises(bridge.FrozenEvidenceError, match="module binding changed"):
        bridge._verify_frozen_selector_source()


@pytest.mark.parametrize("function_name", [
    "verify_frozen_selector_executable", "_check_source_and_code",
    "_check_numeric_dependencies", "_code_signature",
])
def test_bridge_rejects_runtime_witness_function_replacement(monkeypatch, function_name):
    monkeypatch.setattr(witness, function_name, lambda *_args, **_kwargs: {
        "original_selector_unchanged": True,
    })
    with pytest.raises(bridge.FrozenEvidenceError, match="witness live function changed"):
        bridge._verify_frozen_selector_source()


def test_bridge_rejects_runtime_witness_policy_replacement(monkeypatch):
    monkeypatch.setattr(witness, "FROZEN_STATISTICS_SHA256", "0" * 64)
    with pytest.raises(bridge.FrozenEvidenceError, match="witness policy changed"):
        bridge._verify_frozen_selector_source()


def test_bridge_rejects_runtime_witness_literal_inventory_mutation(monkeypatch):
    monkeypatch.setattr(witness, "_LITERALS", {key: frozenset()
                                               for key in witness._LITERALS})
    with pytest.raises(bridge.FrozenEvidenceError, match="literal policy inventory"):
        bridge._verify_frozen_selector_source()


def test_bridge_rejects_forged_live_selector_before_evidence(monkeypatch):
    monkeypatch.setattr(reap_select, "evaluate", lambda *_args: {"gates": {}})
    result = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "executing function differs" in result["reason"]
    assert result["chosen"] is None and result["release_approved"] is False


def test_blocked_response_never_calls_a_rejected_selector(monkeypatch):
    monkeypatch.setattr(reap_select, "select", lambda *_args: {
        "status": "SELECTED", "chosen": "reap128_v2", "reason": "forged",
    })
    result = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert "executing function differs" in result["reason"]
    assert result["selection"]["status"] == "BLOCKED"
    assert result["selection"]["chosen"] is None
    assert result["chosen"] is None and result["release_approved"] is False


def test_missing_original_evidence_stays_blocked_with_intact_selector():
    result = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert result["status"] == "BLOCKED_UNVERIFIED_FROZEN_EVIDENCE"
    assert result["chosen"] is None and result["release_approved"] is False
