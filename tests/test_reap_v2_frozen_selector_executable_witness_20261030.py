"""CPU-only adversarial tests: no model, F: access or production approvals."""

from __future__ import annotations

import ast
import builtins
import hashlib
import fractions
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from types import FunctionType

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


@pytest.mark.parametrize("key,value", [
    ("NaN", 1.0),
    ("Infinity", 0.0),
    ("-Infinity", 0.0),
])
def test_json_nonfinite_constant_map_mutation_refused(monkeypatch, key, value):
    before = json.loads('{"trigram_ratio": NaN}')["trigram_ratio"]
    assert math.isnan(before)
    monkeypatch.setitem(json.decoder._CONSTANTS, key, value)
    if key == "NaN":
        assert json.loads('{"trigram_ratio": NaN}')["trigram_ratio"] == 1.0
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="JSON nonfinite constant map changed"):
        witness.verify_frozen_selector_executable()


def test_json_nonfinite_constant_map_extra_key_refused(monkeypatch):
    monkeypatch.setitem(json.decoder._CONSTANTS, "forged", 0.0)
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="JSON nonfinite constant map changed"):
        witness.verify_frozen_selector_executable()


def test_forged_ast_literal_eval_cannot_authorize_gate1_threshold_mutation(monkeypatch):
    n = 1319
    base = {f"gsm8k-{i}": {"correct": True} for i in range(n)}
    arm = {f"gsm8k-{i}": {"correct": i >= 31} for i in range(n)}
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "FAIL"
    original = ast.literal_eval

    def forged_literal(node):
        value = original(node)
        return -1.0 if type(value) is float and value == -0.03 else value

    monkeypatch.setattr(ast, "literal_eval", forged_literal)
    monkeypatch.setattr(reap_gates, "G1_THRESH", -1.0)
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "PASS"
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="AST runtime function changed"):
        witness.verify_frozen_selector_executable()
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="AST runtime function changed"):
        bridge._verify_frozen_selector_source()


def test_forged_gate_row_reader_path_alias_refused(monkeypatch, tmp_path):
    n = 50
    for arm, correct in (("candidate", False), ("dense", True)):
        directory = tmp_path / arm
        directory.mkdir()
        lines = [json.dumps({"id": f"gsm8k-{i}", "correct": correct,
                             "terminated": True, "trigram_ratio": 1.0})
                 for i in range(n)]
        (directory / "gsm8k.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    original_bytes = (tmp_path / "candidate" / "gsm8k.jsonl").read_bytes()

    def score_from_disk():
        arm = reap_gates._load(tmp_path, "candidate", "gsm8k", expected_n=n)
        base = reap_gates._load(tmp_path, "dense", "gsm8k", expected_n=n)
        return reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"]

    assert score_from_disk() == "FAIL"
    native_path = type(Path())

    class ForgedPath(native_path):
        def read_text(self, *args, **kwargs):
            content = super().read_text(*args, **kwargs)
            if self.parent.name == "candidate" and self.name == "gsm8k.jsonl":
                return content.replace('"correct": false', '"correct": true')
            return content

    monkeypatch.setattr(reap_gates, "Path", ForgedPath)
    assert score_from_disk() == "PASS"
    assert (tmp_path / "candidate" / "gsm8k.jsonl").read_bytes() == original_bytes
    assert reap_select._load is reap_gates._load
    with pytest.raises(witness.SelectorExecutableRefused, match="filesystem Path binding"):
        witness.verify_frozen_selector_executable()


def test_coordinated_witness_and_selector_path_alias_swap_refused(monkeypatch):
    class ForgedPath(type(Path())):
        def read_text(self, *args, **kwargs):
            return super().read_text(*args, **kwargs)

    for module in (witness, reap_gates, reap_select, reap_serve,
                   reap_v2_source_identity):
        monkeypatch.setattr(module, "Path", ForgedPath)
    with pytest.raises(bridge.FrozenEvidenceError,
                       match="bridge frozen selector/gate module binding changed"):
        bridge._verify_frozen_selector_source()


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


def test_original_gate1_false_pass_from_mutable_statistical_precision_refused(monkeypatch):
    """Source-pinned stdev bytecode is insufficient if its precision drifts."""
    n = 1319
    base = {f"gsm8k-{i}": {"correct": True} for i in range(n)}
    arm = {f"gsm8k-{i}": {"correct": i >= 31} for i in range(n)}
    original = reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")
    assert original["status"] == "FAIL"
    assert original["statistic"] < reap_gates.G1_THRESH

    # CPython 3.11 statistics._float_sqrt_of_frac reads this mutable global.
    # Understating stdev makes a genuine original Gate1 FAIL look like PASS.
    monkeypatch.setattr(statistics, "_sqrt_bit_width", 0)
    forged = reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")
    assert forged["status"] == "PASS"
    with pytest.raises(witness.SelectorExecutableRefused, match="statistical/hash/JSON"):
        witness.verify_frozen_selector_executable()
    result = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert result["selection"]["status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


def test_original_gate1_false_pass_from_mutable_fraction_binding_refused(monkeypatch):
    """Unchanged _ss bytecode can compute zero variance through a forged Fraction."""
    n = 1319
    base = {f"gsm8k-{i}": {"correct": True} for i in range(n)}
    arm = {f"gsm8k-{i}": {"correct": i >= 31} for i in range(n)}
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "FAIL"
    monkeypatch.setattr(statistics, "Fraction", lambda *_args: fractions.Fraction(0))
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "PASS"
    with pytest.raises(witness.SelectorExecutableRefused, match="statistical/hash/JSON"):
        witness.verify_frozen_selector_executable()


def test_original_gate1_false_pass_from_fraction_class_method_refused(monkeypatch):
    n = 1319
    base = {f"gsm8k-{i}": {"correct": True} for i in range(n)}
    arm = {f"gsm8k-{i}": {"correct": i >= 31} for i in range(n)}
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "FAIL"
    original = fractions.Fraction.__new__
    monkeypatch.setattr(fractions.Fraction, "__new__",
                        staticmethod(lambda cls, *_args, **_kwargs: original(cls, 0)))
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "PASS"
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="fractional arithmetic method changed"):
        witness.verify_frozen_selector_executable()


def test_original_gate1_false_pass_from_generated_fraction_operator_refused(monkeypatch):
    n = 1319
    base = {f"gsm8k-{i}": {"correct": True} for i in range(n)}
    arm = {f"gsm8k-{i}": {"correct": i >= 31} for i in range(n)}
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "FAIL"
    # _ss uses sum(Fraction(...)); replacing the dynamically installed
    # arithmetic method can zero the variance with unchanged source methods.
    original = fractions.Fraction.__new__
    monkeypatch.setattr(fractions.Fraction, "__add__",
                        lambda self, _other: original(type(self), 0))
    monkeypatch.setattr(fractions.Fraction, "__radd__",
                        lambda self, _other: original(type(self), 0))
    assert reap_gates._lb_gate(arm, base, reap_gates.G1_THRESH, "test")["status"] == "PASS"
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="generated fractional operator changed"):
        witness.verify_frozen_selector_executable()


def test_mutable_float_info_cannot_reauthorize_forged_statistical_precision(monkeypatch):
    monkeypatch.setattr(statistics, "_sqrt_bit_width", 0)
    monkeypatch.setattr(sys, "float_info", SimpleNamespace(mant_dig=-1.5, radix=2))
    with pytest.raises(witness.SelectorExecutableRefused, match="statistical/hash/JSON"):
        witness.verify_frozen_selector_executable()


def test_coordinated_fraction_module_alias_replacement_refused(monkeypatch):
    class FakeFraction(fractions.Fraction):
        pass

    monkeypatch.setattr(statistics, "Fraction", FakeFraction)
    monkeypatch.setattr(fractions, "Fraction", FakeFraction)
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="fractional arithmetic class binding changed"):
        witness.verify_frozen_selector_executable()


def test_original_gate0_false_pass_from_cloned_function_builtins_refused(monkeypatch):
    """A new FunctionType captures forged builtins even after globals restore."""
    n = 200
    base = {f"gsm8k-{i}": {"terminated": True, "trigram_ratio": 0.5}
            for i in range(n)}
    arm = {f"gsm8k-{i}": {"terminated": i >= 50, "trigram_ratio": 0.5}
           for i in range(n)}
    assert reap_gates._gate0(arm, base)["status"] == "FAIL"
    original = reap_gates._rates
    real_builtins = vars(reap_gates)["__builtins__"]
    forged_builtins = dict(vars(builtins), sum=lambda *_args, **_kwargs: 0)
    try:
        vars(reap_gates)["__builtins__"] = forged_builtins
        forged = FunctionType(original.__code__, vars(reap_gates), original.__name__,
                              original.__defaults__, original.__closure__)
    finally:
        vars(reap_gates)["__builtins__"] = real_builtins
    assert forged.__builtins__ is forged_builtins
    assert forged.__globals__ is original.__globals__
    assert forged.__code__ is original.__code__
    monkeypatch.setattr(reap_gates, "_rates", forged)
    assert reap_gates._gate0(arm, base)["status"] == "PASS"
    with pytest.raises(witness.SelectorExecutableRefused,
                       match="executing function differs"):
        witness.verify_frozen_selector_executable()
    result = bridge.bridge_frozen_evidence("Z:/no-original-evidence", "Z:/no-observer")
    assert result["selection"]["status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


def test_cloned_witness_function_with_forged_builtins_refused_by_bridge(monkeypatch):
    original = witness.verify_frozen_selector_executable
    original_builtins = vars(witness)["__builtins__"]
    try:
        vars(witness)["__builtins__"] = dict(vars(builtins), sum=lambda *_args: 0)
        forged = FunctionType(original.__code__, vars(witness), original.__name__,
                              original.__defaults__, original.__closure__)
    finally:
        vars(witness)["__builtins__"] = original_builtins
    monkeypatch.setattr(witness, "verify_frozen_selector_executable", forged)
    with pytest.raises(bridge.FrozenEvidenceError, match="witness live function changed"):
        bridge._verify_frozen_selector_source()


def test_mutable_builtin_sum_refused(monkeypatch):
    monkeypatch.setattr(builtins, "sum", lambda *_args, **_kwargs: 0)
    with pytest.raises(witness.SelectorExecutableRefused, match="statistical/hash/JSON"):
        witness.verify_frozen_selector_executable()


@pytest.mark.parametrize("attribute,replacement", [
    ("Decimal", object()),
    ("groupby", lambda *_args: ()),
    ("defaultdict", dict),
])
def test_original_statistics_import_binding_mutation_refused(
    monkeypatch, attribute, replacement,
):
    monkeypatch.setattr(statistics, attribute, replacement)
    with pytest.raises(witness.SelectorExecutableRefused, match="statistical/hash/JSON"):
        witness.verify_frozen_selector_executable()


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
