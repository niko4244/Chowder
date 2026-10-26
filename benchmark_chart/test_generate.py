"""Synthetic/adversarial audit provenance checks; no model or GPU activity."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from benchmark_chart.generate import (
    AUDIT, MANIFEST, PAIRED, REFERENCES, SOURCE_LOCKS,
    EvidenceError, _validate_reference, load_evidence, render, rows,
)


PROJECT = Path(__file__).resolve().parents[1]


class ChartEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="reap-bench-chart-tests-")
        self.root = Path(self.temp.name)
        for rel in (AUDIT, PAIRED, REFERENCES, MANIFEST):
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((PROJECT / rel).read_bytes())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def fixture(self) -> dict:
        return json.loads((self.root / REFERENCES).read_text(encoding="utf-8"))

    def update_evidence(self, payload: dict) -> None:
        (self.root / REFERENCES).write_text(json.dumps(payload), encoding="utf-8")

    def attempt_manifest_rebaseline(self, rel: Path) -> None:
        manifest = json.loads((self.root / MANIFEST).read_text(encoding="utf-8"))
        normalized = (self.root / rel).read_bytes().replace(b"\r\n", b"\n")
        manifest["sha256"][rel.as_posix()] = hashlib.sha256(normalized).hexdigest()
        (self.root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    def test_baseline_evidence_is_locked_and_complete(self) -> None:
        evidence = load_evidence(self.root)
        self.assertEqual(evidence["sha256"], SOURCE_LOCKS)
        metrics = rows(evidence)
        self.assertEqual(len(metrics), 11)
        self.assertEqual(sum(r["qwen_reference"] != "N/T" for r in metrics), 4)
        self.assertEqual(sum(r["opus_reference"] != "N/T" for r in metrics), 3)
        for row in metrics:
            self.assertEqual((row["reap_direct"], row["qwen_direct"], row["opus_direct"]),
                             ("N/T", "N/T", "N/T"))
        self.assertEqual([m["correct"] for m in evidence["audits"]], [1229, 1220, 151, 152])

    def test_export_is_deterministic_with_eleven_categories(self) -> None:
        first = render(self.root)
        self.assertEqual(first, render(self.root))
        csv_rows = list(csv.DictReader(io.StringIO(first["reap128v2_11_benchmarks.csv"].decode())))
        self.assertEqual(len(csv_rows), 11)
        self.assertEqual(csv_rows[1]["opus_reference"], "66.4")
        self.assertEqual(csv_rows[6]["qwen_metric"], "pass@1")
        self.assertEqual(csv_rows[6]["qwen_reference"], "20.4")
        self.assertEqual(csv_rows[9]["qwen_reference"], "N/T")
        self.assertIn("Q8_0 | HumanEval original Linux | 152/164", first["reap128v2_11_benchmarks.md"].decode())
        self.assertIn("separate independent replay PENDING", first["reap128v2_11_benchmarks.md"].decode())
        picture = ElementTree.fromstring(first["reap128v2_11_benchmarks.svg"])
        self.assertTrue(picture.tag.endswith("svg"))
        self.assertIn("N/T", first["reap128v2_11_benchmarks.svg"].decode())

    def test_source_audit_rewrite_fails_even_if_manifest_rehashed(self) -> None:
        file = self.root / AUDIT
        source = file.read_text(encoding="utf-8")
        file.write_text(source.replace("1,229 (93.17665%)", "1,319 (100.00000%)", 1), encoding="utf-8")
        self.attempt_manifest_rebaseline(AUDIT)
        with self.assertRaisesRegex(EvidenceError, "Pinned inputs unexpectedly changed"):
            load_evidence(self.root)

    def test_paired_audit_rewrite_fails(self) -> None:
        file = self.root / PAIRED
        file.write_text(file.read_text(encoding="utf-8").replace("UNKNOWN / NOT TESTED", "100.0", 1), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "Pinned evidence digest mismatch"):
            load_evidence(self.root)

    def test_references_rewrite_and_manifest_collusion_fail(self) -> None:
        claim = self.fixture()
        claim["publisher_references"][4]["score"] = 99.9
        self.update_evidence(claim)
        self.attempt_manifest_rebaseline(REFERENCES)
        with self.assertRaisesRegex(EvidenceError, "Pinned inputs unexpectedly changed"):
            load_evidence(self.root)

    def test_quote_score_disagreement_detected_even_without_digest_layer(self) -> None:
        claim = self.fixture()
        claim["publisher_references"][4]["score"] = 99.9
        with self.assertRaisesRegex(EvidenceError, "does not agree"):
            _validate_reference(claim)

    def test_original_qwen_ale_metric_cannot_be_swapped_to_aggregate_score(self) -> None:
        claim = self.fixture()
        ale = next(x for x in claim["publisher_references"] if x["benchmark"] == "agents_last_exam")
        ale["score"] = 42.9
        ale["metric"] = "Score"
        with self.assertRaisesRegex(EvidenceError, "does not agree"):
            _validate_reference(claim)

    def test_prohibit_wrong_model_identity_and_provider_url(self) -> None:
        claim = self.fixture()
        claim["competitors"]["qwen"]["identity"] = "Qwen/Qwen3.8-9B"
        with self.assertRaisesRegex(EvidenceError, "Exact model"):
            _validate_reference(claim)
        claim = self.fixture()
        claim["competitors"]["opus"]["source"] = "https://other.invalid/fake-opus"
        with self.assertRaisesRegex(EvidenceError, "Exact model"):
            _validate_reference(claim)

    def test_prohibit_harness_version_and_cohort_promotions(self) -> None:
        for target, bad in (("hle_no_tools", "same_named_benchmark"),
                            ("deepswe_1_1", "same_named_benchmark"),
                            ("automation_v1_0_6", "same_named_benchmark")):
            with self.subTest(target=target):
                claim = self.fixture()
                item = next(x for x in claim["publisher_references"] if x["benchmark"] == target)
                item["comparability"] = bad
                with self.assertRaises(EvidenceError):
                    _validate_reference(claim)

    def test_no_extra_benchmark_or_forged_direct_score(self) -> None:
        claim = self.fixture()
        claim["benchmarks"].append(copy.deepcopy(claim["benchmarks"][0]))
        with self.assertRaisesRegex(EvidenceError, "Benchmark categories"):
            _validate_reference(claim)
        claim = self.fixture()
        claim["benchmarks"][0]["reap_direct"] = "95.1"
        with self.assertRaisesRegex(EvidenceError, "forged direct scores"):
            _validate_reference(claim)

    def test_duplicate_reference_is_rejected(self) -> None:
        claim = self.fixture()
        claim["publisher_references"].append(copy.deepcopy(claim["publisher_references"][0]))
        with self.assertRaisesRegex(EvidenceError, "Duplicate"):
            _validate_reference(claim)


if __name__ == "__main__":
    unittest.main()
