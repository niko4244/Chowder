"""Synthetic adversarial census checks; never execute a model or benchmark task."""

import copy
import json
import unittest
from unittest.mock import patch

from chowder_batch.reap_automationbench_population_20261026 import verify_population
from chowder_batch import reap_q4_automationbench as runner


def fixture():
    sources = []
    tasks = []
    usage = []
    for i in range(600):
        name = f"{('sales', 'marketing', 'operations', 'support', 'finance', 'hr')[i // 100]}.task_{i}"
        prompt = [{"role": "system", "content": "frozen"}, {"role": "user", "content": str(i)}]
        source = {"example_id": i + 1, "name": name, "prompt": prompt,
                  "assertions": [{"type": "field_equals", "value": i}]}
        sources.append(source)
        tasks.append({"id": i + 1, "name": name, "messages": [*copy.deepcopy(prompt),
                     {"role": "assistant", "content": "done"}], "score": 1.0,
                     "passed": True, "assertions_total": 1, "assertions_passed": 1,
                     "assertion_results": [{"type": "field_equals", "params": {"value": i},
                                            "passed": True, "excluded": False}],
                     "steps": 2, "num_model_calls": 1, "num_tool_calls": 0,
                     "input_tokens": 10, "output_tokens": 5})
        usage.append({"task_id": i + 1, "task_name": name, "input_tokens": 10,
                      "output_tokens": 5, "total_tokens": 15})
    summary = {"passed_count": 600, "failed_count": 0, "avg_score": 1.0,
               "pass_rate": 1.0, "aborted_tasks": []}
    return sources, tasks, usage, summary


class PopulationProofTests(unittest.TestCase):
    def test_complete_fixture_and_report_not_scoring_certification(self):
        self.assertEqual(verify_population(*fixture())["binding"],
                         "EXACT_PUBLIC_SOURCE_NAMES_PROMPTS_ASSERTIONS_NOT_WORLDSTATE_CERTIFIED")

    def test_one_substituted_name_rejected(self):
        values = fixture()
        values[1][120]["name"] = "marketing.forged"
        with self.assertRaisesRegex(ValueError, "source name/order"):
            verify_population(*values)

    def test_one_substituted_or_duplicated_id_rejected(self):
        for altered in (0, 3):
            values = fixture()
            values[1][11]["id"] = altered
            with self.assertRaisesRegex(ValueError, "ordinal"):
                verify_population(*values)

    def test_source_prompt_and_assertion_contract_tampering_rejected(self):
        for section in ("prompt", "assertion"):
            values = fixture()
            if section == "prompt":
                values[1][11]["messages"][0]["content"] = "forged"
            else:
                values[1][11]["assertion_results"][0]["params"]["value"] = "forged"
            with self.assertRaisesRegex(ValueError, "prompt|assertion"):
                verify_population(*values)

    def test_inconsistent_partial_and_false_binary_rejected(self):
        for field, value in (("score", 0.5), ("passed", False), ("assertions_passed", 0)):
            values = fixture()
            values[1][11][field] = value
            with self.assertRaisesRegex(ValueError, "metric"):
                verify_population(*values)

    def test_missing_state_is_not_independently_approved(self):
        values = fixture()
        proof = verify_population(*values)
        self.assertFalse(proof["worldstate_recomputed"])
        self.assertFalse(proof["release_approved"])

    def test_aborted_and_usage_forgery_rejected(self):
        values = fixture()
        values[3]["aborted_tasks"] = [values[1][0]["name"]]
        with self.assertRaisesRegex(ValueError, "aborted"):
            verify_population(*values)
        values = fixture()
        values[2][17]["task_name"] = "forged"
        with self.assertRaisesRegex(ValueError, "usage"):
            verify_population(*values)

    def test_silently_laundered_tool_abort_rejected(self):
        values = fixture()
        values[1][11]["messages"].append({"role": "assistant", "tool_calls": [{"id": "unexecuted"}]})
        values[3]["aborted_tasks"] = []
        with self.assertRaisesRegex(ValueError, "aborted"):
            verify_population(*values)

    def test_inconsistent_usage_totals_rejected(self):
        values = fixture()
        values[2][33]["total_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "usage"):
            verify_population(*values)

    def test_wrong_source_population_rejected(self):
        values = fixture()
        values[0][0]["name"] = values[0][1]["name"]
        with self.assertRaisesRegex(ValueError, "official source"):
            verify_population(*values)

    def test_numeric_source_ids_overlap_across_domains_but_not_within(self):
        values = fixture()
        values[0][100]["example_id"] = values[0][0]["example_id"]
        self.assertEqual(verify_population(*values)["population_count"], 600)
        values[0][1]["example_id"] = values[0][0]["example_id"]
        with self.assertRaisesRegex(ValueError, "official source"):
            verify_population(*values)

    def test_full_600_official_triage_rejects_replacement_and_partial_export(self):
        import tempfile
        from pathlib import Path

        values = fixture()
        source, tasks, usage, summary = values
        root = {"meta": {"benchmark_version": "1.0.6", "model": "original-q4",
                         "toolset": "api", "domains": list(runner.SHORT_PUBLIC_TASKS)[:0] +
                         ["sales", "marketing", "operations", "support", "finance", "hr"],
                         "max_steps": 50, "total_tasks": 600},
                "summary": summary, "tasks": tasks, "usage_by_task": usage}
        with tempfile.TemporaryDirectory() as td, \
             patch.object(runner, "_full_official_source_rows", return_value=source):
            path = Path(td) / "raw-private.json"
            def triage():
                path.write_text(json.dumps(root), encoding="utf8")
                return runner._triage_official_output(path, count=600, served_id="original-q4")
            self.assertFalse(triage()["full_public_population_proof"]["release_approved"])
            root["tasks"][1]["name"] = "sales.fake_unique"
            with self.assertRaisesRegex(ValueError, "source name/order"):
                triage()
            root["tasks"][1]["name"] = source[1]["name"]
            root["meta"]["partial"] = True
            with self.assertRaisesRegex(ValueError, "evaluator identity"):
                triage()


if __name__ == "__main__":
    unittest.main()
