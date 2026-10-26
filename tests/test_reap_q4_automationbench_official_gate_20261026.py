"""Synthetic adversarial gates. No imports from upstream, GPUs or generated code."""

import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chowder_batch import reap_q4_automationbench_official_gate_20261026 as gate


GPU = "Tesla T4, 14912, 15360\nTesla T4, 14912, 15360"
GPU_POSTLOAD = "Tesla T4, 9000, 15360\nTesla T4, 9000, 15360"
MEMORY = "MemAvailable: 16777216 kB\nCommitLimit: 33554432 kB\nCommitted_AS: 16777216 kB\n"


class OfficialAutomationGateTests(unittest.TestCase):
    def test_official_source_identity_sha_and_binary_grader_from_git(self):
        # A synthetic checkout tests only the whitelist logic; a real checkout
        # invocation separately validates SHA+Git blobs before deployment.
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "automationbench/tools/api").mkdir(parents=True)
            (base / "automationbench/rubric").mkdir(parents=True)
            (base / "pyproject.toml").write_text(
                '[project]\nname="automation-bench"\nversion="1.0.6"\nrequires-python=">=3.13"\n'
                '[project.scripts]\nauto-bench="automationbench.scripts.eval:main"', encoding="utf-8")
            (base / "automationbench/tools/api/__init__.py").write_text(
                "API_TOOLS = [api_search, api_fetch, base64_encode]", encoding="utf-8")
            (base / "automationbench/rubric/__init__.py").write_text(
                'return float(state.get("partial_credit", 0.0) == 1.0)', encoding="utf-8")
            (base / "automationbench/tools/api/fetch.py").write_text(
                "return router(world, method, internal_path, parsed_params, body_value)", encoding="utf-8")
            committed_sha = gate.OFFICIAL_COMMIT
            def fake_run(argv, **kwargs):
                if argv[1:] == ["rev-parse", "HEAD"]:
                    return committed_sha
                if argv[1:] == ["status", "--porcelain", "--untracked-files=all"]:
                    return ""
                return gate.EXPECTED_BLOBS[argv[-1].removeprefix("HEAD:")]
            with patch.object(gate, "_run_readonly", side_effect=fake_run):
                self.assertEqual(gate.check_official_source(base)["package"], "automation-bench==1.0.6")
                with patch.object(gate, "OFFICIAL_COMMIT", "0" * 40):
                    with self.assertRaisesRegex(gate.GateBlocked, "SHA mismatch"):
                        gate.check_official_source(base)
                (base / "automationbench/rubric/__init__.py").write_text('return 1.0', encoding="utf-8")
                with self.assertRaisesRegex(gate.GateBlocked, "grader"):
                    gate.check_official_source(base)

    def test_source_modified_or_untracked_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(gate, "_run_readonly", side_effect=[gate.OFFICIAL_COMMIT, "?? sitecustomize.py"]):
                with self.assertRaisesRegex(gate.GateBlocked, "modified tracked"):
                    gate.check_official_source(Path(td))

    def test_memory_margin_and_missing_commit_data(self):
        measured = gate.read_memory_floors(MEMORY, 1 * 2**30, 20 * 2**30)
        self.assertEqual(measured["physical_available_gib"], 16)
        self.assertEqual(measured["commit_headroom_gib"], 16)
        with self.assertRaisesRegex(gate.GateBlocked, "RAM/commit/cgroup"):
            gate.read_memory_floors(MEMORY, 17 * 2**30, 20 * 2**30)
        with self.assertRaisesRegex(gate.GateBlocked, "Physical/commit accounting unavailable"):
            gate.read_memory_floors("MemAvailable: 1000000 kB", 0, 100 * 2**30)

    def test_vram_and_cuda_busy_protection(self):
        self.assertEqual(len(gate.parse_gpu_capacity(GPU, "", 0)["cuda_devices"]), 2)
        with self.assertRaisesRegex(gate.GateBlocked, "protected/unknown"):
            gate.parse_gpu_capacity(GPU, "200\n", 0)
        with self.assertRaisesRegex(gate.GateBlocked, "below 12288"):
            gate.parse_gpu_capacity("Tesla T4, 9000, 15360\nTesla T4, 14912, 15360", "", 0)
        with self.assertRaisesRegex(gate.GateBlocked, "Protected or unknown"):
            gate.parse_gpu_capacity(GPU_POSTLOAD, "501\n666", 501)
        self.assertEqual(gate.parse_gpu_capacity(GPU_POSTLOAD, "501", 501)["cuda_pids"], [501])
        with self.assertRaisesRegex(gate.GateBlocked, "under 2 GiB"):
            gate.parse_gpu_capacity("Tesla T4, 1000, 15360\nTesla T4, 9000, 15360", "501", 501)

    def test_wrong_gpu_topology_and_invalid_census_fail(self):
        with self.assertRaisesRegex(gate.GateBlocked, "Exactly two"):
            gate.parse_gpu_capacity("RTX 4090, 18000, 24576", "", 0)
        with self.assertRaisesRegex(gate.GateBlocked, "Unparseable"):
            gate.parse_gpu_capacity(GPU, "other-user-secret-process", 0)

    def test_private_output_rejects_repo_models_existing_and_logs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository = root / "repo"
            repository.mkdir()
            private = root / "private"
            private.mkdir()
            with self.assertRaisesRegex(gate.GateBlocked, "outside"):
                gate.check_private_location(repository / "out.json", (repository,))
            filename = private / "out.json"
            self.assertEqual(gate.check_private_location(filename, (repository,)), filename)
            with self.assertRaisesRegex(gate.GateBlocked, "absolute"):
                gate.check_private_location(Path("relative.json"), (repository,))
            (private / "out.stdout.log").write_text("existing")
            with self.assertRaisesRegex(gate.GateBlocked, "exclusive"):
                gate.check_private_location(filename, (repository,))

    def test_reject_local_credentials_or_python_startup_hooks_in_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "sitecustomize.py").write_text("raise SystemExit('BAD')", encoding="utf-8")
            with self.assertRaisesRegex(gate.GateBlocked, "startup/credential"):
                gate.check_official_source(base)
            (base / "sitecustomize.py").unlink()
            (base / ".env").write_text("UNEXPECTED_CREDENTIAL=BAD")
            with self.assertRaisesRegex(gate.GateBlocked, "startup/credential"):
                gate.check_official_source(base)

    def test_prelaunch_uses_actual_boot_model_and_2h_window(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        before = gate.parse_gpu_capacity(GPU, "", 0)["cuda_devices"]
        after = gate.parse_gpu_capacity(GPU_POSTLOAD, "543", 543)
        receipt = {
            "schema": "reap-q4-automationbench-public-prelaunch.v1",
            "at_utc": now.isoformat(), "boot_id": "fake-boot-uuid",
            "q4_sha256": gate.Q4_SHA256, "q4_bytes": gate.Q4_BYTES,
            "physical_available_gib": 15, "commit_headroom_gib": 20,
            "cgroup_free_gib": 12, "cuda_devices": before, "cuda_pids": [],
        }
        receipt["receipt_sha256"] = gate._receipt_digest(receipt)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "prelaunch.json"
            p.write_text(json.dumps(receipt))
            with patch.object(gate, "_boot_id", return_value="fake-boot-uuid"):
                self.assertIn("prelaunch_sha256", gate.verify_prelaunch(p, after))
                receipt["cuda_devices"][0]["free_mib"] = 1
                p.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(gate.GateBlocked, "changed"):
                    gate.verify_prelaunch(p, after)
                receipt["receipt_sha256"] = gate._receipt_digest(receipt)
                p.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(gate.GateBlocked, "insufficient"):
                    gate.verify_prelaunch(p, after)
                receipt["cuda_devices"][0]["free_mib"] = 14912
                receipt["at_utc"] = (now - datetime.timedelta(hours=3)).isoformat()
                receipt["receipt_sha256"] = gate._receipt_digest(receipt)
                p.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(gate.GateBlocked, "expired"):
                    gate.verify_prelaunch(p, after)

    def test_official_command_exact_public_tasks_and_no_shell(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)
            result = source / "private" / "result.json"
            single = gate.official_command(source, "reap128-v2-q4_k_m-20261025", 8790, result, full=False)
            full = gate.official_command(source, "reap128-v2-q4_k_m-20261025", 8790, result, full=True)
            self.assertEqual(single[:3], [gate.sys.executable, "-m", "automationbench.scripts.eval"])
            self.assertEqual(single[single.index("--num-examples") + 1], "1")
            self.assertEqual(full[full.index("--num-examples") + 1], "-1")
            self.assertEqual(full[full.index("--domains") + 1], ",".join(gate.DOMAINS))
            self.assertIn("--no-ensure-complete", single)
            self.assertIn("--toolset", single)
            self.assertEqual(single[single.index("--toolset") + 1], "api")
            self.assertNotIn("--batch", full)
            self.assertNotIn("-c", single)
            self.assertTrue(all(not x.startswith("https://") for x in single))
            with self.assertRaisesRegex(gate.GateBlocked, "model identifier"):
                gate.official_command(source, "$(touch /tmp/pwned)", 8790, result, full=True)

    def test_source_only_cli_does_not_run_model(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            with patch("sys.argv", ["gate", "--official-source", str(path)]), \
                 patch.object(gate, "check_official_source", return_value={"verified": True}), \
                 patch.object(gate, "check_original_q4", side_effect=AssertionError("unexpected model access")), \
                 patch.object(gate, "inspect_cuda", side_effect=AssertionError("unexpected GPU probe")), \
                 patch("builtins.print") as printed:
                self.assertEqual(gate.main(), 0)
                self.assertIn('"executable": false', printed.call_args[0][0].lower())


if __name__ == "__main__":
    unittest.main()
