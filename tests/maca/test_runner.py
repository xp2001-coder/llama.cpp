import json
import gzip
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import run


class RunnerTests(unittest.TestCase):
    def test_incremental_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "build"
            identity = {"source": "source", "sdk": "sdk"}
            run.claim_build(path, identity)
            run.claim_build(path, identity)
            self.assertEqual(json.loads((path / "maca-probe-owner.json").read_text()), identity)

    def test_nonempty_directory_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            existing = path / "CMakeCache.txt"
            existing.write_text("old build")
            with self.assertRaises(ValueError):
                run.claim_build(path, {})
            self.assertEqual(existing.read_text(), "old build")

    def test_changed_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            run.claim_build(path, {"sdk": "old"})
            with self.assertRaises(ValueError):
                run.claim_build(path, {"sdk": "new"})
            self.assertEqual(json.loads((path / "maca-probe-owner.json").read_text()), {"sdk": "old"})

    def test_build_false_success_rejected(self):
        for error in ("FAILED: object.o", "ninja: error: rebuilding", "ninja: build stopped",
                      "CMake Error at file:1", "compiler: error: bad type", "fatal error: missing.h",
                      "undefined reference to symbol"):
            with self.subTest(error=error):
                self.assertFalse(run.accepted_build({"exit_code": 0}, error))

    def test_build_exit_failure_rejected(self):
        self.assertFalse(run.accepted_build({"exit_code": 1}, ""))

    def test_incremental_no_work_accepted(self):
        self.assertTrue(run.accepted_build({"exit_code": 0}, "ninja: no work to do."))

    def test_warning_not_failure(self):
        self.assertTrue(run.accepted_build({"exit_code": 0}, "warning: option ignored"))

    def test_exact_success_marker(self):
        self.assertTrue(run.accepted_probe("warp", {"exit_code": 0}, "MACA_PROBE_PASS warp\n"))
        for text in ("", "MACA_PROBE_PASS runtime", "prefix MACA_PROBE_PASS warp",
                     "MACA_PROBE_PASS warp\nMACA_PROBE_PASS warp"):
            self.assertFalse(run.accepted_probe("warp", {"exit_code": 0}, text))

    def test_marker_cannot_override_exit_failure(self):
        self.assertFalse(run.accepted_probe("warp", {"exit_code": 134}, "MACA_PROBE_PASS warp"))

    def test_process_records_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, "-c", "print('test output'); raise SystemExit(7)"]
            record, text = run.execute(command, Path(temporary) / "test.log", os.environ.copy(), 30)
            self.assertEqual(record["exit_code"], 7)
            self.assertIn("test output", text)

    def test_missing_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, text = run.execute([str(Path(temporary) / "missing-command")], Path(temporary) / "missing.log", os.environ.copy(), 30)
            self.assertEqual(record["exit_code"], 127)
            self.assertTrue(text)

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("run.subprocess.Popen") as spawn, patch("run.os.killpg", create=True) as kill_group:
                process = spawn.return_value.__enter__.return_value
                process.wait.side_effect = [subprocess.TimeoutExpired(["test"], 1), 0]
                record, _ = run.execute(["test"], Path(temporary) / "timeout.log", os.environ.copy(), 1)
                if os.name == "posix":
                    kill_group.assert_called_once()
                else:
                    process.kill.assert_called_once()
            self.assertEqual(record["exit_code"], 124)

    def test_compiler_target_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            run.pin_target(path, "xcore1000")
            run.pin_target(path, "xcore1000")
            with self.assertRaises(RuntimeError):
                run.pin_target(path, "different-target")
            self.assertEqual(json.loads((path / "maca-compiler-target.json").read_text()), "xcore1000")

    def test_posix_timeout_stops_compiler_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "timeout.log"
            with patch("run.subprocess.Popen") as spawn, patch("run.os.name", "posix"), \
                 patch("run.signal.SIGKILL", 9, create=True), patch("run.os.killpg", create=True) as kill_group:
                process = spawn.return_value.__enter__.return_value
                process.pid = 12345
                process.wait.side_effect = [subprocess.TimeoutExpired(["test"], 1), 0]
                record, _ = run.execute(["test"], log, os.environ.copy(), 1)
                kill_group.assert_called_once_with(12345, run.signal.SIGKILL)
                self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertEqual(record["exit_code"], 124)

    def test_interrupt_stops_compiler_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "interrupt.log"
            with patch("run.subprocess.Popen") as spawn, patch("run.os.name", "posix"), \
                 patch("run.signal.SIGKILL", 9, create=True), patch("run.os.killpg", create=True) as kill_group:
                process = spawn.return_value.__enter__.return_value
                process.pid = 12345
                process.wait.side_effect = [KeyboardInterrupt(), 0]
                with self.assertRaises(KeyboardInterrupt):
                    run.execute(["test"], log, os.environ.copy(), 1)
                kill_group.assert_called_once_with(12345, run.signal.SIGKILL)

    def simulate_run(self, mode="all", bad_step=None, false_success=False, full_ops=False, inventory=False, changed_case=None,
                     flash_attn=False, manifest_flash=None, bad_steps=None, step_outputs=None, interrupt_step=None,
                     manifest_op=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            source = path / "source"
            (source / "tests/maca").mkdir(parents=True)
            (source / "tests/maca/run.py").write_text("# mock source fingerprint\n")
            sdk = path / "sdk with spaces"
            header = sdk / "include/mcr/mc_runtime.h"
            header.parent.mkdir(parents=True)
            header.write_text("// mock SDK, not GPU verification\n")
            for name in ("cmake_mock", "cmake_maca", "multi_target.sh"):
                tool = sdk / "tools/cu-bridge/tools" / name
                tool.parent.mkdir(parents=True, exist_ok=True)
                tool.write_text("mock\n")
            build = path / "build"
            environments = []
            mock_ops = ("ADD", "MUL_MAT", *run.DATA_OPS) + (("FLASH_ATTN_EXT",) if flash_attn else ())

            def fake_execute(command, log, env, timeout):
                environments.append(dict(env))
                code = 0
                text = ""
                if log.stem == "detect-target":
                    text = "xcore1000\n"
                elif log.stem.startswith("build-"):
                    name = log.stem[len("build-"):]
                    kind = "backend" if mode == "backend" else "runtime" if name == "runtime" else "device"
                    binary = build / kind / "bin" / (name if mode == "backend" else "test-maca-" + name)
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(b"mock executable")
                elif log.stem.startswith("run-"):
                    text = "MACA_PROBE_PASS " + log.stem[len("run-"):] + "\n"
                elif log.stem == "backend-devices":
                    text = "Available devices:\n  MACA0: MetaX C500\n"
                elif log.stem == "cpu-sum-oracle":
                    text = "".join("SUM(type=f32,ne=" + shape + ",permute=[1,0,2,3],oracle=integer_sum): OK\n"
                                   for shape in ("[33,256,1,1]", "[3,5,2,2]"))
                    text += "2/2 tests passed\nBackend CPU: OK\n"
                elif log.stem in ("matmul-original-layout", "matmul-quant-layout"):
                    count = 21 if log.stem == "matmul-original-layout" else 64
                    text = "".join("MUL_MAT(mock=%d): OK\n" % i for i in range(count))
                    text += "%d/%d tests passed\n  Backend MACA0: OK\n" % (count, count)
                elif log.stem == "ops-full":
                    text = "".join(op + "(mock=1): OK\n" for op in mock_ops)
                    text += "%d/%d tests passed\n  Backend MACA0: OK\n" % (len(mock_ops), len(mock_ops))
                elif log.stem == "moe-duplicate-ids":
                    text = "".join("MUL_MAT_ID(mock=%d,dup=1): OK\n" % i for i in range(16))
                    text += "16/16 tests passed\n  Backend MACA0: OK\n"
                elif log.stem.startswith("ops-recovery-"):
                    op = log.stem[len("ops-recovery-"):].upper()
                    text = op + "(mock=1): OK\n  1/1 tests passed\n  Backend MACA0: OK\n"
                elif log.stem.startswith("data-"):
                    op = log.stem[len("data-"):].upper()
                    text = op + "(mock=1): OK\n  1/1 tests passed\n  Backend MACA0: OK\n"
                elif log.stem.startswith("ops-"):
                    op = log.stem[len("ops-"):].upper()
                    text = op + "(mock=1): OK\n  1/1 tests passed\n  Backend MACA0: OK\n"
                if log.stem == bad_step:
                    text = "FAILED: deliberate mock failure\n"
                    code = 0 if false_success else 1
                if log.stem == changed_case:
                    text = text.replace("mock=1", "mock=999")
                if bad_steps and log.stem in bad_steps:
                    text = "FAILED: deliberate mock failure\n"
                    code = bad_steps[log.stem]
                if step_outputs and log.stem in step_outputs:
                    code, text = step_outputs[log.stem]
                log.write_text(text)
                if log.stem == interrupt_step:
                    raise KeyboardInterrupt()
                return {"command": command, "exit_code": code, "log": log.name}, text

            argv = ["run.py", "--sdk", str(sdk), "--build-dir", str(build), "--mode", mode]
            if full_ops:
                argv.append("--full-ops")
            if flash_attn:
                argv.append("--flash-attn")
            if inventory:
                manifest = path / "expected.json.gz"
                data = {"base_commit": run.BASE_COMMIT, "device": "MACA0",
                        "flash_attention": flash_attn if manifest_flash is None else manifest_flash,
                        "cases": [[op, "mock=1", "OK"] for op in mock_ops]}
                if manifest_op is not None:
                    data["cases"][0][0] = manifest_op
                manifest.write_bytes(gzip.compress(json.dumps(data).encode()))
                argv += ["--case-manifest", str(manifest)]
            with patch.object(sys, "argv", argv), patch.object(sys, "platform", "linux"), \
                 patch.object(run, "ROOT", source), \
                 patch("run.shutil.which", side_effect=lambda name: name), patch("run.execute", side_effect=fake_execute), \
                 patch.dict(os.environ, {"WCUDA_HOME": "old", "CUCC_TARGETS": "", "CUCC_TARGETS_FROM_DEVICE": ""}), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = run.main()
            report = json.loads(next((build / "results").glob("*/report.json")).read_text())
            if mode == "backend":
                with tarfile.open(next((build / "results").glob("*.tar.gz"))) as archive:
                    members = archive.getnames()
                    saved_report = next(name for name in members if name.endswith("/report.json"))
                    self.assertEqual(json.loads(archive.extractfile(saved_report).read()), report)
                    for step in report["steps"]:
                        self.assertIn(saved_report.rsplit("/", 1)[0] + "/" + step["log"], members)
            return code, report, environments

    def test_flash_attention_candidate_enabled_and_checked(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True)
        self.assertEqual(code, 0)
        self.assertTrue(report["flash_attention_enabled"])
        self.assertTrue(report["data_regressions"]["FLASH_ATTN_EXT"]["accepted"])
        configure = next(item for item in report["steps"] if item["log"] == "configure-backend.log")
        self.assertIn("-DGGML_CUDA_FA=ON", configure["command"])
        self.assertFalse(report["graph_enabled"])
        self.assertFalse(report["fusion_enabled"])

    def test_flash_attention_default_remains_off(self):
        code, report, _ = self.simulate_run(mode="backend")
        self.assertEqual(code, 0)
        self.assertFalse(report["flash_attention_enabled"])
        configure = next(item for item in report["steps"] if item["log"] == "configure-backend.log")
        self.assertIn("-DGGML_CUDA_FA=OFF", configure["command"])

    def test_flash_attention_inventory_mismatch_rejected(self):
        for enabled in (False, True):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.simulate_run(mode="backend", full_ops=True, inventory=True,
                                  flash_attn=enabled, manifest_flash=not enabled)

    def test_flash_attention_not_standalone_probe(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.simulate_run(mode="runtime", flash_attn=True)

    def test_flash_attention_failure_is_not_accepted(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True,
                                              bad_step="data-flash_attn_ext", false_success=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["data_regressions"]["FLASH_ATTN_EXT"]["accepted"])
        self.assertTrue(report["default_ops_completed"])
        self.assertFalse(report["requested_checks_passed"])

    def test_full_orchestration_not_backend_acceptance(self):
        code, report, environments = self.simulate_run()
        self.assertEqual(code, 0)
        self.assertEqual(len(report["probes"]), 6)
        self.assertTrue(report["all_requested_probes_passed"])
        self.assertFalse(report["backend_accepted"])
        self.assertTrue(all("WCUDA_HOME" not in env for env in environments))

    def test_runtime_only(self):
        code, report, _ = self.simulate_run(mode="runtime")
        self.assertEqual(code, 0)
        self.assertEqual(report["probes"], {"runtime": "pass"})

    def test_device_only(self):
        code, report, _ = self.simulate_run(mode="device")
        self.assertEqual(code, 0)
        self.assertNotIn("runtime", report["probes"])

    def test_failed_compile_with_stale_binary_is_rejected(self):
        code, report, _ = self.simulate_run(bad_step="build-warp", false_success=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["probes"]["warp"], "build_failed")
        self.assertEqual(report["probes"]["blas"], "pass")

    def test_failed_configure_marks_every_dependent_probe(self):
        code, report, _ = self.simulate_run(bad_step="configure-device", false_success=True)
        self.assertEqual(code, 1)
        for name in run.DEVICE_PROBES:
            self.assertEqual(report["probes"][name], "configure_failed")

    def test_failed_runtime_result_rejected(self):
        code, report, _ = self.simulate_run(bad_step="run-bf16", false_success=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["probes"]["bf16"], "run_failed")

    def test_target_detection_failure_prevents_device_build(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(bad_step="detect-target", false_success=True)
        self.assertEqual(code, 1)
        self.assertIn("SDK target detection failed", report["error"])
        self.assertNotIn("warp", report["probes"])

    def test_backend_smoke_is_not_full_acceptance(self):
        code, report, environments = self.simulate_run(mode="backend")
        self.assertEqual(code, 0)
        self.assertTrue(report["backend_smoke_completed"])
        self.assertFalse(report["backend_accepted"])
        self.assertFalse(report["operator_coverage_accepted"])
        self.assertEqual(set(report["operators"]), set(run.SMOKE_OPS))
        self.assertTrue(all(env["GGML_CUDA_DISABLE_FUSION"] == "1" for env in environments))
        self.assertFalse(any(item["log"] == "configure-runtime.log" for item in report["steps"]))

    def test_backend_failed_build_stops_before_gpu_tests(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", bad_step="build-llama-bench", false_success=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["backend_smoke_completed"])
        self.assertEqual(report["operators"], {})

    def test_backend_missing_device_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", bad_step="backend-devices", false_success=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["operators"], {})

    def test_backend_failed_operator_continues_later_tests(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", bad_step="ops-rms_norm", false_success=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["operators"]["RMS_NORM"]["completed"])
        self.assertTrue(report["operators"]["SOFT_MAX"]["completed"])
        self.assertTrue(report["operators"]["MUL_MAT"]["completed"])
        self.assertFalse(report["backend_smoke_completed"])
        self.assertTrue(report["test_collection_completed"])

    def test_backend_unsupported_is_not_pass(self):
        text = "  ADD(a=1): OK\n  ADD(a=2): not supported [MACA0]\n  1/1 tests passed\n  Backend MACA0: OK\n"
        counts = run.backend_case_summary({"exit_code": 0}, text, "MACA0", "ADD")
        self.assertEqual(counts["ok"], 1)
        self.assertEqual(counts["unsupported"], 1)
        self.assertTrue(counts["completed"])

    def test_backend_zero_tests_wrong_device_and_summary_rejected(self):
        text = "  ADD(a=1): OK\n  1/1 tests passed\n  Backend MACA0: OK\n"
        for bad in (text.replace("MACA0", "CPU"), text.replace("1/1", "2/2"),
                    "0/0 tests passed\n Backend MACA0: OK\n", text.replace("OK", "FAIL")):
            self.assertFalse(run.backend_case_summary({"exit_code": 0}, bad, "MACA0", "ADD")["completed"])
        self.assertFalse(run.backend_case_summary({"exit_code": 134}, text, "MACA0", "ADD")["completed"])

    def test_backend_console_colors_and_failed_comparison(self):
        text = "  ADD(a=1): \x1b[1;32mOK\x1b[0m\n  1/1 tests passed\n Backend MACA0: OK\n"
        self.assertTrue(run.backend_case_summary({"exit_code": 0}, text, "MACA0", "ADD")["completed"])
        counts = run.backend_case_summary({"exit_code": 1}, text + "ADD(a=2): compare failed FAIL\n", "MACA0", "ADD")
        self.assertEqual(counts["failed"], 1)
        self.assertFalse(counts["completed"])

    def test_backend_changed_output_rejected(self):
        text = "ADD(a=1): OK\nADD(a=2): unfamiliar status\n1/1 tests passed\nBackend MACA0: OK\n"
        counts = run.backend_case_summary({"exit_code": 0}, text, "MACA0", "ADD")
        self.assertEqual(counts["unparsed"], 1)
        self.assertFalse(counts["completed"])

    def test_diagnostic_prefixed_failure_is_counted(self):
        text = ("CPY(x=1): OK\r\n"
                "[CPY] ERR = 0.000000099 > 0.000000060   CPY(x=2): \x1b[31mFAIL\x1b[0m\r\n"
                "1/2 tests passed\r\n Backend MACA0: FAIL\r\n")
        for code in (0, 1):
            counts = run.backend_case_summary({"exit_code": code}, text, "MACA0", "CPY")
            self.assertEqual((counts["ok"], counts["failed"]), (1, 1))
            self.assertFalse(counts["completed"])
        inventory = run.exact_case_summary(text, [["CPY", "x=1", "OK"], ["CPY", "x=2", "OK"]])
        self.assertEqual(inventory["observed_rows"], 2)
        self.assertEqual(inventory["extra"], [["CPY", "x=2", "FAIL", 1]])
        self.assertFalse(inventory["accepted"])

    def test_failure_list_does_not_duplicate_case_rows(self):
        text = "[CPY] ERR = 1 > 0   CPY(x=1): FAIL\nFailing tests:\n CPY(x=1)\n"
        inventory = run.exact_case_summary(text, [["CPY", "x=1", "OK"]])
        self.assertEqual(inventory["observed_rows"], 1)

    def test_backend_crlf_success(self):
        text = " CPY(x=1): OK\r\n1/1 tests passed\r\nBackend MACA0: OK\r\n"
        self.assertTrue(run.backend_case_summary({"exit_code": 0}, text, "MACA0")["completed"])

    def test_full_ops_orchestration(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True)
        self.assertEqual(code, 0)
        self.assertTrue(report["default_ops_completed"])
        self.assertFalse(report["operator_coverage_accepted"])
        self.assertTrue(all(item["accepted"] for item in report["layout_regressions"].values()))
        self.assertTrue(report["moe_duplicate_ids"]["accepted"])
        moe_command = next(item["command"] for item in report["steps"] if item["log"] == "moe-duplicate-ids.log")
        self.assertEqual(moe_command[-4:], ["-o", "MUL_MAT_ID", "-p", r",dup=1$"])
        command = next(item["command"] for item in report["steps"] if item["log"] == "ops-full.log")
        self.assertNotIn("-o", command)
        self.assertNotIn("-p", command)

    def test_failed_layout_continues_full_suite(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_step="matmul-quant-layout", false_success=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["layout_regressions"]["matmul-quant-layout"]["accepted"])
        self.assertTrue(report["moe_duplicate_ids"]["accepted"])
        self.assertTrue(report["default_ops_completed"])

    def test_failed_full_suite_cannot_be_hidden_by_smoke_pass(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_step="ops-full", false_success=True)
        self.assertEqual(code, 1)
        self.assertTrue(report["backend_smoke_completed"])
        self.assertFalse(report["default_ops_completed"])

    def test_failed_moe_regression_continues_full_suite(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_step="moe-duplicate-ids", false_success=True)
        self.assertEqual(code, 1)
        self.assertTrue(report["backend_smoke_completed"])
        self.assertFalse(report["moe_duplicate_ids"]["accepted"])
        self.assertTrue(report["default_ops_completed"])

    def test_full_summary_counts_all_names_and_rejects_abort(self):
        text = "ADD(x=1): OK\nMUL_MAT(x=2): OK\nTOP_K(x=3): not supported [MACA0]\n2/2 tests passed\nBackend MACA0: OK\n"
        counts = run.backend_case_summary({"exit_code": 0}, text, "MACA0")
        self.assertEqual(counts["ok"], 2)
        self.assertEqual(counts["unsupported"], 1)
        self.assertTrue(counts["completed"])
        self.assertFalse(run.backend_case_summary({"exit_code": 134}, text, "MACA0")["completed"])


    def test_failed_data_regression_continues_full_suite(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_step="data-cpy", false_success=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["data_regressions"]["CPY"]["accepted"])
        self.assertTrue(report["data_regressions"]["OUT_PROD"]["accepted"])
        self.assertTrue(report["default_ops_completed"])

    def test_exact_inventory_order_and_ansi(self):
        expected = [["CPY", "type=q8_0", "OK"], ["CPY", "type=iq2_s", "not supported [CPU]"]]
        text = "CPY(type=iq2_s): not supported [CPU]\n\x1b[32mCPY(type=q8_0): OK\x1b[0m\n"
        self.assertTrue(run.exact_case_summary(text, expected)["accepted"])

    def test_exact_inventory_missing_extra_duplicate_status_and_params(self):
        expected = [["CPY", "type=q8_0", "OK"], ["CPY", "type=iq2_s", "not supported [CPU]"]]
        lines = ["CPY(type=q8_0): OK\n", "CPY(type=iq2_s): not supported [CPU]\n"]
        original = "".join(lines)
        for text in (lines[0], original + lines[0], original.replace("iq2_s", "iq2_xs"),
                     original.replace("not supported [CPU]", "OK"), original.replace("OK", "FAIL")):
            with self.subTest(text=text):
                self.assertFalse(run.exact_case_summary(text, expected)["accepted"])

    def test_exact_inventory_preserves_duplicate_multiplicity(self):
        expected = [["CPY", "x=1", "OK"], ["CPY", "x=1", "OK"]]
        text = "CPY(x=1): OK\n"
        self.assertFalse(run.exact_case_summary(text, expected)["accepted"])
        self.assertTrue(run.exact_case_summary(text * 2, expected)["accepted"])

    def test_exact_inventory_filter_and_empty(self):
        expected = [["CPY", "x=1", "OK"], ["ADD", "x=1", "OK"]]
        self.assertTrue(run.exact_case_summary("CPY(x=1): OK\n", expected, "CPY")["accepted"])
        self.assertFalse(run.exact_case_summary("", expected)["accepted"])
        self.assertFalse(run.exact_case_summary("", [])["accepted"])

    def test_exact_inventory_orchestration(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True)
        self.assertEqual(code, 0)
        self.assertTrue(report["exact_case_inventory"]["accepted"])
        self.assertTrue(all(item["inventory"]["accepted"] for item in report["data_regressions"].values()))
        self.assertEqual(len(report["case_manifest_sha256"]), 64)

    def test_changed_case_rejected_despite_successful_counts(self):
        for stage in ("data-cpy", "ops-full"):
            with self.subTest(stage=stage), contextlib.redirect_stderr(io.StringIO()):
                code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, changed_case=stage)
                self.assertEqual(code, 1)
                if stage == "ops-full":
                    self.assertTrue(report["default_ops_completed"])
                    self.assertFalse(report["exact_case_inventory"]["accepted"])
                else:
                    self.assertTrue(report["data_regressions"]["CPY"]["completed"])
                    self.assertFalse(report["data_regressions"]["CPY"]["accepted"])
                    self.assertTrue(report["default_ops_completed"])

    def test_sum_failure_does_not_block_xielu_flash_or_full(self):
        text = ("SUM(mock=1): FAIL\n0/1 tests passed\nBackend MACA0: FAIL\n")
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True,
                                          step_outputs={"data-sum": (1, text)})
        self.assertEqual(code, 1)
        self.assertEqual(report["data_regressions"]["SUM"]["failed"], 1)
        self.assertTrue(report["data_regressions"]["XIELU"]["accepted"])
        self.assertTrue(report["data_regressions"]["FLASH_ATTN_EXT"]["accepted"])
        self.assertTrue(report["default_ops_completed"])
        self.assertTrue(report["test_collection_completed"])
        self.assertEqual([item["stage"] for item in report["test_failures"]], ["data-sum"])

    def test_multiple_failures_all_remain_in_report(self):
        failures = {"ops-add": 1, "data-sum": 1, "data-flash_attn_ext": -6,
                    "matmul-original-layout": 124, "moe-duplicate-ids": 1}
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True,
                                          bad_steps=failures)
        self.assertEqual(code, 1)
        self.assertEqual({item["stage"]: item["exit_code"] for item in report["test_failures"]}, failures)
        self.assertTrue(report["default_ops_completed"])
        self.assertTrue(report["test_collection_completed"])
        self.assertFalse(report["requested_checks_passed"])

    def test_cpu_oracle_uses_cpu_and_does_not_hide_failure(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True)
        self.assertEqual(code, 0)
        self.assertTrue(report["cpu_sum_oracle"]["accepted"])
        command = next(item["command"] for item in report["steps"] if item["log"] == "cpu-sum-oracle.log")
        self.assertEqual(command[command.index("-b") + 1], "CPU")
        for output in ((1, "SUM(x=1): FAIL\nBackend CPU: FAIL\n"),
                       (0, "SUM(x=1): OK\n1/1 tests passed\nBackend CPU: OK\n")):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
                                              step_outputs={"cpu-sum-oracle": output})
            self.assertEqual(code, 1)
            self.assertFalse(report["cpu_sum_oracle"]["accepted"])
            self.assertTrue(report["default_ops_completed"])
            self.assertFalse(report["requested_checks_passed"])

    def test_new_groups_continue_after_independent_failure(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True,
                                          bad_steps={"data-rope": -6})
        self.assertEqual(code, 1)
        for op in ("GET_ROWS", "CONV_2D", "LIGHTNING_INDEXER", "FLASH_ATTN_EXT"):
            self.assertTrue(report["data_regressions"][op]["accepted"])
        self.assertTrue(report["default_ops_completed"])

    def test_operator_crash_timeout_and_missing_binary_continue(self):
        for status in (-6, 124, 127):
            with self.subTest(status=status):
                code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
                                                  bad_steps={"data-cpy": status})
                self.assertEqual(code, 1)
                self.assertTrue(report["data_regressions"]["SUM"]["accepted"])
                self.assertTrue(report["default_ops_completed"])
                self.assertEqual(report["test_failures"][0]["exit_code"], status)

    def test_incomplete_ops_ignore_status_but_preserve_multiplicity(self):
        expected = [["SUM", "a=1", "OK"], ["CPY", "a=2", "OK"], ["CPY", "a=2", "OK"],
                    ["ADD", "a=3", "not supported [CPU]"]]
        text = "SUM(a=1): FAIL\nCPY(a=2): OK\n"
        self.assertEqual(run.incomplete_case_ops(text, expected), ["CPY", "ADD"])
        self.assertEqual(run.incomplete_case_ops(text + "CPY(a=2): OK\nADD(a=3): not supported [CPU]\n", expected), [])

    def test_full_abort_recovers_missing_groups_without_hiding_failure(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True,
                                          step_outputs={"ops-full": (-6, "ADD(mock=1): OK\n")})
        self.assertEqual(code, 1)
        self.assertFalse(report["default_ops_completed"])
        self.assertFalse(report["exact_case_inventory"]["accepted"])
        self.assertNotIn("ADD", report["recovery_requested_ops"])
        self.assertIn("FLASH_ATTN_EXT", report["recovery_requested_ops"])
        self.assertTrue(all(value["accepted"] for value in report["recovery_operators"].values()))
        self.assertTrue(report["test_collection_completed"])
        self.assertFalse(report["requested_checks_passed"])
        self.assertEqual([item["stage"] for item in report["test_failures"]], ["ops-full"])

    def test_recovery_failure_does_not_stop_other_recovery_groups(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True,
                                          bad_steps={"ops-full": -6, "ops-recovery-sum": 124})
        self.assertEqual(code, 1)
        self.assertFalse(report["recovery_operators"]["SUM"]["accepted"])
        self.assertTrue(report["recovery_operators"]["XIELU"]["accepted"])
        self.assertTrue(report["recovery_operators"]["FLASH_ATTN_EXT"]["accepted"])
        self.assertEqual([item["stage"] for item in report["test_failures"]], ["ops-full", "ops-recovery-sum"])

    def test_full_failure_without_manifest_does_not_guess_recovery(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_steps={"ops-full": -6})
        self.assertEqual(code, 1)
        self.assertIn("No reviewed case manifest", report["recovery_unavailable"])
        self.assertNotIn("recovery_operators", report)

    def test_completed_numerical_failure_does_not_repeat_full_ops(self):
        # No cases are missing, even though the only failure is still blocking.
        names = ("ADD", "MUL_MAT", *run.DATA_OPS)
        text = "".join(name + "(mock=1): " + ("FAIL" if name == "SUM" else "OK") + "\n" for name in names)
        text += "%d/%d tests passed\nBackend MACA0: FAIL\n" % (len(names) - 1, len(names))
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
                                          step_outputs={"ops-full": (1, text)})
        self.assertEqual(code, 1)
        self.assertEqual(report["recovery_requested_ops"], [])
        self.assertEqual(report["full_operators"]["failed"], 1)

    def test_user_interrupt_stops_without_recovery(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
                                              interrupt_step="data-sum")
        self.assertEqual(code, 130)
        self.assertEqual(report["error"], "Interrupted")
        self.assertFalse(report["test_collection_completed"])
        self.assertNotIn("XIELU", report["data_regressions"])
        self.assertNotIn("full_operators", report)

    def test_recovery_interrupt_stops_remaining_groups(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
                                              bad_steps={"ops-full": -6}, interrupt_step="ops-recovery-sum")
        self.assertEqual(code, 130)
        self.assertNotIn("XIELU", report["recovery_operators"])
        self.assertFalse(report["test_collection_completed"])

    def test_manifest_names_cannot_be_paths_or_regexes(self):
        for name in ("../ADD", "ADD/MUL", "ADD.*", "ADD,MUL", "", "A\\B"):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.simulate_run(mode="backend", full_ops=True, inventory=True, manifest_op=name)

    def test_prerequisite_failures_stop_before_independent_tests(self):
        for stage in ("configure-backend", "build-test-backend-ops", "backend-devices"):
            with self.subTest(stage=stage), contextlib.redirect_stderr(io.StringIO()):
                code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, bad_steps={stage: 1})
                self.assertEqual(code, 1)
                self.assertFalse(report["test_collection_completed"])
                self.assertFalse(report["requested_checks_passed"])
                self.assertEqual(report["operators"], {})


if __name__ == "__main__":
    unittest.main()
