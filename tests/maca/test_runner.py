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
                     manifest_op=None, host_reference=False, manifest_host=None, fa_seed=None, fa_replay=False):
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
            mock_ops = tuple(dict.fromkeys(("ADD", "MUL_MAT", *run.DATA_OPS))) + (("FLASH_ATTN_EXT",) if flash_attn else ())

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
                elif log.stem == "quantize-host-reference":
                    text = "PASS quantizer host reference: 540/540; NOT a GPU test\n"
                elif log.stem.startswith("fa-replay-"):
                    inputs = "".join("%d:0123456789abcdef:100;" % i for i in range(4))
                    path_name = "maca_f32" if env["GGML_MACA_FA_FORCE_F32"] == "1" else "tile"
                    text = "FA_REPRO seed=%s case=mock=1 inputs=%s\n" % (env["LLAMA_TEST_FA_SEED"], inputs)
                    text += "MACA_FA_PATH kernel=%s k=f32 v=f32 nb=75 kv=1024\n" % path_name
                    text += "FLASH_ATTN_EXT(mock=1): OK\n1/1 tests passed\nBackend MACA0: OK\n"
                elif log.stem.startswith("host-positive-") or log.stem.startswith("host-negative-"):
                    op = command[command.index("-o") + 1]
                    negative = log.stem.startswith("host-negative-")
                    status = "FAIL" if negative else "OK"
                    code = 1 if negative else 0
                    text = "HOST_REFERENCE %s(mock=1) method=%s result=%s\n" % (op, run.HOST_METHODS[op], "FAIL" if negative else "PASS")
                    text += "%s(mock=1): %s\n%d/1 tests passed\nBackend CPU: %s\n" % (op, status, 0 if negative else 1, status)
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
                    if flash_attn and fa_seed is not None:
                        inputs = "".join("%d:0123456789abcdef:100;" % i for i in range(4))
                        text += "FA_REPRO seed=%d case=mock=1 inputs=%s\n" % (fa_seed, inputs)
                elif log.stem.startswith("ops-recovery-"):
                    op = log.stem[len("ops-recovery-"):].upper()
                    text = op + "(mock=1): OK\n  1/1 tests passed\n  Backend MACA0: OK\n"
                elif log.stem.startswith("data-"):
                    op = log.stem[len("data-"):].upper()
                    text = op + "(mock=1): OK\n  1/1 tests passed\n  Backend MACA0: OK\n"
                elif log.stem.startswith("ops-"):
                    op = log.stem[len("ops-"):].upper()
                    text = op + "(mock=1): OK\n  1/1 tests passed\n  Backend MACA0: OK\n"
                if host_reference and not log.stem.startswith("host-"):
                    for op, method in run.HOST_METHODS.items():
                        if op + "(mock=1): OK" in text:
                            text += "HOST_REFERENCE %s(mock=1) method=%s result=PASS\n" % (op, method)
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
            if host_reference:
                argv.append("--host-reference")
            if fa_seed is not None:
                argv += ["--fa-seed", str(fa_seed)]
            if fa_replay:
                argv.append("--fa-replay")
            if inventory:
                manifest = path / "expected.json.gz"
                data = {"base_commit": run.BASE_COMMIT, "device": "MACA0",
                        "flash_attention": flash_attn if manifest_flash is None else manifest_flash,
                        "cases": [[op, "mock=1", "OK"] for op in mock_ops]}
                data["host_reference"] = host_reference if manifest_host is None else manifest_host
                data["fa_replay_case"] = "mock=1"
                if host_reference:
                    data["host_reference_cases"] = [[op, "mock=1", method] for op, method in run.HOST_METHODS.items()]
                    data["host_reference_controls"] = data["host_reference_cases"]
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

    def test_quantize_host_audit_success(self):
        code, report, _ = self.simulate_run(mode="backend")
        self.assertEqual(code, 0)
        self.assertEqual(report["quantize_host_reference"], {"accepted": True, "expected": 540, "gpu_test": False})

    def test_quantize_host_audit_failure_does_not_stop_other_groups(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
                                           bad_step="quantize-host-reference")
        self.assertEqual(code, 1)
        self.assertFalse(report["quantize_host_reference"]["accepted"])
        self.assertTrue(report["exact_case_inventory"]["accepted"])

    def test_quantize_host_audit_missing_marker_is_not_success(self):
        code, report, _ = self.simulate_run(mode="backend", step_outputs={"quantize-host-reference": (0, "")})
        self.assertEqual(code, 1)
        self.assertFalse(report["quantize_host_reference"]["accepted"])

    def test_quantize_host_audit_nonzero_with_marker_is_not_success(self):
        text = "PASS quantizer host reference: 540/540; NOT a GPU test\n"
        code, report, _ = self.simulate_run(mode="backend", step_outputs={"quantize-host-reference": (1, text)})
        self.assertEqual(code, 1)
        self.assertFalse(report["quantize_host_reference"]["accepted"])

    def test_host_reference_success_and_provenance(self):
        code, report, environments = self.simulate_run(mode="backend", full_ops=True, inventory=True, host_reference=True)
        self.assertEqual(code, 0)
        self.assertTrue(report["host_reference_inventory"]["accepted"])
        self.assertEqual(report["host_reference_inventory"]["observed_rows"], len(run.HOST_METHODS))
        self.assertEqual(len(report["host_reference_controls"]), 2*len(run.HOST_METHODS))
        self.assertTrue(all(x["accepted"] for x in report["host_reference_controls"]))
        self.assertTrue(report["comparison_provenance"]["cpu_backend_support_unchanged"])
        self.assertTrue(any(e.get("LLAMA_TEST_HOST_REFERENCE_FAULT") == "1" for e in environments))
        self.assertNotIn("LLAMA_TEST_HOST_REFERENCE_FAULT", environments[-1])
        self.assertEqual(environments[-1]["LLAMA_TEST_HOST_REFERENCE"], "1")

    def test_host_reference_missing_marker_rejected(self):
        text = "CPY(mock=1): OK\n1/1 tests passed\nBackend MACA0: OK\n"
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, host_reference=True,
                                          step_outputs={"data-cpy": (0, text)})
        self.assertEqual(code, 1)
        self.assertFalse(report["data_regressions"]["CPY"]["accepted"])
        self.assertTrue(report["default_ops_completed"])

    def test_host_reference_positive_failure_continues(self):
        code, report, environments = self.simulate_run(mode="backend", full_ops=True, inventory=True, host_reference=True,
                                                      bad_steps={"host-positive-0-cpy": 1})
        self.assertEqual(code, 1)
        self.assertTrue(report["default_ops_completed"])
        self.assertNotIn("LLAMA_TEST_HOST_REFERENCE_FAULT", environments[-1])

    def test_host_reference_wrong_mode_rejected(self):
        with self.assertRaises(SystemExit):
            self.simulate_run(mode="backend", full_ops=True, inventory=True, host_reference=True, manifest_host=False)

    def test_inherited_reference_flags_removed(self):
        with patch.dict(os.environ, {"LLAMA_TEST_HOST_REFERENCE": "check", "LLAMA_TEST_HOST_REFERENCE_FAULT": "1"}):
            _, _, environments = self.simulate_run(mode="backend")
        self.assertTrue(all("LLAMA_TEST_HOST_REFERENCE" not in e and "LLAMA_TEST_HOST_REFERENCE_FAULT" not in e for e in environments))

    def test_inherited_fa_debug_flags_removed(self):
        flags = {"LLAMA_TEST_FA_SEED": "99", "GGML_MACA_FA_FORCE_F32": "1", "GGML_MACA_FA_DIAG": "1"}
        with patch.dict(os.environ, flags):
            _, _, environments = self.simulate_run(mode="backend")
        self.assertTrue(all(all(key not in e for key in flags) for e in environments))

    def test_fa_replay_runs_both_paths_and_all_seeds(self):
        code, report, envs = self.simulate_run(mode="backend", full_ops=True, inventory=True,
            flash_attn=True, fa_seed=7, fa_replay=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(report["fa_replay"]), 32)
        self.assertTrue(all(item["accepted"] for item in report["fa_replay"]))
        self.assertTrue(report["fa_full_replay_inputs_match"])
        self.assertEqual({item["seed"] for item in report["fa_replay"]}, set(range(7, 23)))
        self.assertTrue(all("GGML_MACA_FA_FORCE_F32" not in e for e in envs if "GGML_MACA_FA_DIAG" not in e))

    def test_full_failure_not_hidden_by_fa_replay(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
            flash_attn=True, fa_seed=7, fa_replay=True, bad_steps={"ops-full": 1, "fa-replay-00-tile": -6})
        self.assertEqual(code, 1)
        self.assertEqual(len(report["fa_replay"]), 32)
        self.assertFalse(report["fa_replay"][0]["accepted"])
        self.assertTrue(report["fa_replay"][-1]["accepted"])
        self.assertFalse(report["requested_checks_passed"])

    def test_fa_replay_mismatched_inputs_rejected(self):
        inputs = "".join("%d:ffffffffffffffff:100;" % i for i in range(4))
        text = "FA_REPRO seed=7 case=mock=1 inputs=" + inputs + "\nMACA_FA_PATH kernel=maca_f32\n"
        text += "FLASH_ATTN_EXT(mock=1): OK\n1/1 tests passed\nBackend MACA0: OK\n"
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
            flash_attn=True, fa_seed=7, fa_replay=True, step_outputs={"fa-replay-00-f32": (0, text)})
        self.assertEqual(code, 1)
        self.assertTrue(any("fingerprints do not match" in item["reason"] for item in report["test_failures"]))

    def test_invalid_fa_seed_rejected(self):
        for seed in (-1, 2**64):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.simulate_run(mode="backend", flash_attn=True, fa_seed=seed)

    def test_fa_full_isolated_input_mismatch_rejected(self):
        inputs = "".join("%d:ffffffffffffffff:100;" % i for i in range(4))
        text = "FA_REPRO seed=7 case=mock=1 inputs=" + inputs + "\n"
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True,
            flash_attn=True, fa_seed=7, fa_replay=True, step_outputs={"ops-full": (1, text)})
        self.assertEqual(code, 1)
        self.assertFalse(report["fa_full_replay_inputs_match"])
        self.assertTrue(all(item["accepted"] for item in report["fa_replay"]))
        self.assertTrue(any("full/isolated input fingerprints" in item["reason"] for item in report["test_failures"]))

    def test_fa_replay_requires_seed_exact_case_hashes_and_path(self):
        inputs = "".join("%d:0123456789abcdef:100;" % i for i in range(4))
        text = ("FA_REPRO seed=7 case=mock=1 inputs=" + inputs + "\n"
                "MACA_FA_PATH kernel=tile k=f32 v=f32 nb=75 kv=1024\n"
                "FLASH_ATTN_EXT(mock=1): OK\n1/1 tests passed\nBackend MACA0: OK\n")
        self.assertTrue(run.fa_replay_summary({"exit_code": 0}, text, "mock=1", 7, False)["accepted"])
        self.assertTrue(run.fa_replay_summary({"exit_code": 0}, text.replace("kernel=tile", "kernel=maca_f32"),
                                              "mock=1", 7, True)["accepted"])
        for bad in (text.replace("seed=7", "seed=8"), text.replace("case=mock=1", "case=wrong"),
                    text.replace("inputs=" + inputs, "inputs="), text.replace("kernel=tile", "kernel=maca_f32"),
                    text.replace("1/1", "0/1"), text.replace("FLASH_ATTN_EXT(mock=1): OK", "FLASH_ATTN_EXT(mock=1): FAIL"),
                    text + "FA_REPRO seed=7 case=mock=1 inputs=" + inputs + "\n"):
            self.assertFalse(run.fa_replay_summary({"exit_code": 0}, bad, "mock=1", 7, False)["accepted"])
        for code in (1, 124, -6, 127):
            self.assertFalse(run.fa_replay_summary({"exit_code": code}, text, "mock=1", 7, False)["accepted"])

    def test_reference_marker_exact_multiset(self):
        case = ["CPY", "type=q8_0", "copy_bytes"]
        line = "HOST_REFERENCE CPY(type=q8_0) method=copy_bytes result=PASS\n"
        self.assertTrue(run.host_reference_summary(line, [case])["accepted"])
        for text in ("", line * 2, line.replace("q8_0", "f32"), line.replace("PASS", "FAIL"), line.replace("copy_bytes", "other")):
            self.assertFalse(run.host_reference_summary(text, [case])["accepted"])

    def test_reference_negative_requires_numerical_failure(self):
        case = ["CPY", "type=q8_0", "copy_bytes"]
        text = ("HOST_REFERENCE CPY(type=q8_0) method=copy_bytes result=FAIL\n"
                "CPY(type=q8_0): FAIL\n0/1 tests passed\nBackend CPU: FAIL\n")
        self.assertTrue(run.reference_negative_accepted({"exit_code": 1}, text, case))
        for code in (0, 2, -6, 124, 127, 134):
            self.assertFalse(run.reference_negative_accepted({"exit_code": code}, text, case))
        for bad in (text.replace("0/1 tests passed", ""), text.replace("type=q8_0", "type=f32"),
                    text.replace("CPY(type=q8_0): FAIL", "CPY(type=q8_0): not supported [CPU]")):
            self.assertFalse(run.reference_negative_accepted({"exit_code": 1}, bad, case))

    def test_reference_negative_crash_continues_but_rejects(self):
        code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, host_reference=True,
                                          bad_steps={"host-negative-cpy": -6})
        self.assertEqual(code, 1)
        self.assertTrue(report["default_ops_completed"])
        self.assertFalse(report["host_reference_controls"][1]["accepted"])

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

    def test_large_batch_failures_do_not_cancel_independent_groups(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, inventory=True, flash_attn=True,
                bad_steps={"data-cpy": 1, "data-mul_mat": 134, "data-argsort": 124})
        self.assertEqual(code, 1)
        for op in ("CPY", "MUL_MAT", "ARGSORT"):
            self.assertFalse(report["data_regressions"][op]["accepted"])
        for op in ("MUL_MAT_ID", "FLASH_ATTN_EXT", "GET_ROWS"):
            self.assertTrue(report["data_regressions"][op]["accepted"])
        self.assertTrue(report["default_ops_completed"])
        self.assertFalse(report["requested_checks_passed"])

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
        self.assertFalse(report["moe_duplicate_ids"]["accepted"])
        self.assertEqual(report["moe_duplicate_ids"]["status"], "not_run_upstream_undefined")
        self.assertNotIn("moe-duplicate-ids.log", [item["log"] for item in report["steps"]])
        command = next(item["command"] for item in report["steps"] if item["log"] == "ops-full.log")
        self.assertNotIn("-o", command)
        self.assertNotIn("-p", command)

    def test_failed_layout_continues_full_suite(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_step="matmul-quant-layout", false_success=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["layout_regressions"]["matmul-quant-layout"]["accepted"])
        self.assertFalse(report["moe_duplicate_ids"]["accepted"])
        self.assertTrue(report["default_ops_completed"])

    def test_failed_full_suite_cannot_be_hidden_by_smoke_pass(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_step="ops-full", false_success=True)
        self.assertEqual(code, 1)
        self.assertTrue(report["backend_smoke_completed"])
        self.assertFalse(report["default_ops_completed"])

    def test_failed_moe_regression_continues_full_suite(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code, report, _ = self.simulate_run(mode="backend", full_ops=True, bad_step="data-mul_mat_id", false_success=True)
        self.assertEqual(code, 1)
        self.assertTrue(report["backend_smoke_completed"])
        self.assertFalse(report["data_regressions"]["MUL_MAT_ID"]["accepted"])
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

    def test_historical_inventory_rejected_after_rebase(self):
        manifest = Path(__file__).resolve().parent / "baselines/c500-ops-20261017.json.gz"
        with gzip.open(manifest, "rt", encoding="utf-8") as source:
            historical = json.load(source)
        self.assertNotEqual(historical["base_commit"], run.BASE_COMMIT)
        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run.py", "--build-dir", str(Path(temporary) / "build"),
                    "--mode", "backend", "--full-ops", "--flash-attn", "--host-reference",
                    "--case-manifest", str(manifest)]
            with patch.object(sys, "argv", argv), patch.object(sys, "platform", "linux"), \
                 patch("run.execute") as execute, contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    run.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Case inventory baseline or device does not match", errors.getvalue())
            execute.assert_not_called()
            self.assertFalse((Path(temporary) / "build").exists())

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
                    "matmul-original-layout": 124, "data-mul_mat_id": 1}
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
