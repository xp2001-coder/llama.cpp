#!/usr/bin/env python3
"""Collect MACA probe or experimental backend bring-up evidence."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "dd002b91f583af0b80140f9a140dc04856bfc2e0"
DEVICE_PROBES = ("bridge-runtime", "warp", "fp16", "bf16", "blas")
BUILD_ERROR = re.compile(r"(?m)^FAILED:|^CMake Error|^ninja: (?:error:|build stopped)|\b(?:fatal )?error:|undefined reference to")
BACKEND_TARGETS = ("test-maca-quantize-reference", "llama-bench", "llama-server", "llama-cli", "test-backend-ops")
SMOKE_OPS = ("ADD", "MUL", "RMS_NORM", "SOFT_MAX", "MUL_MAT")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CASE_ROW = re.compile(r"(?m)(?:^\s*|[^\S\r\n]{2,})([A-Z][A-Z0-9_]*)\(([^\n]*)\): ([^\n]*)$")
DATA_OPS = ("IM2COL_3D", "SET_ROWS", "CPY", "OUT_PROD", "NORM", "L2_NORM",
            "ACC", "GET_ROWS_BACK", "REPEAT", "CONCAT", "SUM", "XIELU",
            "GET_ROWS", "ROPE", "CONV_2D", "LIGHTNING_INDEXER", "MUL_MAT", "MUL_MAT_ID", "ARGSORT", "SOFT_MAX_BACK")
HOST_METHODS = {"CPY": "copy_bytes", "OUT_PROD": "outer_product_f64", "CONV_2D": "conv_contiguous_cpu",
                "MUL_MAT": "matmul_f64", "GET_ROWS_BACK": "rows_back_f64", "SOFT_MAX_BACK": "softmax_back_f64",
                "SET_ROWS": "quantize_rows"}
HOST_EXTRA_METHODS = {"CPY": {"quantize_copy"}}
HOST_ROW = re.compile(r"(?m)^HOST_REFERENCE ([A-Z][A-Z0-9_]*)\(([^\n]*)\) method=(\w+) result=(PASS|FAIL)\r?$")


def fa_input_fingerprint(text, params, seed):
    markers = re.findall(r"(?m)^FA_REPRO seed=(\d+) case=(.*?) inputs=([^\r\n]+)\r?$", ANSI.sub("", text))
    selected = [inputs for actual_seed, case, inputs in markers if (actual_seed, case) == (str(seed), params)]
    if len(selected) != 1:
        return None
    value = selected[0]
    if (re.fullmatch(r"(?:[0-3]:[0-9a-f]{16}:[1-9][0-9]*;){4}", value) is None or
            [int(item.split(":")[0]) for item in value.split(";")[:-1]] != [0, 1, 2, 3]):
        return None
    return value


def fa_replay_summary(record, text, params, seed, forced):
    text = ANSI.sub("", text)
    counts = backend_case_summary(record, text, "MACA0", "FLASH_ATTN_EXT")
    exact = exact_case_summary(text, [["FLASH_ATTN_EXT", params, "OK"]])
    markers = re.findall(r"(?m)^FA_REPRO seed=(\d+) case=(.*?) inputs=([^\r\n]+)\r?$", text)
    paths = re.findall(r"MACA_FA_PATH kernel=(\w+)", text)
    fingerprint = fa_input_fingerprint(text, params, seed)
    valid = len(markers) == 1 and fingerprint is not None
    expected_path = "maca_f32" if forced else "tile"
    return dict(accepted=counts["completed"] and exact["accepted"] and valid and set(paths) == {expected_path},
                counts=counts, inventory=exact, markers=markers, paths=paths,
                input_fingerprint=fingerprint if valid else None)


def host_reference_summary(text, expected, op=None, result="PASS"):
    observed = Counter((name, params, method, status) for name, params, method, status in HOST_ROW.findall(ANSI.sub("", text))
                       if op is None or name == op)
    wanted = Counter((*row, result) for row in expected if op is None or row[0] == op)
    missing, extra = wanted - observed, observed - wanted
    return {"accepted": not missing and not extra, "expected_rows": sum(wanted.values()),
            "observed_rows": sum(observed.values()),
            "missing": [[*row, count] for row, count in sorted(missing.items())],
            "extra": [[*row, count] for row, count in sorted(extra.items())]}


def reference_negative_accepted(record, text, case):
    # A crash, timeout, missing test, or unsupported case is not a valid negative control.
    rows = [(name, params, status.strip()) for name, params, status in CASE_ROW.findall(ANSI.sub("", text))]
    return (record["exit_code"] == 1 and len(rows) == 1 and rows[0][:2] == tuple(case[:2])
            and rows[0][2].endswith("FAIL") and not BUILD_ERROR.search(text)
            and host_reference_summary(text, [case], result="FAIL")["accepted"]
            and re.search(r"(?m)^\s*0/1 tests passed\r?$", ANSI.sub("", text)) is not None
            and re.search(r"(?m)^\s*Backend CPU: FAIL\r?$", ANSI.sub("", text)) is not None)


def exact_case_summary(text, expected, op=None):
    observed = Counter((name, params, status.strip()) for name, params, status in CASE_ROW.findall(ANSI.sub("", text))
                       if op is None or name == op)
    wanted = Counter(tuple(row) for row in expected if op is None or row[0] == op)
    missing = wanted - observed
    extra = observed - wanted
    return {"accepted": bool(wanted) and not missing and not extra,
            "expected_rows": sum(wanted.values()), "observed_rows": sum(observed.values()),
            "missing": [[*row, count] for row, count in sorted(missing.items())],
            "extra": [[*row, count] for row, count in sorted(extra.items())]}


def incomplete_case_ops(text, expected):
    observed = Counter((name, params) for name, params, _ in CASE_ROW.findall(ANSI.sub("", text)))
    wanted = Counter((name, params) for name, params, _ in expected)
    missing = {name for name, _ in wanted - observed}
    return list(dict.fromkeys(name for name, _, _ in expected if name in missing))


def backend_case_summary(record, text, device, op=None):
    text = ANSI.sub("", text)
    rows = [status.strip() for name, _, status in CASE_ROW.findall(text) if op is None or name == op]
    counts = {"ok": rows.count("OK"), "failed": sum(row.endswith("FAIL") for row in rows),
              "unsupported": sum(row.startswith("not supported") for row in rows),
              "skipped": sum(row.startswith("skipping large tensors for speed") for row in rows)}
    counts["unparsed"] = len(rows) - sum(counts.values())
    summaries = re.findall(r"(?m)^\s*(\d+)/(\d+) tests passed\r?$", text)
    counts["completed"] = (record["exit_code"] == 0 and counts["ok"] > 0 and counts["failed"] == 0 and counts["unparsed"] == 0
                           and summaries == [(str(counts["ok"]), str(counts["ok"]))]
                           and re.search(r"(?m)^\s*Backend " + re.escape(device) + r": OK\r?$", text) is not None
                           and not BUILD_ERROR.search(text))
    return counts


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def claim_build(path, identity):
    marker = path / "maca-probe-owner.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("Build identity changed. Select a new --build-dir; existing files are preserved.")
    else:
        if path.exists() and any(path.iterdir()):
            raise ValueError("Refusing a nonempty build directory without a MACA probe ownership marker.")
        path.mkdir(parents=True, exist_ok=True)
        write_json(marker, identity)


def execute(command, log, env, timeout):
    print("Running " + log.stem, flush=True)
    code = 127
    try:
        with log.open("w", encoding="utf-8") as output:
            with subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                  env=env, start_new_session=(os.name == "posix")) as process:
                try:
                    code = process.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    # SDK wrappers can spawn compiler children. Stop the whole group.
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait()
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    code = 124
                    output.write("\nProbe command timed out; process group stopped.\n")
    except OSError as error:
        with log.open("a", encoding="utf-8") as output:
            output.write(str(error) + "\n")
    text = log.read_text(encoding="utf-8", errors="replace")
    return {"command": command, "exit_code": code, "log": log.name}, text


def accepted_build(record, text):
    return record["exit_code"] == 0 and not BUILD_ERROR.search(text)


def accepted_probe(name, record, text):
    return record["exit_code"] == 0 and text.splitlines().count("MACA_PROBE_PASS " + name) == 1


def pin_target(build, target):
    marker = build / "maca-compiler-target.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != target:
        raise RuntimeError("Compiler target changed. Select a new --build-dir to avoid stale device objects.")
    write_json(marker, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=Path(os.environ.get("MACA_PATH", "/opt/maca")))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("runtime", "device", "all", "backend"), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--build-timeout", type=int, default=1200)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument("--full-ops", action="store_true", help="After backend smoke tests, run layout regressions and the full default operator suite")
    parser.add_argument("--case-manifest", type=Path, help="Gzipped reviewed case inventory for exact full-ops acceptance")
    parser.add_argument("--flash-attn", action="store_true", help="Enable the generic Flash Attention candidate (backend mode only)")
    parser.add_argument("--host-reference", action="store_true", help="Validate CPU-unsupported cases with explicit test-only host references")
    parser.add_argument("--fa-seed", type=int, help="Reproducible FA input seed (uint64); other operators retain their default initialization")
    parser.add_argument("--fa-replay", action="store_true", help="After the full suite, compare tile/FP32 paths on identical seeded FA inputs")
    parser.add_argument("--quantize-fp-mode", choices=("bridge", "clang", "clang-strict"),
                        help="Override quantizer-only FP options after the device arithmetic probe")
    args = parser.parse_args()
    if args.device < 0 or min(args.jobs, args.build_timeout, args.run_timeout) < 1:
        parser.error("device must be nonnegative; jobs and timeouts must be positive")
    if sys.platform != "linux":
        parser.error("Hardware probes require a Linux MACA installation")
    if args.full_ops and args.mode != "backend":
        parser.error("--full-ops requires --mode backend")
    if args.flash_attn and args.mode != "backend":
        parser.error("--flash-attn requires --mode backend")
    if args.quantize_fp_mode and args.mode != "backend":
        parser.error("--quantize-fp-mode requires --mode backend")
    if args.fa_seed is not None and (not args.flash_attn or not 0 <= args.fa_seed < 2**64):
        parser.error("--fa-seed requires --flash-attn and a uint64 value")
    if args.fa_replay and (args.fa_seed is None or not args.full_ops or not args.case_manifest or args.device != 0):
        parser.error("--fa-replay requires --fa-seed, --full-ops, a reviewed manifest and device 0")
    expected_cases = None
    reference_cases = []
    reference_controls = []
    if args.host_reference and (not args.full_ops or not args.case_manifest):
        parser.error("--host-reference requires --full-ops and a reviewed --case-manifest")
    if args.case_manifest:
        if not args.full_ops:
            parser.error("--case-manifest requires --full-ops")
        with gzip.open(args.case_manifest, "rt", encoding="utf-8") as source:
            inventory = json.load(source)
        if inventory.get("base_commit") != BASE_COMMIT or inventory.get("device") != "MACA" + str(args.device):
            parser.error("Case inventory baseline or device does not match")
        if inventory.get("flash_attention", False) != args.flash_attn:
            parser.error("Case inventory Flash Attention mode does not match --flash-attn")
        if inventory.get("host_reference", False) != args.host_reference:
            parser.error("Case inventory host reference mode does not match --host-reference")
        expected_cases = inventory["cases"]
        if not expected_cases or any(not isinstance(row, list) or len(row) != 3 or
                                     not all(isinstance(value, str) for value in row) for row in expected_cases):
            parser.error("Invalid case inventory")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", row[0]) for row in expected_cases):
            parser.error("Invalid operator name in case inventory")
        if args.host_reference:
            reference_cases = inventory.get("host_reference_cases", [])
            reference_controls = inventory.get("host_reference_controls", [])
            for rows in (reference_cases, reference_controls):
                if not rows or any(not isinstance(row, list) or len(row) != 3 or
                                   not all(isinstance(value, str) for value in row) for row in rows):
                    parser.error("Invalid host reference case list")
                if any((HOST_METHODS.get(row[0]) != row[2] and row[2] not in HOST_EXTRA_METHODS.get(row[0], set())) or
                       [row[0], row[1], "OK"] not in expected_cases for row in rows):
                    parser.error("Host reference method or case does not match inventory")
                if len(set(map(tuple, rows))) != len(rows) or set(row[0] for row in rows) != set(HOST_METHODS):
                    parser.error("Host reference inventory has duplicates or missing operator groups")
            if set(row[2] for row in reference_controls) != set(row[2] for row in reference_cases):
                parser.error("Every host reference method needs positive and negative controls")
        if args.fa_replay and ["FLASH_ATTN_EXT", inventory.get("fa_replay_case"), "OK"] not in expected_cases:
            parser.error("FA replay case is absent from the reviewed inventory")

    sdk = args.sdk.resolve()
    build = args.build_dir.resolve()
    if not (sdk / "include/mcr/mc_runtime.h").is_file():
        parser.error("SDK runtime header not found: " + str(sdk))
    for tool in ("cmake", "ninja"):
        if not shutil.which(tool):
            parser.error("Required tool not found: " + tool)
    bridge = sdk / "tools/cu-bridge"
    if args.mode != "runtime":
        for name in ("cmake_mock", "cmake_maca", "multi_target.sh"):
            if not (bridge / "tools" / name).is_file():
                parser.error("Required SDK tool not found: " + name)
        if not shutil.which("bash"):
            parser.error("Required tool not found: bash")

    env = os.environ.copy()
    env.pop("LLAMA_TEST_HOST_REFERENCE", None)
    env.pop("LLAMA_TEST_HOST_REFERENCE_FAULT", None)
    for key in ("LLAMA_TEST_FA_SEED", "GGML_MACA_FA_FORCE_F32", "GGML_MACA_FA_DIAG"):
        env.pop(key, None)
    if args.fa_seed is not None:
        env["LLAMA_TEST_FA_SEED"] = str(args.fa_seed)
    if args.host_reference:
        env["LLAMA_TEST_HOST_REFERENCE"] = "1"
    sdk_libraries = [str(sdk / "lib64"), str(sdk / "lib"), str(bridge / "lib")]
    env["LD_LIBRARY_PATH"] = ":".join(sdk_libraries + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    # Do not inherit a virtual CUDA tree or compiler settings from an old build.
    for name in ("WCUDA_HOME", "CUDA_HOME", "CUDA_PATH", "CUDACXX", "CUDAHOSTCXX", "MACA_VIRTUAL_ROOT", "MACA_VIRTUAL_CUDA"):
        env.pop(name, None)
    env.update(MACA_PATH=str(sdk), MACA_ROOT=str(sdk), CUCC_PATH=str(bridge))
    if args.mode == "backend":
        env["GGML_CUDA_DISABLE_FUSION"] = "1"
        # Inherited plugins and device filters can select an unrelated backend build.
        for name in ("GGML_BACKEND_PATH", "GGML_CUDA_P2P", "GGML_CUDA_VIRTUAL_DEVICES", "GGML_CUDA_DEVICES"):
            if env.get(name):
                parser.error("Unset " + name + " before backend bring-up")
    identity = {"source": str(ROOT), "sdk": str(sdk), "base_commit": BASE_COMMIT,
                "runtime_header_sha256": hashlib.sha256((sdk / "include/mcr/mc_runtime.h").read_bytes()).hexdigest(),
                "cc": env.get("CC", ""), "cxx": env.get("CXX", ""),
                "cucc_targets": env.get("CUCC_TARGETS", ""),
                "cucc_targets_from_device": env.get("CUCC_TARGETS_FROM_DEVICE", "")}
    claim_build(build, identity)
    results = build / "results"
    results.mkdir(exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="run-", dir=results))
    report = {"base_commit": BASE_COMMIT, "sdk": str(sdk), "device": args.device,
              "scope": "standalone API and device probes, not full ggml operator or model acceptance",
              "backend_accepted": False, "steps": [], "probes": {}, "source_sha256": {}}
    if args.mode == "backend":
        report.update(scope="backend build, device enumeration and selected upstream CPU-reference comparisons",
                      backend_smoke_completed=False, operator_coverage_accepted=False,
                      fusion_enabled=False, graph_enabled=False, flash_attention_enabled=args.flash_attn, operators={},
                      full_ops_requested=args.full_ops, default_ops_completed=False,
                      test_failures=[], test_collection_completed=False, requested_checks_passed=False)
        report["host_reference_enabled"] = args.host_reference
        report["fa_seed"] = args.fa_seed
        report["fa_replay_requested"] = args.fa_replay
        report["quantize_fp_mode_requested"] = args.quantize_fp_mode
        if args.host_reference:
            report["scope"] = "upstream cases with separate CPU-backend and test-only host-reference comparisons; not full backend acceptance"
    report["sdk_file_sha256"] = {}
    for relative in ("include/mcr/mc_runtime.h", "include/mcr/mc_runtime_api.h", "tools/cu-bridge/tools/cmake_mock",
                     "tools/cu-bridge/tools/cmake_maca", "tools/cu-bridge/include/bridge/runtime/cuda_to_maca_mcr_adaptor.h"):
        path = sdk / relative
        if path.is_file():
            report["sdk_file_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = list((ROOT / "tests/maca").glob("*")) + [ROOT / "ggml/src/ggml-cuda/vendors/maca.h"]
    if args.mode == "backend":
        for directory in ("ggml", "src", "common", "tools"):
            sources.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
        sources += [ROOT / "CMakeLists.txt", ROOT / "tests/test-backend-ops.cpp", ROOT / "tests/CMakeLists.txt"]
    for source in sorted(sources):
        if source.is_file():
            report["source_sha256"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    print("Results: " + str(session), flush=True)
    if expected_cases is not None:
        data = args.case_manifest.read_bytes()
        (session / "expected-cases.json.gz").write_bytes(data)
        report["case_manifest_sha256"] = hashlib.sha256(data).hexdigest()

    def step(name, command, timeout):
        record, text = execute(command, session / (name + ".log"), env, timeout)
        report["steps"].append(record)
        write_json(session / "report.json", report)
        return record, text

    def test_failure(record, reason):
        report["test_failures"].append({"stage": Path(record["log"]).stem, "log": record["log"],
                                        "exit_code": record["exit_code"], "reason": reason})
        write_json(session / "report.json", report)
        print("FAILED " + record["log"] + ": " + reason + "; continuing independent checks", flush=True)

    def suite(kind, names, configure, builder):
        record, text = step("configure-" + kind, configure, args.build_timeout)
        if not accepted_build(record, text):
            for name in names:
                report["probes"][name] = "configure_failed"
            return
        for name in names:
            target = "test-maca-" + name
            binary = build / kind / "bin" / target
            command = builder + ["--build", str(build / kind), "--target", target, "-j", str(args.jobs)]
            record, text = step("build-" + name, command, args.build_timeout)
            if not accepted_build(record, text) or not binary.is_file():
                report["probes"][name] = "build_failed"
                continue
            report.setdefault("binary_sha256", {})[name] = hashlib.sha256(binary.read_bytes()).hexdigest()
            record, text = step("run-" + name, [str(binary), str(args.device)], args.run_timeout)
            report["probes"][name] = "pass" if accepted_probe(name, record, text) else "run_failed"

    def backend_suite(configure, builder):
        record, text = step("configure-backend", configure, args.build_timeout)
        if not accepted_build(record, text):
            raise RuntimeError("Backend configure failed; see configure-backend.log")
        for target in BACKEND_TARGETS:
            command = builder + ["--build", str(build / "backend"), "--target", target, "-j", str(args.jobs)]
            record, text = step("build-" + target, command, args.build_timeout)
            binary = build / "backend/bin" / target
            if not accepted_build(record, text) or not binary.is_file():
                raise RuntimeError("Backend build failed: " + target)
            report.setdefault("binary_sha256", {})[target] = hashlib.sha256(binary.read_bytes()).hexdigest()
        for library in sorted((build / "backend/bin").glob("*.so*")):
            if library.is_file():
                report.setdefault("binary_sha256", {})[library.name] = hashlib.sha256(library.read_bytes()).hexdigest()
        env["LD_LIBRARY_PATH"] = str(build / "backend/bin") + ":" + env["LD_LIBRARY_PATH"]
        record, text = step("quantize-host-reference", [str(build / "backend/bin/test-maca-quantize-reference")], args.run_timeout)
        quantize_ok = (record["exit_code"] == 0 and not record.get("timed_out", False) and
                       text.count("PASS quantizer host reference: 540/540; NOT a GPU test") == 1 and
                       not BUILD_ERROR.search(text))
        report["quantize_host_reference"] = dict(accepted=quantize_ok, expected=540, gpu_test=False)
        if not quantize_ok:
            test_failure(record, "Quantizer host reference audit failed; other operator groups will still run")
        device = "MACA" + str(args.device)
        record, text = step("backend-devices", [str(build / "backend/bin/llama-bench"), "--list-devices"], args.run_timeout)
        if record["exit_code"] or not re.search(r"(?m)^\s*" + re.escape(device) + r":", text):
            raise RuntimeError("Requested MACA device was not enumerated; see backend-devices.log")
        for op in SMOKE_OPS:
            command = [str(build / "backend/bin/test-backend-ops"), "test", "-b", device, "-o", op, "-j", "1"]
            if op == "MUL_MAT":
                command += ["-p", "type_a=(f32|q8_0),"]
            record, text = step("ops-" + op.lower(), command, args.run_timeout)
            counts = backend_case_summary(record, text, device, op)
            report["operators"][op] = counts
            write_json(session / "report.json", report)
            if not counts["completed"]:
                test_failure(record, "Selected operator run incomplete or failed: " + op)
        report["backend_smoke_completed"] = all(counts["completed"] for counts in report["operators"].values())
        if args.full_ops:
            command = [str(build / "backend/bin/test-backend-ops"), "test", "-b", device, "-j", "1"]
            if args.host_reference:
                report["host_reference_controls"] = []
                negative_done = set()
                for index, case in enumerate(reference_controls):
                    op, params, method = case
                    control_command = [command[0], "test", "-b", "CPU", "-j", "1", "-o", op,
                                       "-p", "^" + re.escape(params) + "$"]
                    try:
                        env["LLAMA_TEST_HOST_REFERENCE"] = "check"
                        record, text = step("host-positive-%d-%s" % (index, op.lower()), control_command, args.run_timeout)
                        counts = backend_case_summary(record, text, "CPU", op)
                        counts["inventory"] = exact_case_summary(text, [[op, params, "OK"]])
                        counts["reference"] = host_reference_summary(text, [case])
                        counts["accepted"] = counts["completed"] and counts["inventory"]["accepted"] and counts["reference"]["accepted"]
                        report["host_reference_controls"].append(dict(kind="positive", case=case, **counts))
                        if not counts["accepted"]:
                            test_failure(record, "CPU host-reference positive control failed: " + op)
                        if method not in negative_done:
                            negative_done.add(method)
                            env["LLAMA_TEST_HOST_REFERENCE_FAULT"] = "1"
                            suffix = "" if method == HOST_METHODS[op] else "-" + method
                            record, text = step("host-negative-" + op.lower() + suffix, control_command, args.run_timeout)
                            accepted = reference_negative_accepted(record, text, case)
                            report["host_reference_controls"].append(dict(kind="negative", case=case, accepted=accepted))
                            if not accepted:
                                test_failure(record, "Host-reference negative control did not detect the injected error: " + op)
                    finally:
                        env["LLAMA_TEST_HOST_REFERENCE"] = "1"
                        env.pop("LLAMA_TEST_HOST_REFERENCE_FAULT", None)
            oracle_cases = [["SUM", "type=f32,ne=" + shape + ",permute=[1,0,2,3],oracle=integer_sum", "OK"]
                            for shape in ("[33,256,1,1]", "[3,5,2,2]")]
            cpu_command = [str(build / "backend/bin/test-backend-ops"), "test", "-b", "CPU", "-j", "1",
                           "-o", "SUM", "-p", "oracle=integer_sum"]
            record, text = step("cpu-sum-oracle", cpu_command, args.run_timeout)
            counts = backend_case_summary(record, text, "CPU", "SUM")
            counts["inventory"] = exact_case_summary(text, oracle_cases, "SUM")
            counts["accepted"] = counts["completed"] and counts["inventory"]["accepted"]
            report["cpu_sum_oracle"] = counts
            write_json(session / "report.json", report)
            if not counts["accepted"]:
                test_failure(record, "CPU SUM known-answer regression incomplete or failed")
            report["data_regressions"] = {}
            for op in DATA_OPS + (("FLASH_ATTN_EXT",) if args.flash_attn else ()):
                record, text = step("data-" + op.lower(), command + ["-o", op], args.run_timeout)
                counts = backend_case_summary(record, text, device, op)
                if expected_cases is not None:
                    counts["inventory"] = exact_case_summary(text, expected_cases, op)
                counts["accepted"] = counts["completed"] and (expected_cases is None or counts["inventory"]["accepted"])
                if args.host_reference:
                    counts["host_reference"] = host_reference_summary(text, reference_cases, op)
                    counts["accepted"] = counts["accepted"] and counts["host_reference"]["accepted"]
                report["data_regressions"][op] = counts
                write_json(session / "report.json", report)
                if not counts["accepted"]:
                    test_failure(record, "Data operator regression incomplete or changed: " + op)
            regressions = (
                ("matmul-original-layout", r"type_a=q[456]_K,.*k_v=1280,", 21),
                ("matmul-quant-layout", r"m=7,n=9,k=512,", 64),
            )
            report["layout_regressions"] = {}
            for name, params, expected in regressions:
                record, text = step(name, command + ["-o", "MUL_MAT", "-p", params], args.run_timeout)
                counts = backend_case_summary(record, text, device, "MUL_MAT")
                counts["expected_ok"] = expected
                counts["accepted"] = counts["completed"] and counts["ok"] == expected and not counts["unsupported"] and not counts["skipped"]
                report["layout_regressions"][name] = counts
                write_json(session / "report.json", report)
                if not counts["accepted"]:
                    test_failure(record, "Layout regression incomplete or failed: " + name)
            report["moe_duplicate_ids"] = {
                "status": "not_run_upstream_undefined", "accepted": False,
                "reason": "Upstream disabled repeated expert ID cases; MUL_MAT_ID does not guarantee this behavior.",
                "upstream_commit": "8c2134a8755d8a9792af513dab20282affe903e4",
            }
            record, text = step("ops-full", command, args.run_timeout)
            if args.fa_replay:
                report["fa_full_input_fingerprint"] = fa_input_fingerprint(text, inventory["fa_replay_case"], args.fa_seed)
                if report["fa_full_input_fingerprint"] is None:
                    test_failure(record, "Full-run FA input fingerprint missing or invalid; no full/isolated replay claim")
            report["full_operators"] = backend_case_summary(record, text, device)
            report["default_ops_completed"] = report["full_operators"]["completed"]
            if expected_cases is not None:
                report["exact_case_inventory"] = exact_case_summary(text, expected_cases)
            if args.host_reference:
                report["host_reference_inventory"] = host_reference_summary(text, reference_cases)
                report["comparison_provenance"] = {
                    "host_reference_expected": len(reference_cases),
                    "cpu_backend_expected": sum(row[2] == "OK" for row in expected_cases) - len(reference_cases),
                    "cpu_backend_support_unchanged": True}
                if not report["host_reference_inventory"]["accepted"]:
                    test_failure(record, "Host reference markers missing, failed, duplicated, or changed")
            write_json(session / "report.json", report)
            if not report["default_ops_completed"] or (expected_cases is not None and not report["exact_case_inventory"]["accepted"]):
                test_failure(record, "Full operator run failed, incomplete, or case inventory changed")
            if expected_cases is not None:
                # A crashed full run must not hide independent operator groups.
                missing_ops = incomplete_case_ops(text, expected_cases)
                report["recovery_requested_ops"] = missing_ops
                report["recovery_operators"] = {}
                for op in missing_ops:
                    record, recovered = step("ops-recovery-" + op.lower(), command + ["-o", op], args.run_timeout)
                    counts = backend_case_summary(record, recovered, device, op)
                    counts["inventory"] = exact_case_summary(recovered, expected_cases, op)
                    counts["accepted"] = counts["completed"] and counts["inventory"]["accepted"]
                    if args.host_reference:
                        counts["host_reference"] = host_reference_summary(recovered, reference_cases, op)
                        counts["accepted"] = counts["accepted"] and counts["host_reference"]["accepted"]
                    report["recovery_operators"][op] = counts
                    write_json(session / "report.json", report)
                    if not counts["accepted"]:
                        test_failure(record, "Independent recovery incomplete or failed: " + op)
                # Recovery evidence never replaces the failed original full run.
            elif not report["default_ops_completed"]:
                report["recovery_unavailable"] = "No reviewed case manifest; missing operator groups cannot be determined"
        if args.fa_replay:
            # A failed full run remains failed. These independent A/B samples
            # diagnose it, never replace it or widen the numerical tolerance.
            params = inventory["fa_replay_case"]
            report["fa_replay"] = []
            for offset in range(16):
                seed = (args.fa_seed + offset) % 2**64
                pair = []
                for forced in (False, True):
                    env["LLAMA_TEST_FA_SEED"] = str(seed)
                    env["GGML_MACA_FA_DIAG"] = "1"
                    env["GGML_MACA_FA_FORCE_F32"] = "1" if forced else "0"
                    name = "fa-replay-%02d-%s" % (offset, "f32" if forced else "tile")
                    replay_command = [str(build / "backend/bin/test-backend-ops"), "test", "-b", device,
                                      "-j", "1", "-o", "FLASH_ATTN_EXT", "-p", "^" + re.escape(params) + "$"]
                    record, text = step(name, replay_command, args.run_timeout)
                    result = fa_replay_summary(record, text, params, seed, forced)
                    pair.append(result)
                    report["fa_replay"].append(dict(seed=seed, forced_f32=forced, **result))
                    if not result["accepted"]:
                        test_failure(record, "FA reproducible sample failed or incomplete: " + name)
                    write_json(session / "report.json", report)
                if not pair[0]["input_fingerprint"] or pair[0]["input_fingerprint"] != pair[1]["input_fingerprint"]:
                    test_failure(record, "FA A/B input fingerprints do not match for seed " + str(seed))
                if offset == 0:
                    report["fa_full_replay_inputs_match"] = bool(pair[0]["input_fingerprint"]) and (
                        pair[0]["input_fingerprint"] == report.get("fa_full_input_fingerprint"))
                    if not report["fa_full_replay_inputs_match"]:
                        test_failure(record, "FA full/isolated input fingerprints do not match")
            for key in ("GGML_MACA_FA_FORCE_F32", "GGML_MACA_FA_DIAG"):
                env.pop(key, None)
            env["LLAMA_TEST_FA_SEED"] = str(args.fa_seed)
        report["test_collection_completed"] = True

    try:
        cmake = shutil.which("cmake")
        step("cmake-version", [cmake, "--version"], 30)
        step("host-compiler-version", [env.get("CXX") or "c++", "--version"], 30)
        configure = ["-S", str(ROOT / "tests/maca"), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                     "-DMACA_PATH=" + str(sdk)]
        if args.mode in ("runtime", "all"):
            suite("runtime", ("runtime",), [cmake] + configure + ["-B", str(build / "runtime")], [cmake])
        if args.mode != "runtime":
            toolchain = build / "toolchain"
            toolchain.mkdir(exist_ok=True)
            temporary = build / "tmp"
            temporary.mkdir(exist_ok=True)
            env.update(CUBRIDGE_HOME=str(toolchain), TMPDIR=str(temporary))
            env["PATH"] = ":".join([str(sdk / "bin"), str(bridge / "bin"), str(bridge / "tools"), env["PATH"]])
            if not env.get("CUCC_TARGETS") and not env.get("CUCC_TARGETS_FROM_DEVICE"):
                record, text = step("detect-target", ["bash", "-c", 'source "$1"; get_arch_from_device',
                                                       "_", str(bridge / "tools/multi_target.sh")], 30)
                arch = text.strip()
                if record["exit_code"] or not arch or arch == "NOT_FOUND" or not re.fullmatch(r"[A-Za-z0-9_,;.+-]+", arch):
                    raise RuntimeError("SDK target detection failed; see detect-target.log or set CUCC_TARGETS")
                env["CUCC_TARGETS_FROM_DEVICE"] = arch
            report["compiler_target"] = env.get("CUCC_TARGETS") or env["CUCC_TARGETS_FROM_DEVICE"]
            pin_target(build, report["compiler_target"])
            virtual = toolchain / "cu-bridge"
            bridge_flags = [
                "-DCMAKE_CUDA_ARCHITECTURES=52", "-DCUDAToolkit_ROOT=" + str(virtual / "CUDA_DIR"),
                "-DCMAKE_CUDA_COMPILER=" + str(virtual / "CUDA_DIR/bin/nvcc"),
                "-DCMAKE_CUDA_HOST_COMPILER=" + str(virtual / "bin/c++"),
                '-DCMAKE_CUDA_FLAGS=-I"' + str(bridge / "include") + '"',
            ]
            builder = [str(bridge / "tools/cmake_maca")]
            if args.mode == "backend":
                backend_configure = [str(bridge / "tools/cmake_mock"), "-S", str(ROOT), "-B", str(build / "backend"),
                                     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                                     "-DMACA_PATH=" + str(sdk), "-DGGML_MACA=ON", "-DGGML_CUDA=OFF",
                                     "-DGGML_HIP=OFF", "-DGGML_MUSA=OFF", "-DGGML_BACKEND_DL=OFF",
                                     "-DGGML_STATIC=OFF", "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_BUILD_TOOLS=ON",
                                     "-DLLAMA_OPENSSL=OFF", "-DGGML_CUDA_FA=" + ("ON" if args.flash_attn else "OFF")] + bridge_flags
                if args.quantize_fp_mode:
                    backend_configure.append("-DGGML_MACA_QUANTIZE_FP_MODE=" + args.quantize_fp_mode)
                backend_suite(backend_configure, builder)
            else:
                device_configure = [str(bridge / "tools/cmake_mock")] + configure + [
                    "-B", str(build / "device"), "-DMACA_PROBE_DEVICE=ON", "-DMACA_CU_BRIDGE=" + str(bridge)] + bridge_flags
                suite("device", DEVICE_PROBES, device_configure, builder)
    except (RuntimeError, OSError, KeyboardInterrupt) as error:
        report["error"] = str(error) or "Interrupted"
        print(str(error), file=sys.stderr)
    expected = [] if args.mode == "backend" else (["runtime"] if args.mode != "device" else []) + (list(DEVICE_PROBES) if args.mode != "runtime" else [])
    report["expected_probes"] = expected
    report["all_requested_probes_passed"] = bool(expected) and not report.get("error") and all(report["probes"].get(name) == "pass" for name in expected)
    if args.mode == "backend":
        report["requested_checks_passed"] = (report["test_collection_completed"] and report["backend_smoke_completed"]
                                             and not report["test_failures"] and not report.get("error"))
    write_json(session / "report.json", report)
    print(json.dumps(report["probes"], indent=2), flush=True)
    print("Report: " + str(session / "report.json"), flush=True)
    if args.mode == "backend":
        print("Test collection: " + ("finished" if report["test_collection_completed"] else "stopped"), flush=True)
        print("Failed check groups: " + str(len(report["test_failures"])), flush=True)
        for failure in report["test_failures"]:
            print("  " + failure["stage"] + ": " + failure["reason"] + " [" + failure["log"] + "]", flush=True)
        archive = results / ("maca-v2-backend-" + session.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as output:
            for path in sorted(session.iterdir()):
                if path.is_file() and not path.is_symlink():
                    output.add(path, arcname=session.name + "/" + path.name)
        print("Return archive: " + str(archive), flush=True)
        print("SHA256: " + hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
    print("This run does not establish full operator coverage or model correctness.", flush=True)
    passed = report.get("requested_checks_passed") if args.mode == "backend" else report["all_requested_probes_passed"]
    return 0 if passed and not report.get("error") else 130 if report.get("error") == "Interrupted" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
