# MACA backend

The experimental `GGML_MACA` backend runs shared ggml CUDA kernels on MetaX GPUs through the MACA SDK's cu-bridge and mcBLAS. It is disabled by default and appears as `MACA` / `MACA0` at runtime.

## Requirements

- Linux with a working MetaX GPU driver and a complete MACA SDK, including cu-bridge.
- Python 3.8+, CMake 3.18+, Ninja, Bash and a host C++ compiler.
- Validated configuration: one MetaX C500, MACA 3.2.0, hardware target `xcore1000`.

Other GPUs and SDK versions require separate validation. MACA cannot be built together with CUDA, HIP or MUSA because these backends share symbols.

## Build and smoke test

Run from the repository root:

```sh
python3 -B tests/maca/run.py \
    --sdk /opt/maca-3.2.0 \
    --build-dir ./build-maca \
    --mode backend --device 0 --jobs 4 \
    --flash-attn --quantize-fp-mode bridge \
    --build-timeout 7200 --run-timeout 14400
```

The runner configures `GGML_MACA=ON` through the SDK build tools, builds the CLI, server, benchmark and operator tests, then runs smoke tests. Binaries are in `build-maca/backend/bin`; logs and reports are in `build-maca/results`.

Repeat the command for incremental builds. Use a new build directory if the source location, SDK or hardware target changes. The bridge frontend architecture `52` is not the MetaX hardware architecture; the runner detects the hardware target through the SDK.

## Run a model

```sh
export MACA_PATH=/opt/maca-3.2.0
export MACA_BUILD_DIR="$PWD/build-maca/backend"
export LD_LIBRARY_PATH="$MACA_BUILD_DIR/bin:$MACA_PATH/lib64:$MACA_PATH/lib:$MACA_PATH/tools/cu-bridge/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_DISABLE_FUSION=1

"$MACA_BUILD_DIR/bin/llama-cli" --list-devices
"$MACA_BUILD_DIR/bin/llama-cli" \
    -m /path/to/model.gguf --device MACA0 -ngl 99 -c 4096 -n 128
```

Choose a model and context that fit available memory. This is a usage example, not model-quality or performance acceptance. The test runner disables operator fusion automatically; standalone tools need the environment setting above.

## Validation status

The 2026-10-21 C500 run validates the source in commit `1002a536abf2006e20607488065f8f92f31d55d6`, based on upstream `dd002b91f583af0b80140f9a140dc04856bfc2e0`. Flash Attention was enabled; Graph and operator fusion were disabled.

| Check | Result |
| --- | ---: |
| Fixed operator-case inventory | 25,242 / 25,242 passed |
| Failed / unsupported / skipped cases | 0 / 0 / 0 |
| CPU-backend reference comparisons | 22,234 |
| Test-only host-reference comparisons | 3,008 |
| Host-reference positive / fault-injection controls | 42 / 8 passed |
| Flash Attention replays, 16 seeds and two paths | 32 / 32 passed |

Counts refer to tested parameter combinations, not distinct operators or all possible shapes. Host references are test-only oracles, not production CPU implementations. Auxiliary checks overlap the inventory and are not added to its total.

Evidence archive: `maca-rebased-fix-xdbj3w_d.tar.gz`, SHA-256 `62bae2906ce3e1919d45207aba2ebedf0c4d1984c256e7f4f2ea317806a32ab6`. The archive contains the reviewed `expected-cases.json.gz`, source identity and full test results; it is retained separately from Git.

To repeat the full acceptance run, use that reviewed inventory, not the historical 2026-10-17 inventory:

```sh
python3 -B tests/maca/run.py \
    --sdk /opt/maca-3.2.0 --build-dir ./build-maca \
    --mode backend --device 0 --jobs 4 \
    --full-ops --flash-attn --host-reference \
    --case-manifest /path/to/expected-cases.json.gz \
    --fa-seed 20261016 --fa-replay --quantize-fp-mode bridge \
    --build-timeout 7200 --run-timeout 14400
```

Review `requested_checks_passed`, `full_operators` and `exact_case_inventory` in `report.json`. The broad `backend_accepted` and `operator_coverage_accepted` flags remain false by design; this run establishes only fixed-inventory operator correctness.

## Operator support table

The shared [operator table](../ops.md) is generated from backend capability reports. To add or refresh its MACA column, run on the MACA machine after the build, using the environment above:

```sh
"$MACA_BUILD_DIR/bin/test-backend-ops" support -b MACA0 --output csv > docs/ops/MACA.csv
python3 scripts/create_ops_docs.py
```

Inspect the CSV and generated diff before committing them. Support mode queries backend capabilities; it does not run numerical comparisons. Its checkmarks are not a substitute for the correctness tests above. The MACA column is pending a hardware-exported support CSV.

## Limitations

| Area | Current scope |
| --- | --- |
| API provider | Full backend uses cu-bridge. Native `mc*` aliases cover standalone probes only. |
| Matrix multiplication | Reuses shared dispatch and BLAS fallbacks; no NVIDIA MMA or native DP4A acceleration claim. |
| Flash Attention | Shared kernels plus an FP32 fallback; fallback conversions and score buffers add memory and work. |
| Quantized writes | Correctness-first GPU encoders. IQ2_XXS, IQ2_XS and IQ1_S use all-ones importance weights, not calibrated imatrix quantization. |
| Graph / fusion | Graph is not enabled; fusion is disabled in the validated configuration. |
| Multi-GPU / memory features | NCCL, peer copy, VMM and cooperative launch are not enabled. |
| Deployment | Static SDK linking and dynamic backend loading are not validated. |
| Models / performance | Not established by this operator regression; require separate end-to-end tests. |

Some layouts require packing or dequantization into temporary device buffers. Passing correctness tests does not imply an optimized kernel for every case. Duplicate expert IDs per token in `MUL_MAT_ID` are undefined by the pinned upstream revision and are not claimed as supported.

Implementation entry points: [vendor API](../../ggml/src/ggml-cuda/vendors/maca.h), [backend build and adapters](../../ggml/src/ggml-maca), and [validation runner](../../tests/maca/run.py). Earlier development notes remain available in Git history.
