# MACA backend

The `maca-backend` branch is rebased onto upstream commit `dd002b91f583af0b80140f9a140dc04856bfc2e0` (2026-10-20). It adds an experimental `GGML_MACA` backend, disabled by default. MACA compilation and C500 regression on this rebased source are pending; the historical results below do not validate this revision.

The rebase keeps upstream's non-contiguous CPU SUM implementation and new regression cases. Upstream commit `8c2134a8755d8a9792af513dab20282affe903e4` marks duplicate expert IDs per token as undefined for MUL_MAT_ID and disables those tests. The runner no longer requires the old 16-case duplicate-ID group, records it as not run rather than passed, and still runs the independent MUL_MAT_ID group.

The runner now records the new upstream baseline and rejects old case manifests. Keep the 2026-10-17 inventory unchanged as historical evidence. A new reviewed case inventory and C500 run are required before claiming exact coverage on this branch; changing the old manifest's commit field alone is not sufficient. Use a new build directory for the changed baseline.

## Historical accepted operator baseline (2026-10-17)

These results apply to upstream commit `8f574b0ad44c46d7b499ec028c29205aaa3fe80f` plus the MACA changes frozen in the tag below, not to the rebased branch.

Version tag: `maca-v2-c500-ops-baseline-20261017`. Evidence: `maca-quant-rounding-vuf6o60t.tar.gz`, SHA-256 `558cc16d94a39a829031d853f7d3ae9b4ebfdbbacdef404df8beadf42e3a3c88`. The nested full run is `maca-v2-backend-run-cqd6gru4`. The audit checked 413 archive member hashes, 126 nested evidence files, all case identities and multiplicities, 2876 recorded source fingerprints and five SDK fingerprints. The evidence archive is retained separately, not embedded in Git.

Configuration: one MetaX C500, MACA 3.2.0, cu-bridge API provider, hardware target `xcore1000`, Flash Attention enabled, Graph and operator fusion disabled. `GGML_MACA_QUANTIZE_FP_MODE=bridge` adds `--fmad=false` only to the MACA quantized-write translation unit. This is floating-point contraction control, not a global change to matrix multiplication or graph fusion.

| Accepted evidence | Result |
| --- | ---: |
| Full fixed case inventory | 25041 / 25041 OK |
| Failed / unsupported / skipped | 0 / 0 / 0 |
| Comparisons with the CPU backend | 22033 |
| Comparisons with test-only host references | 3008 |
| Previous full-run failures now passing | 111 / 111 |
| Fixed-input GPU quantization groups, byte-exact | 45 / 45 |
| Host quantizer comparisons (not GPU tests) | 540 / 540 |
| Host-reference positive / injected-fault controls | 42 / 8 |
| FA tile / FP32 replays across 16 seeds | 32 / 32 |

The independent operator groups also passed, including CPY (648), SET_ROWS (731) and FLASH_ATTN_EXT (5340). Their cases overlap the full inventory and must not be added to the 25041 total. The 65536 multiply-add probe samples match the host's separate-rounding result; all nine arithmetic fields match under their comparison policy, which excludes 2046 half NaN payload comparisons. IQ init/free helpers are internal, and the same-namespace executable no longer crashes in the control test.

The exact reviewed inventory and machine-readable acceptance record are in `tests/maca/baselines/c500-ops-20261017.json.gz` and `tests/maca/baselines/c500-ops-20261017-acceptance.json`. To reproduce this historical result, use a separate checkout of tag `maca-v2-c500-ops-baseline-20261017`, then run from that checkout's source root:

```sh
python3 -B tests/maca/run.py \
    --sdk /opt/maca-3.2.0 \
    --build-dir ./build-maca-c500-ops \
    --mode backend --device 0 --jobs 4 \
    --full-ops --flash-attn --host-reference \
    --fa-seed 20261016 --fa-replay --quantize-fp-mode bridge \
    --case-manifest tests/maca/baselines/c500-ops-20261017.json.gz \
    --build-timeout 7200 --run-timeout 14400
```

Reuse that source path and build directory for incremental builds. A checkout at a different path needs its own build directory. This command runs the operator, host-control and FA suites; the separate arithmetic, symbol and 45-group diagnostics are recorded in the returned rounding evidence. The retained `backend_accepted=false` and `operator_coverage_accepted=false` report fields do not grant broad backend acceptance; inspect `requested_checks_passed`, `full_operators` and `exact_case_inventory` for this scoped run.

The 3008 host-reference comparisons do not add production CPU operator implementations. The 90 IQ2_XXS, IQ2_XS and IQ1_S cases use the explicitly approved all-ones importance policy, not calibrated imatrix quality. Untested shapes and operations, model inference and quality, performance, Graph/fusion, native-provider integration, other devices/SDKs and multi-GPU still require separate validation. No further local compilation or GPU tests were run while freezing this version. The historical sections below describe earlier candidates; this accepted baseline supersedes their pending counts, not their implementation limits.

## Historical validation status (2026-10-14)

Evidence: `maca-v2-backend-run-_mag605o.tar.gz`, SHA-256 `fc158e5a01e6ed7507419e956614a1a10f664750bf89c432eff1591a575c80d8`. All 2870 recorded source fingerprints matched the delivered candidate. The configuration is one MetaX C500, MACA 3.2.0, target `xcore1000`, generic Flash Attention enabled, Graph and fusion disabled.

| Full-suite status | Cases |
| --- | ---: |
| Numerical comparisons passed | 20497 |
| Numerical comparisons failed | 0 |
| Unsupported by MACA only | 1500 |
| Unsupported by both MACA and CPU | 1351 |
| Unsupported by CPU only; no numerical comparison | 1657 |
| Total records | 25005 |

All 36 requested stages ran. The previous 20086 passing cases stayed passing. The 411 additional passes are FLASH_ATTN_EXT (338), CONV_2D (24), ROPE (20), LIGHTNING_INDEXER (18), GET_ROWS (8), the original SUM failure (1), and two new SUM known-answer cases. The independent CPU SUM oracle tests, 85 quantized layout comparisons and 16 duplicate-expert-ID comparisons also passed.

The expected inventory predicted 20517 passes. Its remaining 20 differences are SET_ROWS cases with `r=1,v=1`: the test builds a view with `r/2` rows, which is zero. The MACA conversion guard rejects zero dimensions. The prior prediction missed this empty-input condition; these cases are not numerical failures or accepted passes. Empty-row writes still need explicit no-op handling and regression tests. Do not relax numerical tolerances or change the expected status merely to pass acceptance.

The 1500 MACA-only gaps comprise FLASH_ATTN_EXT (1001), SET_ROWS (308), CPY (76), MUL_MAT (61), ARGSORT (24), GET_ROWS (16), and MUL_MAT_ID (14). Counts describe tested parameter combinations, not distinct operator kinds. Cases without a CPU implementation need an independent reference. Models, performance, Graph/fusion, multi-GPU and other devices/SDKs remain separate gates. The historical sections below record the implementation sequence; this section supersedes their pending-validation predictions.

## Design

Use `ggml/src/ggml-cuda/vendors/maca.h` as the API entry point. The standalone native host probe maps a small API surface to `mc*`. The full backend selects the SDK's cu-bridge runtime, driver, BLAS and floating-point headers. Do not combine these providers in one translation unit or duplicate aliases already supplied by the SDK. The runtime adapter header guard rejects accidentally selected NVIDIA runtime headers.

Keep shared algorithms in `ggml-cuda`. Add MACA-specific primitives only where the SDK cannot provide the required semantics. Do not infer NVIDIA instruction support from a CUDA-compatible version or architecture number. Do not replace unavailable synchronization with a no-op.

The device build uses the SDK's existing `cmake_mock` / `cmake_maca` path. Direct native compiler integration has not been validated. `CMAKE_CUDA_ARCHITECTURES=52` is the bridge frontend setting, not the MetaX hardware architecture; the SDK detector or `CUCC_TARGETS` selects the hardware target. Other frontend architecture settings are rejected in this first backend candidate.

The backend uses shared CUDA source files, a separate `ggml-maca` library and the existing `ggml_backend_cuda_*` API, as MUSA does. MACA cannot be selected together with CUDA, HIP or MUSA in one build because they share those symbols. The backend is registered and displayed as `MACA` / `MACA0`, not CUDA. Linked-backend mode is the first validation target; dynamic loading is wired but not accepted yet. Do not load MACA and another CUDA-derived plugin into the same process.

MACA gets a separate internal vendor capability tag. NVIDIA MMA, CP async, PDL, native DP4A and CUB paths are not inferred from the SDK's CUDA-compatible version. The integer dot-product helper keeps its portable fallback. The backend retains the BLAS library's default math mode and workspace management. It does not claim TF32 or tensor-core support.

The common dispatch field `warp_size` and `ggml_cuda_get_physical_warp_size()` both use `ggml_maca_kernel_warp_size()` for MACA. Despite the legacy helper name, its MACA value is a logical subgroup size of 32. `cudaDeviceProp::warpSize` is not modified; the log prints the reported physical size separately. MMVQ uses the generic parameter table and the default MoE batch limit in both host and device code. The finite probe coverage below is not proof that every shared kernel satisfies this contract.

Graph, VMM, NCCL, peer copy and cooperative launch are not enabled in this candidate. Generic Flash Attention is optional through `GGML_CUDA_FA` or the runner's `--flash-attn` flag; its supported configurations are limited. MMQ dispatch uses the existing forced-BLAS fallback; MMVQ and MMVF can still be selected by shared dispatch logic. This is not an all-matmul BLAS-only implementation or a performance release. The supplied backend test runner also disables fusion explicitly. Running binaries outside that runner without the environment setting can enable upstream fusion paths that have not been validated.

## Run on Linux with a MACA GPU

Requirements: Python 3.8+, CMake 3.18+, Ninja, a host C++ compiler, Bash, and a complete MACA SDK. No model download is needed. Run with no other workloads using the selected GPU.

From the source root:

```sh
python3 -B tests/maca/test_runner.py
python3 -B tests/maca/run.py \
    --sdk /opt/maca-3.2.0 \
    --build-dir /sw_home/i26439/maca-v2-probe-build \
    --device 0 --mode all
```

Use the installed SDK path; do not install a different SDK merely to run this command. Repeat the same command and build directory for incremental builds. A source-directory or SDK change requires a new build directory. Existing foreign build directories are rejected, not cleaned or overwritten. The script creates a separate result directory for each run and keeps all logs.

`--mode runtime` builds only the native host runtime probe with ordinary CMake. `--mode device` builds only the cu-bridge probes. `--jobs` controls compile concurrency (default 2). The script does not change the parent shell environment, existing backend builds, Git branches, models, or SDK files.

## Accepted probe evidence and limits

The returned `maca-v2-probe-run-6hikdhvf` archive passed all six probes on MetaX C500 with MACA 3.2.0, target `xcore1000`, CMake 3.22.1 and GNU 11.4.0. The bridge compiler identifies itself to CMake as NVIDIA 11.6.55; this is a compatibility identity, not the GPU vendor. The device reported warp size 64 while the logical-32 tests passed at each tested block size. These results cover the phase-1 probe sources, not the new full backend.

| Probe | Coverage |
| --- | --- |
| runtime | Device selection, allocation, pinned host memory, async copies, memset, two-stream event ordering; 17 deterministic iterations |
| bridge-runtime | The same runtime checks through the SDK compatibility headers |
| warp | 32-lane logical broadcast, XOR shuffle, sum, ballot, warp barrier and block barrier; 32/64/128/256 threads and 3 blocks |
| fp16 / bf16 | Eight exactly representable values, conversion and device addition; separate compile/run targets |
| blas | F32 non-square SGEMM with a host reference and an explicit stream |

These are finite smoke tests, not a proof of every memory ordering pattern or data type case. Warp sizes are reported without changing device properties. A warp probe failure must be investigated before enabling kernels that assume its contract. Passing does not establish masked/partial-warp, divergent control flow, all ggml operator layouts, Graph, fusion, attention, models, multi-GPU support, or performance.

`report.json` records the upstream baseline, source and executable SHA-256 hashes, compiler target, commands, exit codes and per-probe results. A zero SDK wrapper exit code is insufficient: error text, missing binaries, timeouts and missing success markers all reject the stage. `backend_accepted` remains false even when every requested probe passes. Return the complete result directory, including failure logs, for the next implementation step.

## Incremental backend bring-up

After updating the phase-1 source directory, reuse the same source and build root:

```sh
python3 -B tests/maca/test_runner.py
python3 -B tests/maca/run.py \
    --sdk /opt/maca-3.2.0 \
    --build-dir /sw_home/i26439/maca-v2-InqZtV/build \
    --mode backend --device 0 --jobs 4 \
    --build-timeout 3600 --run-timeout 1800
```

The first backend run needs a full build in `build/backend`; standalone probe object files cannot replace it. Repeating the command updates this backend build incrementally without deleting caches. The runtime and device probe builds remain in their own subdirectories. Changing common headers can legitimately rebuild many CUDA translation units.

The runner builds `llama-bench`, `llama-server`, `llama-cli` and `test-backend-ops`, checks that `MACA0` is listed, then runs upstream CPU-reference comparisons for ADD, MUL, RMS_NORM, SOFT_MAX and MUL_MAT (F32 and Q8_0 weights). It uses one test worker, no model files and no downloads. Build/configuration/device prerequisite failures stop execution; independent test groups continue after a test failure and preserve its log. Test timeouts are failures, not unsupported cases.

Each run saves `report.json` and logs under `build/results/run-*`. Backend source and binary hashes, commands, SDK identity and the compiler target are recorded. `backend_smoke_completed=true` means the requested stages completed with at least one passed case per selected operation. Unsupported and skipped cases are reported separately. This is not an exact full-case acceptance check: `operator_coverage_accepted` and `backend_accepted` remain false. The results do not establish model correctness, performance or multi-device support.

Return the whole result directory, including logs if configuration or compilation fails. Source and incremental build files remain at the candidate revision; there is no automatic rollback. Do not treat a zero wrapper exit code or an existing binary as evidence that a failed build succeeded.

## Next gates

The first backend build (`run-shpxj436`) configured successfully but failed in the bridge host pass while compiling `conv2d.cu` and `conv3d.cu`. A static MMA tile array bound called a device-only warp-size helper. The MACA helper now has both host and device qualifiers; its logical value is unchanged. Other vendors retain their existing device-only implementation. When `LLAMA_BUILD_TESTS` is enabled, `tests/maca/backend-warp-constexpr.cu` compiles the real shared header and checks host constant expressions and the affected tile layout. This is a compile regression check, not an MMA execution test. The corrected build passed in `run-_im0h6bu`, with 951 selected comparisons OK and 95 unsupported by both MACA and CPU.

The `maca-v2-fullops-0lwfvZ` run used the same test and backend binaries. It stopped at `GGML_ASSERT(convert_func != nullptr)` in the non-contiguous src0 BLAS conversion branch, with exit code 134. Before the abort, its log recorded 7994 OK and 3232 unsupported cases. These are partial counts, not full-suite acceptance. The missing conversion affects quantized types such as Q4_K when a view contains row or batch gaps and dispatch selects BLAS.

`ggml-maca/blas.cuh` packs such quantized src0 views into a temporary device buffer only when the selected non-contiguous converter is absent. It copies whole rows of quantization blocks using their original byte strides, then reuses the shared contiguous conversion and BLAS path. The allocation stays alive through the BLAS call, and copies and computation use the same stream. The original graph and tensor are not modified. Continuous inputs, existing non-contiguous converters, other vendors, and numerical comparison tolerances are unchanged. This fallback adds a packed buffer and device copies for affected layouts; it is not a performance optimization.

The existing `test_mul_mat` suite now has 64 small cases across 16 quantized types, covering contiguous controls, row gaps, batch gaps and combined gaps with broadcast batches. Run them, the 21 existing Q4_K/Q5_K/Q6_K row-gap cases, and the full default suite after an incremental build:

```sh
python3 -B tests/maca/run.py \
    --sdk /opt/maca-3.2.0 --build-dir /sw_home/i26439/maca-v2-InqZtV/build \
    --mode backend --device 0 --jobs 4 --full-ops \
    --build-timeout 3600 --run-timeout 7200
```

`--full-ops` runs the smoke suites and complete targeted groups, and requires exactly 21 and 64 passing layout comparisons without unsupported or skipped cases. It also requires all 16 existing `MUL_MAT_ID` comparisons with `dup=1` to pass, including strided/offset IDs, broadcast activations and more than 1024 experts. It finally runs `test-backend-ops test -b MACA0 -j 1` without operator or parameter filters. Independent test failures remain in the result archive and do not stop later groups. `default_ops_completed` reports completion of this default run, not support for every operator or layout. Exact case-inventory acceptance, models, performance, Graph and fusion remain separate gates.

The returned `run-h0nisyuz` evidence compiled this layout candidate on C500 and passed the 21 original and 64 new layout cases. The five smoke groups remained at 951 OK and 95 unsupported. Its full run then recorded 9527 OK and 3607 unsupported before aborting at `ids_to_sorted_host.size() == size_t(ne_get_rows)` in the `MUL_MAT_ID` fallback. These are partial counts, not full-suite totals. The fallback stopped at the first matching expert slot per token, but the existing upstream tests permit repeated expert IDs. The candidate removes that early exit so every slot gets a sorted input and inverse output mapping. Unique-ID routing and accelerated dispatch are unchanged; the correction is shared by CUDA-derived backends, but only MACA hardware validation is planned in this package. No new kernel, support gate, synchronization or numerical tolerance is introduced.

1. Compile and validate the repeated-expert-ID candidate on C500 with the same incremental build directory.
2. Close actual failures using returned logs, without adding fake synchronization or success returns.
3. Run the full upstream operator suite and establish a new exact case inventory, with unsupported cases kept separate from passes.
4. Validate small models, then enable and measure optional fast paths independently.
5. Repeat on other devices and SDK versions before expanding the supported configuration list.

Old MACA branches and results remain comparison evidence; they are not acceptance evidence for this new source baseline.

## Data operator candidate

The returned `run-0s_0yw4u` completed the full default C500 suite: 24969 printed cases, 14568 OK, 7393 unsupported by MACA alone, 2987 unsupported by both MACA and CPU, and 21 unsupported by CPU alone. All 85 layout and 16 repeated-expert-ID comparisons passed. Unsupported is not a pass; these counts do not establish support for every GGML operation.

The next candidate changes MACA dispatch for three operations:

- `IM2COL_3D`: reuse the existing stride-aware kernel for F32 input views and F32/F16 outputs. Preserve the packed channel/batch dimension in the upstream view test; previously half of the 1024 view cases constructed empty outputs. Both channel divisibility and nonempty output are now asserted in the test. Add two nonempty F16-output view cases.
- `SET_ROWS`: convert nonempty F16 sources to a packed F32 temporary, then reuse existing F32 kernels for F32/BF16 or Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL destinations. Existing F16-to-F16 dispatch stays unchanged. Add four I32-index comparisons with views and broadcast batches. Empty-source cases remain outside this new path.
- `CPY`: copy encoded bytes for same-type quantized tensors with contiguous blocks inside each row, while preserving row and batch strides. F16/BF16-to-Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL reuses conversion to F32 followed by the existing quantizer, with contiguous destinations. Add four quantized reshape/stride cases and two conversion cases with permuted sources. Non-contiguous quantization destinations are excluded from the new conversion path because the CPU reference aborts for them. Numerical tolerances are unchanged.

The small adapter is in `ggml/src/ggml-maca/copy.cuh`. Shared-code hooks are guarded by `GGML_USE_MACA`. Temporary allocations remain alive through queued operations on the same stream; graph tensors are not modified. Conversion adds temporary memory and a kernel launch. This candidate is for correctness, not a speed claim or a replacement quantization algorithm.

`--full-ops` now also runs the three complete targeted operator groups before existing layout/MoE regressions and the unfiltered default suite. An optional `--case-manifest /path/to/expected-cases.json.gz` checks the exact reviewed case identities, status categories and duplicate multiplicities. The incremental package supplies that manifest. Its expectation is a prediction pending hardware execution, not observed acceptance. Any unreviewed missing, extra or changed case fails acceptance, even if totals match. CPU-only unsupported cases stay unvalidated; changing that category to OK is not allowed automatically. Model, Graph, fusion, performance and other devices remain separate gates.

### Quantization boundary follow-up

The returned `run-7n7l6bn0` matches all 2870 recorded source fingerprints. IM2COL_3D completed with 2053 OK; SET_ROWS completed with 255 OK and 476 unsupported. CPY completed 606 case records with 384 OK, five numerical failures and 217 unsupported; its process exited with code 1. The full default suite did not start. The previous parser missed the five diagnostic-prefixed FAIL rows, but the exit code and inventory checks still rejected the run. Both summary and inventory parsing now recognize those rows.

The four failing Q8_0 conversions expose different boundary rules: the x86 CPU SIMD quantizer uses `127/amax` and nearest-even rounding, while the shared GPU code uses `1/(amax/127)` and `roundf`. MACA now uses explicitly rounded arithmetic and nearest-even conversion. Q4_0 also uses explicit separate multiply and add rounding; for an exactly representable input of 13.5 with block maximum 24, contracting the multiply and add can change the encoded value from 4 to 3. This explains a possible source of the BF16-to-Q4_0 failure, but the returned log does not contain generated instructions or raw blocks to prove that contraction occurred.

The changes are guarded by `GGML_USE_MACA` in `cpy-utils.cuh`; other vendors and quantization types are unchanged. The helpers are shared by CPY and SET_ROWS, so both groups must be rerun. Twelve deterministic CPY cases cover F32/F16/BF16, Q4_0/Q8_0, contiguous/permuted sources, half-integer boundaries, zero blocks and both signs. Existing tolerances are not relaxed. Their acceptance target is the current C500/x86 SIMD configuration, not every CPU architecture's quantizer. This remains a candidate pending MACA compilation and execution; no performance improvement is claimed.

The follow-up `run-qmtvfekn` matched all 2870 source fingerprints and passed those twelve fixed boundary cases and all five previously failing identities in both targeted and full runs. The targeted groups passed with 2053 IM2COL_3D, 255 SET_ROWS and 401 CPY comparisons. The full run traversed all 24993 expected records but failed acceptance: 15839 OK, one F16-to-Q5_0 numerical failure and 9153 unsupported. The targeted Q5_0 case passed; the full case failed with error 0.000001414 above the unchanged 0.000000954 limit. Existing random inputs differ between runs, so this is not evidence of a stable pass.

Q5_0 has the same separate multiply/add boundary requirement as Q4_0. With block maximum 48 and input 25.5, separate F32 rounding encodes 8 while a contracted multiply/add can encode 7. The next MACA-only candidate applies explicit division, multiply and add rounding to Q5_0 without changing scale storage, bit packing, support gates or other quantizers. Six more deterministic CPY cases cover F32/F16/BF16 and two source layouts using zero, signed and scaled blocks. The original cases and error thresholds remain unchanged. The log does not include the failing tensor or generated device instructions, so contraction remains a candidate explanation, not a hardware-proven cause. CPY, SET_ROWS and the full exact inventory must pass on C500 before accepting this follow-up; it has no model or performance claim.

## Coverage batch candidate

The returned `run-jyg3vwgg` matched all 2870 source fingerprints and the exact inventory: 24999 records, 15846 OK, no failures and 9153 unsupported. Of the unsupported cases, 6145 were MACA-only, 2945 were unsupported by both backends and 63 had no CPU reference. This accepts the data-ops-v3 default regression on C500, not every operation or model.

The coverage batch reuses shared kernels for eleven operator families. All new compute paths are scoped to MACA; the existing numerical thresholds and quantizers are unchanged.

| Operator | Candidate adaptation | Important limit or cost |
| --- | --- | --- |
| FLASH_ATTN_EXT | Honor the existing `GGML_CUDA_FA` option and use generic tile/vector dispatch | No NVIDIA MMA; head shape, GQA, type and softcap instance limits remain |
| OUT_PROD | Convert F16 or supported quantized src0 to packed F32, then reuse SGEMM | F32 src1/output only; temporary memory; some broadcast cases have no CPU reference |
| CPY | Shared dequantization to packed F32 for additional quantized sources | Existing direct Q4/Q5/Q8 paths are preserved; no new quantization encoders |
| NORM / L2_NORM | Pack non-contiguous F32 rows before the existing normalization kernel | Contiguous output required; extra copy |
| GET_ROWS_BACK | Convert F16 gradients to packed F32 | Existing two-dimensional F32-output kernel and I32 indices |
| ACC | Pack strided F32 src1 before the existing accumulator | Contiguous src0 and output; extra copy |
| REPEAT | Enable BF16 and add exact I16/I32 instantiations of the existing broadcast kernel | Integer copies bypass float; dimensions/strides are bounded |
| CONCAT | Treat quantized rows as byte tensors for the existing strided concat kernel | Dimension 3 only; no re-quantization |
| SUM | Accept a physically contiguous F32 allocation even when tensor axes are permuted | No arbitrary gapped allocation |
| XIELU | Enable the existing BF16 kernel for contiguous tensors | Pending numerical validation on C500 |

`tests/maca/run.py` keeps Flash Attention off unless `--flash-attn` is supplied. That flag sets `GGML_CUDA_FA=ON`; manual CMake users must choose this shared option explicitly. The generic FA selector rejects head-576 GQA ratios without a tile instance and softcap value-head sizes without compiled code. It does not advertise tensor-core capabilities. Graph, fusion, VMM, NCCL and peer copy remain disabled in this validation mode.

The incremental package contains separate exact inventories for FA on and off. The FA-on prediction is 25003 records: 20087 OK, 1908 MACA-only unsupported, 2465 unsupported by both backends and 543 CPU-only unsupported. This includes 4237 newly executable existing cases and four new integer-boundary REPEAT cases. The 480 OUT_PROD cases moving from both-unsupported to CPU-only remain unvalidated, not passed. These are source-derived predictions pending device compilation and execution.

The four new REPEAT cases include I32 values beyond 2^24 and signed limits. Integer REPEAT results are compared byte-for-byte before the existing floating-point comparison, so low-bit loss cannot be hidden. Full validation runs complete targeted groups, existing quantized layout and MoE checks, and then the unfiltered default suite. A missing, extra or changed identity/status/multiplicity rejects the run, even if totals match. No unsupported case is converted into an automatic pass.

Remaining work includes new quantization encoders, TQ conversion, NVFP4/TQ gather, strided RoPE and CONV_2D, long-row ARGSORT, IQ4_NL LIGHTNING_INDEXER, unsupported FA shapes/types, and missing CPU references. The package includes the exact remaining case list. The generated `docs/ops.md`/CSV tables are not updated from predictions; export the native support inventory and regenerate them after device validation. Models, performance, additional devices/SDKs and Graph/fusion remain separate acceptance gates.

## Coverage batch v2: type and layout adapters

The returned `run-qg4gv7_5` matches 2870 source fingerprints and traverses the complete 25003-case inventory. It records 20086 OK, one failed transposed F32 SUM, 1908 MACA-only unsupported, 2465 unsupported by both backends, and 543 CPU-only unsupported. All 31 requested stages ran. This is a complete collection, not accepted numerical regression.

This follow-up keeps the same upstream commit and incremental build directory. It does not rebase, replace the SDK, enable Graph/fusion, or advertise NVIDIA MMA capabilities.

| Area | Implementation | Validation target / cost |
| --- | --- | --- |
| F32 SUM | Correct the CPU reference to use `nb[0]` for a strided inner dimension; preserve its contiguous vector sum | Existing failure plus two exact integer-sum oracles; compare both outputs with the known answer |
| SET_ROWS | Let the existing F16-to-F32 temporary conversion accept strided scalar columns | 20 existing cases; same destination quantizers, extra temporary |
| GET_ROWS | Dequantize NVFP4 to a packed F32 device tensor, then reuse gather | 8 existing cases; temporary covers the entire source table, not only selected rows |
| ROPE | Admit bounded, pair-aligned F32/F16 row-strided views to the existing stride-aware kernels | 20 existing in-place cases; no new kernel or packing |
| CONV_2D | Pack input and kernel views on the GPU, preserving each scalar type and logical axis order | 24 comparable cases; 1114 additional cases still lack a CPU reference |
| FLASH_ATTN_EXT | Supply IQ4_NL non-contiguous F16 conversion and admit existing Q1_0/Q2_0 conversion | 338 existing cases; retain generic head/GQA/softcap constraints, extra dequantization |
| LIGHTNING_INDEXER | Convert IQ4_NL keys to packed F32 and reuse vector/tile kernels | 18 existing cases; temporary key buffer, no native IQ4 compute kernel |

New adapters reject unsafe dimensions/strides and retain existing fast paths. Metadata copies and temporary buffers are local to a call; graph tensors are not mutated. Copies/conversions and consumers use the same stream and pool lifetime. All new GPU branches are MACA-only. The CPU SUM reference correction and its tests are intentionally shared; no numerical tolerance is relaxed.

The FA-on source prediction is 25005 records: 20517 OK, 1480 MACA-only unsupported, 1351 unsupported by both backends and 1657 CPU-only unsupported. This is 428 newly comparable existing cases, the existing SUM fix and two new oracles. Moving 1114 CONV_2D cases from both-unsupported to CPU-only is not execution or correctness acceptance. These are predictions pending a C500 build and run, not benchmark results.

`--full-ops` now runs an independent CPU-only SUM oracle check and complete GET_ROWS, ROPE, CONV_2D and LIGHTNING_INDEXER groups as well as the previous groups. A failed test group is recorded and independent groups continue. If the unfiltered full process aborts, missing groups are recovered separately; recovery never erases the failed full run. Configure/build/device prerequisites still stop unsafe execution, and user interruption stops the run. This continuation policy supersedes the earlier fail-fast description above.

Use the new package's matching FA-on/off exact manifest. Every existing identity and multiplicity is preserved; exactly two oracle cases are added. Remaining work includes 1001 unsupported FA configurations, 288 SET_ROWS encoding cases, 76 CPY cases, 61 MUL_MAT cases, 14 MUL_MAT_ID cases, 24 long-row ARGSORT cases and 16 TQ GET_ROWS cases on the MACA-only list. Cases missing CPU implementations need independent reference tests before they can be accepted. Default-suite success still does not cover all GGML operations, untested layouts, models or devices.

## Operator batch v3 candidate

The last returned cu-bridge run (`_mag605o`) traversed 25005 records: 20497 OK, 1500 MACA-only unsupported, 1351 unsupported by both backends and 1657 CPU-only unsupported. There were no numerical failures, but 20 empty SET_ROWS views did not match the expected inventory, so that run was not fully accepted. Native runtime/API probes are separate evidence and do not establish operator support.

This batch keeps the upstream revision, API provider and existing build directory. The following changes are MACA-only; other vendor dispatch and existing numerical tolerances are unchanged.

- FLASH_ATTN_EXT: preserve the shared tile/vector paths. When no shared instance exists, pack Q/K/V on the GPU, use FP32 BLAS for both matrix products, and use a block-reduction softmax. The fallback handles different K/V head widths, GQA, masks, sinks, ALiBi and softcap. Its score temporary is limited to 64 MiB by chunking queries; packed Q/K/V storage is additional and can be large. This is a correctness fallback, not a fused attention kernel or a performance improvement.
- TQ1_0/TQ2_0: add device conversion with the existing encoded block layout. GET_ROWS and matrix products reuse packed conversion and BLAS. Expert matrix multiplication keeps the existing per-expert grouping/scatter. These paths can allocate a full dequantized source temporary.
- Q1_0/Q2_0/TQ1_0/TQ2_0: add reference-style block encoders for CPY and SET_ROWS, preserving explicit scale/rounding and packing rules. No existing quantizer is replaced. New CPY encoders require contiguous destinations; scalar inputs are packed as needed.
- MUL_MAT_ID: honor explicit F32 source precision by using per-expert FP32 BLAS, avoiding conversion of large activations to F16. Default existing paths remain unchanged.
- ARGSORT: retain shared-memory sorting for short rows. Long contiguous F32 rows use global-memory bitonic stages on one stream, with padded indices and row chunks. This requires many launches and is not a tuned radix sort. New support is bounded to 2^28 columns and INT_MAX elements; arbitrary layouts remain excluded.
- SET_ROWS: a matched empty source/index view is a no-op. The output allocation is retained without packing or launching an empty kernel.

The exact FA-on candidate inventory predicts 1260 newly comparable existing cases plus 36 added regression cases: 24 quantization boundary/layout cases, four long multi-row sorts covering both orders, and eight mismatched-head attention cases with large logits, ALiBi, sinks, softcap and permuted KV views. This predicts 25041 total records, 21793 OK and 3248 unsupported. These numbers are targets, not observed GPU results.

There are still 240 MACA-only encoding cases for MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS. The 3008 cases missing CPU references also remain unvalidated. This batch does not claim all operators are supported. Additional cases absent from the default test suite, models, Graph/fusion, performance, other devices and SDKs need separate acceptance.

The runner adds full independent MUL_MAT, MUL_MAT_ID and ARGSORT groups before the full suite. Each independent group continues after another group fails, while configure/build/device failures still prevent stale-binary execution. Every case identity, status and duplicate multiplicity is checked against the supplied manifest. Unsupported is never counted as a numerical pass. Local host-model tests are not compiled GPU tests; C500 compilation and numerical regression are required before accepting this candidate.

## Test-only host references

The returned cu-bridge `run-taobwycj` matches the v3 source fingerprints and exact 25041-case inventory: 21793 OK, 240 MACA-only unsupported, 1351 unsupported by both backends, and 1657 CPU-only unsupported. A CPU-only unsupported record means the target advertised support but the comparison did not execute. It is not a numerical pass.

The opt-in `tests/maca/run.py --host-reference` mode addresses those 1657 cases without changing GPU kernels or the CPU backend's capability declarations. It requires a matching reviewed case manifest. The original case identities, multiplicities and numerical tolerances are unchanged.

| Group | Cases | Reference computation |
| --- | ---: | --- |
| CPY | 63 | Copy identical-type quantized blocks in logical tensor order; compare exact bytes |
| OUT_PROD | 480 | Dequantize CPU-side inputs using GGML type traits; accumulate each dot product in FP64 with explicit batch broadcasting |
| CONV_2D | 1114 | Pack the independent CPU-side kernel and input into contiguous tensors, preserving their scalar types; reuse the CPU convolution |

The graph and its initialized data are copied to the CPU before target execution. CPU preprocessing nodes and reference outputs are computed from that independent copy. The target retains the original layouts. References must never consume target-computed intermediate results. This is test infrastructure, not a production CPU fallback and not a new native CPU implementation.

The runner first compares the host methods against CPU-supported small cases and injects an error into each reference method in a separate negative-control process. A negative control is accepted only for the expected numerical test failure, never a crash, timeout or unsupported case. The normal test processes cannot inherit fault injection. Independent groups continue after a failed group; a failed control or full run still rejects acceptance.

Each reference comparison prints a `HOST_REFERENCE` marker with its exact case, method and result. The runner verifies the full marker multiset separately from the normal test rows and records both comparison provenances. The unchanged default mode keeps CPU-unsupported cases unsupported. Manual test use is `LLAMA_TEST_HOST_REFERENCE=1`; `check` forces the reference on CPU-supported controls. `LLAMA_TEST_HOST_REFERENCE_FAULT` is only for negative controls and must not be set during normal validation.

The target inventory is 23450 OK (21793 CPU-backend comparisons plus 1657 test-only host-reference comparisons) and 1591 unsupported (240 MACA-only plus 1351 both-unsupported). This is a prediction until C500 execution is returned and checked. Even a matching successful run does not establish all GGML operations, arbitrary shapes, models, other hardware, Graph/fusion, performance or the native API provider.

### Host-reference context correction

The returned `run-ig3_5qn5` matches the host-reference v1 source and manifest. Its independent CPY and OUT_PROD groups passed all 63 and 480 new host-reference comparisons respectively, including exact case/method markers. The CONV_2D reference crashed before any of its 1114 new comparisons completed; the full run is not accepted.

The CONV_2D helper accidentally called the test class's sentinel-adding `ggml_new_tensor` wrapper twice. That created five tensors in a context budgeted for four, and also added temporary reference tensors to the tested graph's sentinel list. The correction explicitly calls `::ggml_new_tensor`, budgets exactly three tensors plus the graph, and asserts that the sentinel list is unchanged. Existing target-side sentinel checks, comparison tolerances, the 25041-case manifest, GPU kernels and backend capabilities are unchanged. The existing CPU convolution positive/negative controls reproduce this failure and must pass after rebuilding before full acceptance.

## Mixed-input and FA reproducibility candidate (2026-10-16)

The returned host-reference v2 full run traversed all 25041 cases: 23449 OK, one FA numerical failure and 1591 unsupported. Its independent FA group passed, and a subsequent diagnostic sampled the failing configuration and variants 35 times without reproducing it. These fresh random inputs do not clear the original full-run failure.

This incremental candidate targets 1141 existing unsupported cases across four operators. It does not rebase, change the API provider, modify production CPU capabilities, or enable Graph/fusion.

| Operator | Existing cases targeted | MACA execution / independent reference |
| --- | ---: | --- |
| MUL_MAT | 359 | Pack a mixed FP16 right operand to FP32 on-device; reuse BLAS and existing weight conversion. Preserve the FP16/FP16 path and MUL_MAT_ID restrictions. Test-only FP64 dot products handle batch broadcasting. |
| OUT_PROD | 704 | Convert the FP16 right operand on-device, including transposed layouts; reuse FP32 BLAS. Extend the existing FP64 host oracle to read FP16 right operands. |
| GET_ROWS_BACK | 54 | Convert BF16, quantized or I32 gradient inputs on-device; reuse the existing FP32 scatter-add equivalent. The host reference independently accumulates duplicate row IDs in FP64. Fix I32 test initialization so gradient data is not mistaken for row IDs. |
| SOFT_MAX_BACK | 24 | Permit constant ALiBi bias with the existing backward formula, given the forward output. No mask/slope gradient is produced. Compare against an explicit FP64 formula. |

Conversion may allocate a full FP32 temporary; these are correctness adapters, not performance claims. Support guards retain bounded indices, layouts, scalar types and shape constraints. Existing optimized paths outside these new combinations are unchanged. The source-derived target is **24591 OK and 450 unsupported**, not a hardware result. The remaining 450 cases are CPY (90) and SET_ROWS (360) quantized destination encoders: MXFP4, NVFP4, Q2_K through Q6_K, IQ4_XS, IQ2_XXS/XS/S, IQ3_XXS/S and IQ1_S/M. No host round-trip is advertised as GPU support for those encoders.

FA diagnosis is separate from capability acceptance. `LLAMA_TEST_FA_SEED` enables per-case/tensor deterministic inputs in the FA tests; the unconfigured default remains random. `FA_REPRO` records the seed, exact case and FNV-1a fingerprints of logical input bytes. The large-logit variant also uses the seed; the sparse mask retains its existing fixed generator. Full tests and isolated reruns must match input fingerprints before their results can be compared. These new seeded inputs cannot reconstruct the original failed tensors, which were not saved.

For MACA-only diagnostics, `GGML_MACA_FA_DIAG=1` prints the chosen kernel. `GGML_MACA_FA_FORCE_F32=1` selects the existing FP32 fallback when supported; without that explicit diagnostic override dispatch is unchanged. This is an A/B comparison, not the default fix for the unresolved tile failure. `tests/maca/run.py --fa-seed 20261016 --fa-replay` runs the full normal-dispatch suite, then 16 seeds on both tile and FP32 paths. It checks case identity, seed, input fingerprints, actual path, numerical result and process exit. Independent failures continue; successful reruns cannot erase a failed full run. Numerical tolerances are unchanged. Host reference controls include positive comparisons with existing CPU paths and a separate injected-fault negative control for each of six reference methods.

Local Python, source-contract and packaging checks are not MACA compilation or C500 numerical acceptance. Return the hardware archive before accepting this candidate; model inference, performance, native provider, Graph/fusion and other devices remain separate gates.

## Quantized write candidate (2026-10-16)

The accepted `run-tnj7bn8a` covers 25041 fixed cases: 24591 OK, no numerical failures, and 450 unsupported. Its 2871 source fingerprints match the mixed-input candidate. The remaining inventory comprises 90 CPY and 360 SET_ROWS cases, not 450 distinct operators.

This candidate adds MACA-only GPU block encoders for MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S and IQ1_M. Each format accounts for 30 existing cases. CPY requires a contiguous quantized destination. SET_ROWS preserves index broadcasting, source/index views and empty-source no-op behavior. Scalar sources are packed to FP32 on the device. Existing read, matrix multiply and other vendor paths are not replaced.

The private `ggml-maca/quantize-ref.cuh` scalar encoders and `quantize-tables.h` immutable codebook builders derive from the pinned upstream `ggml-quants.c`. Changes include device qualifiers, explicit table arguments, uint16 half storage, device half conversion, and a bounded insertion sort in place of IQ1's qsort. The IQ1 exhaustive-neighbour fallback indexes its eight-element input with the inner coordinate rather than the grid index, avoiding an upstream out-of-range access in that fallback. `generate_reference.py` and `reference-provenance.json` in the delivery package record the source and transformations. The upstream CPU encoder files remain unchanged.

**Semantic extension:** IQ2_XXS, IQ2_XS and IQ1_S normally require an importance matrix. CPY and SET_ROWS do not accept one. With explicit user approval, MACA uses an all-ones importance vector for their 90 cases. The encoded block layout remains the existing GGML format, but this is not calibrated imatrix quantization and makes no model-quality claim. The test-only reference calls the upstream offline encoder with the same documented importance policy. This does not add production CPU operator support.

The implementation is a correctness-first scalar-per-block GPU path, not a tuned parallel quantizer. Codebook initialization is host-only and independent of tensor values; tensor quantization stays on the GPU. Immutable host tables are uploaded to context-owned temporary buffers on the current stream for each invocation. Initialization, table uploads and per-thread local storage can be expensive; performance and Graph capture are not established.

Acceptance keeps all 25041 case identities and numerical thresholds. The forecast is 25041 OK and zero unsupported, comprising 22033 CPU-backend comparisons and 3008 test-only host-reference comparisons. An additional host executable compares the scalar port with unchanged offline encoders on 540 deterministic zero, constant, tie and random patterns. It does not replace GPU verification. Forty-two CPU-backed positive controls and eight injected-fault negative controls check the reference methods; the existing FA replay and independent operator groups remain enabled. All counts are acceptance targets until the new hardware result archive is audited.

## Quantized write rounding candidate (2026-10-16)

The quant-write-v1 hardware run recorded 24930 OK and 111 failures, with no unsupported cases. The later diagnostic completed 45 groups after isolating its IQ helper namespace, but 27 groups still differed from the CPU reference. In its arithmetic probe, all 65536 GPU multiply-add outputs matched single-rounding FMA; the CPU outputs matched separate multiplication and addition, with 15956 differing samples. This identifies a rounding difference, not proof that every quantization failure has the same cause.

`GGML_MACA_QUANTIZE_FP_MODE` applies contraction control to `ggml-maca/quantize.cu` only. `bridge` uses `--fmad=false`, `clang` uses `-ffp-contract=off`, and `clang-strict` also uses `-fno-fast-math`. The delivery runner tests these compiler interfaces on the installed SDK and selects only a mode with matching device arithmetic. An unsupported or ignored option is not accepted just because compilation succeeds. Other translation units, matrix multiplication, attention, CPU quantizers, numerical tolerances and the fixed case inventory are unchanged. Quantized write performance remains unmeasured.

IQ codebook init/free helpers now have internal linkage, matching their file-local tables. This prevents an executable from interposing a helper while the backend reads a different table instance. Both isolated and same-namespace diagnostic executables are checked after rebuilding.

The package reuses the existing incremental build and full operator runner, then collects all 45 fixed-input quantization groups independently. Arithmetic, symbol isolation and complete operator acceptance have separate result fields. Local compilation and tests were not run for this candidate at the user's request. Device acceptance, model tests and performance must not be inferred from packaging success.
