# MACA v2 bring-up

This development snapshot starts from upstream commit `8f574b0ad44c46d7b499ec028c29205aaa3fe80f` (2026-10-13). It adds an experimental `GGML_MACA` backend, disabled by default. The latest C500 run completed the full default operator suite with 20497 numerical comparisons passed and no numerical failures, but exact inventory acceptance is still blocked by 20 SET_ROWS cases that remain unsupported. This is a development checkpoint, not a fully supported backend or a performance release. Do not apply old branch acceptance counts to this snapshot.

## Current validation status (2026-10-14)

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
