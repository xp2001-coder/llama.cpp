# MACA backend

The MACA backend builds the existing `ggml-cuda` implementation through the
MetaX MACA CUDA compatibility bridge. It produces a distinct `ggml-maca`
backend library while sharing CUDA-family backend APIs and kernels.

## Requirements

- A MetaX GPU with a compatible driver
- A MACA SDK installation that includes `tools/cu-bridge`
- CMake and Ninja

Set `MACA_PATH` when the SDK is not available through `/opt/maca`:

```bash
export MACA_PATH=/path/to/maca
```

The SDK compiler wrapper creates a virtual CUDA layout under
`${MACA_VIRTUAL_ROOT:-$HOME/cu-bridge}`. Both paths can be overridden without
modifying the source tree.

## Build

Configure with `GGML_MACA=ON`. The helper script detects the device target when
possible and accepts `CUCC_TARGETS` or `CUCC_TARGETS_FROM_DEVICE` overrides:

```bash
export MACA_PATH=/opt/maca
export CUCC_TARGETS_FROM_DEVICE=xcore1000  # example for a tested C500 setup
scripts/maca/build.sh
```

The CMake CUDA architecture value used by the compatibility toolchain is a
placeholder. The actual MACA target is selected by the `CUCC_TARGETS` family of
environment variables.

## Validation

Run the Q8_0 backend operation tests and model benchmark with:

```bash
scripts/maca/validate.sh /path/to/model-q8_0.gguf
```

The output directory defaults to
`${MACA_TEST_OUTPUT:-${TMPDIR:-/tmp}/llama-maca-validation}`.

Validated configurations include:

- MetaX C500, single GPU
- MetaX C600-A, single GPU and four-device enumeration/execution
- Qwen2.5-3B-Instruct in F16, Q8_0, and Q4_0
- Llama-3.2-3B-Instruct in F16 and Q8_0
- Q8_0 prefill through dequantization plus mcBLAS
- Q8_0 single-token decode through logical-warp32 MMVQ plus MACA `sdot4`
- optional CUDA-compatible graph capture and replay (`-DGGML_MACA_GRAPHS=ON`); disabled by default while graph-update fallback is being hardened
- Interactive and OpenAI-compatible HTTP inference, including UTF-8 prompts

## Current feature policy

The initial backend keeps features without dedicated validation on conservative
fallbacks:

- Flash Attention: disabled
- NCCL: disabled
- virtual memory management: disabled
- peer copy: disabled
- quantized fusion and MMVQ `MUL_MAT_ID`: fallback paths

Q8_0 has a MACA-specific decode optimization. Other quantization formats use
the shared compatibility paths and remain candidates for follow-up performance
work.
