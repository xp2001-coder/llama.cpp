#pragma once

#include "../ggml-cuda/common.cuh"

static inline bool ggml_maca_can_quantize_extended(ggml_type type) {
    switch (type) {
        case GGML_TYPE_MXFP4: case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

bool ggml_maca_quantize_supports(const ggml_tensor * src, const ggml_tensor * dst, const ggml_tensor * indices);
void ggml_maca_quantize(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * indices);
