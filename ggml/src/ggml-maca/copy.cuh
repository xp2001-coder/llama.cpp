#pragma once

#include "../ggml-cuda/cpy.cuh"
#include "../ggml-cuda/convert.cuh"
#include "blas.cuh"

#include <limits>

static inline bool ggml_maca_can_quantize(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 ||
           type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL;
}

static inline bool ggml_maca_copy_layout(const ggml_tensor * tensor, bool contiguous_rows = true) {
    const size_t limit = std::numeric_limits<int64_t>::max() - CUDA_CPY_BLOCK_SIZE;
    const size_t type_size = ggml_type_size(tensor->type);
    const int64_t block_size = ggml_blck_size(tensor->type);
    if (((contiguous_rows || block_size != 1) && tensor->nb[0] != type_size) ||
        tensor->ne[0] <= 0 || tensor->ne[0] % block_size != 0) {
        return false;
    }
    size_t elements = 1;
    size_t span = type_size;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0 || (uint64_t) tensor->ne[d] > limit / elements || tensor->nb[d] > limit ||
            (block_size == 1 && tensor->nb[d] % type_size != 0)) {
            return false;
        }
        elements *= tensor->ne[d];
        const size_t count = d == 0 ? tensor->ne[d] / block_size : tensor->ne[d];
        if (count > 1 && tensor->nb[d] > (limit - span) / (count - 1)) {
            return false;
        }
        span += (count - 1) * tensor->nb[d];
    }
    return elements / block_size <= limit / type_size;
}

static inline bool ggml_maca_can_copy_f32(const ggml_tensor * src) {
    if (!ggml_maca_copy_layout(src, false)) {
        return false;
    }
    if (ggml_is_quantized(src->type) && src->nb[1] < ggml_row_size(src->type, src->ne[0])) {
        return false;
    }
    // Some shared conversion kernels use 32-bit element indices.
    return (src->type == GGML_TYPE_F32 || ggml_get_to_fp32_cuda(src->type) != nullptr) &&
           ggml_nelements(src) <= std::numeric_limits<int>::max() - CUDA_CPY_BLOCK_SIZE;
}

static inline ggml_tensor ggml_maca_pack_scalar(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_cuda_pool_alloc<char> & storage) {
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_BF16);
    GGML_ASSERT(ggml_maca_copy_layout(src, false));
    ggml_tensor packed = *src;
    size_t bytes = ggml_type_size(src->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        packed.nb[d] = bytes;
        GGML_ASSERT((uint64_t) src->ne[d] <= std::numeric_limits<int64_t>::max() / bytes);
        bytes *= src->ne[d];
    }
    packed.data = storage.alloc(bytes);
    ggml_cuda_cpy(ctx, src, &packed);
    return packed;
}

static inline ggml_tensor ggml_maca_copy_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_cuda_pool_alloc<char> & storage) {
    GGML_ASSERT(ggml_maca_can_copy_f32(src));
    ggml_tensor packed = *src;
    packed.type = GGML_TYPE_F32;
    size_t bytes = sizeof(float);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        packed.nb[d] = bytes;
        GGML_ASSERT((uint64_t) src->ne[d] <= std::numeric_limits<int64_t>::max() / bytes);
        bytes *= src->ne[d];
    }
    packed.data = storage.alloc(bytes);
    if (ggml_is_quantized(src->type)) {
        ggml_cuda_pool_alloc<char> encoded(ctx.pool());
        ggml_tensor rows = *src;
        if (!ggml_is_contiguous(src)) {
            rows = ggml_maca_pack_quantized_rows(ctx, src, encoded);
        }
        ggml_get_to_fp32_cuda(src->type)(rows.data, (float *) packed.data, ggml_nelements(src), ctx.stream());
    } else {
        ggml_cuda_cpy(ctx, src, &packed);
    }
    return packed;
}
