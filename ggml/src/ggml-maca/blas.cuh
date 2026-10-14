#pragma once

#include "../ggml-cuda/common.cuh"

#include <limits>

static ggml_tensor ggml_maca_pack_quantized_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_cuda_pool_alloc<char> & storage) {
    GGML_ASSERT(ggml_is_quantized(src->type));
    const size_t type_size = ggml_type_size(src->type);
    const int64_t block_size = ggml_blck_size(src->type);
    const size_t size_max = std::numeric_limits<size_t>::max();
    GGML_ASSERT(src->nb[0] == type_size && src->ne[0] > 0 && src->ne[0] % block_size == 0);
    GGML_ASSERT((uint64_t) (src->ne[0] / block_size) <= size_max / type_size);

    ggml_tensor packed = *src;
    packed.nb[1] = (src->ne[0] / block_size) * type_size;
    size_t bytes = packed.nb[1];
    size_t source_span = bytes;
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        GGML_ASSERT(src->ne[d] > 0 && (uint64_t) src->ne[d] <= size_max / bytes);
        if (src->ne[d] > 1) {
            GGML_ASSERT(src->nb[d] <= (size_max - source_span) / (src->ne[d] - 1));
            source_span += (src->ne[d] - 1) * src->nb[d];
        }
        bytes *= src->ne[d];
        if (d + 1 < GGML_MAX_DIMS) {
            packed.nb[d + 1] = bytes;
        }
    }
    GGML_ASSERT(src->nb[1] >= packed.nb[1]);
    packed.data = storage.alloc(bytes);

    // Copy whole quantization blocks, without reading row or batch gaps.
    for (int64_t i3 = 0; i3 < src->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src->ne[2]; ++i2) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                (char *) packed.data + i3 * packed.nb[3] + i2 * packed.nb[2], packed.nb[1],
                (const char *) src->data + i3 * src->nb[3] + i2 * src->nb[2], src->nb[1],
                packed.nb[1], src->ne[1], cudaMemcpyDeviceToDevice, ctx.stream()));
        }
    }
    return packed;
}
