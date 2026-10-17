#include "quantize.cuh"
#include "copy.cuh"
#include <cfloat>
#include "quantize-ref.cuh"
#include "quantize-tables.h"

#include <algorithm>

namespace {

using ggml_maca_quant_ref::tables;

struct host_tables {
    std::mutex mutex;
    ~host_tables() {
        using namespace ggml_maca_quant_tables;
        for (ggml_type type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S}) {
            ggml_maca_quant_tables::iq2xs_free_impl(type);
        }
        ggml_maca_quant_tables::iq3xs_free_impl(256);
        ggml_maca_quant_tables::iq3xs_free_impl(512);
    }
};

// The immutable host tables outlive asynchronous uploads. Pool buffers use the context stream.
static tables upload_tables(ggml_backend_cuda_context & ctx, ggml_type type,
                            ggml_cuda_pool_alloc<char> & grid, ggml_cuda_pool_alloc<int> & map,
                            ggml_cuda_pool_alloc<uint16_t> & neighbours) {
    using namespace ggml_maca_quant_tables;
    static host_tables owner;
    std::lock_guard<std::mutex> lock(owner.mutex);
    tables result = {};
    const void * hgrid = nullptr;
    const int * hmap = nullptr;
    const uint16_t * hneighbours = nullptr;
    size_t grid_bytes = 0;
    int map_size = 0;
    const bool iq3 = type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
    if (iq3) {
        const int n = type == GGML_TYPE_IQ3_XXS ? 256 : 512;
        ggml_maca_quant_tables::iq3xs_init_impl(n);
        const auto & t = iq3_data[iq3_data_index(n)];
        hgrid = t.grid; hmap = t.map; hneighbours = t.neighbours;
        grid_bytes = n*sizeof(uint32_t);
        map_size = 4096;
    } else if (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
               type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
        ggml_maca_quant_tables::iq2xs_init_impl(type);
        const auto & t = iq2_data[iq2_data_index(type)];
        hgrid = t.grid; hmap = t.map; hneighbours = t.neighbours;
        grid_bytes = iq2_grid_size(type)*sizeof(uint64_t);
        map_size = 43692;
    } else {
        return result;
    }
    size_t neighbour_count = 0;
    for (int i = 0; i < map_size; ++i) {
        if (hmap[i] < 0) {
            const size_t offset = -(int64_t) hmap[i] - 1;
            neighbour_count = std::max(neighbour_count, offset + 1 + hneighbours[offset]);
        }
    }
    char * device_grid = grid.alloc(grid_bytes);
    int * device_map = map.alloc(map_size);
    uint16_t * device_neighbours = neighbours.alloc(neighbour_count);
    CUDA_CHECK(cudaMemcpyAsync(device_grid, hgrid, grid_bytes, cudaMemcpyHostToDevice, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(device_map, hmap, map_size*sizeof(int), cudaMemcpyHostToDevice, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(device_neighbours, hneighbours, neighbour_count*sizeof(uint16_t), cudaMemcpyHostToDevice, ctx.stream()));
    if (iq3) { result.grid32 = (const uint32_t *) device_grid; }
    else { result.grid64 = (const uint64_t *) device_grid; }
    result.map = device_map;
    result.neighbours = device_neighbours;
    return result;
}

struct row_layout {
    int64_t ne0, ne1, ne2, ne3;
    int64_t idx_ne1, idx_ne2, dst_ne1;
    size_t idx_nb0, idx_nb1, idx_nb2, dst_nb1, dst_nb2, dst_nb3;
    ggml_type index_type;
};

template<ggml_type type>
__device__ static void encode_block(const tables & table, const float * x, void * y) {
    using namespace ggml_maca_quant_ref;
    if constexpr (type == GGML_TYPE_MXFP4) { quantize_row_mxfp4_ref(x, (ggml_maca_quant_ref::block_mxfp4 *) y, QK_MXFP4); }
    if constexpr (type == GGML_TYPE_NVFP4) { quantize_row_nvfp4_ref(x, (ggml_maca_quant_ref::block_nvfp4 *) y, QK_NVFP4); }
    if constexpr (type == GGML_TYPE_Q2_K) { quantize_row_q2_K_ref(x, (ggml_maca_quant_ref::block_q2_K *) y, QK_K); }
    if constexpr (type == GGML_TYPE_Q3_K) { quantize_row_q3_K_ref(x, (ggml_maca_quant_ref::block_q3_K *) y, QK_K); }
    if constexpr (type == GGML_TYPE_Q4_K) { quantize_row_q4_K_ref(x, (ggml_maca_quant_ref::block_q4_K *) y, QK_K); }
    if constexpr (type == GGML_TYPE_Q5_K) { quantize_row_q5_K_ref(x, (ggml_maca_quant_ref::block_q5_K *) y, QK_K); }
    if constexpr (type == GGML_TYPE_Q6_K) { quantize_row_q6_K_ref(x, (ggml_maca_quant_ref::block_q6_K *) y, QK_K); }
    if constexpr (type == GGML_TYPE_IQ4_XS) { ggml_maca_quant_ref::quantize_iq4_xs(x, y, 1, QK_K, nullptr); }
    if constexpr (type == GGML_TYPE_IQ2_S) { quantize_row_iq2_s_impl(table, x, y, QK_K, nullptr); }
    if constexpr (type == GGML_TYPE_IQ3_XXS) { quantize_row_iq3_xxs_impl(table, 256, x, y, QK_K, nullptr); }
    if constexpr (type == GGML_TYPE_IQ3_S) { ggml_maca_quant_ref::quantize_iq3_s(table, x, y, 1, QK_K, nullptr); }
    if constexpr (type == GGML_TYPE_IQ1_M) { ggml_maca_quant_ref::quantize_iq1_m(table, x, y, 1, QK_K, nullptr); }
    if constexpr (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ1_S) {
        // Operator APIs have no imatrix. This MACA extension uses uniform importance, not calibration data.
        float weights[QK_K];
        for (int i = 0; i < QK_K; ++i) { weights[i] = 1.0f; }
        if constexpr (type == GGML_TYPE_IQ2_XXS) { quantize_row_iq2_xxs_impl(table, x, y, QK_K, weights); }
        if constexpr (type == GGML_TYPE_IQ2_XS) { quantize_row_iq2_xs_impl(table, x, y, QK_K, weights); }
        if constexpr (type == GGML_TYPE_IQ1_S) { ggml_maca_quant_ref::quantize_iq1_s(table, x, y, 1, QK_K, weights); }
    }
}

template<ggml_type type>
__global__ static void quantize_blocks(const float * src, char * dst, const char * indices,
                                       row_layout layout, int qk, size_t block_bytes, int64_t count, tables table) {
    const int64_t block = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (block >= count) { return; }
    size_t offset = block*block_bytes;
    if (indices) {
        const int64_t column = block % (layout.ne0/qk);
        int64_t row = block / (layout.ne0/qk);
        const int64_t i1 = row % layout.ne1;
        row /= layout.ne1;
        const int64_t i2 = row % layout.ne2;
        const int64_t i3 = row / layout.ne2;
        const char * address = indices + i1*layout.idx_nb0 + (i2%layout.idx_ne1)*layout.idx_nb1 +
                                (i3%layout.idx_ne2)*layout.idx_nb2;
        const int64_t index = layout.index_type == GGML_TYPE_I64 ? *(const int64_t *) address : *(const int32_t *) address;
        assert(index >= 0 && index < layout.dst_ne1);
        if (index < 0 || index >= layout.dst_ne1) { return; }
        offset = index*layout.dst_nb1 + i2*layout.dst_nb2 + i3*layout.dst_nb3 + column*block_bytes;
    }
    encode_block<type>(table, src + block*qk, dst + offset);
}

} // namespace

bool ggml_maca_quantize_supports(const ggml_tensor * src, const ggml_tensor * dst, const ggml_tensor * indices) {
    if (!ggml_maca_can_quantize_extended(dst->type) || !ggml_maca_copy_layout(dst) ||
        (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
        return false;
    }
    if (!indices) {
        return ggml_maca_can_copy_f32(src) && ggml_is_contiguous(dst) && ggml_nelements(src) == ggml_nelements(dst);
    }
    if ((indices->type != GGML_TYPE_I32 && indices->type != GGML_TYPE_I64) || indices->ne[3] != 1 ||
        src->ne[0] != dst->ne[0] || src->ne[1] != indices->ne[0] || src->ne[2] != dst->ne[2] || src->ne[3] != dst->ne[3] ||
        indices->ne[1] <= 0 || indices->ne[2] <= 0 ||
        src->ne[2] % indices->ne[1] != 0 || src->ne[3] % indices->ne[2] != 0) {
        return false;
    }
    if (src->ne[1] == 0) { return true; }
    return ggml_maca_can_copy_f32(src) && ggml_maca_copy_layout(indices, false) &&
           dst->nb[1] >= ggml_row_size(dst->type, dst->ne[0]);
}

void ggml_maca_quantize(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, const ggml_tensor * indices) {
    GGML_ASSERT(ggml_maca_quantize_supports(src, dst, indices));
    if (indices && src->ne[1] == 0) { return; }
    ggml_cuda_pool_alloc<char> packed(ctx.pool());
    const ggml_tensor input = ggml_maca_copy_f32(ctx, src, packed);
    ggml_cuda_pool_alloc<char> grid(ctx.pool());
    ggml_cuda_pool_alloc<int> map(ctx.pool());
    ggml_cuda_pool_alloc<uint16_t> neighbours(ctx.pool());
    const tables table = upload_tables(ctx, dst->type, grid, map, neighbours);
    row_layout layout = {};
    if (indices) {
        layout = {src->ne[0], src->ne[1], src->ne[2], src->ne[3], indices->ne[1], indices->ne[2], dst->ne[1],
                  indices->nb[0], indices->nb[1], indices->nb[2], dst->nb[1], dst->nb[2], dst->nb[3], indices->type};
    }
    const int qk = ggml_blck_size(dst->type);
    const int64_t count = ggml_nelements(src)/qk;
    const size_t block_bytes = ggml_type_size(dst->type);
    const int threads = 32;
    const unsigned blocks = (unsigned) ((count + threads - 1)/threads);
    const char * rows = indices ? (const char *) indices->data : nullptr;
#define MACA_ENCODE(TYPE) case GGML_TYPE_##TYPE: \
    quantize_blocks<GGML_TYPE_##TYPE><<<blocks, threads, 0, ctx.stream()>>>( \
        (const float *) input.data, (char *) dst->data, rows, layout, qk, block_bytes, count, table); break
    switch (dst->type) {
        MACA_ENCODE(MXFP4); MACA_ENCODE(NVFP4);
        MACA_ENCODE(Q2_K); MACA_ENCODE(Q3_K); MACA_ENCODE(Q4_K); MACA_ENCODE(Q5_K); MACA_ENCODE(Q6_K);
        MACA_ENCODE(IQ4_XS); MACA_ENCODE(IQ2_XXS); MACA_ENCODE(IQ2_XS); MACA_ENCODE(IQ2_S);
        MACA_ENCODE(IQ3_XXS); MACA_ENCODE(IQ3_S); MACA_ENCODE(IQ1_S); MACA_ENCODE(IQ1_M);
        default: GGML_ABORT("Unsupported MACA quantization type");
    }
#undef MACA_ENCODE
    CUDA_CHECK(cudaGetLastError());
}
