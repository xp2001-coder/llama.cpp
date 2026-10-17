// Host audit of the scalar port against the unchanged upstream offline encoders.
#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define __device__
#define __half_as_ushort(x) (x)
#define __ushort_as_half(x) (x)
#define __float2half_rn ggml_fp32_to_fp16
#define __half2float ggml_fp16_to_fp32
#include "ggml-maca/quantize-ref.cuh"
#include "ggml-maca/quantize-tables.h"
#undef __device__
#undef __half_as_ushort
#undef __ushort_as_half
#undef __float2half_rn
#undef __half2float

static ggml_maca_quant_ref::tables get_tables(ggml_type type) {
    using namespace ggml_maca_quant_tables;
    if (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S) {
        const int size = type == GGML_TYPE_IQ3_XXS ? 256 : 512;
        ggml_maca_quant_tables::iq3xs_init_impl(size);
        const auto & entry = iq3_data[iq3_data_index(size)];
        return {nullptr, entry.grid, entry.map, entry.neighbours};
    }
    if (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
        type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
        ggml_maca_quant_tables::iq2xs_init_impl(type);
        const auto & entry = iq2_data[iq2_data_index(type)];
        return {entry.grid, nullptr, entry.map, entry.neighbours};
    }
    return {};
}

static void encode(ggml_type type, const float * x, void * y, const float * weights,
                   const ggml_maca_quant_ref::tables & table) {
    namespace ref = ggml_maca_quant_ref;
    switch (type) {
#define TEST_REF(TYPE, block, qk) case GGML_TYPE_##TYPE: ref::quantize_row_##block##_ref(x, (ref::block_##block *) y, qk); break
        TEST_REF(MXFP4, mxfp4, QK_MXFP4);
        TEST_REF(NVFP4, nvfp4, QK_NVFP4);
        TEST_REF(Q2_K, q2_K, QK_K); TEST_REF(Q3_K, q3_K, QK_K); TEST_REF(Q4_K, q4_K, QK_K);
        TEST_REF(Q5_K, q5_K, QK_K); TEST_REF(Q6_K, q6_K, QK_K);
#undef TEST_REF
        case GGML_TYPE_IQ4_XS: ref::quantize_iq4_xs(x, y, 1, QK_K, nullptr); break;
        case GGML_TYPE_IQ2_XXS: ref::quantize_row_iq2_xxs_impl(table, x, y, QK_K, weights); break;
        case GGML_TYPE_IQ2_XS: ref::quantize_row_iq2_xs_impl(table, x, y, QK_K, weights); break;
        case GGML_TYPE_IQ2_S: ref::quantize_row_iq2_s_impl(table, x, y, QK_K, nullptr); break;
        case GGML_TYPE_IQ3_XXS: ref::quantize_row_iq3_xxs_impl(table, 256, x, y, QK_K, nullptr); break;
        case GGML_TYPE_IQ3_S: ref::quantize_iq3_s(table, x, y, 1, QK_K, nullptr); break;
        case GGML_TYPE_IQ1_S: ref::quantize_iq1_s(table, x, y, 1, QK_K, weights); break;
        case GGML_TYPE_IQ1_M: ref::quantize_iq1_m(table, x, y, 1, QK_K, nullptr); break;
        default: GGML_ABORT("Unknown test type");
    }
}

int main() {
    ggml_init_params params = {16*1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);
    int checked = 0;
    for (ggml_type type : {GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
                          GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS,
                          GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                          GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M}) {
        const auto table = get_tables(type);
        const int qk = ggml_blck_size(type);
        const size_t bytes = ggml_type_size(type);
        std::vector<float> x(qk), weights(qk, 1.0f), a(qk), b(qk);
        std::vector<uint8_t> actual(bytes), expected(bytes);
        for (int pattern = 0; pattern < 36; ++pattern) {
            uint32_t rng = 0x31415926u + pattern;
            for (int i = 0; i < qk; ++i) {
                rng = rng*1664525u + 1013904223u;
                float value = ((int32_t) (rng >> 8) - 8388608)*(1.0f/8388608);
                if (pattern == 0) { value = 0; }
                if (pattern == 1) { value = 1; }
                if (pattern == 2) { value = (i%5 - 2)*0.25f; }
                if (pattern == 3) { value = i%2 ? -0.0f : 0.0f; }
                if (pattern >= 4) { value *= pattern%3 == 0 ? 150.0f : pattern%3 == 1 ? 0.001f : 1.0f; }
                x[i] = value;
            }
            std::fill(actual.begin(), actual.end(), 0xa5);
            std::fill(expected.begin(), expected.end(), 0xa5);
            const float * importance = ggml_quantize_requires_imatrix(type) ? weights.data() : nullptr;
            const auto written = ggml_quantize_chunk(type, x.data(), expected.data(), 0, 1, qk, importance);
            GGML_ASSERT(written == bytes);
            encode(type, x.data(), actual.data(), importance, table);
            const auto traits = ggml_get_type_traits(type);
            traits->to_float(expected.data(), a.data(), qk);
            traits->to_float(actual.data(), b.data(), qk);
            double diff = 0, norm = 0;
            for (int i = 0; i < qk; ++i) {
                diff += (double) (a[i] - b[i])*(a[i] - b[i]);
                norm += (double) a[i]*a[i];
            }
            if (!std::isfinite(diff) || (norm == 0 ? diff != 0 : diff/norm > 1e-7)) {
                fprintf(stderr, "FAIL %s pattern=%d nmse=%g\n", ggml_type_name(type), pattern, norm ? diff/norm : diff);
                return 1;
            }
            ++checked;
        }
        printf("PASS quantizer host reference: %s (36 patterns)\n", ggml_type_name(type));
    }
    ggml_free(ctx);
    printf("PASS quantizer host reference: %d/540; NOT a GPU test\n", checked);
    return checked == 540 ? 0 : 1;
}
