// Derived from ggml-quants.c at 8f574b0ad44c46d7b499ec028c29205aaa3fe80f.
// Regenerate with generate_reference.py in the quant-write delivery package.


#pragma once

namespace ggml_maca_quant_ref {


typedef struct {
    uint8_t e; // E8M0
    uint8_t qs[QK_MXFP4/2];
} block_mxfp4;

static_assert(sizeof(block_mxfp4) == sizeof(::block_mxfp4), "block layout changed");

typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;

static_assert(sizeof(block_nvfp4) == sizeof(::block_nvfp4), "block layout changed");

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            uint16_t d;    // super-block scale for quantized scales
            uint16_t dmin; // super-block scale for quantized mins
        } ;
        uint32_t dm;
    } ;
} block_q2_K;

static_assert(sizeof(block_q2_K) == sizeof(::block_q2_K), "block layout changed");

typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    uint16_t d;           // super-block scale
} block_q3_K;

static_assert(sizeof(block_q3_K) == sizeof(::block_q3_K), "block layout changed");

typedef struct {
    union {
        struct {
            uint16_t d;    // super-block scale for quantized scales
            uint16_t dmin; // super-block scale for quantized mins
        } ;
        uint32_t dm;
    } ;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

static_assert(sizeof(block_q4_K) == sizeof(::block_q4_K), "block layout changed");

typedef struct {
    union {
        struct {
            uint16_t d;    // super-block scale for quantized scales
            uint16_t dmin; // super-block scale for quantized mins
        } ;
        uint32_t dm;
    } ;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;

static_assert(sizeof(block_q5_K) == sizeof(::block_q5_K), "block layout changed");

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    uint16_t d;             // super-block scale
} block_q6_K;

static_assert(sizeof(block_q6_K) == sizeof(::block_q6_K), "block layout changed");

typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;

static_assert(sizeof(block_iq4_xs) == sizeof(::block_iq4_xs), "block layout changed");

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;

static_assert(sizeof(block_iq2_xxs) == sizeof(::block_iq2_xxs), "block layout changed");

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;

static_assert(sizeof(block_iq2_xs) == sizeof(::block_iq2_xs), "block layout changed");

typedef struct {
    uint16_t d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;

static_assert(sizeof(block_iq2_s) == sizeof(::block_iq2_s), "block layout changed");

typedef struct {
    uint16_t d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;

static_assert(sizeof(block_iq3_xxs) == sizeof(::block_iq3_xxs), "block layout changed");

typedef struct {
    uint16_t d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;

static_assert(sizeof(block_iq3_s) == sizeof(::block_iq3_s), "block layout changed");

typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;

static_assert(sizeof(block_iq1_s) == sizeof(::block_iq1_s), "block layout changed");

typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (4-bit if QK_K == 64)
} block_iq1_m;

static_assert(sizeof(block_iq1_m) == sizeof(::block_iq1_m), "block layout changed");

union iq1m_scale_t { uint16_t f16; uint16_t u16; };

struct tables { const uint64_t * grid64; const uint32_t * grid32; const int * map; const uint16_t * neighbours; };

__device__ static uint16_t to_half(float x) { return __half_as_ushort(__float2half_rn(x)); }

__device__ static float from_half(uint16_t x) { return __half2float(__ushort_as_half(x)); }

__device__ static void sort_pairs(float * pairs, int n) {
    for (int i = 1; i < n; ++i) {
        const float value = pairs[2*i];
        uint32_t index;
        memcpy(&index, pairs + 2*i + 1, sizeof(index));
        int j = i;
        while (j > 0 && pairs[2*(j-1)] > value) {
            memcpy(pairs + 2*j, pairs + 2*(j-1), 2*sizeof(float));
            --j;
        }
        pairs[2*j] = value;
        memcpy(pairs + 2*j + 1, &index, sizeof(index));
    }
}

static constexpr float group_max_eps = 1e-15f;

static constexpr float group_max_eps_iq3_xxs = 1e-8f;

static constexpr float group_max_eps_iq2_s = 1e-8f;

static constexpr float group_max_eps_iq1_m = 1e-7f;

static constexpr float group_max_eps_iq1_s = 1e-12f;

static constexpr int IQ3S_BLOCK_SIZE = 32;

static constexpr int IQ1S_BLOCK_SIZE = 32;

static constexpr int IQ1M_BLOCK_SIZE = 16;

__device__ static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    if (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    else {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

__device__ static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

__device__ static inline uint8_t ggml_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    int fp32_man  = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;
        }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

__device__ static int best_index_int8(int n, const int8_t * val, float x) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

__device__ static int best_index_mxfp4(float x, float e) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    int best_index = 0;
    float best_err = fabsf(kvalues_mxfp4[0]*e - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i]*e - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return best_index;
}

__device__ static int nearest_int(float fval) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

__device__ static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
        const float * GGML_RESTRICT qw) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < group_max_eps) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {

        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

__device__ static float make_q3_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, bool do_rmse) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < group_max_eps) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

__device__ static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {

        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

__device__ static void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

__device__ static float make_qp_quants(int n, int nmax, const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT L, const float * quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    if (max < group_max_eps) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        float w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            float diff = x[i] - scale_is*l;
            float w = quant_weights[i];
            mse += w*diff*diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

__device__ static int iq2_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint64_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

__device__ static int iq3_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint32_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

__device__ static int iq1_find_best_neighbour2(const uint16_t * GGML_RESTRICT neighbours, const uint64_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, const float * GGML_RESTRICT xg, int8_t * GGML_RESTRICT L, int ngrid) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_score = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = xg[(pg[i] - 1)/2];
            float w = weight[i];
            float diff = scale*q - xval[i];
            d2 += w*diff*diff;
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = neighbours[j];
        }
    }
    if (grid_index < 0) {
        for (int i = 0; i < ngrid; ++i) {
            const int8_t * grid_i = (const int8_t *)(grid + i);
            float d2 = 0;
            for (int j = 0; j < 8; ++j) {
                float w = weight[j];
                float q = xg[(grid_i[j] - 1)/2];
                float diff = scale*q - xval[j];
                d2 += w*diff*diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    assert(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

__device__ static void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    static const int qk = QK_MXFP4;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (amax < fabsf(v)) {
                amax = fabsf(v);
            }
        }

        const uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;

        const float d = ggml_e8m0_to_fp32_half(e);

        y[i].e = e;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t x0 = best_index_mxfp4(x[i*qk + 0    + j], d);
            const uint8_t x1 = best_index_mxfp4(x[i*qk + qk/2 + j], d);

            y[i].qs[j]  = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

__device__ static void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float * xb = x + i*qk + s*qk_sub;

            float amax = 0.0f;
            for (int j = 0; j < qk_sub; j++) {
                if (amax < fabsf(xb[j])) {
                    amax = fabsf(xb[j]);
                }
            }

            // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
            const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
            y[i].d[s] = ue;
            const float d = ggml_ue4m3_to_fp32(ue);

            for (int j = 0; j < qk_sub/2; ++j) {
                const uint8_t x0 = best_index_mxfp4(xb[0        + j], d);
                const uint8_t x1 = best_index_mxfp4(xb[qk_sub/2 + j], d);

                y[i].qs[s*(qk_sub/2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

__device__ static void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = to_half(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = to_half(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = to_half(max_min/q4scale);
        } else {
            y[i].dmin = to_half(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = from_half(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = from_half(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = MAX(0, MIN(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

__device__ static void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = to_half(1/iscale);
        } else {
            y[i].d = to_half(0.f);
        }

        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = from_half(y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

__device__ static void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = to_half(max_scale/63.f);
        y[i].dmin = to_half(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = from_half(y[i].d) * sc;
            if (!d) continue;
            const float dm = from_half(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

__device__ static void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float weights[32];
    uint8_t Laux[32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = to_half(max_scale/63.f);
        y[i].dmin = to_half(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = from_half(y[i].d) * sc;
            if (!d) continue;
            const float dm = from_half(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * GGML_RESTRICT qh = y[i].qh;
        uint8_t * GGML_RESTRICT ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

__device__ static void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < group_max_eps) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = to_half(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = to_half(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = from_half(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

__device__ static void quantize_row_iq2_xxs_impl(const tables & table, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif


    const uint64_t * kgrid_q2xs      = table.grid64;
    const int      * kmap_q2xs       = table.map;
    const uint16_t * kneighbors_q2xs = table.neighbours;

    assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    assert(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    assert(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_xxs * y = (block_iq2_xxs *) vy;

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = to_half(0.f);
        memset(q2, 0, QK_K/4);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            const float * qw = quant_weights + QK_K*ibl + 32*ib;
            for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < group_max_eps) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
            float eff_max = scale*kMaxQ;
            if (eff_max <= 0) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/eff_max;
                float this_scale = 1/id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            if (scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + grid_index);
                    for (int i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                assert(grid_index >= 0);
                q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
                q2[2*ib+1] |= (block_signs[k] << 7*k);
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = to_half(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            q2[2*ib+1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}

__device__ static void quantize_row_iq2_xs_impl(const tables & table, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif


    const uint64_t * kgrid_q2xs      = table.grid64;
    const int      * kmap_q2xs       = table.map;
    const uint16_t * kneighbors_q2xs = table.neighbours;

    assert(quant_weights   && "missing quantization weights");
    assert(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    assert(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    assert(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_xs * y = (block_iq2_xs *) vy;

    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2*(QK_K/16)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = to_half(0.f);
        memset(q2, 0, QK_K/4);
        memset(y[ibl].scales, 0, QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            const float * qw = quant_weights + QK_K*ibl + 16*ib;
            for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            if (max < group_max_eps) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                assert(grid_index >= 0);
                q2[2*ib+k] = grid_index | (block_signs[k] << 9);
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = to_half(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);

    }
}

__device__ static void quantize_row_iq3_xxs_impl(const tables & table, int grid_size, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n,
        const float * GGML_RESTRICT quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif


    const uint32_t * kgrid_q3xs      = table.grid32;
    const int      * kmap_q3xs       = table.map;
    const uint16_t * kneighbors_q3xs = table.neighbours;

    //assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    assert(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    assert(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    assert(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    uint16_t * dh;
    uint8_t * qs;
    int block_size;
    if (grid_size == 256) {
        block_iq3_xxs * y = (block_iq3_xxs *) vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_xxs);
    } else {
        block_iq3_s * y = (block_iq3_s *) vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_s);
    }
    int quant_size = block_size - sizeof(uint16_t);

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    bool   is_on_grid[8];
    bool   is_on_grid_aux[8];
    uint8_t block_signs[8];
    alignas(4) uint8_t q3[3*(QK_K/8)+QK_K/32];
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4);
    uint8_t  * qh = q3 + 3*(QK_K/8);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        dh[0] = to_half(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 32; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 32);
            if (max < group_max_eps_iq3_xxs) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < 8; ++k) is_on_grid[k] = true;
            for (int is = -15; is <= 15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 8; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                assert(grid_index >= 0);
                if (grid_size == 256) {
                    q3[8*ib+k] = grid_index;
                } else {
                    q3[8*ib+k] = grid_index & 255;
                    qh[ib] |= ((grid_index >> 8) << k);
                }

            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size/sizeof(uint16_t);
            qs += block_size;
            continue;
        }

        float d = max_scale/31;
        dh[0] = to_half(d * 1.0125f);  // small improvement via this fudge factor
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(qs, q3, quant_size);

        dh += block_size/sizeof(uint16_t);
        qs += block_size;

    }
}

__device__ static void quantize_row_iq3_s_impl(const tables & table, int block_size, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int n,
        const float * GGML_RESTRICT quant_weights,
        float   * scales,
        float   * weight,
        float   * xval,
        int8_t  * L,
        int8_t  * Laux,
        float   * waux,
        bool    * is_on_grid,
        bool    * is_on_grid_aux,
        uint8_t * block_signs) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif


    const uint32_t * kgrid_q3xs      = table.grid32;
    const int      * kmap_q3xs       = table.map;
    const uint16_t * kneighbors_q3xs = table.neighbours;

    //assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    assert(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    assert(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    assert(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    block_iq3_s * y = (block_iq3_s *) vy;

    const int bs4 = block_size/4;
    const int bs8 = block_size/8;

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = to_half(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
            memset(L, 0, block_size);
            if (!max) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < bs4; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                    for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < bs4; ++k) {
                    //if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                assert(grid_index >= 0);
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = to_half(d * 1.033f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ib += 2) {
            int l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
            l1 = MAX(0, MIN(15, l1));
            int l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
            l2 = MAX(0, MIN(15, l2));
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }

    }
}

__device__ static size_t quantize_iq3_s(const tables & table, const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    float scales[QK_K/IQ3S_BLOCK_SIZE];
    float weight[IQ3S_BLOCK_SIZE];
    float xval[IQ3S_BLOCK_SIZE];
    int8_t L[IQ3S_BLOCK_SIZE];
    int8_t Laux[IQ3S_BLOCK_SIZE];
    float  waux[IQ3S_BLOCK_SIZE];
    bool   is_on_grid[IQ3S_BLOCK_SIZE/4];
    bool   is_on_grid_aux[IQ3S_BLOCK_SIZE/4];
    uint8_t block_signs[IQ3S_BLOCK_SIZE/8];
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq3_s_impl(table, IQ3S_BLOCK_SIZE, src, qrow, n_per_row, quant_weights,
                scales, weight, xval, L, Laux, waux, is_on_grid, is_on_grid_aux, block_signs);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq3_s);
    }
    return nrow * nblock * sizeof(block_iq3_s);
}

__device__ static void quantize_row_iq1_s_impl(const tables & table, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights,
        float    * scales,
        float    * weight,
        float    * sumx,
        float    * sumw,
        float    * pairs,
        int8_t   * L,
        uint16_t * index,
        int8_t   * shifts) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif


    const uint64_t * kgrid_q2xs      = table.grid64;
    const int      * kmap_q2xs       = table.map;
    const uint16_t * kneighbors_q2xs = table.neighbours;

    assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    assert(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    assert(n%QK_K == 0);

    block_iq1_s * y = (block_iq1_s *) vy;

    const int64_t nbl = n/QK_K;

    const int block_size = IQ1S_BLOCK_SIZE;

    const float x_p[3] = {-1 + IQ1S_DELTA,  IQ1S_DELTA, 1 + IQ1S_DELTA};
    const float x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};


    int * idx = (int *)(pairs + 1);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = to_half(0.f);
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            const float * qw = quant_weights + QK_K*ibl + block_size*ib;
            for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
            if (max < group_max_eps_iq1_s) {
                scales[ib] = 0;
                shifts[ib] = 1;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            sort_pairs(pairs, block_size);
            {
                sumx[0] = sumw[0] = 0;
                for (int j = 0; j < block_size; ++j) {
                    int i = idx[2*j];
                    sumx[j+1] = sumx[j] + weight[i]*xb[i];
                    sumw[j+1] = sumw[j] + weight[i];
                }
            }
            float best_score = -FLT_MAX, scale = max;
            int besti1 = -1, besti2 = -1, best_shift = 0;
            for (int i1 = 0; i1 <= block_size; ++i1) {
                for (int i2 = i1; i2 <= block_size; ++i2) {
                    float sumqx = (sumx[i1] - sumx[0])*x_p[0] + (sumx[i2] - sumx[i1])*x_p[1] + (sumx[block_size] - sumx[i2])*x_p[2];
                    float sumq2 = (sumw[i1] - sumw[0])*x_p[0]*x_p[0] + (sumw[i2] - sumw[i1])*x_p[1]*x_p[1] + (sumw[block_size] - sumw[i2])*x_p[2]*x_p[2];
                    if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                        scale = sumqx/sumq2; best_score = scale*sumqx;
                        besti1 = i1; besti2 = i2; best_shift = 1;
                    }
                    sumqx = (sumx[i1] - sumx[0])*x_m[0] + (sumx[i2] - sumx[i1])*x_m[1] + (sumx[block_size] - sumx[i2])*x_m[2];
                    sumq2 = (sumw[i1] - sumw[0])*x_m[0]*x_m[0] + (sumw[i2] - sumw[i1])*x_m[1]*x_m[1] + (sumw[block_size] - sumw[i2])*x_m[2]*x_m[2];
                    if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                        scale = sumqx/sumq2; best_score = scale*sumqx;
                        besti1 = i1; besti2 = i2; best_shift = -1;
                    }
                }
            }
            if (besti1 < 0 || besti2 < 0 || best_shift == 0) {
                scales[ib] = 0;
                shifts[ib] = 1;
                memset(L, 1, block_size);
                continue;
            }
            for (int j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale; best_shift = -best_shift;
            }
            bool all_on_grid = true;
            const float * xx = best_shift == 1 ? x_p : x_m;
            for (int k = 0; k < block_size/8; ++k) {
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
                    assert(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            if (!all_on_grid) {
                float sumqx = 0, sumq2 = 0;
                for (int k = 0; k < block_size/8; ++k) {
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8*k + j];
                        float q = xx[(pg[j] - 1)/2];
                        sumqx += w*q*xb[8*k+j];
                        sumq2 += w*q*q;
                    }
                }
                if (sumqx > 0 && sumq2 > 0) scale = sumqx/sumq2;
            }
            uint16_t h = 0;
            for (int k = 0; k < block_size/8; ++k) {
                y[ibl].qs[(block_size/8)*ib + k] = index[k] & 255;
                h |= (index[k] >> 8) << 3*k;
            }
            y[ibl].qh[ib] = h;
            assert(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/15;
        y[ibl].d = to_half(d*1.125f); // 1.125f is another fudge factor. Don't ask me why it is needed.
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(7, l));
            if (shifts[ib] == -1) l |= 8;
            y[ibl].qh[ib] |= (l << 12);
        }
    }
}

__device__ static size_t quantize_iq1_s(const tables & table, const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(n_per_row%QK_K == 0);
    float  scales[QK_K/IQ1S_BLOCK_SIZE];
    float  weight[IQ1S_BLOCK_SIZE];
    int8_t L[IQ1S_BLOCK_SIZE];
    float  sumx[IQ1S_BLOCK_SIZE+1];
    float  sumw[IQ1S_BLOCK_SIZE+1];
    float  pairs[2*IQ1S_BLOCK_SIZE];
    uint16_t index[IQ1S_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1S_BLOCK_SIZE];
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq1_s_impl(table, src, qrow, n_per_row, quant_weights, scales, weight, sumx, sumw, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq1_s);
    }
    return nrow * nblock * sizeof(block_iq1_s);
}

__device__ static void quantize_row_iq1_m_impl(const tables & table, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights,
        float    * scales,
        float    * weight,
        float    * pairs,
        int8_t   * L,
        uint16_t * index,
        int8_t   * shifts) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif


    const uint64_t * kgrid_q2xs      = table.grid64;
    const int      * kmap_q2xs       = table.map;
    const uint16_t * kneighbors_q2xs = table.neighbours;

    //assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    assert(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    assert(n%QK_K == 0);

    block_iq1_m * y = (block_iq1_m *) vy;

    const int64_t nbl = n/QK_K;

    const int block_size = IQ1M_BLOCK_SIZE;

    const float x_p[3] = {-1 + IQ1M_DELTA,  IQ1M_DELTA, 1 + IQ1M_DELTA};
    const float x_m[3] = {-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA};
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};

    int * idx = (int *)(pairs + 1);

    float sumqx[4], sumq2[4];

    iq1m_scale_t s;
    const float * xx;

    for (int ibl = 0; ibl < nbl; ++ibl) {
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);
        memset(y[ibl].scales, 0, QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
            if (max < group_max_eps_iq1_m) {
                scales[ib] = 0;
                shifts[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            sort_pairs(pairs, block_size);
            float best_score = -FLT_MAX, scale = max;
            int besti1 = -1, besti2 = -1, best_k = -1;
            // 0: +, +
            // 1: +, -
            // 2: -, +
            // 3: -, -
            // prefix sums are kept per half of the block because each half can use a different sign (x_p or x_m)
            // since v[0]-v[1] = v[1]-v[2] = -1 for both x_p and x_m, the 3-group sum for a split collapses to T*v[2] - px[i1] - px[i2]
            {
                float px[2][IQ1M_BLOCK_SIZE+1];
                float pw[2][IQ1M_BLOCK_SIZE+1];
                px[0][0] = px[1][0] = 0;
                pw[0][0] = pw[1][0] = 0;
                for (int j = 0; j < block_size; ++j) {
                    const int i = idx[2*j];
                    const int h = i < block_size/2 ? 0 : 1;
                    px[h][j+1]   = px[h][j] + weight[i]*xb[i];
                    px[1-h][j+1] = px[1-h][j];
                    pw[h][j+1]   = pw[h][j] + weight[i];
                    pw[1-h][j+1] = pw[1-h][j];
                }
                const float txs[2] = {px[0][block_size], px[1][block_size]}; // total weight*x per half
                const float tws[2] = {pw[0][block_size], pw[1][block_size]}; // total weight per half
                const float p2   = x_p[2], m2 = x_m[2];
                const float cp1  = x_p[0]*x_p[0] - x_p[1]*x_p[1];
                const float cp2  = x_p[1]*x_p[1] - x_p[2]*x_p[2];
                const float cm1  = x_m[0]*x_m[0] - x_m[1]*x_m[1];
                const float cm2  = x_m[1]*x_m[1] - x_m[2]*x_m[2];
                for (int i1 = 0; i1 <= block_size; ++i1) {
                    for (int i2 = i1; i2 <= block_size; ++i2) {
                        float qx_p[2], qx_m[2], q2_p[2], q2_m[2];
                        for (int h = 0; h < 2; ++h) {
                            const float sx = px[h][i1] + px[h][i2];
                            qx_p[h] = txs[h]*p2 - sx;
                            qx_m[h] = txs[h]*m2 - sx;
                            q2_p[h] = tws[h]*p2*p2 + pw[h][i1]*cp1 + pw[h][i2]*cp2;
                            q2_m[h] = tws[h]*m2*m2 + pw[h][i1]*cm1 + pw[h][i2]*cm2;
                        }
                        sumqx[0] = qx_p[0] + qx_p[1];
                        sumqx[1] = qx_p[0] + qx_m[1];
                        sumqx[2] = qx_m[0] + qx_p[1];
                        sumqx[3] = qx_m[0] + qx_m[1];
                        sumq2[0] = q2_p[0] + q2_p[1];
                        sumq2[1] = q2_p[0] + q2_m[1];
                        sumq2[2] = q2_m[0] + q2_p[1];
                        sumq2[3] = q2_m[0] + q2_m[1];
                        for (int k = 0; k < 4; ++k) {
                            if (sumq2[k] > 0 && sumqx[k]*sumqx[k] > best_score*sumq2[k]) {
                                scale = sumqx[k]/sumq2[k]; best_score = scale*sumqx[k];
                                besti1 = i1; besti2 = i2; best_k = k;
                            }
                        }
                    }
                }
            }
            if (besti1 < 0 || besti2 < 0 || best_k < 0) {
                scales[ib] = 0;
                shifts[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            for (int j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale;
                best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
            }
            bool all_on_grid = true;
            for (int k = 0; k < block_size/8; ++k) {
                if (k == 0) xx = best_k < 2 ? x_p : x_m;
                else xx = best_k%2 == 0 ? x_p : x_m;
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
                    assert(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            if (!all_on_grid) {
                float sumqx_f = 0, sumq2_f = 0;
                for (int k = 0; k < block_size/8; ++k) {
                    if (k == 0) xx = best_k < 2 ? x_p : x_m;
                    else xx = best_k%2 == 0 ? x_p : x_m;
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8*k + j];
                        float q = xx[(pg[j] - 1)/2];
                        sumqx_f += w*q*xb[8*k+j];
                        sumq2_f += w*q*q;
                    }
                }
                if (sumqx_f > 0 && sumq2_f > 0) scale = sumqx_f/sumq2_f;
            }
            y[ibl].qs[2*ib + 0] = index[0] & 255;
            y[ibl].qs[2*ib + 1] = index[1] & 255;
            y[ibl].qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
            assert(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_k;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        uint16_t * sc = (uint16_t *)y[ibl].scales;
        float d = max_scale/15;
        float id = 1/d;
        float sumqx_f = 0, sumq2_f = 0;
        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib+0]-1));
            l = MAX(0, MIN(7, l));
            sc[ib/4] |= (l << 3*(ib%4));
            y[ibl].qh[ib] |= masks[shifts[ib]];
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int k = 0; k < block_size/8; ++k) {
                if (k == 0) xx = shifts[ib] < 2 ? x_p : x_m;
                else xx = shifts[ib]%2 == 0 ? x_p : x_m;
                const int8_t * pg = (const int8_t *)(kgrid_q2xs + y[ibl].qs[2*ib+k] + ((y[ibl].qh[ib] << (8 - 4*k)) & 0x700));
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8*k + j];
                    float q = xx[(pg[j] - 1)/2]*(2*l+1);
                    sumqx_f += w*q*xb[8*k+j];
                    sumq2_f += w*q*q;
                }
            }
        }
        if (sumq2_f > 0) d = sumqx_f/sumq2_f;
        s.f16 = to_half(d*1.1125f); // 1.1125f is another fudge factor. Don't ask me why it is needed.
        sc[0] |= ((s.u16 & 0x000f) << 12);
        sc[1] |= ((s.u16 & 0x00f0) <<  8);
        sc[2] |= ((s.u16 & 0x0f00) <<  4);
        sc[3] |= ((s.u16 & 0xf000) <<  0);
    }
}

__device__ static size_t quantize_iq1_m(const tables & table, const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(n_per_row%QK_K == 0);
    float  scales[QK_K/IQ1M_BLOCK_SIZE];
    float  weight[IQ1M_BLOCK_SIZE];
    int8_t L[IQ1M_BLOCK_SIZE];
    float  pairs[2*IQ1M_BLOCK_SIZE];
    uint16_t index[IQ1M_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1M_BLOCK_SIZE];
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_iq1_m_impl(table, src, qrow, n_per_row, quant_weights, scales, weight, pairs, L, index, shifts);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq1_m);
    }
    return nrow * nblock * sizeof(block_iq1_m);
}

__device__ static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float * GGML_RESTRICT x,
        uint16_t * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(q4, 0, super_block_size/2);
    dh[0] = to_half(0.f);

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < group_max_eps) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            float al = id*xb[j];
            int l = best_index_int8(16, values, al);
            Lb[j] = l;
            float q = values[l];
            float w = weight[j];
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        d = sumq2 > 0 ? sumqx/sumq2 : 0.f;
        float best = d*sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d * sumqx;
            }
        }
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size/block_size > 1) {
        int nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
        float d = -max_scale/32;
        dh[0] = to_half(d);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            uint8_t * Lb = L + ib*block_size;
            const float * xb = x + ib*block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_int8(16, values, idl*xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = to_half(scales[0]);
        if (ntry > 0) {
            float id = scales[0] ? 1/scales[0] : 0;
            for (int j = 0; j < super_block_size; ++j) {
                L[j] = best_index_int8(16, values, id*x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

__device__ static size_t quantize_iq4_xs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    assert(n_per_row%QK_K == 0);
    int64_t nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    uint8_t L[QK_K];
    float weight[32];
    float scales[QK_K/32];
    for (int64_t row = 0; row < nrow; ++row) {
        block_iq4_xs * iq4 = (block_iq4_xs *)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = quant_weights ? quant_weights + QK_K*ibl : NULL;
            quantize_row_iq4_nl_impl(QK_K, 32, src + QK_K*ibl, &iq4[ibl].d, iq4[ibl].qs, &iq4[ibl].scales_h, iq4[ibl].scales_l,
                    scales, weight, L, kvalues_iq4nl, qw, 7);
        }
        src += n_per_row;
        qrow += nblock*sizeof(block_iq4_xs);
    }
    return nrow * nblock * sizeof(block_iq4_xs);
}

__device__ static void quantize_row_iq2_s_impl(const tables & table, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t n, const float * GGML_RESTRICT quant_weights) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif


    const uint64_t * kgrid_q2xs      = table.grid64;
    const int      * kmap_q2xs       = table.map;
    const uint16_t * kneighbors_q2xs = table.neighbours;

    assert(kmap_q2xs       && "forgot to call ggml_quantize_init()?");
    assert(kgrid_q2xs      && "forgot to call ggml_quantize_init()?");
    assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
    assert(n%QK_K == 0);

    const int kMaxQ = 3;

    const int64_t nbl = n/QK_K;

    block_iq2_s * y = (block_iq2_s *) vy;

    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_s));
        y[ibl].d = to_half(0.f);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 16*ib;
                for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 16; ++i) weight[i] = 0.25f*sigma2 + xb[i]*xb[i];
            }
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
            memset(L, 0, 16);
            if (max < group_max_eps_iq2_s) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                assert(grid_index >= 0);
                const int i8 = 2*ib + k;
                y[ibl].qs[i8] = grid_index & 255;
                y[ibl].qh[i8/4] |= ((grid_index >> 8) << 2*(i8%4));
                y[ibl].qs[QK_K/8 + i8] = block_signs[k];
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = to_half(d * 0.9875f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
    }
}

} // namespace ggml_maca_quant_ref
