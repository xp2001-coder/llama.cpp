#pragma once

#include "copy.cuh"
#include <cmath>

// Use FP32 BLAS only when the shared attention kernels have no matching instance.
static bool ggml_maca_fattn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (dst->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        !ggml_maca_copy_layout(dst) || !ggml_maca_can_copy_f32(q) ||
        !ggml_maca_can_copy_f32(k) || !ggml_maca_can_copy_f32(v)) {
        return false;
    }
    if (q->ne[0] != k->ne[0] || k->ne[1] != v->ne[1] ||
        q->ne[2] % k->ne[2] || q->ne[2] % v->ne[2] ||
        q->ne[3] % k->ne[3] || q->ne[3] % v->ne[3] ||
        dst->ne[0] != v->ne[0] || dst->ne[1] != q->ne[2] ||
        dst->ne[2] != q->ne[1] || dst->ne[3] != q->ne[3] ||
        dst->nb[2] / sizeof(float) > INT_MAX) {
        return false;
    }
    // Limit a score row to the 64 MiB scratch budget before computing products.
    if (k->ne[1] > (1 << 24)) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || !ggml_maca_copy_layout(mask) ||
        mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1] ||
        q->ne[2] % mask->ne[2] || q->ne[3] % mask->ne[3])) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) ||
        !ggml_maca_copy_layout(sinks) || ggml_nelements(sinks) != q->ne[2])) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(ggml_get_op_params_f32(dst, i))) {
            return false;
        }
    }
    return true;
}

static __global__ void ggml_maca_fattn_softmax(float * scores, const char * mask,
        const float * sinks, int nk, int query_start, int head, size_t mask_stride,
        float scale, float softcap, float slope) {
    const int lane = threadIdx.x;
    float * row = scores + int64_t(blockIdx.x) * nk;
    const half * mask_row = mask ? (const half *) (mask + (query_start + blockIdx.x) * mask_stride) : nullptr;
    __shared__ float scratch[256];
    float maximum = -INFINITY;
    for (int i = lane; i < nk; i += blockDim.x) {
        float value = row[i] * scale;
        if (softcap != 0.0f) {
            value = softcap * tanhf(value / softcap);
        }
        if (mask_row) {
            value += slope * __half2float(mask_row[i]);
        }
        row[i] = value;
        maximum = fmaxf(maximum, value);
    }
    if (lane == 0 && sinks) {
        maximum = fmaxf(maximum, sinks[head]);
    }
    scratch[lane] = maximum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (lane < stride) {
            scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        }
        __syncthreads();
    }
    maximum = scratch[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = lane; i < nk; i += blockDim.x) {
        const float value = maximum == -INFINITY ? 0.0f : expf(row[i] - maximum);
        row[i] = value;
        sum += value;
    }
    if (lane == 0 && sinks && maximum != -INFINITY) {
        sum += expf(sinks[head] - maximum);
    }
    scratch[lane] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    const float inverse = scratch[0] > 0.0f ? 1.0f / scratch[0] : 0.0f;
    for (int i = lane; i < nk; i += blockDim.x) {
        row[i] *= inverse;
    }
}

static void ggml_maca_fattn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_maca_fattn_supported(dst));
    ggml_cuda_pool_alloc<char> q_storage(ctx.pool()), k_storage(ctx.pool()), v_storage(ctx.pool());
    const ggml_tensor q = ggml_maca_copy_f32(ctx, dst->src[0], q_storage);
    const ggml_tensor k = ggml_maca_copy_f32(ctx, dst->src[1], k_storage);
    const ggml_tensor v = ggml_maca_copy_f32(ctx, dst->src[2], v_storage);
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const int dk = q.ne[0], dv = v.ne[0], nk = k.ne[1];
    const int chunk = std::min<int64_t>(q.ne[1], (1 << 24) / nk);
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), size_t(nk) * chunk);
    const float scale = ggml_get_op_params_f32(dst, 0);
    const float bias = ggml_get_op_params_f32(dst, 1);
    const float softcap = ggml_get_op_params_f32(dst, 2);
    int head_power = 1;
    while (head_power <= q.ne[2] / 2) {
        head_power *= 2;
    }
    const float m0 = powf(2.0f, -bias / head_power);
    const float m1 = powf(2.0f, -bias / (2.0f * head_power));
    const float one = 1.0f, zero = 0.0f;
    const cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, ctx.stream()));
    for (int64_t batch = 0; batch < q.ne[3]; ++batch) {
        for (int64_t head = 0; head < q.ne[2]; ++head) {
            const float * kp = (const float *) ((const char *) k.data +
                head / (q.ne[2] / k.ne[2]) * k.nb[2] + batch / (q.ne[3] / k.ne[3]) * k.nb[3]);
            const float * vp = (const float *) ((const char *) v.data +
                head / (q.ne[2] / v.ne[2]) * v.nb[2] + batch / (q.ne[3] / v.ne[3]) * v.nb[3]);
            const char * mp = mask ? (const char *) mask->data +
                (head % mask->ne[2]) * mask->nb[2] + (batch % mask->ne[3]) * mask->nb[3] : nullptr;
            const float slope = bias > 0.0f ? (head < head_power ? powf(m0, head + 1) : powf(m1, 2 * (head - head_power) + 1)) : 1.0f;
            for (int64_t query = 0; query < q.ne[1]; query += chunk) {
                const int count = std::min<int64_t>(chunk, q.ne[1] - query);
                const float * qp = (const float *) ((const char *) q.data + query * q.nb[1] + head * q.nb[2] + batch * q.nb[3]);
                float * out = (float *) ((char *) dst->data + head * dst->nb[1] + query * dst->nb[2] + batch * dst->nb[3]);
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nk, count, dk,
                    &one, kp, dk, qp, dk, &zero, scores.get(), nk));
                ggml_maca_fattn_softmax<<<count, 256, 0, ctx.stream()>>>(scores.get(), mp,
                    sinks ? (const float *) sinks->data : nullptr, nk, query, head,
                    mask ? mask->nb[1] : 0, scale, softcap, slope);
                CUDA_CHECK(cudaGetLastError());
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dv, count, nk,
                    &one, vp, dv, scores.get(), nk, &zero, out, dst->nb[2] / sizeof(float)));
            }
        }
    }
}
