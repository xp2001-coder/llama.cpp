#include "probe-common.h"
#include <cublas_v2.h>

#include <cmath>

#define CHECK_BLAS(call) do { \
    const auto status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s: status=%d\n", #call, static_cast<int>(status)); \
        std::exit(1); \
    } \
} while (0)

int main(int argc, char ** argv) {
    select_device(argc, argv);
    constexpr int m = 3, n = 2, k = 4;
    const float a[m * k] = {1, 2, 3, 4, 5, 6, -1, -2, -3, 0.5f, 1, 1.5f};
    const float b[k * n] = {1, 0, 2, -1, 0.5f, -2, 1, 3};
    float c[m * n] = {};
    float * buffer = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), sizeof(a) + sizeof(b) + sizeof(c)));
    float * da = buffer;
    float * db = da + m * k;
    float * dc = db + k * n;
    CHECK(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dc, 0, sizeof(c)));
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle;
    CHECK_BLAS(cublasCreate(&handle));
    CHECK_BLAS(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, da, m, db, k, &beta, dc, m));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
    CHECK_BLAS(cublasDestroy(handle));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(buffer));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float expected = 0.0f;
            for (int p = 0; p < k; ++p) {
                expected += a[i + p * m] * b[p + j * k];
            }
            if (!std::isfinite(c[i + j * m]) || std::fabs(c[i + j * m] - expected) > 1e-5f) {
                std::fprintf(stderr, "SGEMM mismatch at [%d,%d]: %.9g != %.9g\n", i, j, c[i + j * m], expected);
                return 1;
            }
        }
    }
    std::puts("MACA_PROBE_PASS blas");
    return 0;
}
