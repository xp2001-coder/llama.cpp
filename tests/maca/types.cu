#include "probe-common.h"
#ifdef MACA_PROBE_BF16
#include <cuda_bf16.h>
#else
#include <cuda_fp16.h>
#endif

__global__ void probe_type(const float * input, float * output) {
    const int i = threadIdx.x;
#ifdef MACA_PROBE_BF16
    const auto a = __float2bfloat16(input[i]);
    const auto b = __float2bfloat16(0.5f);
    output[i] = __bfloat162float(__hadd(a, b));
#else
    const auto a = __float2half_rn(input[i]);
    const auto b = __float2half_rn(0.5f);
    output[i] = __half2float(__hadd(a, b));
#endif
}

int main(int argc, char ** argv) {
    select_device(argc, argv);
    const float input[] = {-3.5f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.25f, 8.0f};
    float output[8] = {};
    float * buffer = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), 2 * sizeof(input)));
    CHECK(cudaMemcpy(buffer, input, sizeof(input), cudaMemcpyHostToDevice));
    probe_type<<<1, 8>>>(buffer, buffer + 8);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(output, buffer + 8, sizeof(output), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(buffer));
    for (int i = 0; i < 8; ++i) {
        if (output[i] != input[i] + 0.5f) {
            std::fprintf(stderr, "Type conversion/addition mismatch at %d: %.9g\n", i, output[i]);
            return 1;
        }
    }
#ifdef MACA_PROBE_BF16
    std::puts("MACA_PROBE_PASS bf16");
#else
    std::puts("MACA_PROBE_PASS fp16");
#endif
    return 0;
}
