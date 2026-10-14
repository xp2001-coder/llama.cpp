#include "probe-common.h"

#include <vector>

struct result {
    int broadcast;
    int exchanged;
    int sum;
    unsigned int ballot;
    int warp_memory;
    int block_memory;
    int device_warp_size;
};

// Test a 32-lane logical contract without changing the reported warp size.
__global__ void probe_warp(result * output) {
    __shared__ int scratch[256];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int group = tid - lane;
    const int value = 1000 * blockIdx.x + tid;
    result row;
    row.broadcast = __shfl_sync(0xffffffffu, value, 11, 32);
    row.exchanged = __shfl_xor_sync(0xffffffffu, value, 16, 32);
    int sum = value;
    for (int offset = 1; offset < 32; offset *= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
    }
    row.sum = sum;
    row.ballot = __ballot_sync(0xffffffffu, lane % 2 == 0);
    scratch[tid] = value;
    __syncwarp(0xffffffffu);
    row.warp_memory = scratch[group + (lane ^ 1)];
    __syncthreads();
    row.block_memory = scratch[(tid + 33) % blockDim.x];
    row.device_warp_size = warpSize;
    output[blockIdx.x * blockDim.x + tid] = row;
}

int main(int argc, char ** argv) {
    const int device = select_device(argc, argv);
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("device=%s reported_warp_size=%d\n", properties.name, properties.warpSize);
    int failures = 0;
    for (const int threads : {32, 64, 128, 256}) {
        std::vector<result> actual(3 * threads);
        result * buffer = nullptr;
        CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), actual.size() * sizeof(result)));
        CHECK(cudaMemset(buffer, 0, actual.size() * sizeof(result)));
        probe_warp<<<3, threads>>>(buffer);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(actual.data(), buffer, actual.size() * sizeof(result), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(buffer));
        for (int block = 0; block < 3; ++block) {
            for (int tid = 0; tid < threads; ++tid) {
                const int group = tid - tid % 32;
                const int start = 1000 * block + group;
                const result & row = actual[block * threads + tid];
                if (row.broadcast != start + 11 || row.exchanged != start + ((tid % 32) ^ 16) ||
                    row.sum != 32 * start + 496 || row.ballot != 0x55555555u ||
                    row.warp_memory != start + ((tid % 32) ^ 1) ||
                    row.block_memory != 1000 * block + (tid + 33) % threads || row.device_warp_size <= 0) {
                    if (failures < 8) {
                        std::fprintf(stderr, "threads=%d block=%d tid=%d broadcast=%d xor=%d sum=%d ballot=%x warp_memory=%d block_memory=%d\n",
                                     threads, block, tid, row.broadcast, row.exchanged, row.sum, row.ballot, row.warp_memory, row.block_memory);
                    }
                    ++failures;
                }
            }
        }
        std::printf("threads=%d device_warp_size=%d cumulative_failures=%d\n", threads, actual[0].device_warp_size, failures);
    }
    if (failures) {
        return 1;
    }
    std::puts("MACA_PROBE_PASS warp");
    return 0;
}
