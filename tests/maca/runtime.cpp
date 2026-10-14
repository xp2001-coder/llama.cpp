#include "probe-common.h"

#include <cstring>

int main(int argc, char ** argv) {
    const int device = select_device(argc, argv);
    int selected = -1;
    CHECK(cudaGetDevice(&selected));
    if (selected != device) {
        return 1;
    }

    constexpr size_t n = 4096;
    constexpr size_t bytes = n * sizeof(int);
    int * input = nullptr;
    int * output = nullptr;
    int * buffer = nullptr;
    CHECK(cudaMallocHost(reinterpret_cast<void **>(&input), bytes));
    CHECK(cudaMallocHost(reinterpret_cast<void **>(&output), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), bytes));
    cudaStream_t producer;
    cudaStream_t consumer;
    cudaEvent_t ready;
    CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));

    for (int iteration = 0; iteration < 17; ++iteration) {
        for (size_t i = 0; i < n; ++i) {
            input[i] = iteration ? static_cast<int>(i) * 13 - iteration * 41 : 0;
            output[i] = -1;
        }
        CHECK(cudaMemsetAsync(buffer, 0, bytes, producer));
        if (iteration) {
            CHECK(cudaMemcpyAsync(buffer, input, bytes, cudaMemcpyHostToDevice, producer));
        }
        CHECK(cudaEventRecord(ready, producer));
        CHECK(cudaStreamWaitEvent(consumer, ready, 0));
        CHECK(cudaMemcpyAsync(output, buffer, bytes, cudaMemcpyDeviceToHost, consumer));
        CHECK(cudaStreamSynchronize(consumer));
        if (std::memcmp(input, output, bytes) != 0) {
            std::fprintf(stderr, "Copy/order mismatch at iteration %d\n", iteration);
            return 1;
        }
    }
    CHECK(cudaEventSynchronize(ready));
    CHECK(cudaEventDestroy(ready));
    CHECK(cudaStreamDestroy(consumer));
    CHECK(cudaStreamDestroy(producer));
    CHECK(cudaFree(buffer));
    CHECK(cudaFreeHost(output));
    CHECK(cudaFreeHost(input));
#ifdef MACA_PROBE_BRIDGE_RUNTIME
    std::puts("MACA_PROBE_PASS bridge-runtime");
#else
    std::puts("MACA_PROBE_PASS runtime");
#endif
    return 0;
}
