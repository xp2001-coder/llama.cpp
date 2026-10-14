#pragma once

#include "maca.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

static int select_device(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s DEVICE_INDEX\n", argv[0]);
        std::exit(2);
    }
    char * end = nullptr;
    errno = 0;
    const long device = std::strtol(argv[1], &end, 10);
    const bool invalid_number = errno != 0 || !argv[1][0] || *end || device < 0;
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (invalid_number || device >= count) {
        std::fprintf(stderr, "Invalid device index; visible devices: %d\n", count);
        std::exit(2);
    }
    CHECK(cudaSetDevice(static_cast<int>(device)));
    return static_cast<int>(device);
}
