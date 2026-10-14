#pragma once

// Select one API provider for each translation unit.
#if defined(GGML_MACA_USE_CU_BRIDGE)
#include <cuda_runtime.h>
#if !defined(__CUDA_TO_MACA_ADAPTOR_H__)
#error "The selected CUDA runtime is not the MACA cu-bridge runtime"
#endif
#if defined(GGML_USE_MACA)
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Shared kernels use a logical subgroup, not cudaDeviceProp::warpSize.
static constexpr __host__ __device__ int ggml_maca_kernel_warp_size() {
    return 32;
}
#endif
#else
#if defined(GGML_USE_MACA)
#error "The full MACA backend requires the tested cu-bridge API provider"
#endif
#if defined(__CUDA_TO_MACA_ADAPTOR_H__) || defined(__CUDA_RUNTIME_H__)
#error "Native MACA aliases must not be mixed with CUDA compatibility headers"
#endif

#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>

// Keep the native surface limited to APIs exercised by the runtime probe.
#define cudaError_t                 mcError_t
#define cudaSuccess                 mcSuccess
#define cudaGetErrorString          mcGetErrorString
#define cudaGetDeviceCount          mcGetDeviceCount
#define cudaGetDevice               mcGetDevice
#define cudaSetDevice               mcSetDevice
#define cudaMalloc                  mcMalloc
#define cudaFree                    mcFree
#define cudaMallocHost              mcMallocHost
#define cudaFreeHost                mcFreeHost
#define cudaMemcpyAsync             mcMemcpyAsync
#define cudaMemcpyHostToDevice      mcMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      mcMemcpyDeviceToHost
#define cudaMemsetAsync             mcMemsetAsync
#define cudaStream_t                mcStream_t
#define cudaStreamCreateWithFlags   mcStreamCreateWithFlags
#define cudaStreamNonBlocking       mcStreamNonBlocking
#define cudaStreamDestroy           mcStreamDestroy
#define cudaStreamSynchronize       mcStreamSynchronize
#define cudaStreamWaitEvent         mcStreamWaitEvent
#define cudaEvent_t                 mcEvent_t
#define cudaEventCreateWithFlags    mcEventCreateWithFlags
#define cudaEventDisableTiming      mcEventDisableTiming
#define cudaEventRecord             mcEventRecord
#define cudaEventSynchronize        mcEventSynchronize
#define cudaEventDestroy            mcEventDestroy
#endif
