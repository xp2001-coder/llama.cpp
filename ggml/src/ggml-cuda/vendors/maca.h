#pragma once

// cu-bridge provides the CUDA-compatible headers and CUDA-to-MACA ABI mappings.
// Direct API remapping here would bypass its wrappers and can cause ABI mismatches.

#if defined(__has_include)
#  if !__has_include(<bridge/runtime/cuda_to_maca_mcr_adaptor.h>)
#    error "MACA cu-bridge headers were not found; configure with MACA tools/cu-bridge"
#  endif
#  if !__has_include(<bridge/blas/cublas_v2_wrapper.h>)
#    error "MACA cu-bridge mcBLAS wrapper was not found"
#  endif
#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <common/__clang_maca_mckl_lib_header.h>

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL

// Match the CUDA vendor compatibility aliases for toolkits older than 11.2.
#if defined(CUDART_VERSION) && CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED \
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020

// VMM, peer copy, NCCL, and Flash Attention remain disabled pending validation.
// Graph support is experimental and disabled by default.

// cu-bridge supplies the CUDA-compatible FP16 and BF16 type aliases.
