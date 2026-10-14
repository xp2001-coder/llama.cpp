cmake_minimum_required(VERSION 3.18)

set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "MACA SDK root")
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The MACA bring-up backend currently requires Linux")
endif()
if (NOT EXISTS "${MACA_PATH}/include/mcr/mc_runtime.h")
    message(FATAL_ERROR "Set MACA_PATH to a complete MACA SDK installation")
endif()
if (GGML_STATIC)
    message(FATAL_ERROR "Static SDK linking is not validated for the MACA bring-up backend")
endif()
if (GGML_CUDA_FORCE_MMQ)
    message(FATAL_ERROR "MACA bring-up uses BLAS fallback; forced MMQ is not enabled")
endif()

# This is the bridge frontend target, not a MetaX hardware capability.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52)
endif()
if (NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    message(FATAL_ERROR "MACA bring-up requires bridge architecture 52; select hardware with CUCC_TARGETS")
endif()
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(MACA_CUDA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda")
file(GLOB MACA_SOURCES "${MACA_CUDA_SOURCE_DIR}/*.cu")
file(GLOB MACA_HEADERS "${MACA_CUDA_SOURCE_DIR}/*.cuh")
foreach(pattern fattn-tile fattn-mma mmq mmf)
    file(GLOB MACA_INSTANCES "${MACA_CUDA_SOURCE_DIR}/template-instances/${pattern}*.cu")
    list(APPEND MACA_SOURCES ${MACA_INSTANCES})
endforeach()
ggml_cuda_fattn_vec_instances("${MACA_CUDA_SOURCE_DIR}" MACA_INSTANCES)
list(APPEND MACA_SOURCES ${MACA_INSTANCES})

if (LLAMA_BUILD_TESTS)
    list(PREPEND MACA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/maca/backend-warp-constexpr.cu")
endif()

ggml_add_backend_library(ggml-maca ${MACA_SOURCES} ${MACA_HEADERS}
    "${MACA_CUDA_SOURCE_DIR}/vendors/maca.h" "../../include/ggml-cuda.h")
target_include_directories(ggml-maca PRIVATE
    "${MACA_PATH}/tools/cu-bridge/include" "${MACA_PATH}/include" "${MACA_CUDA_SOURCE_DIR}")
target_compile_definitions(ggml-maca PRIVATE
    GGML_USE_MACA GGML_MACA_USE_CU_BRIDGE
    GGML_CUDA_NO_VMM GGML_CUDA_NO_PEER_COPY GGML_CUDA_FORCE_CUBLAS)
if (NOT GGML_CUDA_FA)
    target_compile_definitions(ggml-maca PRIVATE GGML_CUDA_NO_FA)
endif()
target_compile_features(ggml-maca PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(ggml-maca PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-extended-lambda>")
target_link_libraries(ggml-maca PRIVATE CUDA::cudart CUDA::cublas)

message(STATUS "MACA experimental backend: SDK=${MACA_PATH}, logical warp=32, BLAS fallback")
message(STATUS "MACA generic Flash Attention candidate: GGML_CUDA_FA=${GGML_CUDA_FA}")
message(STATUS "MACA Graph, VMM, NCCL, peer copy and NVIDIA MMA paths are not enabled")
