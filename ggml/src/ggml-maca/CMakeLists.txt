set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "Path to the MACA SDK")

if (MACA_PATH STREQUAL "")
    if (EXISTS /opt/maca)
        set(MACA_PATH /opt/maca)
    else()
        message(FATAL_ERROR "MACA_PATH is not set and /opt/maca does not exist")
    endif()
endif()

set(MACA_CU_BRIDGE "${MACA_PATH}/tools/cu-bridge")
if (NOT EXISTS "${MACA_CU_BRIDGE}")
    message(FATAL_ERROR "MACA cu-bridge was not found at ${MACA_CU_BRIDGE}")
endif()

# Reuse ggml-cuda sources with a distinct MACA backend identity.
set(GGML_CUDA_BACKEND_TARGET ggml-maca)
set(GGML_CUDA_VENDOR_MACA ON)

# Keep unvalidated features on conservative fallbacks.
set(GGML_CUDA_FA OFF)
set(GGML_CUDA_NCCL OFF)
set(GGML_CUDA_NO_VMM ON)
set(GGML_CUDA_NO_PEER_COPY ON)
set(GGML_CUDA_FORCE_CUBLAS ON)
set(GGML_CUDA_GRAPHS ${GGML_MACA_GRAPHS})

add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../ggml-cuda"
    "${CMAKE_CURRENT_BINARY_DIR}/cuda-sources"
)

# Register the shared CUDA-family ABI for static builds.
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()
