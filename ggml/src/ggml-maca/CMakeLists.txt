set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "Path to the MACA SDK")

if (MACA_PATH STREQUAL "")
    if (EXISTS /opt/maca)
        set(MACA_PATH /opt/maca)
    else()
        message(FATAL_ERROR "MACA_PATH is not set and /opt/maca does not exist")
    endif()
endif()

set(MACA_CU_BRIDGE "$ENV{MACA_CU_BRIDGE}" CACHE PATH "Path to MACA cu-bridge")
if (MACA_CU_BRIDGE STREQUAL "")
    set(MACA_CU_BRIDGE "${MACA_PATH}/tools/cu-bridge")
endif()
if (NOT EXISTS "${MACA_CU_BRIDGE}")
    message(FATAL_ERROR "MACA cu-bridge was not found at ${MACA_CU_BRIDGE}")
endif()

find_package(CUDAToolkit)
if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "MACA CUDA-compatible toolkit was not found")
endif()

enable_language(CUDA)

file(GLOB GGML_HEADERS_MACA "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_MACA
    "../../include/ggml-cuda.h"
    "../ggml-cuda/vendors/maca.h"
    "maca.cuh"
    "maca-policy.cuh"
)

file(GLOB GGML_SOURCES_MACA "../ggml-cuda/*.cu")
file(GLOB SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_MACA ${SRCS})
file(GLOB SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_MACA ${SRCS})
file(GLOB SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_MACA ${SRCS})
file(GLOB SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_MACA ${SRCS})

ggml_cuda_fattn_vec_instances("${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda" SRCS)
list(APPEND GGML_SOURCES_MACA ${SRCS})

ggml_add_backend_library(ggml-maca
    ${GGML_HEADERS_MACA}
    ${GGML_SOURCES_MACA}
)

target_compile_definitions(ggml-maca PRIVATE GGML_USE_MACA)

add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
add_compile_definitions(GGML_CUDA_NO_FA)
add_compile_definitions(GGML_CUDA_NO_VMM)
add_compile_definitions(GGML_CUDA_NO_PEER_COPY)

if (GGML_MACA_GRAPHS)
    add_compile_definitions(GGML_CUDA_USE_GRAPHS)
endif()

if (GGML_STATIC)
    target_link_libraries(ggml-maca PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
else()
    target_link_libraries(ggml-maca PRIVATE CUDA::cudart CUDA::cublas)
endif()

set(MACA_CUDA_FLAGS -use_fast_math -extended-lambda)
if (NOT MSVC)
    list(APPEND MACA_CUDA_FLAGS -Xcompiler -Wno-pedantic)
endif()
target_compile_options(ggml-maca PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${MACA_CUDA_FLAGS}>")

if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()
