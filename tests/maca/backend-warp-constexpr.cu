#include "mma.cuh"

// Compile the real shared header in both bridge passes; no GPU execution.
static_assert(ggml_cuda_get_physical_warp_size() == ggml_maca_kernel_warp_size(), "MACA warp contract mismatch");

struct maca_warp_layout {
    int lanes[ggml_cuda_get_physical_warp_size()];
};
static_assert(sizeof(maca_warp_layout) == sizeof(int) * ggml_maca_kernel_warp_size(), "Invalid host array bound");

using maca_mma_layout = ggml_cuda_mma::tile<16, 16, half2, ggml_cuda_mma::DATA_LAYOUT_I_MAJOR_SCRAMBLED>;
static_assert(maca_mma_layout::ne == 16 * 16 / ggml_maca_kernel_warp_size(), "Invalid MMA tile array bound");
static_assert(sizeof(maca_mma_layout) == sizeof(half2) * maca_mma_layout::ne, "Invalid MMA tile layout");
