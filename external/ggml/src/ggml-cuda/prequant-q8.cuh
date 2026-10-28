#pragma once

#include "common.cuh"

#include <memory>

// q8_1 rows written by the kernel that produces a mul_mat_vec_q input (a norm,
// SwiGLU, a rounding), so mmvq can skip its separate quantize_q8_1 pass.
//
// Stores what quantize_q8_1 stores for value xi at linear index i_cont. Like that
// kernel, the 32 lanes of the warp must hold the 32 consecutive values of one
// block (lane == i_cont % QK8_1) and all of them must be active.
static __device__ __forceinline__ void prequant_q8_1_store(const float xi, block_q8_1 * y, const int64_t i_cont) {
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;

    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

// Host hand-off from the producer to the mmvq that reads its output. The rows come
// from the pool, which may free strictly last-in first-out: clear the slot before
// allocating new rows, and take or clear it before mmvq allocates anything.
struct ggml_cuda_prequant_q8_1 {
    const ggml_backend_cuda_context * ctx = nullptr;
    const ggml_tensor * weights = nullptr;  // src0 of the consuming MUL_MAT
    const void * src1 = nullptr;            // the f32 data the rows were made from
    std::unique_ptr<ggml_cuda_pool_alloc<char>> rows;
};

inline ggml_cuda_prequant_q8_1 & ggml_cuda_prequant_q8_1_slot() {
    thread_local ggml_cuda_prequant_q8_1 slot;
    return slot;
}
