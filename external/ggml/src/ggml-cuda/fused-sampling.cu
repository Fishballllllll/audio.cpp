#include "fused-sampling.cuh"

#include <cfloat>

// Same expressions as the binbcast SUB/ADD kernels (a - b, a + b) and scale_f32
// (scale * x + bias).
template <bool u_first>
static __global__ void cfg_scale_fused_kernel(const float * c, const float * u, float * dst, const float s1, const float b1,
                                              const float s2, const float b2, const int64_t k) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float d = c[i] - u[i];
    const float e = s1 * d + b1;
    const float f = u_first ? u[i] + e : e + u[i];
    dst[i] = s2 * f + b2;
}

void ggml_cuda_op_cfg_scale_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * cond, const ggml_tensor * uncond,
                                  float s1, float b1, bool u_first, float s2, float b2, ggml_tensor * dst) {
    GGML_ASSERT(cond->type == GGML_TYPE_F32 && uncond->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(cond) && ggml_is_contiguous(uncond) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(cond, uncond) && ggml_are_same_shape(cond, dst));

    const int64_t k = ggml_nelements(dst);
    const int block = 256;
    const int64_t blocks = (k + block - 1) / block;
    if (u_first) {
        cfg_scale_fused_kernel<true><<<(unsigned int) blocks, block, 0, ctx.stream()>>>(
            (const float *) cond->data, (const float *) uncond->data, (float *) dst->data, s1, b1, s2, b2, k);
    } else {
        cfg_scale_fused_kernel<false><<<(unsigned int) blocks, block, 0, ctx.stream()>>>(
            (const float *) cond->data, (const float *) uncond->data, (float *) dst->data, s1, b1, s2, b2, k);
    }
}

// argmax_f32 (argmax.cu) over values computed on the fly; launched with the thread
// count ggml_cuda_argmax would use for `candidates` columns.
template <bool values_first>
static __global__ void pick_top_candidate_kernel(const char * __restrict__ scores, const size_t scores_stride,
                                                 const char * __restrict__ noise, const size_t noise_stride,
                                                 const int32_t * __restrict__ order, int32_t * __restrict__ token,
                                                 const int64_t ncols) {
    float maxval = -FLT_MAX;
    int   argmax = -1;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const int32_t id = order[col];
        const float v = *(const float *) (scores + id*scores_stride);
        const float n = *(const float *) (noise  + id*noise_stride);
        const float val = values_first ? v + n : n + v;
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        *token = order[argmax];
    }
}

void ggml_cuda_op_pick_top_candidate(ggml_backend_cuda_context & ctx, const ggml_tensor * scores, const ggml_tensor * noise,
                                     const int32_t * order, int64_t candidates, bool values_first, ggml_tensor * token) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && noise->type == GGML_TYPE_F32 && token->type == GGML_TYPE_I32);
    GGML_ASSERT(scores->ne[0] == 1 && noise->ne[0] == 1 && candidates > 0 && candidates <= scores->ne[1]);

    const int64_t num_threads = std::min<int64_t>(1024, (candidates + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    if (values_first) {
        pick_top_candidate_kernel<true><<<1, (unsigned int) num_threads, 0, ctx.stream()>>>(
            (const char *) scores->data, scores->nb[1], (const char *) noise->data, noise->nb[1], order,
            (int32_t *) token->data, candidates);
    } else {
        pick_top_candidate_kernel<false><<<1, (unsigned int) num_threads, 0, ctx.stream()>>>(
            (const char *) scores->data, scores->nb[1], (const char *) noise->data, noise->nb[1], order,
            (int32_t *) token->data, candidates);
    }
}
