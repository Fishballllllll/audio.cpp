#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

// Optional bf16 rounding of the stored value (ROPE -> ROUND_BF16 fusion).
template <bool round_bf16>
static __device__ __forceinline__ float rope_round(float v) {
    if constexpr (round_bf16) {
        return __bfloat162float(__float2bfloat16(v));
    }
    return v;
}

template <bool forward, bool has_ff, typename T, typename D, bool packed_rows = false, bool round_bf16 = false>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride) {
    const int i0 = packed_rows ? 2*threadIdx.x : 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = packed_rows ? blockDim.y*blockIdx.x + threadIdx.y : blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        x0 = rope_round<round_bf16>(x0);
        x1 = rope_round<round_bf16>(x1);
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 >= n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

// round_in: the input is rounded to bf16 as it is read (ROUND_BF16 -> ROPE fusion);
// the arithmetic on the rounded values is unchanged.
// One NEOX rope work item (row row_dst, dims i0 and i0 + n_dims/2); shared by rope_neox and
// rope_neox_qkv so both evaluate the same expressions.
template <bool forward, bool has_ff, typename T, typename D, bool round_bf16 = false, bool round_in = false>
static __device__ __forceinline__ void rope_neox_row(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                                     const int            row_dst,
                                                     const int            i0) {
    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 >= n_dims) {
        if constexpr (round_bf16 || round_in) {
            dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(rope_round<true>(x[ix + i0 / 2 + 0]));
            dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(rope_round<true>(x[ix + i0 / 2 + 1]));
        } else {
            dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
            dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);
        }

        return;
    }

    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = rope_round<round_in>(x[ix + 0]);
    const float x1 = rope_round<round_in>(x[ix + n_dims/2]);

    dst[idst + 0]          = ggml_cuda_cast<D>(rope_round<round_bf16>(x0 * cos_theta - x1 * sin_theta));
    dst[idst + n_dims / 2] = ggml_cuda_cast<D>(rope_round<round_bf16>(x0 * sin_theta + x1 * cos_theta));
}

template <bool forward, bool has_ff, typename T, typename D, bool round_bf16 = false, bool round_in = false>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    rope_neox_row<forward, has_ff, T, D, round_bf16, round_in>(x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims,
                                                                pos, freq_scale, ext_factor, attn_factor, corr_dims,
                                                                theta_scale, freq_factors, row_indices, set_rows_stride,
                                                                row_dst, i0);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T *            x,
                                  T *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    if (i0 >= n_dims) {
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, i0 / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, i0 / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, i0 / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D, bool round_bf16 = false>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // One warp per 64-wide head, eight rows per block. The generic launch
    // leaves seven of its eight warps idle for these narrow heads. Keep the
    // same scalar arithmetic and require full blocks to avoid tail reads.
    if constexpr (std::is_same_v<T, float> && std::is_same_v<D, float>) {
        if (ne00 == 64 && nr % 8 == 0 && freq_factors == nullptr && set_rows_stride == 0) {
            rope_norm<forward, false, T, D, true, round_bf16><<<nr / 8, dim3(32, 8), 0, stream>>>(
                x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
                attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
            return;
        }
    }
#endif

    if (freq_factors == nullptr) {
        rope_norm<forward, false, T, D, false, round_bf16><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    } else {
        rope_norm<forward, true, T, D, false, round_bf16><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    }
}

template <bool forward, typename T, typename D, bool round_bf16 = false, bool round_in = false>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_neox<forward, false, T, D, round_bf16, round_in><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    } else {
        rope_neox<forward, true, T, D, round_bf16, round_in><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T *            x,
                            T *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_multi<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    } else {
        rope_multi<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr,
                            const ggml_tensor *         round_bf16 = nullptr,
                            const ggml_tensor *         src0_in = nullptr,
                            const bool                  round_in = false) {
    // src0_in: read the input from here instead of dst->src[0] (same shape, own strides).
    const ggml_tensor * src0 = src0_in != nullptr ? src0_in : dst->src[0];
    GGML_ASSERT(src0_in == nullptr || ggml_are_same_shape(src0_in, dst->src[0]));
    GGML_ASSERT(!round_in || (round_bf16 != nullptr && src0->type == GGML_TYPE_F32));
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    // ROPE -> ROUND_BF16: write the rounded values straight into the ROUND node.
    const bool do_round = round_bf16 != nullptr;
    if (do_round) {
        GGML_ASSERT(forward && set_rows == nullptr);
        // An f16 output (the rounded value cast to f16) only comes from the round_in fusion.
        GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_are_same_shape(dst, round_bf16) &&
                    (round_bf16->type == GGML_TYPE_F32 || (round_bf16->type == GGML_TYPE_F16 && round_in)));
        dst_d = round_bf16->data;
    }
    const ggml_tensor * out = do_round ? round_bf16 : dst;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = out->nb[1] / ggml_type_size(out->type);
    const size_t s2 = out->nb[2] / ggml_type_size(out->type);
    const size_t s3 = out->nb[3] / ggml_type_size(out->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    mrope_sections sections;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            if (do_round && round_in && out->type == GGML_TYPE_F16) {
                rope_neox_cuda<forward, float, half, true, true>((const float *) src0_d, (half *) dst_d, ne00, ne01,
                                                                 ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos,
                                                                 freq_scale, freq_base, ext_factor, attn_factor,
                                                                 corr_dims, freq_factors, row_indices,
                                                                 set_rows_stride, stream);
            } else if (do_round && round_in) {
                rope_neox_cuda<forward, float, float, true, true>((const float *) src0_d, (float *) dst_d, ne00, ne01,
                                                                  ne02, s01, s02, s03, s1, s2, s3, n_dims, nr, pos,
                                                                  freq_scale, freq_base, ext_factor, attn_factor,
                                                                  corr_dims, freq_factors, row_indices,
                                                                  set_rows_stride, stream);
            } else if (do_round) {
                rope_neox_cuda<forward, float, float, true>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02,
                                                            s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                                                            freq_base, ext_factor, attn_factor, corr_dims,
                                                            freq_factors, row_indices, set_rows_stride, stream);
            } else {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, stream);
            }
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        GGML_ASSERT(!do_round);
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        GGML_ASSERT(!do_round);
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            if (do_round) {
                rope_norm_cuda<forward, float, float, true>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02,
                                                            s01, s02, s03, s1, s2, s3, n_dims, nr, pos, freq_scale,
                                                            freq_base, ext_factor, attn_factor, corr_dims,
                                                            freq_factors, row_indices, set_rows_stride, stream);
            } else {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, stream);
            }
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

void ggml_cuda_op_rope_round_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * round_bf16) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, nullptr, round_bf16);
}

void ggml_cuda_op_round_rope_round_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * rope, const ggml_tensor * src_unrounded,
                                        const ggml_tensor * out) {
    const int mode = ((const int32_t *) rope->op_params)[2];
    GGML_ASSERT(mode == GGML_ROPE_TYPE_NEOX);
    ggml_cuda_op_rope_impl<true>(ctx, rope, nullptr, out, src_unrounded, true);
}

// Arguments of one rope_neox launch, derived as in ggml_cuda_op_rope_impl.
struct rope_neox_launch {
    const float *  x;
    void *         dst;
    int            ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, nr;
    const int32_t * pos;
    float          freq_scale, ext_factor, attn_factor, theta_scale;
    rope_corr_dims corr_dims;
    const float *  freq_factors;
};

static rope_neox_launch rope_neox_launch_for(const ggml_tensor * rope, const ggml_tensor * src0, const ggml_tensor * out) {
    rope_neox_launch l;
    l.x    = (const float *) src0->data;
    l.dst  = out->data;
    l.ne00 = src0->ne[0];
    l.ne01 = src0->ne[1];
    l.ne02 = src0->ne[2];
    l.nr   = ggml_nrows(src0);
    l.s01  = src0->nb[1] / ggml_type_size(src0->type);
    l.s02  = src0->nb[2] / ggml_type_size(src0->type);
    l.s03  = src0->nb[3] / ggml_type_size(src0->type);
    l.s1   = out->nb[1] / ggml_type_size(out->type);
    l.s2   = out->nb[2] / ggml_type_size(out->type);
    l.s3   = out->nb[3] / ggml_type_size(out->type);
    l.n_dims = ((const int32_t *) rope->op_params)[1];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];
    float freq_base;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base,     (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&l.freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&l.ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&l.attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,     (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,     (const int32_t *) rope->op_params + 10, sizeof(float));
    ggml_rope_yarn_corr_dims(l.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, l.corr_dims.v);
    l.theta_scale  = powf(freq_base, -2.0f / l.n_dims);
    l.pos          = (const int32_t *) rope->src[1]->data;
    l.freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;
    return l;
}

// Blocks [0, q.nr) rope q rows, [q.nr, q.nr + k.nr) rope k rows (to f16), the rest round v to f16.
template <bool has_ff>
static __global__ void rope_neox_qkv(const rope_neox_launch q, const rope_neox_launch k,
                                     const char * v_x, half * v_dst, const int64_t v_k,
                                     const int64_t v_ne0, const int64_t v_ne1, const int64_t v_ne2,
                                     const int64_t v_nb01, const int64_t v_nb02, const int64_t v_nb03) {
    const int i0 = 2*threadIdx.y;
    if (blockIdx.x < q.nr) {
        if (i0 < q.ne00) {
            rope_neox_row<true, has_ff, float, float, true, true>(
                q.x, (float *) q.dst, q.ne00, q.ne01, q.ne02, q.s01, q.s02, q.s03, q.s1, q.s2, q.s3, q.n_dims, q.pos,
                q.freq_scale, q.ext_factor, q.attn_factor, q.corr_dims, q.theta_scale, q.freq_factors, nullptr, 0,
                blockIdx.x, i0);
        }
        return;
    }
    if (blockIdx.x < q.nr + k.nr) {
        if (i0 < k.ne00) {
            rope_neox_row<true, has_ff, float, half, true, true>(
                k.x, (half *) k.dst, k.ne00, k.ne01, k.ne02, k.s01, k.s02, k.s03, k.s1, k.s2, k.s3, k.n_dims, k.pos,
                k.freq_scale, k.ext_factor, k.attn_factor, k.corr_dims, k.theta_scale, k.freq_factors, nullptr, 0,
                blockIdx.x - q.nr, i0);
        }
        return;
    }
    const int64_t i = (int64_t) (blockIdx.x - q.nr - k.nr)*blockDim.y + threadIdx.y;
    if (i >= v_k) {
        return;
    }
    const int64_t j0 = i % v_ne0;
    const int64_t j1 = (i / v_ne0) % v_ne1;
    const int64_t j2 = (i / (v_ne0*v_ne1)) % v_ne2;
    const int64_t j3 = i / (v_ne0*v_ne1*v_ne2);
    const float * x = (const float *) (v_x + j1*v_nb01 + j2*v_nb02 + j3*v_nb03);
    v_dst[i] = __float2half(rope_round<true>(x[j0]));
}

void ggml_cuda_op_round_rope_qkv(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor * q_rope, const ggml_tensor * q_src, const ggml_tensor * q_out,
                                 const ggml_tensor * k_rope, const ggml_tensor * k_src, const ggml_tensor * k_out,
                                 const ggml_tensor * v_x, const ggml_tensor * v_dst) {
    for (const ggml_tensor * rope : { q_rope, k_rope }) {
        GGML_ASSERT(((const int32_t *) rope->op_params)[2] == GGML_ROPE_TYPE_NEOX);
    }
    GGML_ASSERT(q_src->type == GGML_TYPE_F32 && k_src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(q_src, q_rope->src[0]) && ggml_are_same_shape(k_src, k_rope->src[0]));
    GGML_ASSERT(q_out->type == GGML_TYPE_F32 && k_out->type == GGML_TYPE_F16);
    GGML_ASSERT(q_src->ne[0] % 2 == 0 && q_src->ne[0] <= 2*CUDA_ROPE_BLOCK_SIZE);
    GGML_ASSERT(k_src->ne[0] % 2 == 0 && k_src->ne[0] <= 2*CUDA_ROPE_BLOCK_SIZE);
    GGML_ASSERT((q_rope->src[2] == nullptr) == (k_rope->src[2] == nullptr));
    GGML_ASSERT(v_x->type == GGML_TYPE_F32 && v_x->nb[0] == sizeof(float));
    GGML_ASSERT(v_dst->type == GGML_TYPE_F16 && ggml_is_contiguous(v_dst) && ggml_nelements(v_dst) == ggml_nelements(v_x));

    const rope_neox_launch q = rope_neox_launch_for(q_rope, q_src, q_out);
    const rope_neox_launch k = rope_neox_launch_for(k_rope, k_src, k_out);
    const int64_t v_k      = ggml_nelements(v_dst);
    const int64_t v_blocks = (v_k + CUDA_ROPE_BLOCK_SIZE - 1) / CUDA_ROPE_BLOCK_SIZE;
    GGML_ASSERT(q.nr + k.nr + v_blocks < INT_MAX);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const dim3 block_nums((unsigned int) (q.nr + k.nr + v_blocks), 1, 1);
    if (q.freq_factors != nullptr) {
        rope_neox_qkv<true><<<block_nums, block_dims, 0, ctx.stream()>>>(
            q, k, (const char *) v_x->data, (half *) v_dst->data, v_k, v_x->ne[0], v_x->ne[1], v_x->ne[2],
            v_x->nb[1], v_x->nb[2], v_x->nb[3]);
    } else {
        rope_neox_qkv<false><<<block_nums, block_dims, 0, ctx.stream()>>>(
            q, k, (const char *) v_x->data, (half *) v_dst->data, v_k, v_x->ne[0], v_x->ne[1], v_x->ne[2],
            v_x->nb[1], v_x->nb[2], v_x->nb[3]);
    }
}

// One row per block: ROUND_BF16(x) -> RMS_NORM -> MUL -> ROUND_BF16 exactly as rms_norm_f32<block_size>
// (multiply, round, round_in) computes it, then ROPE(NEOX) -> ROUND_BF16 of that rounded row as
// rope_neox computes it, reading the row from shared memory.
template <int block_size, bool has_ff>
static __global__ void round_rms_norm_rope(const float * x, const int64_t stride_row, const int64_t stride_channel,
                                           const int64_t stride_sample, const float eps, const float * mul,
                                           const rope_neox_launch r) {
    __shared__ float s_sum[32];
    __shared__ float row[2*CUDA_ROPE_BLOCK_SIZE];
    const int ncols     = r.ne00;
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
    const int tid       = threadIdx.x;

    x += blockIdx.z*stride_sample + blockIdx.y*stride_channel + blockIdx.x*stride_row;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = rope_round<true>(x[col]);
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float xv    = rope_round<true>(x[col]);
        const float value = scale * xv * mul[col];
        row[col]          = rope_round<true>(value);
    }
    __syncthreads();

    const int i0 = 2*tid;
    if (i0 < ncols) {
        const int row_dst = (blockIdx.z*nchannels + blockIdx.y)*nrows + blockIdx.x;
        rope_neox_row<true, has_ff, float, float, true, false>(
            row, (float *) r.dst, r.ne00, r.ne01, r.ne02, 0, 0, 0, r.s1, r.s2, r.s3, r.n_dims, r.pos, r.freq_scale,
            r.ext_factor, r.attn_factor, r.corr_dims, r.theta_scale, r.freq_factors, nullptr, 0, row_dst, i0);
    }
}

void ggml_cuda_op_round_rms_norm_rope(ggml_backend_cuda_context & ctx, const ggml_tensor * src_unrounded,
                                      const ggml_tensor * rms, const ggml_tensor * mul, const ggml_tensor * rope,
                                      const ggml_tensor * out) {
    const int64_t ncols = src_unrounded->ne[0];
    GGML_ASSERT(((const int32_t *) rope->op_params)[2] == GGML_ROPE_TYPE_NEOX);
    GGML_ASSERT(src_unrounded->type == GGML_TYPE_F32 && src_unrounded->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_are_same_shape(src_unrounded, rms->src[0]) && ggml_nrows(src_unrounded) == ggml_nrows(rope));
    GGML_ASSERT(rope->ne[0] == ncols && ncols % 2 == 0 && ncols <= 2*CUDA_ROPE_BLOCK_SIZE && ncols < 1024);
    GGML_ASSERT(ggml_is_contiguous(rope->src[0]) && rope->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && ggml_nelements(mul) == ncols);
    GGML_ASSERT(out->type == GGML_TYPE_F32 && ggml_is_contiguous(out) && ggml_are_same_shape(out, rope));

    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    const rope_neox_launch r = rope_neox_launch_for(rope, rope->src[0], out);
    const dim3 blocks_num(src_unrounded->ne[1], src_unrounded->ne[2], src_unrounded->ne[3]);
    const int64_t s01 = src_unrounded->nb[1] / sizeof(float);
    const int64_t s02 = src_unrounded->nb[2] / sizeof(float);
    const int64_t s03 = src_unrounded->nb[3] / sizeof(float);
    // rms_norm_f32 uses 256 threads below 1024 columns.
    if (r.freq_factors != nullptr) {
        round_rms_norm_rope<256, true><<<blocks_num, 256, 0, ctx.stream()>>>(
            (const float *) src_unrounded->data, s01, s02, s03, eps, (const float *) mul->data, r);
    } else {
        round_rms_norm_rope<256, false><<<blocks_num, 256, 0, ctx.stream()>>>(
            (const float *) src_unrounded->data, s01, s02, s03, eps, (const float *) mul->data, r);
    }
}
