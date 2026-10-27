#include "cuda_depth.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio> // TMPDIAG
#include <cstdlib> // TMPDIAG
#include <stdexcept>
#include <string>

namespace engine::models::breeze_tts {
namespace {

constexpr int kWarp = 32;
constexpr int kMatvecWarps = 8;
constexpr int kMaxBatch = 2;
constexpr int kSampleThreads = 1024;

void check(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("BreezeTTS CUDA depth: ") + what + ": " + cudaGetErrorString(status));
    }
}

// Same value as GGML_UNARY_OP_ROUND_BF16 (f32 -> bf16 -> f32, round to nearest even).
__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

// Programmatic dependent launch (sm_90+): a kernel may start while its predecessor
// finishes, prefetch its weights, and only wait before reading the previous outputs.
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0F + expf(-x));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset);
    }
    return x;
}

template <int type>
__device__ __forceinline__ void load8(const void * row, int k, float * out) {
    if constexpr (type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) {
        const uint4 v = *reinterpret_cast<const uint4 *>(static_cast<const char *>(row) + static_cast<size_t>(k) * 2);
        const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f;
            if constexpr (type == GGML_TYPE_BF16) {
                f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[i]));
            } else {
                f = __half22float2(*reinterpret_cast<const half2 *>(&words[i]));
            }
            out[2 * i] = f.x;
            out[2 * i + 1] = f.y;
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // block_q8_0: f16 scale + 32 int8. Eight consecutive values never straddle a block.
        const char * block = static_cast<const char *>(row) + static_cast<size_t>(k / 32) * 34;
        const float scale = __half2float(*reinterpret_cast<const half *>(block));
        const uint16_t * q = reinterpret_cast<const uint16_t *>(block + 2 + k % 32);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint16_t pair = q[i];
            out[2 * i] = scale * static_cast<float>(static_cast<int8_t>(pair & 0xff));
            out[2 * i + 1] = scale * static_cast<float>(static_cast<int8_t>(pair >> 8));
        }
    } else {
        const float4 * p = reinterpret_cast<const float4 *>(static_cast<const float *>(row) + k);
        const float4 a = p[0];
        const float4 b = p[1];
        out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
        out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
    }
}

__device__ __forceinline__ float load1(const void * row, int type, int k) {
    if (type == GGML_TYPE_BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(row)[k]);
    }
    if (type == GGML_TYPE_F16) {
        return __half2float(reinterpret_cast<const half *>(row)[k]);
    }
    return reinterpret_cast<const float *>(row)[k];
}

struct AttentionArgs {
    const float * qkv = nullptr;
    int qkv_stride = 0;
    half * k_cache = nullptr;  // [batch, kv_heads, max_pos, dim]
    half * v_cache = nullptr;
    int max_pos = 0;
    float * ctx = nullptr;
    int ctx_stride = 0;
    int pos = 0;
    int heads = 0;
    int kv_heads = 0;
    float theta_scale = 0.0F;
    const float * rope_factors = nullptr;
    float scale = 0.0F;
};

enum MatvecMode : int {
    kPlain = 0,      // y = W x
    kRound = 1,      // y = r(W x)
    kRoundResid = 2, // y = r(r(W x) + residual)
    kSwiGlu = 3,     // y = r(r(silu(r(W_gate x))) * r(W_up x))
    kAttnO = 4,      // x = attention (computed per block), then as kRoundResid
};

struct MatvecArgs {
    const void * w = nullptr;
    size_t row_bytes = 0;
    int k = 0;
    int n = 0;
    int batch = 1;
    // Input rows: either `x` ([x_batches, k], stride x_stride) or one embedding-table
    // row selected on the device by *token (+ table_base); a single input row is
    // broadcast to every output row.
    const float * x = nullptr;
    int x_stride = 0;
    int x_batches = 1;
    const void * table = nullptr;
    size_t table_row_bytes = 0;
    int table_type = GGML_TYPE_F32;
    const int32_t * token = nullptr;
    int64_t table_base = 0;
    // Optional RMSNorm of the input (times norm_w), rounded to bf16 when norm_round.
    const float * norm_w = nullptr;
    float eps = 0.0F;
    bool norm_round = true;
    float * out = nullptr;
    int out_stride = 0;
    const float * residual = nullptr;
    int residual_stride = 0;
    int up_offset = 0;
    AttentionArgs attn;  // kAttnO only
};

// Shared scratch the kAttnO prologue needs after the staged inputs.
size_t attention_scratch_bytes(const AttentionArgs & t, int dim) {
    return static_cast<size_t>(t.heads) * dim * sizeof(float) +
           2 * static_cast<size_t>(t.kv_heads) * t.max_pos * dim * sizeof(half) + static_cast<size_t>(t.heads) * 33 * sizeof(float);
}

// Attention of one batch row into `ctx` (shared, heads * dim), by the whole block:
// NEOX RoPE of the new q/k (rounded to bf16, q pre-scaled like flash attention), the
// f16 K (transposed, [kv][dim][max_pos]) and V ([kv][max_pos][dim]) rows 0..pos staged
// in shared memory; one warp per head with lane l scoring position l. Block 0 appends
// the new k/v row to the cache.
template <int dim>
__device__ void attention_into_shared(const AttentionArgs & t, int b, float * ctx, float * scratch) {
    constexpr int half_dim = dim / 2;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    const int warps = blockDim.x / kWarp;
    const int group = t.heads / t.kv_heads;
    const int len = t.pos + 1;
    float * q = scratch;
    half * kt = reinterpret_cast<half *>(q + t.heads * dim);  // [kv][dim][max_pos]
    half * vc = kt + t.kv_heads * t.max_pos * dim;
    float * probs = reinterpret_cast<float *>(vc + t.kv_heads * t.max_pos * dim);  // [heads][32]
    float * sums = probs + t.heads * 32;
    const float * row = t.qkv + b * t.qkv_stride;
    const int q_out = t.heads * dim;
    const int kv_out = t.kv_heads * dim;
    const size_t cache_row = static_cast<size_t>(t.max_pos) * dim;
    half * k_cache = t.k_cache + static_cast<size_t>(b * t.kv_heads) * cache_row;
    half * v_cache = t.v_cache + static_cast<size_t>(b * t.kv_heads) * cache_row;
    const bool writer = blockIdx.x == 0;

    // Earlier positions: copy the cached rows (V as is, 8 halves at a time; K transposed).
    const int chunks = t.pos * dim / 8;
    for (int i = tid; i < t.kv_heads * chunks; i += blockDim.x) {
        const int g = i / chunks;
        const int c = i % chunks;
        reinterpret_cast<uint4 *>(vc + g * cache_row)[c] = reinterpret_cast<const uint4 *>(v_cache + g * cache_row)[c];
    }
    for (int i = tid; i < t.kv_heads * t.pos * dim; i += blockDim.x) {
        const int g = i / (t.pos * dim);
        const int l = (i / dim) % t.pos;
        const int d = i % dim;
        kt[g * cache_row + static_cast<size_t>(d) * t.max_pos + l] = k_cache[g * cache_row + static_cast<size_t>(l) * dim + d];
    }
    // New position: rotate q (all heads) and k, take v.
    for (int i = tid; i < (t.heads + t.kv_heads) * half_dim; i += blockDim.x) {
        const int d = i % half_dim;
        const int head = i / half_dim;
        // As ggml's rope: theta_base / freq_factor with theta_base = pos * theta_scale^d.
        const float theta_base = static_cast<float>(t.pos) * powf(t.theta_scale, static_cast<float>(d));
        const float theta = t.rope_factors != nullptr ? theta_base / t.rope_factors[d] : theta_base;
        const float cos_t = cosf(theta);
        const float sin_t = sinf(theta);
        if (head < t.heads) {
            const float x0 = row[head * dim + d];
            const float x1 = row[head * dim + d + half_dim];
            // Flash attention scales q before the dot product.
            q[head * dim + d] = round_bf16(x0 * cos_t - x1 * sin_t) * t.scale;
            q[head * dim + d + half_dim] = round_bf16(x0 * sin_t + x1 * cos_t) * t.scale;
        } else {
            const int g = head - t.heads;
            const float x0 = row[q_out + g * dim + d];
            const float x1 = row[q_out + g * dim + d + half_dim];
            const half k0 = __float2half(round_bf16(x0 * cos_t - x1 * sin_t));
            const half k1 = __float2half(round_bf16(x0 * sin_t + x1 * cos_t));
            const size_t at = g * cache_row + static_cast<size_t>(t.pos) * dim + d;
            kt[g * cache_row + static_cast<size_t>(d) * t.max_pos + t.pos] = k0;
            kt[g * cache_row + static_cast<size_t>(d + half_dim) * t.max_pos + t.pos] = k1;
            if (writer) {
                k_cache[at] = k0;
                k_cache[at + half_dim] = k1;
            }
        }
    }
    for (int i = tid; i < kv_out; i += blockDim.x) {
        const int g = i / dim;
        const int d = i % dim;
        const half v = __float2half(row[q_out + kv_out + i]);
        const size_t at = g * cache_row + static_cast<size_t>(t.pos) * dim + d;
        vc[at] = v;
        if (writer) {
            v_cache[at] = v;
        }
    }
    __syncthreads();

    // Scores and softmax: one warp per head, lane l scores position l on its own.
    for (int h = warp; h < t.heads; h += warps) {
        const int g = h / group;
        float mine = -INFINITY;
        if (lane < len) {
            const half * kcol = kt + g * cache_row + lane;
            const float * qh = q + h * dim;
            float s = 0.0F;
#pragma unroll 8
            for (int i = 0; i < dim; ++i) {
                s += qh[i] * __half2float(kcol[static_cast<size_t>(i) * t.max_pos]);
            }
            mine = s;
        }
        float m = mine;
        for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, offset));
        }
        const float e = lane < len ? expf(mine - m) : 0.0F;
        probs[h * 32 + lane] = e;
        const float sum = warp_sum(e);
        if (lane == 0) {
            sums[h] = sum;
        }
    }
    __syncthreads();
    for (int i = tid; i < t.heads * dim; i += blockDim.x) {
        const int h = i / dim;
        const int d = i % dim;
        const int g = h / group;
        float out = 0.0F;
        for (int l = 0; l < len; ++l) {
            out += probs[h * 32 + l] * __half2float(vc[g * cache_row + static_cast<size_t>(l) * dim + d]);
        }
        ctx[i] = round_bf16(out / sums[h]);
    }
    __syncthreads();
}

// A block stages (and normalizes) the input once, then its warps walk the output
// rows (one row per warp at a time, grid-strided). The grid is capped at a few blocks
// per SM so the staging cost is paid once per SM slot, and each warp prefetches its
// next row's first chunk while it reduces the current one.
template <int wtype, int mode>
__global__ void matvec_kernel(MatvecArgs a) {
    extern __shared__ float xs[];
    __shared__ float partial[kMatvecWarps];
    // Eight weights (16 bytes of bf16) per lane and step; k is a multiple of 256. Each
    // chunk issues kUnroll steps of loads before using them.
    constexpr int kUnroll = 4;
    constexpr int kStep = 8 * kWarp;
    const bool staged = mode == kAttnO || a.norm_w != nullptr || a.table != nullptr;
    const int tid = threadIdx.x;
    const int warp = tid / kWarp;
    const int lane = tid % kWarp;
    // Split-K: a row's k range is cut into segments of kSegment values, one warp each,
    // so long rows (the 8192-wide down projection) keep all their loads in flight.
    constexpr int kSegment = kUnroll * kStep;  // 1024
    const int wpr = a.k % kSegment == 0 ? min(a.k / kSegment, kMatvecWarps) : 1;  // warps per row
    const int rows_per_block = kMatvecWarps / wpr;
    const int seg = warp % wpr;
    const int seg_len = a.k / wpr;
    const int seg_begin = seg * seg_len;
    const int row_stride = gridDim.x * rows_per_block;
    const int block_first = blockIdx.x * rows_per_block;
    int n = block_first + warp / wpr;
    __shared__ float red[kMatvecWarps][2 * kMaxBatch];
    float w[kUnroll][8];
    float u[kUnroll][8];
    auto load_chunk = [&](int row_index, int k0) {
        const char * row = static_cast<const char *>(a.w) + static_cast<size_t>(row_index) * a.row_bytes;
        const char * up_row = static_cast<const char *>(a.w) + static_cast<size_t>(row_index + a.up_offset) * a.row_bytes;
#pragma unroll
        for (int s = 0; s < kUnroll; ++s) {
            if (k0 + s * kStep < seg_begin + seg_len) {
                load8<wtype>(row, k0 + s * kStep, w[s]);
                if constexpr (mode == kSwiGlu) {
                    load8<wtype>(up_row, k0 + s * kStep, u[s]);
                }
            }
        }
    };
    // Weights do not depend on the previous kernel: fetch the first chunk before waiting.
    if (n < a.n) {
        load_chunk(n, seg_begin + lane * 8);
    }
    pdl_launch_dependents();
    pdl_wait();
    if constexpr (mode == kAttnO) {
        for (int b = 0; b < a.x_batches; ++b) {
            attention_into_shared<128>(a.attn, b, xs + b * a.k, xs + a.x_batches * a.k);
        }
    } else if (staged) {
        for (int b = 0; b < a.x_batches; ++b) {
            if (a.table != nullptr) {
                const char * row = static_cast<const char *>(a.table) +
                                   static_cast<size_t>(a.table_base + *a.token) * a.table_row_bytes;
                for (int k = tid; k < a.k; k += blockDim.x) {
                    xs[b * a.k + k] = load1(row, a.table_type, k);
                }
            } else {
                for (int k = tid; k < a.k; k += blockDim.x) {
                    xs[b * a.k + k] = a.x[b * a.x_stride + k];
                }
            }
            if (a.norm_w != nullptr) {
                float sum = 0.0F;
                for (int k = tid; k < a.k; k += blockDim.x) {
                    const float v = xs[b * a.k + k];
                    sum += v * v;
                }
                sum = warp_sum(sum);
                if (tid % kWarp == 0) {
                    partial[tid / kWarp] = sum;
                }
                __syncthreads();
                float total = 0.0F;
                for (int i = 0; i < kMatvecWarps; ++i) {
                    total += partial[i];
                }
                const float scale = rsqrtf(total / static_cast<float>(a.k) + a.eps);
                for (int k = tid; k < a.k; k += blockDim.x) {
                    const float v = xs[b * a.k + k] * scale * a.norm_w[k];
                    xs[b * a.k + k] = a.norm_round ? round_bf16(v) : v;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    for (int base = block_first; base < a.n; base += row_stride, n += row_stride) {
        float acc[kMaxBatch] = {0.0F, 0.0F};
        float acc_up[kMaxBatch] = {0.0F, 0.0F};
        if (n < a.n) {
            for (int k0 = seg_begin + lane * 8; k0 < seg_begin + seg_len; k0 += kUnroll * kStep) {
                if (k0 != seg_begin + lane * 8) {
                    load_chunk(n, k0);
                }
#pragma unroll
                for (int s = 0; s < kUnroll; ++s) {
                    const int k = k0 + s * kStep;
                    if (k >= seg_begin + seg_len) {
                        break;
                    }
#pragma unroll
                    for (int b = 0; b < kMaxBatch; ++b) {
                        if (b < a.x_batches) {
                            const float4 * xp =
                                reinterpret_cast<const float4 *>(staged ? xs + b * a.k + k : a.x + b * a.x_stride + k);
                            const float4 x0 = xp[0];
                            const float4 x1 = xp[1];
                            const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[b] += w[s][i] * x[i];
                                if constexpr (mode == kSwiGlu) {
                                    acc_up[b] += u[s][i] * x[i];
                                }
                            }
                        }
                    }
                }
            }
            // The current row's weights are consumed: start fetching the next row.
            if (n + row_stride < a.n) {
                load_chunk(n + row_stride, seg_begin + lane * 8);
            }
        }
#pragma unroll
        for (int b = 0; b < kMaxBatch; ++b) {
            acc[b] = warp_sum(acc[b]);
            if constexpr (mode == kSwiGlu) {
                acc_up[b] = warp_sum(acc_up[b]);
            }
        }
        if (wpr > 1) {
            if (lane == 0) {
#pragma unroll
                for (int b = 0; b < kMaxBatch; ++b) {
                    red[warp][b] = acc[b];
                    red[warp][kMaxBatch + b] = acc_up[b];
                }
            }
            __syncthreads();
            if (seg == 0) {
#pragma unroll
                for (int b = 0; b < kMaxBatch; ++b) {
                    float total = 0.0F;
                    float total_up = 0.0F;
                    for (int i = 0; i < wpr; ++i) {
                        total += red[warp + i][b];
                        total_up += red[warp + i][kMaxBatch + b];
                    }
                    acc[b] = total;
                    acc_up[b] = total_up;
                }
            }
            __syncthreads();
        }
        if (lane == 0 && seg == 0 && n < a.n) {
            for (int b = 0; b < a.batch; ++b) {
                const int src = a.x_batches == 1 ? 0 : b;
                float y = acc[src];
                if constexpr (mode == kRound) {
                    y = round_bf16(y);
                } else if constexpr (mode == kRoundResid || mode == kAttnO) {
                    y = round_bf16(round_bf16(y) + a.residual[b * a.residual_stride + n]);
                } else if constexpr (mode == kSwiGlu) {
                    const float gate = round_bf16(silu(round_bf16(y)));
                    y = round_bf16(gate * round_bf16(acc_up[src]));
                }
                a.out[b * a.out_stride + n] = y;
            }
        }
    }
}

// One block per (batch row, query head), one thread per head dimension: NEOX RoPE
// of the new q/k (rounded to bf16), then attention over positions 0..pos. Earlier
// positions come from the f16 cache (as flash attention reads it); the new k/v row
// is used from shared memory and appended to the cache by the group's first head.
template <int dim>
__global__ void attention_kernel(AttentionArgs a) {
    constexpr int half_dim = dim / 2;
    constexpr int warps = dim / kWarp;
    __shared__ float q[dim];
    __shared__ float kn[dim];
    __shared__ float vn[dim];
    __shared__ float scores[32];
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int group = a.heads / a.kv_heads;
    const int g = h / group;
    const int d = threadIdx.x;
    const int warp = d / kWarp;
    const int lane = d % kWarp;
    const float * row = a.qkv + b * a.qkv_stride;
    const int q_out = a.heads * dim;
    const int kv_out = a.kv_heads * dim;
    half * k_cache = a.k_cache + static_cast<size_t>(b * a.kv_heads + g) * a.max_pos * dim;
    half * v_cache = a.v_cache + static_cast<size_t>(b * a.kv_heads + g) * a.max_pos * dim;
    const bool writer = h % group == 0;
    pdl_launch_dependents();
    pdl_wait();

    if (d < half_dim) {
        // As ggml's rope: theta_base / freq_factor with theta_base = pos * theta_scale^d.
        const float theta_base = static_cast<float>(a.pos) * powf(a.theta_scale, static_cast<float>(d));
        const float theta = a.rope_factors != nullptr ? theta_base / a.rope_factors[d] : theta_base;
        const float cos_t = cosf(theta);
        const float sin_t = sinf(theta);
        const float k0 = row[q_out + g * dim + d];
        const float k1 = row[q_out + g * dim + d + half_dim];
        const half ka = __float2half(round_bf16(k0 * cos_t - k1 * sin_t));
        const half kb = __float2half(round_bf16(k0 * sin_t + k1 * cos_t));
        kn[d] = __half2float(ka);
        kn[d + half_dim] = __half2float(kb);
        if (writer) {
            k_cache[a.pos * dim + d] = ka;
            k_cache[a.pos * dim + d + half_dim] = kb;
        }
        const float q0 = row[h * dim + d];
        const float q1 = row[h * dim + d + half_dim];
        // Flash attention scales q before the dot product.
        q[d] = round_bf16(q0 * cos_t - q1 * sin_t) * a.scale;
        q[d + half_dim] = round_bf16(q0 * sin_t + q1 * cos_t) * a.scale;
    }
    const half v = __float2half(row[q_out + kv_out + g * dim + d]);
    vn[d] = __half2float(v);
    if (writer) {
        v_cache[a.pos * dim + d] = v;
    }
    __syncthreads();

    for (int l = warp; l <= a.pos; l += warps) {
        float s = 0.0F;
        for (int i = lane; i < dim; i += kWarp) {
            const float k = l == a.pos ? kn[i] : __half2float(k_cache[l * dim + i]);
            s += q[i] * k;
        }
        s = warp_sum(s);
        if (lane == 0) {
            scores[l] = s;
        }
    }
    __syncthreads();
    float m = scores[0];
    for (int l = 1; l <= a.pos; ++l) {
        m = fmaxf(m, scores[l]);
    }
    float sum = 0.0F;
    float out = 0.0F;
    for (int l = 0; l <= a.pos; ++l) {
        const float p = expf(scores[l] - m);
        sum += p;
        out += p * (l == a.pos ? vn[d] : __half2float(v_cache[l * dim + d]));
    }
    a.ctx[b * a.ctx_stride + h * dim + d] = round_bf16(out / sum);
}

struct SampleArgs {
    const float * logits = nullptr;
    int logits_stride = 0;
    int batch = 1;
    float guidance = 1.0F;
    float inv_temperature = 1.0F;
    int n = 0;  // power of two, <= 2 * kSampleThreads
    int candidates = 0;
    float top_p = 1.0F;
    const float * noise = nullptr;
    int32_t * token = nullptr;
};

// CFG mix, temperature, top-k, top-p and the exponential race of the fused graph:
// argmax over the kept candidates of (score + noise[id]).
__global__ void sample_kernel(SampleArgs a) {
    __shared__ float vals[2 * kSampleThreads];
    __shared__ int ids[2 * kSampleThreads];
    __shared__ float best_v[kSampleThreads / kWarp];
    __shared__ int best_i[kSampleThreads / kWarp];
    const int tid = threadIdx.x;
    pdl_launch_dependents();
    pdl_wait();
    for (int i = tid; i < a.n; i += blockDim.x) {
        float s = a.logits[i];
        if (a.batch == 2 && a.guidance != 1.0F) {
            const float u = a.logits[a.logits_stride + i];
            s = u + (s - u) * a.guidance;
        }
        vals[i] = s * a.inv_temperature;
        ids[i] = i;
    }
    __syncthreads();
    // Bitonic sort, descending by score, ties by ascending id.
    for (int size = 2; size <= a.n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < a.n; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const bool i_first = vals[i] > vals[j] || (vals[i] == vals[j] && ids[i] < ids[j]);
                    if (i_first != desc) {
                        const float tv = vals[i];
                        vals[i] = vals[j];
                        vals[j] = tv;
                        const int ti = ids[i];
                        ids[i] = ids[j];
                        ids[j] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (a.top_p < 1.0F && tid == 0) {
        // Keep a candidate while the mass strictly before it is below top_p.
        const float m = vals[0];
        float sum = 0.0F;
        for (int i = 0; i < a.candidates; ++i) {
            sum += expf(vals[i] - m);
        }
        float cumulative = 0.0F;
        for (int i = 0; i < a.candidates; ++i) {
            const float p = expf(vals[i] - m) / sum;
            cumulative += p;
            const float before = cumulative - p;
            if (!(a.top_p - before > 0.0F)) {
                vals[i] -= 1.0e30F;
            }
        }
    }
    __syncthreads();
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = tid; i < a.candidates; i += blockDim.x) {
        const float v = vals[i] + a.noise[ids[i]];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, offset);
        const int oi = __shfl_xor_sync(0xffffffff, bi, offset);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (tid % kWarp == 0) {
        best_v[tid / kWarp] = bv;
        best_i[tid / kWarp] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < static_cast<int>(blockDim.x) / kWarp; ++w) {
            if (best_v[w] > bv || (best_v[w] == bv && best_i[w] < bi)) {
                bv = best_v[w];
                bi = best_i[w];
            }
        }
        if (best_v[0] > bv || (best_v[0] == bv && best_i[0] < bi)) {
            bv = best_v[0];
            bi = best_i[0];
        }
        *a.token = ids[bi];
    }
}

bool g_use_pdl = false;
int g_sm_count = 0;
int g_waves = 1;  // TMPDIAG: resident-block waves per launch (TMP_WAVES)

// One wave of resident blocks: every block stages its input once and walks rows.
template <int wtype, int mode>
void launch_matvec_kernel(const MatvecArgs & a, int rows_blocks, size_t shared, cudaStream_t stream) {
    const auto kernel = matvec_kernel<wtype, mode>;
    int per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kMatvecWarps * kWarp, shared), "occupancy");
    const int cap = std::max(1, per_sm) * g_sm_count * g_waves;
    launch(kernel, dim3(std::min(rows_blocks, cap)), dim3(kMatvecWarps * kWarp), shared, stream, a);
}

template <typename... Params, typename... Args>
void launch(void (*kernel)(Params...), dim3 grid, dim3 block, size_t shared, cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = g_use_pdl ? 1 : 0;
    check(cudaLaunchKernelEx(&config, kernel, args...), "launch");
}

template <int mode>
void launch_matvec_mode(int wtype, const MatvecArgs & a, cudaStream_t stream) {
    const bool staged = mode == kAttnO || a.norm_w != nullptr || a.table != nullptr;
    size_t shared = staged ? static_cast<size_t>(a.x_batches) * a.k * sizeof(float) : 0;
    if constexpr (mode == kAttnO) {
        shared = static_cast<size_t>(a.x_batches) * a.k * sizeof(float) + attention_scratch_bytes(a.attn, 128);
    }
    const int wpr = a.k % 1024 == 0 ? std::min(a.k / 1024, kMatvecWarps) : 1;
    const int rows_per_block = kMatvecWarps / wpr;
    const int rows_blocks = (a.n + rows_per_block - 1) / rows_per_block;
    switch (wtype) {
        case GGML_TYPE_BF16:
            launch_matvec_kernel<GGML_TYPE_BF16, mode>(a, rows_blocks, shared, stream);
            break;
        case GGML_TYPE_F16:
            launch_matvec_kernel<GGML_TYPE_F16, mode>(a, rows_blocks, shared, stream);
            break;
        case GGML_TYPE_F32:
            launch_matvec_kernel<GGML_TYPE_F32, mode>(a, rows_blocks, shared, stream);
            break;
        case GGML_TYPE_Q8_0:
            launch_matvec_kernel<GGML_TYPE_Q8_0, mode>(a, rows_blocks, shared, stream);
            break;
        default:
            throw std::runtime_error("BreezeTTS CUDA depth: unsupported weight type");
    }
}

void launch_matvec(int mode, int wtype, const MatvecArgs & a, cudaStream_t stream) {
    switch (mode) {
        case kPlain: launch_matvec_mode<kPlain>(wtype, a, stream); break;
        case kRound: launch_matvec_mode<kRound>(wtype, a, stream); break;
        case kRoundResid: launch_matvec_mode<kRoundResid>(wtype, a, stream); break;
        case kAttnO: launch_matvec_mode<kAttnO>(wtype, a, stream); break;
        default: launch_matvec_mode<kSwiGlu>(wtype, a, stream); break;
    }
}

bool supported_type(int type) {
    return type == GGML_TYPE_BF16 || type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_Q8_0;
}

class BreezeCudaDepthImpl final : public BreezeCudaDepth {
public:
    explicit BreezeCudaDepthImpl(const BreezeCudaDepthDesc & desc) : d_(desc) {
        if (d_.head_dim != 128 || d_.heads % d_.kv_heads != 0 || d_.steps + 1 > 32 ||
            d_.codebook_size > 2 * kSampleThreads || (d_.codebook_size & (d_.codebook_size - 1)) != 0 ||
            d_.depth_hidden % 256 != 0 || d_.intermediate % 256 != 0 || d_.hidden % 256 != 0 || (d_.heads * d_.head_dim) % 256 != 0) {
            throw std::runtime_error("BreezeTTS CUDA depth: unsupported shape");
        }
        auto check_matrix = [](const BreezeCudaDepthMatrix & m, const char * name) {
            if (m.data == nullptr || !supported_type(m.type)) {
                throw std::runtime_error(std::string("BreezeTTS CUDA depth: unsupported ") + name);
            }
        };
        check_matrix(d_.projector, "projector");
        check_matrix(d_.table, "embedding table");
        if (d_.table.type == GGML_TYPE_Q8_0) {
            throw std::runtime_error("BreezeTTS CUDA depth: unsupported embedding table type");
        }
        check_matrix(d_.heads_w, "codebook heads");
        for (const auto & layer : d_.layers) {
            check_matrix(layer.qkv, "qkv");
            check_matrix(layer.o, "o");
            check_matrix(layer.gate_up, "gate_up");
            check_matrix(layer.down, "down");
            if (layer.input_norm == nullptr || layer.post_norm == nullptr) {
                throw std::runtime_error("BreezeTTS CUDA depth: missing layer norm");
            }
        }
        const int64_t b = kMaxBatch;
        const int64_t qkv = (d_.heads + 2 * d_.kv_heads) * d_.head_dim;
        const size_t cache = static_cast<size_t>(b * d_.kv_heads * (d_.steps + 1) * d_.head_dim);
        alloc(x_, b * d_.depth_hidden);
        alloc(qkv_, b * qkv);
        alloc(ctx_, b * d_.heads * d_.head_dim);
        alloc(inter_, b * d_.intermediate);
        alloc(logits_, b * d_.codebook_size);
        check(cudaMalloc(&tokens_, static_cast<size_t>(d_.steps) * sizeof(int32_t)), "cudaMalloc tokens");
        check(cudaMemset(tokens_, 0, static_cast<size_t>(d_.steps) * sizeof(int32_t)), "cudaMemset tokens");
        k_cache_.resize(d_.layers.size());
        v_cache_.resize(d_.layers.size());
        for (size_t l = 0; l < d_.layers.size(); ++l) {
            check(cudaMalloc(&k_cache_[l], cache * sizeof(half)), "cudaMalloc k cache");
            check(cudaMalloc(&v_cache_[l], cache * sizeof(half)), "cudaMalloc v cache");
        }
        check(cudaMallocHost(&tokens_host_, static_cast<size_t>(d_.steps) * sizeof(int32_t)), "cudaMallocHost");
        int device = 0;
        int major = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "compute capability");
        g_use_pdl = major >= 9 && getenv("TMP_NO_PDL") == nullptr; // TMPDIAG env
        int sms = 0;
        check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SM count");
        g_sm_count = sms;
        const char * waves = getenv("TMP_WAVES"); // TMPDIAG env
        g_waves = waves != nullptr ? atoi(waves) : 1;
    }

    ~BreezeCudaDepthImpl() override {
        prof_report(); // TMPDIAG
        if (graph_exec_ != nullptr) {
            cudaGraphExecDestroy(graph_exec_);
        }
        for (float * p : {x_, qkv_, ctx_, inter_, logits_}) {
            cudaFree(p);
        }
        cudaFree(tokens_);
        for (half * p : k_cache_) {
            cudaFree(p);
        }
        for (half * p : v_cache_) {
            cudaFree(p);
        }
        cudaFreeHost(tokens_host_);
    }

    std::vector<int32_t> run(
        void * stream_ptr,
        const BreezeCudaDepthSampling & sampling,
        const float * hidden,
        const int32_t * first_token,
        const float * noise) override {
        auto * stream = static_cast<cudaStream_t>(stream_ptr);
        if (graph_exec_ == nullptr || !(sampling == key_) || hidden != hidden_ || first_token != first_token_ ||
            noise != noise_ || stream != stream_) {
            if (graph_exec_ != nullptr) {
                cudaGraphExecDestroy(graph_exec_);
                graph_exec_ = nullptr;
            }
            cudaGraph_t graph = nullptr;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
            try {
                enqueue(stream, sampling, hidden, first_token, noise);
            } catch (...) {
                cudaStreamEndCapture(stream, &graph);
                if (graph != nullptr) {
                    cudaGraphDestroy(graph);
                }
                throw;
            }
            check(cudaStreamEndCapture(stream, &graph), "end capture");
            const cudaError_t status = cudaGraphInstantiate(&graph_exec_, graph, 0);
            cudaGraphDestroy(graph);
            check(status, "graph instantiate");
            key_ = sampling;
            hidden_ = hidden;
            first_token_ = first_token;
            noise_ = noise;
            stream_ = stream;
        }
        if (prof_) { // TMPDIAG
            check(cudaStreamSynchronize(stream), "synchronize");
            cudaEvent_t start;
            cudaEventCreate(&start);
            cudaEventRecord(start, stream);
            prof_record_ = true;
            enqueue(stream, sampling, hidden, first_token, noise);
            prof_record_ = false;
            check(cudaStreamSynchronize(stream), "synchronize");
            cudaEvent_t prev = start;
            for (auto & [cat, e] : marks_) {
                float ms = 0;
                cudaEventElapsedTime(&ms, prev, e);
                prof_ms_[cat] += ms;
                if (prev != start) cudaEventDestroy(prev);
                prev = e;
            }
            if (prev != start) cudaEventDestroy(prev);
            cudaEventDestroy(start);
            marks_.clear();
            cudaEvent_t g0, g1;
            cudaEventCreate(&g0);
            cudaEventCreate(&g1);
            cudaEventRecord(g0, stream);
            check(cudaGraphLaunch(graph_exec_, stream), "graph launch");
            cudaEventRecord(g1, stream);
            check(cudaStreamSynchronize(stream), "synchronize");
            float gms = 0;
            cudaEventElapsedTime(&gms, g0, g1);
            prof_graph_ms_ += gms;
            ++prof_frames_;
            cudaEventDestroy(g0);
            cudaEventDestroy(g1);
            return std::vector<int32_t>(tokens_host_, tokens_host_ + d_.steps);
        }
        check(cudaGraphLaunch(graph_exec_, stream), "graph launch");
        check(cudaStreamSynchronize(stream), "synchronize");
        return std::vector<int32_t>(tokens_host_, tokens_host_ + d_.steps);
    }

private:
    template <typename T>
    static void alloc(T *& p, int64_t count) {
        check(cudaMalloc(&p, static_cast<size_t>(count) * sizeof(T)), "cudaMalloc");
    }

    void layers(cudaStream_t stream, int batch, int pos) {
        const int dh = static_cast<int>(d_.depth_hidden);
        const int qkv = static_cast<int>((d_.heads + 2 * d_.kv_heads) * d_.head_dim);
        const int inter = static_cast<int>(d_.intermediate);
        for (size_t l = 0; l < d_.layers.size(); ++l) {
            const auto & layer = d_.layers[l];
            MatvecArgs a;
            a.w = layer.qkv.data;
            a.row_bytes = layer.qkv.row_bytes;
            a.k = dh;
            a.n = qkv;
            a.batch = batch;
            a.x = x_;
            a.x_stride = dh;
            a.x_batches = batch;
            a.norm_w = layer.input_norm;
            a.eps = d_.rms_norm_eps;
            a.out = qkv_;
            a.out_stride = qkv;
            if (!skip(kQkv)) launch_matvec(kRound, layer.qkv.type, a, stream);
            mark(stream, kQkv);

            AttentionArgs t;
            t.qkv = qkv_;
            t.qkv_stride = qkv;
            t.k_cache = k_cache_[l];
            t.v_cache = v_cache_[l];
            t.max_pos = static_cast<int>(d_.steps + 1);
            t.ctx = ctx_;
            t.ctx_stride = static_cast<int>(d_.heads * d_.head_dim);
            t.pos = pos;
            t.heads = static_cast<int>(d_.heads);
            t.kv_heads = static_cast<int>(d_.kv_heads);
            t.theta_scale = powf(d_.rope_theta, -2.0F / static_cast<float>(d_.head_dim));
            t.rope_factors = layer.rope_factors;
            t.scale = 1.0F / std::sqrt(static_cast<float>(d_.head_dim));

            MatvecArgs o;
            o.w = layer.o.data;
            o.row_bytes = layer.o.row_bytes;
            o.k = static_cast<int>(d_.heads * d_.head_dim);
            o.n = dh;
            o.batch = batch;
            o.x_batches = batch;
            o.attn = t;
            static const bool attn_fused = getenv("TMP_ATTN_FUSED") != nullptr; // TMPDIAG
            if (!attn_fused) {
                if (!skip(kAttn)) {
                    launch(attention_kernel<128>, dim3(batch, static_cast<unsigned>(d_.heads)), dim3(128), 0, stream, t);
                }
                mark(stream, kAttn);
                o.x = ctx_;
                o.x_stride = static_cast<int>(d_.heads * d_.head_dim);
            }
            o.out = x_;
            o.out_stride = dh;
            o.residual = x_;
            o.residual_stride = dh;
            if (!skip(kO)) launch_matvec(attn_fused ? kAttnO : kRoundResid, layer.o.type, o, stream);
            mark(stream, kO);

            MatvecArgs m;
            m.w = layer.gate_up.data;
            m.row_bytes = layer.gate_up.row_bytes;
            m.k = dh;
            m.n = inter;
            m.batch = batch;
            m.x = x_;
            m.x_stride = dh;
            m.x_batches = batch;
            m.norm_w = layer.post_norm;
            m.eps = d_.rms_norm_eps;
            m.out = inter_;
            m.out_stride = inter;
            m.up_offset = inter;
            if (!skip(kGateUp)) launch_matvec(kSwiGlu, layer.gate_up.type, m, stream);
            mark(stream, kGateUp);

            MatvecArgs down;
            down.w = layer.down.data;
            down.row_bytes = layer.down.row_bytes;
            down.k = inter;
            down.n = dh;
            down.batch = batch;
            down.x = inter_;
            down.x_stride = inter;
            down.x_batches = batch;
            down.out = x_;
            down.out_stride = dh;
            down.residual = x_;
            down.residual_stride = dh;
            if (!skip(kDown)) launch_matvec(kRoundResid, layer.down.type, down, stream);
            mark(stream, kDown);
        }
    }

    void embed(cudaStream_t stream, int batch, const int32_t * token, int64_t codebook) {
        MatvecArgs a;
        a.w = d_.projector.data;
        a.row_bytes = d_.projector.row_bytes;
        a.k = static_cast<int>(d_.hidden);
        a.n = static_cast<int>(d_.depth_hidden);
        a.batch = batch;
        a.x_batches = 1;
        a.table = d_.table.data;
        a.table_row_bytes = d_.table.row_bytes;
        a.table_type = d_.table.type;
        a.token = token;
        a.table_base = codebook * d_.vocab;
        a.out = x_;
        a.out_stride = static_cast<int>(d_.depth_hidden);
        launch_matvec(kPlain, d_.projector.type, a, stream);
        mark(stream, kProj);
    }

    void head_and_sample(cudaStream_t stream, const BreezeCudaDepthSampling & s, const float * noise, int64_t step) {
        const int batch = static_cast<int>(s.batch);
        MatvecArgs h;
        h.w = static_cast<const char *>(d_.heads_w.data) + static_cast<size_t>(step) * d_.head_step_bytes;
        h.row_bytes = d_.heads_w.row_bytes;
        h.k = static_cast<int>(d_.depth_hidden);
        h.n = static_cast<int>(d_.codebook_size);
        h.batch = batch;
        h.x = x_;
        h.x_stride = h.k;
        h.x_batches = batch;
        h.norm_w = d_.final_norm;
        h.eps = d_.rms_norm_eps;
        h.norm_round = false;
        h.out = logits_;
        h.out_stride = h.n;
        if (!skip(kHead)) launch_matvec(kPlain, d_.heads_w.type, h, stream);
        mark(stream, kHead);

        SampleArgs a;
        a.logits = logits_;
        a.logits_stride = static_cast<int>(d_.codebook_size);
        a.batch = batch;
        a.guidance = s.guidance;
        a.inv_temperature = 1.0F / s.temperature;
        a.n = static_cast<int>(d_.codebook_size);
        a.candidates = static_cast<int>(s.candidates);
        a.top_p = s.top_p;
        a.noise = noise + step * d_.codebook_size;
        a.token = tokens_ + step;
        if (!skip(kSample)) launch(sample_kernel, dim3(1), dim3(kSampleThreads), 0, stream, a);
        mark(stream, kSample);
    }

    void enqueue(
        cudaStream_t stream,
        const BreezeCudaDepthSampling & s,
        const float * hidden,
        const int32_t * first_token,
        const float * noise) {
        const int batch = static_cast<int>(s.batch);
        // Position 0: the projected backbone hidden state of each row.
        MatvecArgs p;
        p.w = d_.projector.data;
        p.row_bytes = d_.projector.row_bytes;
        p.k = static_cast<int>(d_.hidden);
        p.n = static_cast<int>(d_.depth_hidden);
        p.batch = batch;
        p.x = hidden;
        p.x_stride = p.k;
        p.x_batches = batch;
        p.out = x_;
        p.out_stride = p.n;
        launch_matvec(kPlain, d_.projector.type, p, stream);
        mark(stream, kProj);
        layers(stream, batch, 0);
        // Position 1: the first codebook's token.
        embed(stream, batch, first_token, 0);
        layers(stream, batch, 1);
        head_and_sample(stream, s, noise, 0);
        for (int64_t step = 1; step < d_.steps; ++step) {
            embed(stream, batch, tokens_ + step - 1, step);
            layers(stream, batch, static_cast<int>(step + 1));
            head_and_sample(stream, s, noise, step);
        }
        check(cudaGetLastError(), "kernel launch");
        check(cudaMemcpyAsync(tokens_host_, tokens_, static_cast<size_t>(d_.steps) * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream),
              "copy tokens");
        mark(stream, kSample);
    }

    BreezeCudaDepthDesc d_;
    float * x_ = nullptr;
    float * qkv_ = nullptr;
    float * ctx_ = nullptr;
    float * inter_ = nullptr;
    float * logits_ = nullptr;
    int32_t * tokens_ = nullptr;
    int32_t * tokens_host_ = nullptr;
    std::vector<half *> k_cache_;
    std::vector<half *> v_cache_;
    cudaGraphExec_t graph_exec_ = nullptr;
    BreezeCudaDepthSampling key_;
    const float * hidden_ = nullptr;
    const int32_t * first_token_ = nullptr;
    const float * noise_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // TMPDIAG: TMP_DEPTH_PROF=1 times each kernel class (outside the CUDA graph)
    // and the captured graph itself, then prints per-frame averages.
    enum Cat { kProj, kQkv, kAttn, kO, kGateUp, kDown, kHead, kSample, kCats };
    bool prof_ = getenv("TMP_DEPTH_PROF") != nullptr;
    bool prof_record_ = false;
    std::vector<std::pair<int, cudaEvent_t>> marks_;
    double prof_ms_[kCats] = {0};
    double prof_graph_ms_ = 0.0;
    int64_t prof_frames_ = 0;
    int skip_ = getenv("TMP_DEPTH_SKIP") ? atoi(getenv("TMP_DEPTH_SKIP")) : -1;
    bool skip(int cat) const { return skip_ == cat; }
    void mark(cudaStream_t stream, int cat) {
        if (!prof_record_) return;
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, stream);
        marks_.push_back({cat, e});
    }
    void prof_report() {
        if (prof_frames_ == 0) return;
        static const char * names[kCats] = {"proj/embed", "norm+qkv", "attention", "o+resid", "norm+gateup+swiglu",
                                             "down+resid", "head", "sample"};
        double total = 0;
        for (double v : prof_ms_) total += v;
        fprintf(stderr, "TMPDIAG depth prof frames=%lld graph=%.3fms/frame  eager-sum=%.3fms/frame\n",
                (long long) prof_frames_, prof_graph_ms_ / prof_frames_, total / prof_frames_);
        for (int c = 0; c < kCats; ++c) {
            fprintf(stderr, "TMPDIAG   %-20s %.3fms/frame\n", names[c], prof_ms_[c] / prof_frames_);
        }
    }
};

}  // namespace

std::unique_ptr<BreezeCudaDepth> BreezeCudaDepth::create(const BreezeCudaDepthDesc & desc) {
    return std::make_unique<BreezeCudaDepthImpl>(desc);
}

}  // namespace engine::models::breeze_tts
