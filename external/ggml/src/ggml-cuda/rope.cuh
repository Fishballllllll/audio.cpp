#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rope_round_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * round_bf16);

// ROUND_BF16(x) -> [RESHAPE] -> ROPE -> ROUND_BF16 in one NEOX rope kernel: reads the
// unrounded x (src_unrounded, rope's input shape with its own strides), rounds each
// value as it is read, and writes the rounded result to out (rope's shape, any strides).
void ggml_cuda_op_round_rope_round_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * rope, const ggml_tensor * src_unrounded,
                                        const ggml_tensor * out);

// The q, k and v chains of a packed QKV projection in one launch: q and k as in
// ggml_cuda_op_round_rope_round_bf16 (k written as f16), and ROUND_BF16(v_x) -> CPY(f16) into
// v_dst. Same per-element expressions as the three separate kernels.
void ggml_cuda_op_round_rope_qkv(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor * q_rope, const ggml_tensor * q_src, const ggml_tensor * q_out,
                                 const ggml_tensor * k_rope, const ggml_tensor * k_src, const ggml_tensor * k_out,
                                 const ggml_tensor * v_x, const ggml_tensor * v_dst);

// ROUND_BF16(src_unrounded) -> RMS_NORM(rms) -> MUL(mul) -> ROUND_BF16 -> [RESHAPE] -> ROPE(NEOX) ->
// ROUND_BF16(out) in one kernel, one row per block; same expressions as the separate kernels.
void ggml_cuda_op_round_rms_norm_rope(ggml_backend_cuda_context & ctx, const ggml_tensor * src_unrounded,
                                      const ggml_tensor * rms, const ggml_tensor * mul, const ggml_tensor * rope,
                                      const ggml_tensor * out);
