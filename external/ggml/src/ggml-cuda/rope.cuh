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
