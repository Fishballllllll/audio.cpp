#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// SET_ROWS reading its rows from src_unrounded (dst->src[0]'s shape, own strides), each value
// rounded to bf16 as it is read (ROUND_BF16 -> [RESHAPE] -> VIEW -> SET_ROWS).
void ggml_cuda_op_round_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * src_unrounded);
