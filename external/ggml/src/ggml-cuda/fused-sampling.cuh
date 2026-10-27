#pragma once

#include "common.cuh"

// Fused tails of an on-device sampling graph. Each kernel evaluates the same
// per-element expressions, in the same order, as the unfused nodes, so the result
// is bit-identical to running them one by one.

// SUB(c, u) -> SCALE(s1, b1) -> ADD(u, .) -> SCALE(s2, b2): classifier-free guidance
// mix followed by the temperature scale. u_first: the ADD reads u as src0.
void ggml_cuda_op_cfg_scale_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * cond, const ggml_tensor * uncond,
                                  float s1, float b1, bool u_first, float s2, float b2, ggml_tensor * dst);

// GET_ROWS(scores, order) + GET_ROWS(noise, order) -> ADD -> ARGMAX -> GET_ROWS(order, .):
// picks order[argmax_c(scores[order[c]] + noise[order[c]])] with ARGMAX's own scan and
// reduction order. scores/noise are [1, N] f32 rows read with their row stride.
void ggml_cuda_op_pick_top_candidate(ggml_backend_cuda_context & ctx, const ggml_tensor * scores, const ggml_tensor * noise,
                                     const int32_t * order, int64_t candidates, bool values_first, ggml_tensor * token);
