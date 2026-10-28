#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_channels(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_channels_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_channels_add_bias_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// round_bf16_tensor: optional trailing GGML_UNARY_OP_ROUND_BF16 node that receives the rounded result.
// ROUND(res_x) -> ADD(res_other) -> ROUND = residual, then RMS_NORM(residual) -> MUL -> ROUND, in one
// kernel; the residual is still stored (later layers read it).
void ggml_cuda_op_round_add_round_rms_norm_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * res_x,
                                                 const ggml_tensor * res_other, bool x_first, ggml_tensor * residual,
                                                 ggml_tensor * rms, ggml_tensor * mul_tensor, ggml_tensor * round_tensor,
                                                 void * q8 = nullptr);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 ggml_tensor * round_bf16_tensor = nullptr, void * q8 = nullptr);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
