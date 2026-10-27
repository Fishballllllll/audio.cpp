#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Hand-written CUDA implementation of the BreezeTTS depth decoder frame
// (breeze_tts.exec_plan depth.impl=cuda). It reads the weights the ggml
// backend already holds on the device and replaces the fused depth graph with
// a fixed sequence of fused kernels that is captured once as a CUDA graph.
namespace engine::models::breeze_tts {

struct BreezeCudaDepthMatrix {
    const void * data = nullptr;  // device pointer, rows of `cols` elements
    size_t row_bytes = 0;
    int type = 0;                 // ggml_type
    int64_t rows = 0;
    int64_t cols = 0;
};

struct BreezeCudaDepthLayer {
    const float * input_norm = nullptr;
    const float * post_norm = nullptr;
    const float * rope_factors = nullptr;  // optional [head_dim / 2] RoPE divisors
    BreezeCudaDepthMatrix qkv;
    BreezeCudaDepthMatrix o;
    BreezeCudaDepthMatrix gate_up;
    BreezeCudaDepthMatrix down;
};

struct BreezeCudaDepthDesc {
    int64_t hidden = 0;        // backbone hidden size (projector input)
    int64_t depth_hidden = 0;
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate = 0;
    int64_t vocab = 0;         // rows per codebook in the embedding table and heads
    int64_t codebook_size = 0; // sampled ids (reserved ids above are never drawn)
    int64_t steps = 0;         // codebooks produced per frame
    float rms_norm_eps = 0.0F;
    float rope_theta = 0.0F;
    BreezeCudaDepthMatrix projector;
    BreezeCudaDepthMatrix table;   // [num_codebooks * vocab, hidden]
    BreezeCudaDepthMatrix heads_w; // rows: vocab per step, `head_step_bytes` apart
    size_t head_step_bytes = 0;
    const float * final_norm = nullptr;
    std::vector<BreezeCudaDepthLayer> layers;
};

struct BreezeCudaDepthSampling {
    int64_t batch = 1;  // 2 when the conditional and unconditional rows are paired
    float guidance = 1.0F;
    float temperature = 1.0F;
    int64_t candidates = 0;
    float top_p = 1.0F;

    bool operator==(const BreezeCudaDepthSampling & other) const {
        return batch == other.batch && guidance == other.guidance && temperature == other.temperature &&
               candidates == other.candidates && top_p == other.top_p;
    }
};

class BreezeCudaDepth {
public:
    // Throws when a weight type or shape is not supported; callers fall back to ggml.
    static std::unique_ptr<BreezeCudaDepth> create(const BreezeCudaDepthDesc & desc);
    virtual ~BreezeCudaDepth() = default;

    // hidden: device [batch, hidden] f32; first_token: device int32; noise: device
    // [steps, codebook_size] f32. Runs on `stream` and returns the sampled ids.
    virtual std::vector<int32_t> run(
        void * stream,
        const BreezeCudaDepthSampling & sampling,
        const float * hidden,
        const int32_t * first_token,
        const float * noise) = 0;
};

}  // namespace engine::models::breeze_tts
