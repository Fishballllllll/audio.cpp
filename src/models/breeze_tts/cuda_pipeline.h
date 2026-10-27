#pragma once

#include <cstddef>
#include <cstdint>

// Device helpers for pipelined BreezeTTS frames (breeze_tts.pipelined_frames):
// the sampled depth tokens are embedded on the device so the next backbone step
// can be queued right behind the depth graph.
namespace engine::models::breeze_tts {

constexpr int kBreezeFrameEmbedMaxCodebooks = 16;

struct BreezeFrameEmbedArgs {
    const void * table = nullptr;  // device embedding table, rows of `dim` values
    size_t row_bytes = 0;
    int type = 0;                  // ggml_type: GGML_TYPE_F32, GGML_TYPE_F16 or GGML_TYPE_BF16
    int64_t rows = 0;              // num_codebooks * vocab
    int64_t vocab = 0;             // rows per codebook
    int64_t dim = 0;
    const int32_t * tokens[kBreezeFrameEmbedMaxCodebooks] = {};  // one device id per codebook
    int num_codebooks = 0;
    float * out = nullptr;         // [dim, batch]; every row receives the same sum
    int batch = 1;
    int32_t * codes = nullptr;     // [num_codebooks] copy of the ids, for one readback
};

// Sums row codebook * vocab + id of every codebook, in codebook order starting
// from +0: the same float additions as the host frame_embedding().
void breeze_frame_embed(void * stream, const BreezeFrameEmbedArgs & args);

void breeze_copy_device_async(void * stream, void * dst, const void * src, size_t bytes);

}  // namespace engine::models::breeze_tts
