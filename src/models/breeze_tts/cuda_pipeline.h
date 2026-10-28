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

// A gate lets the host queue a graph before its first token is known: the device
// waits on a host-mapped word, so the costly graph launch overlaps the previous
// frame instead of sitting between the host's sampling and the device.
struct BreezeGate {
    unsigned long long * host = nullptr;    // [0]: (token << 32) | ticket, [1]: error flag
    unsigned long long * device = nullptr;  // device alias of host
};

// Returns false (gate left empty) if the device cannot map pinned host memory.
bool breeze_gate_create(BreezeGate & gate);
void breeze_gate_destroy(BreezeGate & gate);
// Queues a wait until the host releases `ticket`, then writes the released token
// to token_dst. If no release arrives within ~10 s the wait gives up and sets the
// error flag, so a stuck host shows up as an error instead of a hang.
void breeze_gate_wait(void * stream, const BreezeGate & gate, unsigned int ticket, int32_t * token_dst);
void breeze_gate_release(const BreezeGate & gate, unsigned int ticket, int32_t token);
bool breeze_gate_failed(const BreezeGate & gate);

void * breeze_event_create();
void breeze_event_destroy(void * event);
void breeze_event_record(void * event, void * stream);
void breeze_event_synchronize(void * event);

}  // namespace engine::models::breeze_tts
