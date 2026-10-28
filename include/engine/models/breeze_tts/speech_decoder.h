#pragma once

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/breeze_tts/assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models {

namespace breeze_tts {

struct BreezeSpeechCodes {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    int64_t code_groups = 0;
};

struct BreezeSpeechDecoderWeights;
class BreezeSpeechDecoderGraph;

class BreezeSpeechDecoderRuntime {
public:
    BreezeSpeechDecoderRuntime(
        std::shared_ptr<const BreezeTTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type,
        core::AttentionPreference attention_preference = core::AttentionPreference::Auto);
    ~BreezeSpeechDecoderRuntime();

    runtime::AudioBuffer decode(const BreezeSpeechCodes & codec_codes) const;
    static constexpr int64_t kLeftContextFrames = 25;
    runtime::AudioBuffer decode_stream_chunk(
        const BreezeSpeechCodes & codec_codes,
        int64_t context_frames,
        int64_t new_frames,
        int64_t frames_per_chunk) const;
    // decode_stream_chunk() split in two, so decoding overlaps the caller: queue
    // a chunk on the decoder's backend, then collect its audio later (at most one
    // chunk is queued; collecting with none queued returns empty audio).
    void enqueue_stream_chunk(
        const BreezeSpeechCodes & codec_codes,
        int64_t context_frames,
        int64_t new_frames,
        int64_t frames_per_chunk) const;
    runtime::AudioBuffer collect_stream_chunk() const;
    bool stream_chunk_queued() const { return queued_graph_ != nullptr; }
    // Waits for and drops a queued chunk.
    void discard_stream_chunk() const;
    runtime::AudioBuffer decode_and_trim_reference(
        const BreezeSpeechCodes & reference_codes,
        const BreezeSpeechCodes & generated_codes) const;
    void release_runtime_graphs() const;
    void release_weights();
    void restore_weights();

private:
    std::vector<float> decode_window_samples(
        const std::vector<int32_t> & chunk,
        int64_t chunk_frames,
        int64_t context_frames) const;

    std::shared_ptr<const BreezeTTSAssets> assets_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<const BreezeSpeechDecoderWeights> weights_;
    size_t graph_arena_bytes_ = 0;
    bool allow_flash_attention_ = true;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    mutable std::unique_ptr<BreezeSpeechDecoderGraph> graph_;
    mutable std::unique_ptr<BreezeSpeechDecoderGraph> stream_initial_graph_;
    mutable std::unique_ptr<BreezeSpeechDecoderGraph> stream_graph_;
    // Always present to keep this public class layout identical when the private
    // Strix Halo compile definition differs between translation units.
    mutable std::array<std::unique_ptr<BreezeSpeechDecoderGraph>, 2> optimized_graphs_;
    // The chunk queued by enqueue_stream_chunk(), if any.
    mutable BreezeSpeechDecoderGraph * queued_graph_ = nullptr;
    mutable int64_t queued_context_frames_ = 0;
    mutable int64_t queued_new_frames_ = 0;
};

}  // namespace breeze_tts
}  // namespace engine::models
