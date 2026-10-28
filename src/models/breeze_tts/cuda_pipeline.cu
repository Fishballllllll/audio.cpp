#include "cuda_pipeline.h"

#include "ggml.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace engine::models::breeze_tts {
namespace {

void check(cudaError_t status, const char * what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("BreezeTTS pipelined frame: ") + what + ": " + cudaGetErrorString(status));
    }
}

template <int Type>
__device__ __forceinline__ float load_value(const char * row, int64_t i) {
    if constexpr (Type == GGML_TYPE_F32) {
        return reinterpret_cast<const float *>(row)[i];
    } else if constexpr (Type == GGML_TYPE_F16) {
        return __half2float(reinterpret_cast<const __half *>(row)[i]);
    } else {
        // bf16 -> f32 is exact: the bf16 bits are the high half of the f32.
        const uint16_t bits = reinterpret_cast<const uint16_t *>(row)[i];
        return __uint_as_float(static_cast<uint32_t>(bits) << 16);
    }
}

template <int Type>
__global__ void frame_embed_kernel(BreezeFrameEmbedArgs args) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 && threadIdx.x < args.num_codebooks) {
        args.codes[threadIdx.x] = *args.tokens[threadIdx.x];
    }
    if (i >= args.dim) {
        return;
    }
    const char * table = static_cast<const char *>(args.table);
    float sum = 0.0F;
    for (int codebook = 0; codebook < args.num_codebooks; ++codebook) {
        const int64_t row = codebook * args.vocab + *args.tokens[codebook];
        // An out-of-range id is rejected by the host after the codes are read back.
        if (row < 0 || row >= args.rows) {
            continue;
        }
        sum = __fadd_rn(sum, load_value<Type>(table + row * args.row_bytes, i));
    }
    for (int b = 0; b < args.batch; ++b) {
        args.out[static_cast<int64_t>(b) * args.dim + i] = sum;
    }
}

}  // namespace

void breeze_frame_embed(void * stream, const BreezeFrameEmbedArgs & args) {
    if (args.num_codebooks <= 0 || args.num_codebooks > kBreezeFrameEmbedMaxCodebooks || args.dim <= 0 ||
        args.batch <= 0 || args.table == nullptr || args.out == nullptr || args.codes == nullptr) {
        throw std::runtime_error("BreezeTTS pipelined frame: invalid embedding arguments");
    }
    constexpr int threads = 256;
    const dim3 blocks(static_cast<unsigned>((args.dim + threads - 1) / threads));
    auto * s = static_cast<cudaStream_t>(stream);
    switch (args.type) {
        case GGML_TYPE_F32: frame_embed_kernel<GGML_TYPE_F32><<<blocks, threads, 0, s>>>(args); break;
        case GGML_TYPE_F16: frame_embed_kernel<GGML_TYPE_F16><<<blocks, threads, 0, s>>>(args); break;
        case GGML_TYPE_BF16: frame_embed_kernel<GGML_TYPE_BF16><<<blocks, threads, 0, s>>>(args); break;
        default: throw std::runtime_error("BreezeTTS pipelined frame: unsupported embedding table type");
    }
    check(cudaGetLastError(), "embedding kernel launch");
}

void breeze_copy_device_async(void * stream, void * dst, const void * src, size_t bytes) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "device copy");
}

namespace {

__global__ void fetch_kernel(uint4 * dst, const uint4 * src, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // Fetch again every time: an earlier frame read the same host memory.
        dst[i] = __ldcv(src + i);
    }
}

}  // namespace

void * breeze_device_alias(void * host) {
    void * device = nullptr;
    if (host == nullptr || cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return device;
}

void breeze_fetch_async(void * stream, void * dst, const void * src_alias, size_t bytes) {
    if (bytes % sizeof(uint4) != 0 || reinterpret_cast<uintptr_t>(dst) % sizeof(uint4) != 0 ||
        reinterpret_cast<uintptr_t>(src_alias) % sizeof(uint4) != 0) {
        throw std::runtime_error("BreezeTTS pipelined frame: fetch needs 16-byte aligned sizes and addresses");
    }
    const size_t count = bytes / sizeof(uint4);
    constexpr int threads = 256;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count + threads - 1) / threads, 64));
    fetch_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint4 *>(dst), static_cast<const uint4 *>(src_alias), count);
    check(cudaGetLastError(), "fetch launch");
}

namespace {

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void gate_kernel(volatile unsigned long long * words, unsigned int ticket, int32_t * token_dst) {
    const unsigned long long start = global_ns();
    unsigned long long word = words[0];
    // Tickets only grow; compare as a signed difference so a wrap is harmless.
    while (static_cast<int>(static_cast<unsigned int>(word) - ticket) < 0) {
        if (global_ns() - start > 10ull * 1000 * 1000 * 1000) {
            // Token 0 is in range, so the queued graph still runs safely.
            *token_dst = 0;
            words[1] = 1;
            return;
        }
        __nanosleep(200);
        word = words[0];
    }
    *token_dst = static_cast<int32_t>(static_cast<unsigned int>(word >> 32));
}

}  // namespace

bool breeze_gate_create(BreezeGate & gate) {
    gate = {};
    void * host = nullptr;
    if (cudaHostAlloc(&host, 2 * sizeof(unsigned long long), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    void * device = nullptr;
    if (cudaHostGetDevicePointer(&device, host, 0) != cudaSuccess || device == nullptr) {
        cudaGetLastError();
        cudaFreeHost(host);
        return false;
    }
    gate.host = static_cast<unsigned long long *>(host);
    gate.device = static_cast<unsigned long long *>(device);
    gate.host[0] = 0;
    gate.host[1] = 0;
    return true;
}

void breeze_gate_destroy(BreezeGate & gate) {
    if (gate.host != nullptr) {
        cudaFreeHost(gate.host);
    }
    gate = {};
}

void breeze_gate_wait(void * stream, const BreezeGate & gate, unsigned int ticket, int32_t * token_dst) {
    gate_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(gate.device, ticket, token_dst);
    check(cudaGetLastError(), "gate launch");
}

void breeze_gate_release(const BreezeGate & gate, unsigned int ticket, int32_t token) {
    // One 64-bit store: the device sees the ticket and its token together.
    const unsigned long long word =
        (static_cast<unsigned long long>(static_cast<unsigned int>(token)) << 32) | ticket;
    __atomic_store_n(gate.host, word, __ATOMIC_RELEASE);
}

bool breeze_gate_failed(const BreezeGate & gate) {
    // Reading clears the flag, so one failure does not fail every later request.
    return __atomic_exchange_n(gate.host + 1, 0ull, __ATOMIC_ACQ_REL) != 0;
}

void * breeze_event_create() {
    cudaEvent_t event = nullptr;
    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event create");
    return event;
}

void breeze_event_destroy(void * event) {
    if (event != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(event));
    }
}

void breeze_event_record(void * event, void * stream) {
    check(cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)), "event record");
}

void breeze_event_synchronize(void * event) {
    check(cudaEventSynchronize(static_cast<cudaEvent_t>(event)), "event synchronize");
}

}  // namespace engine::models::breeze_tts
