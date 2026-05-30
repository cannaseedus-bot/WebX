/**
 * dx12_inference_pipeline.h
 * DX12 compute inference pipeline — multi-layer, pipelined, descriptor-ring.
 *
 * Architecture:
 *   Single root signature for all kernels (SRV table + UAV table + 8 inline constants)
 *   One descriptor heap; each layer-slot has a fixed window of descriptor handles
 *   Per-token command list: embed → QKV → KV write → attn → res → LN → FFN → res → logits
 *   Pipeline parallelism: layer l processes token t while layer l+1 processes t+1
 *
 * Usage:
 *   DX12InferencePipeline pipe;
 *   pipe.init(device, queue, config);
 *   pipe.uploadWeights(layer, mat_type, int4_data, bytes);  // from stream thread
 *   pipe.generateAsync(prompt_tokens, max_new, callback);
 */

#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <array>

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// ── Model config ──────────────────────────────────────────────────────────────

struct InferConfig {
    uint32_t vocab     = 256;
    uint32_t dim       =  64;
    uint32_t n_heads   =   1;
    uint32_t head_dim  =  64;    // dim / n_heads
    uint32_t ffn_dim   = 256;
    uint32_t n_layers  =   1;
    uint32_t max_seq   = 512;
    uint32_t top_k     =  40;
    float    temp      = 0.8f;
};

// ── Weight matrix types ───────────────────────────────────────────────────────

enum class WeightType : uint8_t { Wq=0, Wk, Wv, Wo, W1, W2, Embed, LmHead, COUNT };

// ── Descriptor slot layout per layer (indices into heap) ─────────────────────
// Each layer occupies DESCS_PER_LAYER consecutive descriptors.
// Layout: [X_SRV, Wq_SRV, Wk_SRV, Wv_SRV, Wo_SRV, W1_SRV, W2_SRV, Emb_SRV,
//          Kcache_SRV, Vcache_SRV,          -- SRVs end here
//          X_UAV, Q_UAV, K_UAV, V_UAV, Out_UAV, H_UAV, Logits_UAV]

static constexpr uint32_t SRV_PER_LAYER  = 10;
static constexpr uint32_t UAV_PER_LAYER  =  7;
static constexpr uint32_t DESCS_PER_LAYER= SRV_PER_LAYER + UAV_PER_LAYER;

// ── Inline 32-bit constants layout (b0) ──────────────────────────────────────
// Matches cbuffer in each HLSL kernel

struct KernelConstants {
    uint32_t dim;
    uint32_t ffn_dim;
    uint32_t vocab;
    uint32_t n_heads;
    uint32_t token_id;   // for embed kernel
    uint32_t seq_pos;    // current KV cache write position
    uint32_t layer;
    uint32_t _pad;
};
static_assert(sizeof(KernelConstants) == 32);
static constexpr uint32_t NUM_CONSTANTS = sizeof(KernelConstants) / 4;

// ── Fences — one per layer for pipeline stall ─────────────────────────────────

static constexpr uint32_t MAX_LAYERS = 48;

// ══════════════════════════════════════════════════════════════════════════════

class DX12InferencePipeline {
public:
    using TokenCB = std::function<void(uint32_t tok, const char* text)>;
    using PerfCB  = std::function<void(double tps)>;

    // ── lifecycle ──────────────────────────────────────────────────────────

    /**
     * init — attach to existing DX12 device + queue, allocate all GPU memory.
     * Call once. Thread-safe thereafter for uploadWeights() + generateAsync().
     */
    bool init(ID3D12Device* dev, ID3D12CommandQueue* queue, const InferConfig& cfg);
    void shutdown();

    // ── weight streaming (call from IO/stream thread) ──────────────────────

    /**
     * Upload INT4-packed weight chunk for one matrix.
     * byteOffset: byte position within the layer's weight matrix.
     * Thread-safe — uses a staging ring.
     */
    void uploadWeights(uint32_t layer, WeightType type,
                       const void* int4Data, size_t bytes,
                       uint32_t byteOffset = 0);

    /** Mark a layer fully loaded. After this call, inference can use it. */
    void markLayerReady(uint32_t layer);

    // ── generation (call from inference thread) ────────────────────────────

    /**
     * Non-blocking — launches pipeline threads, streams tokens via cb.
     */
    void generateAsync(const std::vector<uint32_t>& promptTokens,
                       uint32_t maxNew, TokenCB cb, PerfCB perfCb = {});

    void stop() { stopFlag_.store(true); }
    bool isRunning() const { return running_.load(); }

    // ── status ─────────────────────────────────────────────────────────────

    bool layerReady(uint32_t l) const {
        return l < cfg_.n_layers && layerReady_[l].load();
    }
    float weightProgress() const;

private:
    // ── config + device ─────────────────────────────────────────────────────
    InferConfig          cfg_;
    ID3D12Device*        dev_   = nullptr;
    ID3D12CommandQueue*  queue_ = nullptr;

    // ── root signature + PSOs ────────────────────────────────────────────────
    ComPtr<ID3D12RootSignature> rootSig_;
    ComPtr<ID3D12PipelineState> psoEmbed_;   // embed.hlsl
    ComPtr<ID3D12PipelineState> psoMM_;      // mm_int4.hlsl (reused for all matmuls)
    ComPtr<ID3D12PipelineState> psoAttn_;    // attn_kvcache.hlsl
    ComPtr<ID3D12PipelineState> psoGelu_;    // gelu.hlsl (reuse existing)
    ComPtr<ID3D12PipelineState> psoAdd_;     // add.hlsl  (reuse existing)
    ComPtr<ID3D12PipelineState> psoLN_;      // ln.hlsl   (reuse existing)

    // ── descriptor heap ──────────────────────────────────────────────────────
    // Layout: [layer_0 descs | layer_1 descs | ... | layer_N | global descs]
    ComPtr<ID3D12DescriptorHeap> heap_;
    uint32_t                     descSize_   = 0;
    uint32_t                     heapOffset_ = 0;  // next free slot

    D3D12_CPU_DESCRIPTOR_HANDLE CpuHandle(uint32_t idx) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GpuHandle(uint32_t idx) const;

    // ── per-layer GPU resources ───────────────────────────────────────────────
    struct LayerGPURes {
        // Weight buffers (DEFAULT heap, SRV — INT4 packed)
        ComPtr<ID3D12Resource> Wq, Wk, Wv, Wo, W1, W2;
        // LayerNorm params (DEFAULT heap, SRV — float)
        ComPtr<ID3D12Resource> LNgamma1, LNbeta1;     // [dim]
        ComPtr<ID3D12Resource> LNgamma2, LNbeta2;     // [dim]
        // Intermediate float buffers (DEFAULT heap, UAV)
        ComPtr<ID3D12Resource> X, Q, K, V, Out, H;   // [dim] each (H is [ffn_dim])
        // KV cache — persistent across tokens
        ComPtr<ID3D12Resource> Kcache, Vcache;        // [max_seq × dim]
        // Descriptor base in heap_
        uint32_t descBase = 0;
        std::atomic<bool> weightsLoaded{false};
    };
    std::vector<LayerGPURes> layers_;

    // Embedding table + LM head
    ComPtr<ID3D12Resource> embed_;   // [vocab × dim]
    ComPtr<ID3D12Resource> lmHead_; // [dim × vocab]
    ComPtr<ID3D12Resource> logits_; // [vocab]
    ComPtr<ID3D12Resource> logitsReadback_;

    // ── staging ring for weight uploads ──────────────────────────────────────
    static constexpr uint32_t STAGING_SLOTS = 8;
    static constexpr uint32_t STAGING_SZ    = 64 * 1024;  // 64 KB each
    struct StagingSlot {
        ComPtr<ID3D12Resource> buf;
        void*                  mapped = nullptr;
        std::atomic<bool>      free{true};
    };
    std::array<StagingSlot, STAGING_SLOTS> staging_;

    int  acquireStaging();
    void releaseStaging(int slot);

    // ── fence + sync ──────────────────────────────────────────────────────────
    ComPtr<ID3D12Fence>       fence_;
    HANDLE                    fenceEvent_ = nullptr;
    std::atomic<uint64_t>     fenceVal_{0};
    void waitGPU();

    // ── per-token command infrastructure ─────────────────────────────────────
    // Ring of (allocator, list) pairs so GPU and CPU can work concurrently
    static constexpr uint32_t CMD_RING = 4;
    struct CmdFrame {
        ComPtr<ID3D12CommandAllocator>    alloc;
        ComPtr<ID3D12GraphicsCommandList> list;
        uint64_t                          fenceVal = 0;
    };
    std::array<CmdFrame, CMD_RING> cmdRing_;
    uint32_t                       cmdHead_ = 0;

    CmdFrame& nextFrame();

    // ── pipeline state ────────────────────────────────────────────────────────
    std::vector<std::atomic<bool>> layerReady_;
    std::atomic<bool>              running_{false};
    std::atomic<bool>              stopFlag_{false};
    std::thread                    genThread_;

    // ── internal pipeline parallelism scheduler ───────────────────────────────

    /**
     * recordTokenLayer — record GPU commands for layer l, token at seq pos t.
     * Does NOT call Execute — caller batches multiple layers into one submit.
     */
    void recordTokenLayer(ID3D12GraphicsCommandList* cl,
                          uint32_t layer, uint32_t seqPos, uint32_t tokenId);

    /** Dispatch embed kernel: embed[tokenId] → layers_[0].X */
    void recordEmbed(ID3D12GraphicsCommandList* cl, uint32_t tokenId);

    /** Dispatch logits kernel: layers_[last].X → logits_ */
    void recordLogits(ID3D12GraphicsCommandList* cl);

    // ── generation loop (runs on genThread_) ──────────────────────────────────

    void runGenLoop(std::vector<uint32_t> tokens, uint32_t maxNew,
                    TokenCB cb, PerfCB perfCb);

    // ── PSO construction helpers ──────────────────────────────────────────────
    bool buildRootSignature();
    bool buildPSOs();
    bool allocGPUBuffers();
    bool buildDescriptors();
    void writeSRV(uint32_t heapIdx, ID3D12Resource* res, uint32_t elements, uint32_t stride);
    void writeUAV(uint32_t heapIdx, ID3D12Resource* res, uint32_t elements, uint32_t stride);

    ComPtr<ID3D12Resource> makeDefaultBuf(uint64_t bytes, bool uav = true);
    ComPtr<ID3D12Resource> makeUploadBuf(uint64_t bytes);

    // Sampling helpers (run on CPU after logits readback)
    uint32_t    sampleTopK(const std::vector<float>& logits, float temp, uint32_t k);
    static std::string detok(uint32_t tok);

    static uint32_t PK(uint32_t n) { return (n + 7) / 8; }  // INT4 packed uint32 per row
};
