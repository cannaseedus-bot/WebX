/**
 * scx_stream_engine.h
 * SCX streaming inference engine — DX12 GPU-first path
 *
 * Architecture:
 *   IO thread     → reads SCX chunks from socket/file/mesh
 *   Decode threads → unpack INT4 tiles into upload heap
 *   GPU worker    → CopyBufferRegion → dispatch int4_decode.hlsl
 *   Infer thread  → runs layer attention when weights are ready
 *
 * Usage:
 *   ScxStreamEngine eng;
 *   eng.init(device, queue);
 *   eng.submitChunk(layer, tileType, data, bytes);   // from IO thread
 *   eng.inferBatch(tokens, batchSz);                 // from infer thread
 */

#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <vector>
#include <string>
#include <functional>
#include <memory>

#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr uint32_t SCX_MAX_LAYERS    = 48;
static constexpr uint32_t SCX_TILE_ALIGN    = 256;       // GPU read alignment
static constexpr uint32_t SCX_CHUNK_SIZE    = 16384;     // 16 KB per chunk
static constexpr uint32_t SCX_UPLOAD_RING   = 8;         // ring slots
static constexpr uint32_t SCX_DECODE_GROUPS = 64;        // HLSL numthreads(64,1,1)

// ── Tile types (Q / K / V / FFN) ─────────────────────────────────────────────

enum class TileType : uint8_t {
    Q   = 0,
    K   = 1,
    V   = 2,
    O   = 3,   // output projection
    FFN1= 4,   // feed-forward up
    FFN2= 5,   // feed-forward down
};

// ── SCX wire header (must match encoder) ─────────────────────────────────────

#pragma pack(push,1)
struct ScxChunkHeader {
    uint8_t  type;       // 0x01=META 0x02=CHUNK 0x03=ACK 0x04=COMPLETE 0x05=ERROR
    uint8_t  flags;      // 0x01=compressed 0x02=SCX 0x04=last-chunk
    uint16_t seq;
    uint32_t len;
    uint8_t  hash[32];   // sha256(payload)
};
static_assert(sizeof(ScxChunkHeader) == 40, "frame header must be 40 bytes");

struct ScxTileMeta {
    uint32_t layer_id;
    uint8_t  tile_type;  // TileType
    uint32_t tile_row;
    uint32_t tile_col;
    uint32_t tile_rows;  // height of this tile
    uint32_t tile_cols;  // width  of this tile
    uint32_t packed_bytes;
};
#pragma pack(pop)

// ── Weight tile (decoded, lives in VRAM) ─────────────────────────────────────

struct WeightTile {
    TileType type;
    uint32_t row, col;
    uint32_t rows, cols;
    uint64_t vram_offset;  // byte offset in weight_buf_[layer]
    bool     ready = false;
};

// ── Layer state ───────────────────────────────────────────────────────────────

struct LayerState {
    std::atomic<int>  tilesLoaded{0};
    std::atomic<int>  tilesTotal{-1};   // set from META frame; -1 = unknown
    std::atomic<bool> ready{false};

    LayerState() = default;
    LayerState(const LayerState& other)
        : tilesLoaded(other.tilesLoaded.load()),
          tilesTotal(other.tilesTotal.load()),
          ready(other.ready.load()) {}
    LayerState& operator=(const LayerState& other) {
        tilesLoaded.store(other.tilesLoaded.load());
        tilesTotal.store(other.tilesTotal.load());
        ready.store(other.ready.load());
        return *this;
    }
    LayerState(LayerState&& other) noexcept
        : tilesLoaded(other.tilesLoaded.load()),
          tilesTotal(other.tilesTotal.load()),
          ready(other.ready.load()) {}
    LayerState& operator=(LayerState&& other) noexcept {
        tilesLoaded.store(other.tilesLoaded.load());
        tilesTotal.store(other.tilesTotal.load());
        ready.store(other.ready.load());
        return *this;
    }

    bool isReady() const {
        return tilesTotal.load() >= 0 &&
               tilesLoaded.load() >= tilesTotal.load();
    }
};

// ── Incoming work item (IO → decode threads) ──────────────────────────────────

struct ChunkWork {
    uint32_t    layer;
    TileType    tileType;
    uint32_t    dtype = 3;       // XSQ2 dtype; 3 = INT4
    uint32_t    tileRow, tileCol;
    uint32_t    tileRows, tileCols;
    std::vector<uint8_t> packed;   // raw INT4 payload
    bool        last = false;
};

// ── Opcode callback (wires to KernelTrace / opcode kernel) ───────────────────

using OpcodeCallback = std::function<void(const std::string& op, const std::string& json)>;

// ── ScxStreamEngine ───────────────────────────────────────────────────────────

class ScxStreamEngine {
public:
    // ── lifecycle ──────────────────────────────────────────────────────────

    /**
     * Attach to an already-initialised DX12 device + queue.
     * Allocates: upload ring, per-layer weight buffers, decode PSO.
     * @param numLayers  transformer depth (e.g. 12 for GPT-2 small)
     * @param hiddenDim  model hidden size (e.g. 768)
     */
    bool init(ID3D12Device* dev, ID3D12CommandQueue* queue,
              uint32_t numLayers = 12, uint32_t hiddenDim = 768);

    /** Shut down threads and release GPU resources. */
    void shutdown();

    // ── stream API (called from IO / network thread) ───────────────────────

    /**
     * Submit a raw SCX chunk.  Thread-safe.
     * Parses ScxChunkHeader, enqueues ChunkWork for decode threads.
     */
    void submitFrame(const uint8_t* frame, size_t frameBytes);

    /**
     * Direct tile submit (already parsed — useful for mesh loader).
     */
    void submitChunk(uint32_t layer, TileType t,
                     uint32_t tileRow, uint32_t tileCol,
                     uint32_t tileRows, uint32_t tileCols,
                     const uint8_t* packed, size_t bytes,
                     bool lastTile = false, uint32_t dtype = 3);

    // ── inference API (called from inference thread) ───────────────────────

    /**
     * Block until layer is ready, then run attention + FFN for one batch.
     * Returns false if engine is shutting down.
     */
    bool inferBatch(uint32_t layer,
                    const float* inputTokens,    // [batchSz × hiddenDim]
                    float* output,               // [batchSz × hiddenDim]
                    uint32_t batchSz);

    /**
     * Generate N tokens autoregressively.
     * Blocks until all layers have streamed in enough for each step.
     */
    bool generate(const uint32_t* promptTokens, uint32_t promptLen,
                  uint32_t* outTokens, uint32_t genLen);

    // ── status ─────────────────────────────────────────────────────────────

    bool  layerReady(uint32_t layer) const;
    float streamProgress() const;   // 0.0 – 1.0 across all layers
    const std::string& lastError() const { return initError_; }
    void setOpcodeCallback(OpcodeCallback cb) { opcodeCallback_ = cb; }

private:
    // ── DX12 handles (not owned) ────────────────────────────────────────────
    ID3D12Device*       dev_   = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;

    // ── config ──────────────────────────────────────────────────────────────
    uint32_t numLayers_  = 12;
    uint32_t hiddenDim_  = 768;
    std::string initError_;

    // ── per-layer state ─────────────────────────────────────────────────────
    std::vector<LayerState>                    layers_;
    std::vector<ComPtr<ID3D12Resource>>        weightBuf_;    // DEFAULT heap, one per layer
    std::vector<D3D12_RESOURCE_STATES>         weightStates_;
    std::vector<std::unique_ptr<std::atomic<uint64_t>>> writeOffset_;
    std::vector<uint8_t>                       decodeShader_;
    std::vector<uint8_t>                       attentionShader_;

    // ── upload ring buffer (CPU UPLOAD heap) ────────────────────────────────
    struct UploadSlot {
        ComPtr<ID3D12Resource> buf;
        void*                  mapped = nullptr;
        std::atomic<bool>      free{true};

        UploadSlot() = default;
        UploadSlot(const UploadSlot& other)
            : buf(other.buf), mapped(other.mapped),
              free(other.free.load()) {}
        UploadSlot& operator=(const UploadSlot& other) {
            buf = other.buf;
            mapped = other.mapped;
            free.store(other.free.load());
            return *this;
        }
        UploadSlot(UploadSlot&& other) noexcept
            : buf(std::move(other.buf)), mapped(other.mapped),
              free(other.free.load()) {
            other.mapped = nullptr;
        }
        UploadSlot& operator=(UploadSlot&& other) noexcept {
            buf = std::move(other.buf);
            mapped = other.mapped;
            free.store(other.free.load());
            other.mapped = nullptr;
            return *this;
        }
    };
    std::vector<UploadSlot> uploadRing_;
    std::vector<D3D12_RESOURCE_STATES> uploadStates_;

    // ── decode pipeline ─────────────────────────────────────────────────────
    ComPtr<ID3D12RootSignature> decodeSig_;
    ComPtr<ID3D12PipelineState> decodePso_;   // int4_decode.hlsl

    // ── attention pipeline ──────────────────────────────────────────────────
    ComPtr<ID3D12RootSignature> attnSig_;
    ComPtr<ID3D12PipelineState> attnPso_;     // scx_attention.hlsl

    // ── per-frame command allocator + list ──────────────────────────────────
    ComPtr<ID3D12CommandAllocator>     cmdAlloc_;
    ComPtr<ID3D12GraphicsCommandList>  cmdList_;
    ComPtr<ID3D12Fence>                fence_;
    HANDLE                             fenceEvent_ = nullptr;
    uint64_t                           fenceVal_   = 0;

    // ── decode work queue ───────────────────────────────────────────────────
    std::mutex              workMu_;
    std::mutex              gpuMu_;
    std::condition_variable workCv_;
    std::queue<ChunkWork>   workQ_;
    std::atomic<bool>       running_{false};

    // ── threads ─────────────────────────────────────────────────────────────
    std::vector<std::thread> decodeThreads_;

    // ── opcode callback ─────────────────────────────────────────────────────
    OpcodeCallback opcodeCallback_;

    // ── internals ───────────────────────────────────────────────────────────
    bool createDecodeRootSigAndPso();
    bool createAttnRootSigAndPso();
    bool loadShaderBytecode(const char* name, std::vector<uint8_t>& bytecode);
    bool allocWeightBuffers();
    bool allocUploadRing();
    void decodeWorker();                        // thread body

    /**
     * Copy packed INT4 data into an upload slot, record GPU commands:
     *   CopyBufferRegion → Dispatch(int4_decode)
     * Caller must have workMu_ released before calling.
     */
    void uploadAndDecode(const ChunkWork& work);

    /** Find a free upload ring slot (spins briefly). */
    int  acquireUploadSlot();
    void releaseUploadSlot(int slot);

    void waitGPU();
    void emitOp(const std::string& op, const std::string& json) const;

    static uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
};

// ── Inline impl: status helpers ───────────────────────────────────────────────

inline bool ScxStreamEngine::layerReady(uint32_t layer) const {
    if (layer >= layers_.size()) return false;
    return layers_[layer].ready.load();
}

inline float ScxStreamEngine::streamProgress() const {
    if (layers_.empty()) return 0.f;
    int total = 0, loaded = 0;
    for (auto& l : layers_) {
        int t = l.tilesTotal.load();
        if (t > 0) { total += t; loaded += std::min(l.tilesLoaded.load(), t); }
    }
    return total > 0 ? float(loaded) / float(total) : 0.f;
}
