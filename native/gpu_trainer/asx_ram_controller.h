// asx_ram_controller.h
// ASX RAM Controller — slot types and GPU window interface
//
// Three-plane model:
//   [C] Loading   — disk → CPU buffer  (future: background thread)
//   [B] Ready     — CPU buffer filled, waiting for upload slot
//   [A] Computing — data on GPU, shader executing
//
// The GPU window is a fixed set of D3D11 buffers. Tiles stream through it;
// the GPU never holds more than one head's Q/K/V at a time.

#pragma once
#include "d3d11_engine.h"
#include <cstdint>
#include <vector>
#include <string>

// ── slot state ────────────────────────────────────────────────────────────────

enum class SlotPhase : uint8_t {
    Empty     = 0,   // slot unused
    Loading   = 1,   // [C] disk read in progress
    Ready     = 2,   // [B] CPU data valid, waiting to upload
    Computing = 3,   // [A] data on GPU, dispatch issued
};

struct TileKey {
    uint32_t layer_id = 0xFFFFFFFF;
    uint32_t head_id  = 0xFFFFFFFF;
    bool valid() const { return layer_id != 0xFFFFFFFF; }
    bool operator==(const TileKey& o) const {
        return layer_id == o.layer_id && head_id == o.head_id;
    }
};

// CPU-side slot: holds Q/K/V tile data for one attention head
struct TileSlot {
    SlotPhase          phase    = SlotPhase::Empty;
    TileKey            key      = {};
    float              load_ms  = 0.f;

    // 3 × 256KB = 768KB per slot
    std::vector<float> q;   // [65536] = [64 rows × 1024 cols]
    std::vector<float> k;   // [65536]
    std::vector<float> v;   // [65536]

    void alloc() {
        q.resize(65536); k.resize(65536); v.resize(65536);
    }
    void reset() {
        phase = SlotPhase::Empty; key = {};
    }
};

// Priority score — higher = more urgent to keep resident / compute next
// Used for LRU eviction ordering (full LRU cache is a future gate)
inline float tile_priority(uint32_t layer_id, uint32_t head_id,
                            uint32_t current_layer, uint32_t lru_age_ms) {
    float layer_dist = (float)(layer_id >= current_layer
                               ? layer_id - current_layer
                               : current_layer - layer_id);
    float proximity  = 1.f / (1.f + layer_dist);
    float freshness  = 1.f / (1.f + (float)lru_age_ms * 0.001f);
    (void)head_id;
    return proximity * freshness;
}

// ── GPU window ────────────────────────────────────────────────────────────────
// Fixed D3D11 resources. Tiles stream through this window — never all resident.

struct GPUWindow {
    // tile inputs (Q/K/V) — 256KB each
    ComPtr<ID3D11Buffer>              qBuf, kBuf, vBuf;
    ComPtr<ID3D11ShaderResourceView>  q_srv, k_srv, v_srv;

    // softmax intermediate P[64,64] = 16KB
    ComPtr<ID3D11Buffer>              pBuf;
    ComPtr<ID3D11ShaderResourceView>  p_srv;
    ComPtr<ID3D11UnorderedAccessView> p_uav;

    // context output [64,1024] = 256KB + staging
    ComPtr<ID3D11Buffer>              ctxBuf, ctxStaging;
    ComPtr<ID3D11UnorderedAccessView> ctx_uav;

    // shaders
    ComPtr<ID3D11ComputeShader>       softmax_cs;
    ComPtr<ID3D11ComputeShader>       vmul_cs;

    bool  init(ID3D11Device* dev, const char* softmax_hlsl, const char* vmul_hlsl);
    void  upload(ID3D11DeviceContext* ctx, const TileSlot& slot);
    void  dispatch(ID3D11DeviceContext* ctx);
    bool  readback(ID3D11DeviceContext* ctx, float* out_65536);  // false = device lost
};
