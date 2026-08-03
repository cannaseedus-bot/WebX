/**
 * gpu_model.h
 * GPUModel — transformer inference with real KV cache + token streaming.
 *
 * CPU math baseline that runs immediately.
 * Every compute-heavy function is marked [GPU_HOOK] — swap for DX11/DX12
 * dispatch without touching the generation loop or the WebView2 bridge.
 *
 * Usage (wires directly into cockpit_host.cpp):
 *
 *   GPUModel model;
 *   model.init();
 *   model.setTokenCallback([](uint32_t tok, const char* txt){
 *       JS(L"window.GPT_TOKEN&&window.GPT_TOKEN(" +
 *          std::to_wstring(tok) + L",'" + ... + L"');" );
 *   });
 *   model.generateAsync("Hello world");   // non-blocking, streams via callback
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>

// ── Model dimensions ──────────────────────────────────────────────────────────
// Change these to match a real checkpoint (e.g. GPT-2 small: 768/12/64/3072)

struct ModelConfig {
    uint32_t vocab     = 256;   // byte-level tokenizer for demo
    uint32_t dim       =  64;   // embedding / hidden dim
    uint32_t n_heads   =   1;
    uint32_t head_dim  =  64;   // dim / n_heads
    uint32_t ffn_dim   = 256;   // 4 × dim typical
    uint32_t n_layers  =   1;
    uint32_t max_seq   = 512;
    float    temp      = 0.8f;
    uint32_t top_k     =  40;
};

// ── KV cache ──────────────────────────────────────────────────────────────────

struct KVCache {
    uint32_t           n_layers;
    uint32_t           head_dim;
    uint32_t           max_seq;
    std::vector<float> K;  // [n_layers × max_seq × dim]
    std::vector<float> V;  // same

    void alloc(const ModelConfig& c) {
        n_layers = c.n_layers; head_dim = c.head_dim; max_seq = c.max_seq;
        K.assign(c.n_layers * c.max_seq * c.dim, 0.f);
        V.assign(c.n_layers * c.max_seq * c.dim, 0.f);
    }

    float* Krow(uint32_t layer, uint32_t pos) {
        return K.data() + (layer * max_seq + pos) * head_dim;
    }
    float* Vrow(uint32_t layer, uint32_t pos) {
        return V.data() + (layer * max_seq + pos) * head_dim;
    }
};

// ── Token callback ────────────────────────────────────────────────────────────
// tok  = raw token id
// text = decoded utf-8 fragment (for byte-level: just the char)

using TokenCallback = std::function<void(uint32_t tok, const std::string& text)>;
using PerfCallback  = std::function<void(double tok_per_sec, uint32_t step)>;

// ── GPUModel ──────────────────────────────────────────────────────────────────

class GPUModel {
public:
    explicit GPUModel(ModelConfig cfg = {}) : cfg_(cfg) {}

    // Call once. Initialises weights + KV cache.
    bool init();

    // Set callbacks before calling generate.
    void setTokenCallback(TokenCallback cb) { tokenCb_ = std::move(cb); }
    void setPerfCallback(PerfCallback  cb) { perfCb_  = std::move(cb); }

    // Non-blocking. Launches inference thread; tokens stream via callback.
    void generateAsync(const std::string& prompt, uint32_t maxNewTokens = 256);

    // Blocking version (for tests).
    std::string generateSync(const std::string& prompt, uint32_t maxNewTokens = 128);

    // Stop any in-progress generation.
    void stop() { stopFlag_.store(true); }
    bool isRunning() const { return running_.load(); }

    const ModelConfig& config() const { return cfg_; }

private:
    ModelConfig   cfg_;
    KVCache       kv_;
    TokenCallback tokenCb_;
    PerfCallback  perfCb_;

    std::atomic<bool> running_{false};
    std::atomic<bool> stopFlag_{false};
    std::thread       inferThread_;
    std::mutex        genMu_;  // prevents concurrent generations

    // ── weight tensors (CPU baseline; swap with GPU buffers) ──────────────────
    // Stored as float for readability; replace with INT4-packed vectors
    // and call DX12 dispatch in the [GPU_HOOK] functions below.

    std::vector<float> embed_;        // [vocab × dim]

    // Per-layer weights (n_layers entries each)
    std::vector<std::vector<float>> Wq_, Wk_, Wv_, Wo_;  // [dim × dim]
    std::vector<std::vector<float>> ln1g_, ln1b_, ln2g_, ln2b_; // [dim]
    std::vector<std::vector<float>> W1_, W2_;   // dim×ffn, ffn×dim

    std::vector<float> lmHead_;   // [dim × vocab]

    // ── internal helpers ──────────────────────────────────────────────────────

    void runGenLoop(std::vector<uint32_t> tokens, uint32_t maxNew);
    void runLayer(uint32_t layer, float* x, uint32_t pos);

    // [GPU_HOOK] — replace body with DX12 CSQProj dispatch
    void matmul(const float* A, const float* B, float* C,
                uint32_t M, uint32_t K, uint32_t N);

    // [GPU_HOOK] — replace with gpt2_layernorm_fwd.hlsl CSMain dispatch
    void layerNorm(float* x, const float* g, const float* b, uint32_t n);

    // [GPU_HOOK] — replace with gpt2_attn_fwd.hlsl CSMain dispatch
    void attention(float* q, uint32_t layer, uint32_t pos, float* out);

    // [GPU_HOOK] — replace with gpt2_gelu_fwd.hlsl CSMain dispatch
    void gelu(float* x, uint32_t n);

    // Sampling
    uint32_t sampleTopK(const float* logits, uint32_t n, float temp, uint32_t k);
    static float dot(const float* a, const float* b, uint32_t n);
    static void  softmax(float* x, uint32_t n);

    // Simple byte-level detokenizer
    static std::string detok(uint32_t tok) {
        if (tok >= 32 && tok < 127) return std::string(1, char(tok));
        if (tok == '\n') return "\n";
        return "";
    }
};
