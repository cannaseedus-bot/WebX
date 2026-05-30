/**
 * speculative_decoder.h
 * Speculative decoding fused into the DX12 multi-layer pipeline.
 *
 * Architecture:
 *   draft_  — small DX12InferencePipeline (e.g., 1-layer, dim=64)
 *   main_   — large DX12InferencePipeline (e.g., n-layer, dim=768)
 *
 *   Each step:
 *     1. Draft generates K candidate tokens autoregressively (fast)
 *     2. Main model runs ONE pass that verifies all K positions in parallel
 *     3. Accept the longest prefix where draft matches main's distribution
 *     4. If all K accepted: bonus token from main's last position
 *     5. If some rejected: fallback to main's correction token at reject point
 *
 *   KV management:
 *     - draft KV: kept in draft_ pipeline, rolled back on rejection
 *     - main KV:  committed only for accepted tokens
 *
 * Usage:
 *   SpeculativeDecoder dec;
 *   dec.init(device, queue, draftCfg, mainCfg);
 *   dec.uploadDraftWeights(...);
 *   dec.uploadMainWeights(...);
 *   dec.markDraftReady(); dec.markMainReady();
 *   dec.generateAsync(promptTokens, maxNew, tokenCb, perfCb);
 */

#pragma once

#include "dx12_inference_pipeline.h"
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>

// ── Speculative decoder config ────────────────────────────────────────────────

struct SpecConfig {
    InferConfig draft;          // small / fast model
    InferConfig main;           // large / accurate model
    uint32_t    K       = 4;    // draft tokens per speculation step
    float       threshold = 1.f; // acceptance threshold (1.0 = strict, <1 = lenient)
};

// ─────────────────────────────────────────────────────────────────────────────

class SpeculativeDecoder {
public:
    using TokenCB = DX12InferencePipeline::TokenCB;
    using PerfCB  = DX12InferencePipeline::PerfCB;

    // ── lifecycle ──────────────────────────────────────────────────────────

    /**
     * Initialise both pipelines on the same device + queue.
     * Passing the same queue causes GPU-side serialisation; for true
     * parallelism provide separate compute queues (optional).
     */
    bool init(ID3D12Device* dev,
              ID3D12CommandQueue* draftQueue,
              ID3D12CommandQueue* mainQueue,
              const SpecConfig& cfg);

    void shutdown();

    // ── weight loading ─────────────────────────────────────────────────────

    void uploadDraftWeights(uint32_t layer, WeightType type,
                            const void* data, size_t bytes,
                            uint32_t byteOffset = 0);

    void uploadMainWeights(uint32_t layer, WeightType type,
                           const void* data, size_t bytes,
                           uint32_t byteOffset = 0);

    void markDraftLayerReady(uint32_t layer) { draft_.markLayerReady(layer); }
    void markMainLayerReady(uint32_t layer)  { main_.markLayerReady(layer);  }

    // Mark all layers ready at once
    void markDraftReady() {
        for (uint32_t l = 0; l < cfg_.draft.n_layers; ++l)
            draft_.markLayerReady(l);
    }
    void markMainReady() {
        for (uint32_t l = 0; l < cfg_.main.n_layers; ++l)
            main_.markLayerReady(l);
    }

    // ── generation ─────────────────────────────────────────────────────────

    /**
     * Non-blocking — launches scheduler thread, streams tokens via cb.
     * perfCb receives (tokens/sec) every 8 accepted tokens.
     */
    void generateAsync(const std::vector<uint32_t>& promptTokens,
                       uint32_t maxNew, TokenCB cb, PerfCB perfCb = {});

    void stop()       { stopFlag_.store(true); draft_.stop(); main_.stop(); }
    bool isRunning()  const { return running_.load(); }

    float draftProgress() const { return draft_.weightProgress(); }
    float mainProgress()  const { return main_.weightProgress();  }

private:
    SpecConfig              cfg_;
    DX12InferencePipeline   draft_;
    DX12InferencePipeline   main_;

    std::atomic<bool>       running_{false};
    std::atomic<bool>       stopFlag_{false};
    std::thread             schedThread_;

    // ── scheduler (runs on schedThread_) ───────────────────────────────────

    void runScheduler(std::vector<uint32_t> tokens,
                      uint32_t maxNew, TokenCB cb, PerfCB perfCb);

    /**
     * Draft step: generate K tokens from current context.
     * Returns the K candidate token ids and their per-token probabilities
     * under the draft distribution p(tok | draft).
     */
    struct DraftResult {
        std::vector<uint32_t> tokens;   // K draft token ids
        std::vector<float>    probs;    // p_draft[i] for each candidate
    };
    DraftResult draftGenerate(const std::vector<uint32_t>& ctx, uint32_t K);

    /**
     * Verify step: run main model on ctx + draft_tokens (K positions).
     * Returns the main model's probability for each draft token AND
     * the main model's own greedy / sampled token at each position.
     *
     * Internally this runs ONE forward pass of the main model over the
     * full context + K speculative tokens and reads back K+1 logit vectors.
     */
    struct VerifyResult {
        std::vector<float>    p_main;       // p_main[i] = main prob of draft_tok[i]
        std::vector<uint32_t> main_tokens;  // main model's token at each position
    };
    VerifyResult verifyTokens(const std::vector<uint32_t>& ctx,
                              const DraftResult& draft);

    /**
     * Acceptance step: compare draft and main probabilities.
     * Returns the number of accepted tokens (0..K).
     * Uses the speculative sampling acceptance criterion:
     *   accept token i if uniform() < min(1, p_main[i] / p_draft[i])
     */
    uint32_t acceptTokens(const DraftResult& draft,
                          const VerifyResult& verify);

    // Synchronous single-token generation helpers (used internally)
    uint32_t draftStep(const std::vector<uint32_t>& ctx);
    uint32_t mainStep(const std::vector<uint32_t>& ctx);

    // Sample from a flat logit vector
    static uint32_t sampleFromLogits(const std::vector<float>& logits,
                                     float temp, uint32_t top_k);
    static float    probOf(const std::vector<float>& logits,
                           uint32_t tok, float temp);
};
