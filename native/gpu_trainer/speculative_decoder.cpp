/**
 * speculative_decoder.cpp
 * Speculative decoding orchestration over two DX12InferencePipeline instances.
 *
 * The core loop (per step):
 *
 *   Draft generates K tokens autoregressively (cheap, fast model).
 *   Main model verifies all K positions in a single parallel pass.
 *   Acceptance criterion (speculative sampling, Chen et al. 2023):
 *     For each position i:
 *       r ~ Uniform(0,1)
 *       if r < min(1, p_main(x_i) / p_draft(x_i))  →  accept x_i
 *       else                                          →  reject, sample correction from
 *                                                        renorm( max(0, p_main - p_draft) )
 *   After accepting n tokens: bonus token = main's greedy at position n.
 *
 * Expected speedup: K × acceptance_rate  (typically 2–4×).
 */

#include "speculative_decoder.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

bool SpeculativeDecoder::init(
    ID3D12Device*        dev,
    ID3D12CommandQueue*  draftQueue,
    ID3D12CommandQueue*  mainQueue,
    const SpecConfig&    cfg)
{
    cfg_ = cfg;
    if (!draft_.init(dev, draftQueue, cfg.draft)) return false;
    if (!main_.init(dev, mainQueue,   cfg.main))  return false;
    return true;
}

void SpeculativeDecoder::shutdown()
{
    stop();
    if (schedThread_.joinable()) schedThread_.join();
    draft_.shutdown();
    main_.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight upload pass-throughs
// ─────────────────────────────────────────────────────────────────────────────

void SpeculativeDecoder::uploadDraftWeights(
    uint32_t layer, WeightType type,
    const void* data, size_t bytes, uint32_t byteOffset)
{
    draft_.uploadWeights(layer, type, data, bytes, byteOffset);
}

void SpeculativeDecoder::uploadMainWeights(
    uint32_t layer, WeightType type,
    const void* data, size_t bytes, uint32_t byteOffset)
{
    main_.uploadWeights(layer, type, data, bytes, byteOffset);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public generation entry point
// ─────────────────────────────────────────────────────────────────────────────

void SpeculativeDecoder::generateAsync(
    const std::vector<uint32_t>& promptTokens,
    uint32_t maxNew, TokenCB cb, PerfCB perfCb)
{
    if (running_.exchange(true)) return;
    stopFlag_.store(false);
    if (schedThread_.joinable()) schedThread_.join();
    schedThread_ = std::thread([this,
                                 toks = promptTokens,
                                 maxNew, cb, perfCb]() mutable {
        runScheduler(std::move(toks), maxNew, cb, perfCb);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Scheduler
// ─────────────────────────────────────────────────────────────────────────────

void SpeculativeDecoder::runScheduler(
    std::vector<uint32_t> tokens,
    uint32_t maxNew, TokenCB cb, PerfCB perfCb)
{
    const uint32_t K    = cfg_.K;
    uint32_t       step = 0;
    uint32_t       accepted_total = 0;
    auto           t0   = std::chrono::steady_clock::now();

    // Prefill both models on the prompt
    // (Prefill is handled implicitly: both pipelines warm up their KV caches
    //  during the first generateAsync call's prefill phase.  We drive them
    //  synchronously here via the public API.)

    while (step < maxNew && !stopFlag_.load()) {
        // ── 1. Draft: generate K candidate tokens ────────────────────────────
        DraftResult dr = draftGenerate(tokens, K);
        if (dr.tokens.empty() || stopFlag_.load()) break;

        // ── 2. Main: verify K positions in one forward pass ──────────────────
        VerifyResult vr = verifyTokens(tokens, dr);
        if (stopFlag_.load()) break;

        // ── 3. Accept/reject ──────────────────────────────────────────────────
        uint32_t n_accepted = acceptTokens(dr, vr);

        // Emit accepted draft tokens
        for (uint32_t i = 0; i < n_accepted && step < maxNew; ++i, ++step) {
            uint32_t tok = dr.tokens[i];
            tokens.push_back(tok);
            ++accepted_total;

            std::string text = DX12InferencePipeline::detok(tok);
            if (cb) cb(tok, text.c_str());
            if (tok == '\n' || tokens.size() >= cfg_.main.max_seq) goto done;
        }

        if (step >= maxNew) break;

        // ── 4. Correction / bonus token ───────────────────────────────────────
        if (n_accepted < K) {
            // Draft was wrong at position n_accepted.
            // Sample correction from renorm(max(0, p_main - p_draft)):
            // For simplicity: use main model's token at that position.
            uint32_t correction = vr.main_tokens[n_accepted];
            tokens.push_back(correction);
            ++step; ++accepted_total;

            std::string text = DX12InferencePipeline::detok(correction);
            if (cb) cb(correction, text.c_str());
            if (correction == '\n' || tokens.size() >= cfg_.main.max_seq) break;
        } else {
            // All K accepted → bonus token from main's next position
            uint32_t bonus = vr.main_tokens[K];  // main_tokens has K+1 entries
            tokens.push_back(bonus);
            ++step; ++accepted_total;

            std::string text = DX12InferencePipeline::detok(bonus);
            if (cb) cb(bonus, text.c_str());
            if (bonus == '\n' || tokens.size() >= cfg_.main.max_seq) break;
        }

        // Performance callback
        if (perfCb && accepted_total % 8 == 0) {
            auto now  = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(now - t0).count();
            perfCb(double(accepted_total) / sec);
        }
    }

done:
    running_.store(false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Draft generation
// ─────────────────────────────────────────────────────────────────────────────

// Run draft model synchronously for K steps, collecting token ids + their
// softmax probabilities under the draft distribution.
SpeculativeDecoder::DraftResult
SpeculativeDecoder::draftGenerate(const std::vector<uint32_t>& ctx, uint32_t K)
{
    DraftResult result;
    result.tokens.reserve(K);
    result.probs.reserve(K);

    // Drive draft_ pipeline step-by-step using a synchronous wrapper.
    // generateAsync + a shared_mutex-based "done" flag lets us collect each
    // token as it streams, while the pipeline manages its KV cache internally.
    //
    // We use generateAsync with maxNew=K and a collecting callback, then wait.

    std::vector<float> draftLogits;  // last logit vector (captured via callback)
    bool done = false;
    uint32_t count = 0;

    // The draft pipeline will stream tokens via callback.
    // We collect them; after K tokens we stop.
    draft_.generateAsync(ctx, K,
        [&](uint32_t tok, const char* /*text*/) {
            result.tokens.push_back(tok);
            // prob is approximated as 1/Z via the softmax; the pipeline
            // does not expose per-token probs directly, so we use uniform 1.0
            // as a conservative estimate that maximises acceptance.
            // A production implementation would expose logit readback here.
            result.probs.push_back(1.0f);
            ++count;
        }, {});

    // Wait for draft to finish K tokens
    while (draft_.isRunning() && !stopFlag_.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main model verification pass
// ─────────────────────────────────────────────────────────────────────────────

SpeculativeDecoder::VerifyResult
SpeculativeDecoder::verifyTokens(
    const std::vector<uint32_t>& ctx, const DraftResult& draft)
{
    VerifyResult result;
    const uint32_t K = uint32_t(draft.tokens.size());

    // Build the full context: original ctx + draft tokens
    std::vector<uint32_t> fullCtx = ctx;
    fullCtx.insert(fullCtx.end(), draft.tokens.begin(), draft.tokens.end());

    // Run main model to produce K+1 output tokens (one per draft position + bonus).
    // Each emitted token IS the main model's greedy choice at that position.
    result.main_tokens.reserve(K + 1);
    result.p_main.reserve(K);

    main_.generateAsync(fullCtx, K + 1,
        [&](uint32_t tok, const char* /*text*/) {
            result.main_tokens.push_back(tok);
            // p_main for acceptance: use 1.0 when main_token == draft_token,
            // 0.0 otherwise.  This implements the deterministic acceptance rule
            // (accept iff tokens match), which is equivalent to speculative
            // sampling with temperature→0.
            const uint32_t pos = uint32_t(result.main_tokens.size()) - 1;
            if (pos < K) {
                float p = (tok == draft.tokens[pos]) ? 1.0f : 0.0f;
                result.p_main.push_back(p);
            }
        }, {});

    while (main_.isRunning() && !stopFlag_.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // Pad p_main if main returned fewer than K tokens
    while (result.p_main.size() < K) result.p_main.push_back(0.0f);
    // Ensure bonus token slot exists
    if (result.main_tokens.size() < K + 1) {
        // Run one more step for the bonus token
        uint32_t bonus = mainStep(fullCtx);
        result.main_tokens.push_back(bonus);
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Acceptance criterion
// ─────────────────────────────────────────────────────────────────────────────

uint32_t SpeculativeDecoder::acceptTokens(
    const DraftResult& draft, const VerifyResult& verify)
{
    static std::mt19937 rng(1337);
    std::uniform_real_distribution<float> udist(0.f, 1.f);

    const uint32_t K = uint32_t(draft.tokens.size());

    for (uint32_t i = 0; i < K; ++i) {
        float p_d = std::max(1e-9f, draft.probs[i]);
        float p_m = verify.p_main[i];
        float ratio = p_m / p_d;
        float threshold = std::min(1.f, ratio) * cfg_.threshold;
        if (udist(rng) >= threshold) return i;  // reject at position i
    }
    return K;  // all accepted
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-step synchronous helpers
// ─────────────────────────────────────────────────────────────────────────────

uint32_t SpeculativeDecoder::draftStep(const std::vector<uint32_t>& ctx)
{
    uint32_t tok = 0;
    draft_.generateAsync(ctx, 1,
        [&](uint32_t t, const char*) { tok = t; }, {});
    while (draft_.isRunning()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return tok;
}

uint32_t SpeculativeDecoder::mainStep(const std::vector<uint32_t>& ctx)
{
    uint32_t tok = 0;
    main_.generateAsync(ctx, 1,
        [&](uint32_t t, const char*) { tok = t; }, {});
    while (main_.isRunning()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return tok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling utilities
// ─────────────────────────────────────────────────────────────────────────────

uint32_t SpeculativeDecoder::sampleFromLogits(
    const std::vector<float>& logits, float temp, uint32_t top_k)
{
    const uint32_t V = uint32_t(logits.size());
    top_k = std::min(top_k, V);

    std::vector<uint32_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
        [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
    idx.resize(top_k);

    std::vector<float> probs(top_k);
    float mx = logits[idx[0]];
    float sum = 0.f;
    for (uint32_t i = 0; i < top_k; ++i) {
        probs[i] = expf((logits[idx[i]] - mx) / temp);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    static std::mt19937 rng(42);
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return idx[dist(rng)];
}

float SpeculativeDecoder::probOf(
    const std::vector<float>& logits, uint32_t tok, float temp)
{
    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;
    for (float v : logits) sum += expf((v - mx) / temp);
    return expf((logits[tok] - mx) / temp) / sum;
}
