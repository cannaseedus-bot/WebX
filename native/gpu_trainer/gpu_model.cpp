/**
 * gpu_model.cpp
 * GPUModel — CPU-math baseline, fully wired for DX12 swap-in.
 *
 * [GPU_HOOK] comments mark every function that should become a GPU dispatch.
 * The generation loop, KV cache, and WebView2 bridge stay untouched.
 */

#include "gpu_model.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>

// ── Math helpers ──────────────────────────────────────────────────────────────

float GPUModel::dot(const float* a, const float* b, uint32_t n) {
    float s = 0.f;
    for (uint32_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void GPUModel::softmax(float* x, uint32_t n) {
    float mx = *std::max_element(x, x + n);
    float sum = 0.f;
    for (uint32_t i = 0; i < n; ++i) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (uint32_t i = 0; i < n; ++i) x[i] /= sum;
}

// ── [GPU_HOOK] matmul: C[M,N] = A[M,K] × B[K,N] ─────────────────────────────
// Replace body with:
//   1. Upload A, B into GPU SRVs (or use already-resident buffers)
//   2. SetCB(QProjParams{M,K,N,...})
//   3. Dispatch(CSQProj, ceil(N/16), ceil(M/16), 1)
//   4. Readback C  (or keep on GPU and pass UAV to next stage)

void GPUModel::matmul(const float* A, const float* B, float* C,
                      uint32_t M, uint32_t K, uint32_t N)
{
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (uint32_t k = 0; k < K; ++k) acc += A[m*K+k] * B[k*N+n];
            C[m*N+n] = acc;
        }
}

// ── [GPU_HOOK] layerNorm ──────────────────────────────────────────────────────
// Replace body with: Dispatch(gpt2_layernorm_fwd.hlsl CSMain, S, 1, 1)

void GPUModel::layerNorm(float* x, const float* g, const float* b, uint32_t n)
{
    float mean = 0.f, var = 0.f;
    for (uint32_t i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    for (uint32_t i = 0; i < n; ++i) { float d = x[i]-mean; var += d*d; }
    var /= n;
    float istd = 1.f / sqrtf(var + 1e-5f);
    for (uint32_t i = 0; i < n; ++i)
        x[i] = g[i] * (x[i] - mean) * istd + b[i];
}

// ── [GPU_HOOK] GELU ───────────────────────────────────────────────────────────
// Replace body with: Dispatch(gpt2_gelu_fwd.hlsl CSMain, ceil(n/256), 1, 1)

void GPUModel::gelu(float* x, uint32_t n)
{
    static const float SQRT_2_PI = 0.79788456f;
    static const float COEFF     = 0.044715f;
    for (uint32_t i = 0; i < n; ++i) {
        float v = x[i];
        float k = SQRT_2_PI * (v + COEFF * v * v * v);
        k = std::max(-10.f, std::min(10.f, k));  // clamp (HD 4600 tanh safety)
        x[i] = 0.5f * v * (1.f + tanhf(k));
    }
}

// ── [GPU_HOOK] attention ──────────────────────────────────────────────────────
// Replace body with:
//   1. Copy q, kv_.Krow, kv_.Vrow into GPU buffers
//   2. Dispatch(gpt2_attn_fwd.hlsl CSMain, n_head, 1, 1) with pos+1 as seq_len
//   3. Readback out

void GPUModel::attention(float* q, uint32_t layer, uint32_t pos, float* out)
{
    const uint32_t D = cfg_.head_dim;
    float scale = 1.f / sqrtf(float(D));

    // Scores: Q · K^T for tokens 0..pos
    std::vector<float> scores(pos + 1);
    for (uint32_t j = 0; j <= pos; ++j)
        scores[j] = dot(q, kv_.Krow(layer, j), D) * scale;
    softmax(scores.data(), pos + 1);

    // Weighted sum over V
    std::fill(out, out + D, 0.f);
    for (uint32_t j = 0; j <= pos; ++j) {
        const float* v = kv_.Vrow(layer, j);
        for (uint32_t d = 0; d < D; ++d) out[d] += scores[j] * v[d];
    }
}

// ── init ──────────────────────────────────────────────────────────────────────

bool GPUModel::init()
{
    const uint32_t V = cfg_.vocab;
    const uint32_t D = cfg_.dim;
    const uint32_t F = cfg_.ffn_dim;
    const uint32_t L = cfg_.n_layers;

    // Deterministic init (sine-based) — replace with real checkpoint loading
    auto sinInit = [](std::vector<float>& v, uint32_t n, float scale, uint32_t seed=0) {
        v.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            v[i] = sinf(float(i + seed + 1) * 0.01f) * scale;
    };

    sinInit(embed_,   V * D, 0.1f);
    sinInit(lmHead_,  D * V, 0.05f);

    Wq_.resize(L); Wk_.resize(L); Wv_.resize(L); Wo_.resize(L);
    ln1g_.resize(L); ln1b_.resize(L); ln2g_.resize(L); ln2b_.resize(L);
    W1_.resize(L);  W2_.resize(L);

    for (uint32_t l = 0; l < L; ++l) {
        sinInit(Wq_[l], D*D, 0.1f, l*7+1);
        sinInit(Wk_[l], D*D, 0.1f, l*7+2);
        sinInit(Wv_[l], D*D, 0.1f, l*7+3);
        sinInit(Wo_[l], D*D, 0.1f, l*7+4);
        sinInit(W1_[l], D*F, 0.1f, l*7+5);
        sinInit(W2_[l], F*D, 0.1f, l*7+6);
        ln1g_[l].assign(D, 1.f); ln1b_[l].assign(D, 0.f);
        ln2g_[l].assign(D, 1.f); ln2b_[l].assign(D, 0.f);
    }

    kv_.alloc(cfg_);
    return true;
}

// ── runLayer — one full transformer layer ─────────────────────────────────────

void GPUModel::runLayer(uint32_t layer, float* x, uint32_t pos)
{
    const uint32_t D = cfg_.dim;
    const uint32_t F = cfg_.ffn_dim;

    std::vector<float> xn(D), Q(D), K(D), V(D), attnOut(D);
    std::vector<float> ffnH(F), ffnOut(D);

    // ── LayerNorm 1 ──────────────────────────────────────────────────────────
    memcpy(xn.data(), x, D*4);
    layerNorm(xn.data(), ln1g_[layer].data(), ln1b_[layer].data(), D);

    // ── QKV projections [GPU_HOOK: 3 × CSQProj dispatches] ──────────────────
    matmul(xn.data(), Wq_[layer].data(), Q.data(), 1, D, D);
    matmul(xn.data(), Wk_[layer].data(), K.data(), 1, D, D);
    matmul(xn.data(), Wv_[layer].data(), V.data(), 1, D, D);

    // ── Store KV in cache ────────────────────────────────────────────────────
    memcpy(kv_.Krow(layer, pos), K.data(), D*4);
    memcpy(kv_.Vrow(layer, pos), V.data(), D*4);

    // ── Attention [GPU_HOOK: gpt2_attn_fwd.hlsl CSMain] ─────────────────────
    attention(Q.data(), layer, pos, attnOut.data());

    // ── Output projection [GPU_HOOK: CSQProj] ────────────────────────────────
    std::vector<float> attnProj(D);
    matmul(attnOut.data(), Wo_[layer].data(), attnProj.data(), 1, D, D);

    // ── Residual 1 [GPU_HOOK: gpt2_residual_add.hlsl CSMain_addto] ──────────
    for (uint32_t i = 0; i < D; ++i) x[i] += attnProj[i];

    // ── LayerNorm 2 ──────────────────────────────────────────────────────────
    memcpy(xn.data(), x, D*4);
    layerNorm(xn.data(), ln2g_[layer].data(), ln2b_[layer].data(), D);

    // ── FFN W1 [GPU_HOOK: CSQProj] ───────────────────────────────────────────
    matmul(xn.data(), W1_[layer].data(), ffnH.data(), 1, D, F);

    // ── GELU [GPU_HOOK: gpt2_gelu_fwd.hlsl CSMain] ──────────────────────────
    gelu(ffnH.data(), F);

    // ── FFN W2 [GPU_HOOK: CSQProj] ───────────────────────────────────────────
    matmul(ffnH.data(), W2_[layer].data(), ffnOut.data(), 1, F, D);

    // ── Residual 2 ───────────────────────────────────────────────────────────
    for (uint32_t i = 0; i < D; ++i) x[i] += ffnOut[i];
}

// ── Sampling ──────────────────────────────────────────────────────────────────

uint32_t GPUModel::sampleTopK(const float* logits, uint32_t n, float temp, uint32_t k)
{
    // Collect top-k indices
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
        [&](uint32_t a, uint32_t b){ return logits[a] > logits[b]; });
    idx.resize(k);

    // Apply temperature and softmax over top-k
    std::vector<float> probs(k);
    for (uint32_t i = 0; i < k; ++i) probs[i] = logits[idx[i]] / temp;
    softmax(probs.data(), k);

    // Categorical sample
    static std::mt19937 rng(42);
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return idx[dist(rng)];
}

// ── Generation loop ───────────────────────────────────────────────────────────

void GPUModel::runGenLoop(std::vector<uint32_t> tokens, uint32_t maxNew)
{
    const uint32_t D   = cfg_.dim;
    const uint32_t V   = cfg_.vocab;
    const uint32_t L   = cfg_.n_layers;

    kv_.K.assign(kv_.K.size(), 0.f);
    kv_.V.assign(kv_.V.size(), 0.f);

    std::vector<float> x(D), logits(V);

    // ── Prefill: process prompt tokens ────────────────────────────────────────
    for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
        uint32_t tok = tokens[pos];
        memcpy(x.data(), embed_.data() + tok * D, D * 4);
        for (uint32_t l = 0; l < L; ++l) runLayer(l, x.data(), pos);
    }

    // ── Generation ───────────────────────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();

    for (uint32_t step = 0; step < maxNew && !stopFlag_.load(); ++step) {
        uint32_t pos = uint32_t(tokens.size()) - 1;

        // lm_head projection: logits = x × lmHead^T
        // [GPU_HOOK: CSQProj dispatch for lm_head]
        for (uint32_t v = 0; v < V; ++v)
            logits[v] = dot(x.data(), lmHead_.data() + v * D, D);

        // Sample next token
        uint32_t next = sampleTopK(logits.data(), V, cfg_.temp, cfg_.top_k);
        tokens.push_back(next);

        // Emit to caller
        std::string text = detok(next);
        if (tokenCb_) tokenCb_(next, text);

        // Performance callback every 8 steps
        if (perfCb_ && step % 8 == 7) {
            auto now = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(now - t0).count();
            perfCb_(double(step + 1) / secs, step + 1);
        }

        if (next == '\n' || pos + 1 >= cfg_.max_seq) break;

        // Embed next token and run layers
        memcpy(x.data(), embed_.data() + next * D, D * 4);
        uint32_t nextPos = pos + 1;
        for (uint32_t l = 0; l < L; ++l) runLayer(l, x.data(), nextPos);
    }

    running_.store(false);
}

// ── Public API ────────────────────────────────────────────────────────────────

void GPUModel::generateAsync(const std::string& prompt, uint32_t maxNew)
{
    if (running_.exchange(true)) return;  // one generation at a time
    stopFlag_.store(false);

    // Tokenise (byte-level for demo)
    std::vector<uint32_t> tokens;
    for (unsigned char c : prompt) tokens.push_back(c);

    if (inferThread_.joinable()) inferThread_.join();
    inferThread_ = std::thread([this, t=std::move(tokens), maxNew]() mutable {
        std::lock_guard<std::mutex> lk(genMu_);
        runGenLoop(std::move(t), maxNew);
    });
}

std::string GPUModel::generateSync(const std::string& prompt, uint32_t maxNew)
{
    std::string out;
    setTokenCallback([&](uint32_t, const std::string& txt){ out += txt; });
    generateAsync(prompt, maxNew);
    if (inferThread_.joinable()) inferThread_.join();
    return out;
}
