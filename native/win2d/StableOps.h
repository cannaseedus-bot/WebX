//================================================================================
// StableOps.h — Numerically stable ops for iGPU training (DirectXMath SIMD)
//
// Used in the KXML Sek-phase stabilizer node and backward pass.
// All ops are pure CPU SIMD — complement Win2D GPU dispatch for large batches.
//
// KXML binding:
//   <node id="stabilizer" phase="Sek" domain="compute" device="cpu">
//     <bind from="logits_raw" to="logits_stable" transform="stable.clamp_softmax" />
//   </node>
//================================================================================
#pragma once

#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace DirectX;
using namespace DirectX::PackedVector;

namespace KXML { namespace Stable {

// ─── StableOps ────────────────────────────────────────────────────────────────

class StableOps {
public:
    // Clamp 4-lane logit vector to [-maxAbs, +maxAbs]
    static XMVECTOR ClampLogits(XMVECTOR v, float maxAbs = 20.f) {
        return XMVectorClamp(v, XMVectorReplicate(-maxAbs),
                                XMVectorReplicate(maxAbs));
    }

    // Numerically stable softmax: shift by max, clamp, exp, divide
    static XMVECTOR StableSoftmax(XMVECTOR v) {
        XMVECTOR mx  = XMVectorSplatX(XMVector4Dot(v, XMVectorSplatOne()));
        // Single-lane max: use FMADD reduction
        float f[4]; XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(f), v);
        float mv = std::max({f[0], f[1], f[2], f[3]});
        XMVECTOR shifted = XMVectorSubtract(v, XMVectorReplicate(mv));
        shifted = XMVectorClamp(shifted, XMVectorReplicate(-20.f),
                                         XMVectorReplicate(20.f));
        XMVECTOR ev  = XMVectorExp(shifted);
        float ef[4]; XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(ef), ev);
        float s = ef[0]+ef[1]+ef[2]+ef[3];
        return XMVectorDivide(ev, XMVectorReplicate(s > 0.f ? s : 1e-7f));
    }

    // L2 gradient clip — scale whole vector if norm > maxNorm
    static XMVECTOR ClipGradient(XMVECTOR g, float maxNorm = 1.f) {
        XMVECTOR nSq  = XMVector4Dot(g, g);
        float norm    = sqrtf(XMVectorGetX(nSq));
        if (norm > maxNorm && norm > 0.f)
            g = XMVectorMultiply(g, XMVectorReplicate(maxNorm / norm));
        return g;
    }

    static float ClampLoss(float loss, float maxLoss = 10.f) {
        return std::min(loss, maxLoss);
    }

    static XMHALF4 ToFP16(XMVECTOR v)   { XMHALF4 r; XMStoreHalf4(&r, v); return r; }
    static XMVECTOR FromFP16(XMHALF4 v) { return XMLoadHalf4(&v); }
};

// ─── Scalar stable cross-entropy ──────────────────────────────────────────────
// Operates on a 4-element logit SIMD vector; targetToken selects which lane.

inline float StableCrossEntropy(XMVECTOR logits, uint32_t targetLane) {
    logits           = StableOps::ClampLogits(logits, 20.f);
    XMVECTOR probs   = StableOps::StableSoftmax(logits);
    float    f[4];   XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(f), probs);
    float    p       = std::max(f[targetLane & 3], 1e-7f);
    return   StableOps::ClampLoss(-logf(p), 10.f);
}

// ─── Batch gradient clipper ────────────────────────────────────────────────────

inline std::vector<float> ClipGradientsL2(const std::vector<float>& g,
                                          float maxNorm = 1.f) {
    float norm = 0.f;
    for (float v : g) norm += v * v;
    norm = sqrtf(norm);
    float scale = (norm > maxNorm && norm > 0.f) ? maxNorm / norm : 1.f;
    std::vector<float> out(g.size());
    for (size_t i = 0; i < g.size(); i++) out[i] = g[i] * scale;
    return out;
}

}} // namespace KXML::Stable
