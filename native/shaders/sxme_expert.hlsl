// ============================================================
// SCX-MoE EXPERT MLP FORWARD SHADER v3.1.0
// ============================================================
// One thread per (token, expert_slot).
// SwiGLU FFN: gate = silu(Wg·x), up = Wu·x, out = Wd·(gate*up)
// cs_5_0 DXBC — Intel HD 4600 compatible, NO WaveOps.
//
// Expert 2 (amplify) reads addon adapter tensors from t6/t7:
//   t6 = adapterRouteEmb  [adapterN × 256]  float32
//   t7 = adapterRouteBias [adapterN]         float32
// Project hidden[1024] → 256 via stride-4 mean-pool, find best
// route match, scale output by (1 + saturate(score)*bias*strength).
//
// Compile:
//   fxc /T cs_5_0 /E CS_ExpertForward /O3 /Fo sxme_expert.cso     sxme_expert.hlsl
//   fxc /T cs_5_0 /E CS_ExpertReduce  /O3 /Fo sxme_expert_reduce.cso sxme_expert.hlsl
// ============================================================

cbuffer ExpertParams : register(b0)
{
    uint  batchSize;        // tokens in batch
    uint  hiddenSize;       // d_model (1024)
    uint  intermediateSize; // FFN inner dim (2816)
    uint  topK;             // experts per token (2)
    uint  numExperts;       // total experts (8)
    uint  layerIdx;         // transformer layer index
    uint  pad0;
    uint  pad1;
};

// Expert 2 adapter params — bound to zero-sized buffers when no addon loaded
cbuffer AdapterParams : register(b1)
{
    uint  adapterN;         // number of adapter records (0 = disabled)
    float adapterStrength;  // scale factor for adapter bias (default 0.1)
    uint  apad0;
    uint  apad1;
};

// --- Core MoE buffers ---
StructuredBuffer<float>   hiddenIn      : register(t0); // [batchSize × hiddenSize]
StructuredBuffer<uint>    expertIds     : register(t1); // [batchSize × topK]
StructuredBuffer<float>   expertWeights : register(t2); // [batchSize × topK]

// Expert weight banks (all experts × all layers, packed):
//   bankOffset = (expertId * 8 + layerIdx) * hiddenSize * intermediateSize
StructuredBuffer<float>   wGate : register(t3); // gate proj
StructuredBuffer<float>   wUp   : register(t4); // up   proj
StructuredBuffer<float>   wDown : register(t5); // down proj

// --- Expert 2 adapter buffers (bound unconditionally; unused unless expertId==2) ---
StructuredBuffer<float>   adapterRouteEmb  : register(t6); // [adapterN × 256]
StructuredBuffer<float>   adapterRouteBias : register(t7); // [adapterN]

// Output: per-slot contributions [batchSize × topK × hiddenSize]
RWStructuredBuffer<float> hiddenOut : register(u0);

// ============================================================
// ADAPTER SCALE  (Expert 2 only, inline helper)
// Projects hidden[1024] → 256 via stride-4 mean-pool,
// finds best cosine match in adapterRouteEmb,
// returns scale = 1 + saturate(score) * adapterRouteBias[best] * adapterStrength.
// When adapterN == 0, returns 1.0 (no-op).
// ============================================================
float AdapterScale(uint tokBase)
{
    if (adapterN == 0) return 1.0;

    // Project: proj[d] = mean(hidden[d*4 .. d*4+3])
    // hiddenSize must be 4 × 256 = 1024 for this to be exact.
    // For other sizes we clamp: stride = hiddenSize / 256.
    uint stride = max(hiddenSize / 256u, 1u);

    float proj[256];
    float pnorm = 0.0;
    for (uint d = 0; d < 256; d++)
    {
        float s = 0.0;
        uint base = tokBase + d * stride;
        for (uint k = 0; k < stride; k++)
            s += hiddenIn[base + k];
        proj[d] = s / (float)stride;
        pnorm += proj[d] * proj[d];
    }
    float inv_pnorm = 1.0 / sqrt(max(pnorm, 1e-8));

    // Find best matching route record (dot-product similarity)
    float bestScore = -1e9;
    uint  bestIdx   = 0;
    for (uint r = 0; r < adapterN; r++)
    {
        float dot = 0.0;
        uint  rBase = r * 256u;
        for (uint d2 = 0; d2 < 256; d2++)
            dot += proj[d2] * inv_pnorm * adapterRouteEmb[rBase + d2];
        if (dot > bestScore) { bestScore = dot; bestIdx = r; }
    }

    float bias = adapterRouteBias[bestIdx];
    return 1.0 + saturate(bestScore) * bias * adapterStrength;
}

// ============================================================
// CS_ExpertForward  —  one thread per (token × top-K slot)
// ============================================================
[numthreads(64, 1, 1)]
void CS_ExpertForward(uint3 DTid : SV_DispatchThreadID)
{
    uint tid  = DTid.x;
    uint tok  = tid / topK;
    uint slot = tid % topK;

    if (tok >= batchSize) return;

    uint  expertId = expertIds[tok * topK + slot];
    float w        = expertWeights[tok * topK + slot];
    uint  tokBase  = tok * hiddenSize;

    // Weight bank offsets
    uint gateStride = hiddenSize * intermediateSize;
    uint downStride = intermediateSize * hiddenSize;
    uint bankOff    = (expertId * 8u + layerIdx) * gateStride;
    uint downOff    = (expertId * 8u + layerIdx) * downStride;

    // Expert 2: compute adapter scale before MLP loop
    float aScale = (expertId == 2u) ? AdapterScale(tokBase) : 1.0;

    // SwiGLU MLP: output_d = sum_i [ silu(Wg[i]·x) * Wu[i]·x * wDown[i][d] ]
    for (uint d = 0; d < hiddenSize; d++)
    {
        float acc = 0.0;
        for (uint i = 0; i < intermediateSize; i++)
        {
            // Gate dot-product
            float gVal = 0.0;
            for (uint dd = 0; dd < hiddenSize; dd++)
                gVal += hiddenIn[tokBase + dd] * wGate[bankOff + i * hiddenSize + dd];

            // Up dot-product
            float uVal = 0.0;
            for (uint dd2 = 0; dd2 < hiddenSize; dd2++)
                uVal += hiddenIn[tokBase + dd2] * wUp[bankOff + i * hiddenSize + dd2];

            // SiLU gate (silu(x) = x * sigmoid(x))
            float sig   = 1.0 / (1.0 + exp(-gVal));
            float gated = gVal * sig * uVal;

            acc += gated * wDown[downOff + i * hiddenSize + d];
        }

        hiddenOut[tok * topK * hiddenSize + slot * hiddenSize + d] = acc * w * aScale;
    }
}

// ============================================================
// CS_ExpertReduce  —  sum top-K slots back to [batchSize × hiddenSize]
// One thread per (token × hidden_dim).
// ============================================================
[numthreads(64, 1, 1)]
void CS_ExpertReduce(uint3 DTid : SV_DispatchThreadID)
{
    uint flat = DTid.x;
    uint tok  = flat / hiddenSize;
    uint d    = flat % hiddenSize;

    if (tok >= batchSize) return;

    float sum = 0.0;
    for (uint slot = 0; slot < topK; slot++)
        sum += hiddenOut[tok * topK * hiddenSize + slot * hiddenSize + d];

    // Reduce back to contiguous [tok, d] in-place
    hiddenOut[tok * hiddenSize + d] = sum;
}
