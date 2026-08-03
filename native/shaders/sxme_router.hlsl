// ============================================================
// SCX-MoE ROUTER SHADER v3.1.0
// ============================================================
// Expert routing: compute gate scores, select top-2 experts.
// cs_6_0 — HD 4600 compatible, NO WaveOps, NO SM 6.5+ required.
//
// Compile: dxc -T cs_6_0 -E CS_Router -O3 sxme_router.hlsl -Fo sxme_router.cso
// ============================================================

cbuffer RouterParams : register(b0)
{
    uint batchSize;       // Number of tokens in batch
    uint hiddenSize;      // d_model (e.g. 1024)
    uint numExperts;      // Total number of experts (e.g. 8)
    uint topK;            // Experts selected per token (e.g. 2)
};

// Input: hidden states [batchSize × hiddenSize] (float32)
StructuredBuffer<float>   hiddenStates : register(t0);
// Gate matrix [numExperts × hiddenSize] (float32, row-major)
StructuredBuffer<float>   routerGate   : register(t1);

// Output: selected expert IDs per token [batchSize × topK] (uint)
RWStructuredBuffer<uint>  expertIds    : register(u0);
// Output: softmax weights per token [batchSize × topK] (float32)
RWStructuredBuffer<float> expertWeights : register(u1);

// ============================================================
// ONE THREAD PER TOKEN
// No cross-lane communication — thread-local serial scan.
// ============================================================
[numthreads(64, 1, 1)]
void CS_Router(uint3 DTid : SV_DispatchThreadID)
{
    uint tok = DTid.x;
    if (tok >= batchSize) return;

    uint hidBase = tok * hiddenSize;

    // --- Compute dot-product gate scores (thread-local) ---
    // Max 16 experts supported without dynamic alloc
    float scores[16];
    uint  ecount = min(numExperts, 16u);

    for (uint e = 0; e < ecount; e++)
    {
        float dot = 0.0;
        uint gateBase = e * hiddenSize;
        for (uint d = 0; d < hiddenSize; d++)
        {
            dot += hiddenStates[hidBase + d] * routerGate[gateBase + d];
        }
        scores[e] = dot;
    }

    // --- Top-K selection (serial, no WaveOps needed) ---
    uint   selId[2];
    float  selScore[2];

    for (uint k = 0; k < min(topK, 2u); k++)
    {
        float best = -1e30;
        uint  bestE = 0;
        for (uint e2 = 0; e2 < ecount; e2++)
        {
            // Skip already selected
            bool already = false;
            for (uint j = 0; j < k; j++)
                if (selId[j] == e2) already = true;

            if (!already && scores[e2] > best)
            {
                best  = scores[e2];
                bestE = e2;
            }
        }
        selId[k]    = bestE;
        selScore[k] = best;
    }

    // --- Softmax over top-K scores ---
    float maxS = selScore[0];
    for (uint k2 = 1; k2 < min(topK, 2u); k2++)
        maxS = max(maxS, selScore[k2]);

    float sumExp = 0.0;
    for (uint k3 = 0; k3 < min(topK, 2u); k3++)
        sumExp += exp(selScore[k3] - maxS);

    // --- Write outputs ---
    for (uint k4 = 0; k4 < min(topK, 2u); k4++)
    {
        uint outIdx = tok * topK + k4;
        expertIds[outIdx]     = selId[k4];
        expertWeights[outIdx] = exp(selScore[k4] - maxS) / (sumExp + 1e-9);
    }
}
