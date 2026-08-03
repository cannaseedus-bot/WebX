// ============================================================
// EVOLUTION SHADER (cs_6_0)
// Selects and recombines adapter candidates.
//
// Candidate tensors are adapter folds, not base model folds.
// The shader performs deterministic elite/crossover selection
// from reward scores and emits an evolution trace.
//
// Compile: dxc -T cs_6_0 -E main -O3 evolution.hlsl -Fo evolution.cso
// ============================================================

#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=8), \
     SRV(t0), SRV(t1), SRV(t2), \
     UAV(u0), UAV(u1)"

cbuffer EvolutionCB : register(b0)
{
    uint  ElementCount;
    uint  CandidateCount;
    uint  CandidateStride;
    uint  FrameIdx;
    float EliteRatio;
    float CrossMix;
    float DiversityGain;
    float _pad0;
};

StructuredBuffer<float> candidates     : register(t0);
StructuredBuffer<float> reward_scores  : register(t1);
StructuredBuffer<uint2> parent_pairs   : register(t2);

RWStructuredBuffer<float> evolved_out   : register(u0);
RWStructuredBuffer<uint>  evolution_log : register(u1);

float deterministic_mix(uint idx, uint a, uint b)
{
    uint h = idx * 1103515245u + a * 12345u + b * 2654435761u + FrameIdx;
    h ^= h >> 13;
    return (float)(h & 0xFFFFu) / 65535.0f;
}

[RootSignature(ROOT_SIG)]
[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= ElementCount || CandidateCount == 0 || CandidateStride == 0) return;

    uint slot = idx % CandidateStride;
    uint child = idx / CandidateStride;
    uint pair_idx = child % CandidateCount;

    uint2 parents = parent_pairs[pair_idx];
    uint parent_a = parents.x % CandidateCount;
    uint parent_b = parents.y % CandidateCount;

    float score_a = reward_scores[parent_a];
    float score_b = reward_scores[parent_b];
    uint elite = (score_a >= score_b) ? parent_a : parent_b;

    float a = candidates[parent_a * CandidateStride + slot];
    float b = candidates[parent_b * CandidateStride + slot];
    float e = candidates[elite * CandidateStride + slot];

    float mix_gate = deterministic_mix(idx, parent_a, parent_b);
    float crossover = lerp(a, b, saturate(CrossMix + (mix_gate - 0.5f) * DiversityGain));
    float elite_keep = (mix_gate < EliteRatio) ? 1.0f : 0.0f;
    float next = lerp(crossover, e, elite_keep);

    evolved_out[idx] = next;
    evolution_log[idx] = (elite << 24) | (parent_a << 16) | (parent_b << 8) | ((uint)(mix_gate * 255.0f) & 0xFFu);
}
