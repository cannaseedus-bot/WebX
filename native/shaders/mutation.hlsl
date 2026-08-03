// ============================================================
// MUTATION SHADER (cs_6_0)
// Adapter-only mutation stage for SCXQ2/Micronaut folds.
//
// This kernel never writes base model tensors. It reads adapter
// weights and reward scores, emits bounded adapter deltas, and
// writes a compact mutation trace for replay/proof.
//
// Compile: dxc -T cs_6_0 -E main -O3 mutation.hlsl -Fo mutation.cso
// ============================================================

#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=8), \
     SRV(t0), SRV(t1), SRV(t2), \
     UAV(u0), UAV(u1)"

cbuffer MutationCB : register(b0)
{
    uint  ElementCount;
    uint  CandidateCount;
    uint  FrameIdx;
    uint  Seed;
    float MutationRate;
    float MutationLimit;
    float RewardGain;
    float Decay;
};

StructuredBuffer<float> adapter_in     : register(t0);
StructuredBuffer<float> reward_scores  : register(t1);
StructuredBuffer<uint>  mutation_mask  : register(t2);

RWStructuredBuffer<float> adapter_out   : register(u0);
RWStructuredBuffer<uint>  mutation_log  : register(u1);

uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rand_signed(uint idx)
{
    uint h = hash_u32(idx ^ Seed ^ (FrameIdx * 1664525u));
    float u = (float)(h & 0x00FFFFFFu) / 16777215.0f;
    return u * 2.0f - 1.0f;
}

[RootSignature(ROOT_SIG)]
[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= ElementCount) return;

    uint candidate = (CandidateCount > 0) ? (idx % CandidateCount) : 0;
    float reward = reward_scores[candidate];
    uint enabled = mutation_mask[idx];

    float base = adapter_in[idx];
    float gate = (enabled != 0u && reward > 0.0f) ? 1.0f : 0.0f;
    float noise = rand_signed(idx);
    float delta = clamp(noise * MutationRate * (1.0f + reward * RewardGain), -MutationLimit, MutationLimit);
    float next = base * (1.0f - Decay) + gate * delta;

    adapter_out[idx] = next;

    uint qdelta = (uint)(saturate((delta / max(MutationLimit, 1e-6f)) * 0.5f + 0.5f) * 65535.0f);
    mutation_log[idx] = (candidate << 24) | (enabled << 23) | (qdelta & 0xFFFFu);
}
