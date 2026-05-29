// ============================================================
// REWARD SHADER (cs_6_0)
// Computes bounded reward for Micronaut adapter candidates.
//
// Reward is derived from route confidence, replay success,
// latency/cost pressure, and safety score. It writes reward
// values and a compact proof trace for adapter training.
//
// Compile: dxc -T cs_6_0 -E main -O3 reward.hlsl -Fo reward.cso
// ============================================================

#define ROOT_SIG \
    "RootConstants(b0, num32BitConstants=8), \
     SRV(t0), SRV(t1), SRV(t2), SRV(t3), \
     UAV(u0), UAV(u1)"

cbuffer RewardCB : register(b0)
{
    uint  CandidateCount;
    uint  FrameIdx;
    uint  RewardMode;
    uint  _pad0;
    float ConfidenceWeight;
    float SuccessWeight;
    float LatencyPenalty;
    float SafetyWeight;
};

StructuredBuffer<float> route_confidence : register(t0);
StructuredBuffer<float> replay_success   : register(t1);
StructuredBuffer<float> latency_ms       : register(t2);
StructuredBuffer<float> safety_score     : register(t3);

RWStructuredBuffer<float> reward_scores  : register(u0);
RWStructuredBuffer<uint>  reward_log     : register(u1);

float bounded_reward(float confidence, float success, float latency, float safety)
{
    float latency_term = exp(-max(latency, 0.0f) * LatencyPenalty);
    float raw =
        confidence * ConfidenceWeight +
        success * SuccessWeight +
        latency_term +
        safety * SafetyWeight;

    float norm = raw / max(ConfidenceWeight + SuccessWeight + SafetyWeight + 1.0f, 1e-6f);
    return saturate(norm);
}

[RootSignature(ROOT_SIG)]
[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= CandidateCount) return;

    float confidence = saturate(route_confidence[idx]);
    float success = saturate(replay_success[idx]);
    float latency = max(latency_ms[idx], 0.0f);
    float safety = saturate(safety_score[idx]);

    float reward = bounded_reward(confidence, success, latency, safety);

    if (RewardMode == 1u)
    {
        reward = min(reward, safety);
    }
    else if (RewardMode == 2u)
    {
        reward = reward * success;
    }

    reward_scores[idx] = reward;

    uint q_conf = (uint)(confidence * 255.0f) & 0xFFu;
    uint q_succ = (uint)(success * 255.0f) & 0xFFu;
    uint q_safe = (uint)(safety * 255.0f) & 0xFFu;
    uint q_rewd = (uint)(reward * 255.0f) & 0xFFu;
    reward_log[idx] = (q_rewd << 24) | (q_safe << 16) | (q_succ << 8) | q_conf;
}
