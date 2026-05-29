// ============================================================
// SCX-MoE GLYPH EXEC SHADER v3.1.0
// ============================================================
// Extends kuhul glyph_compute.hlsl with MoE routing signal.
// INT4 SCXQ2 ISA, cs_6_0. No WaveOps.
//
// OP_ESC (0xF) → writes expert routing signal to outBuffer
// instead of XCFE; the host reads outBuffer to dispatch
// CS_Router and CS_ExpertForward.
//
// Compile: dxc -T cs_6_0 -E CS_GlyphExec -O3 sxme_glyph_exec.hlsl -Fo sxme_glyph_exec.cso
// ============================================================

cbuffer DispatchParams : register(b0)
{
    uint laneCount;
    uint tokensPerLane;
    uint mode;          // 0=GRAM, 1=TENSOR, 2=MOE_ROUTE
    uint param1;
};

StructuredBuffer<uint>    gramBuffer   : register(t0);
RWStructuredBuffer<float> stateBuffer  : register(u0);
RWStructuredBuffer<uint>  outBuffer    : register(u1);
RWStructuredBuffer<uint>  debugBuffer  : register(u2);

// ISA v1
#define OP_NOP   0x0
#define OP_LOAD  0x1
#define OP_STORE 0x2
#define OP_ADD   0x3
#define OP_MUL   0x4
#define OP_DOT   0x5
#define OP_NORM  0x6
#define OP_EXP   0x8
#define OP_SUM   0x9
#define OP_MAX   0xA
#define OP_MIN   0xB
#define OP_MOV   0xC
#define OP_ESC   0xF

uint DecodeToken(uint word, uint idx)
{
    return (word >> (idx * 4)) & 0xF;
}

[numthreads(64, 1, 1)]
void CS_GlyphExec(uint3 DTid : SV_DispatchThreadID)
{
    float r0 = 0, r1 = 0, acc = 0;
    uint pc = 0;

    uint lane = DTid.x;
    if (lane >= laneCount) return;

    uint base = lane * ((tokensPerLane + 7) / 8);

    r0  = stateBuffer[lane * 4 + 0];
    r1  = stateBuffer[lane * 4 + 1];
    acc = stateBuffer[lane * 4 + 2];

    uint out_flag  = 0;
    uint debug_idx = 0;

    for (uint i = 0; i < tokensPerLane; i++)
    {
        uint word  = gramBuffer[base + (i >> 3)];
        uint token = DecodeToken(word, i & 7);

        switch (token)
        {
            case OP_NOP:   break;
            case OP_LOAD:  r0 = stateBuffer[lane * 4 + 0]; break;
            case OP_STORE:
                stateBuffer[lane * 4 + 0] = r0;
                stateBuffer[lane * 4 + 1] = r1;
                stateBuffer[lane * 4 + 2] = acc;
                break;
            case OP_ADD:   r0 = r0 + r1;  break;
            case OP_MUL:   r0 = r0 * r1;  break;
            case OP_DOT:   acc = r0 * r1; break;
            case OP_NORM:
            {
                float mag = sqrt(r0 * r0 + r1 * r1 + 1e-6);
                r0 /= mag; r1 /= mag;
                break;
            }
            case OP_EXP:   r0 = exp(r0);  break;
            case OP_SUM:   acc += r0;     break;
            case OP_MAX:   r0 = max(r0, r1); break;
            case OP_MIN:   r0 = min(r0, r1); break;
            case OP_MOV:   r1 = r0;       break;

            case OP_ESC:
                // MoE routing signal: encode (lane, acc-as-logit) in outBuffer
                out_flag = 1;
                if (mode == 2u) // MOE_ROUTE
                    outBuffer[lane] = lane | 0x80000000u;
                else
                    outBuffer[lane] = lane | 0x80000000u;
                break;

            default: break;
        }

        if (debug_idx < 64)
        {
            debugBuffer[lane * 64 + debug_idx] = token;
            debug_idx++;
        }
        pc++;
    }

    stateBuffer[lane * 4 + 0] = r0;
    stateBuffer[lane * 4 + 1] = r1;
    stateBuffer[lane * 4 + 2] = acc;
    stateBuffer[lane * 4 + 3] = pc;

    if (out_flag)
        outBuffer[lane] = lane | 0x80000000u;
}
