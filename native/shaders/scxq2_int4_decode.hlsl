/**
 * scxq2_int4_decode.hlsl
 * GPU-side SCXQ2 INT4 → float32 decode kernel.
 *
 * Each thread unpacks 2 INT4 values from one packed uint32 byte.
 * Dispatch: ceil(numElements / 2 / 64) thread groups, [64,1,1] threads.
 *
 * Root signature bindings (see ScxStreamEngine::createDecodeRootSigAndPso):
 *   b0  = ScxDecodeCB  (constant buffer)
 *   t0  = IN           (packed INT4,  StructuredBuffer<uint>)
 *   u0  = OUT          (decoded f32,  RWStructuredBuffer<float>)
 */

// ── Constant buffer ───────────────────────────────────────────────────────────

cbuffer ScxDecodeCB : register(b0)
{
    uint g_numPacked;    // number of uint elements in IN  (= numFloats / 2)
    uint g_dstOffset;    // element offset into OUT to write at (for ring append)
    float g_scale;       // per-tensor scale factor (SCXQ2 quantisation)
    float g_zero;        // per-tensor zero-point
};

// ── Resources ─────────────────────────────────────────────────────────────────

StructuredBuffer<uint>    IN  : register(t0);  // packed INT4 pairs
RWStructuredBuffer<float> OUT : register(u0);  // decoded float32

// ── INT4 unpack helpers ───────────────────────────────────────────────────────

// Each byte holds two 4-bit values: high nibble = v0, low nibble = v1.
// INT4 range [-8, 7]; dequant: f = (nibble - 8) * scale + zero

float unpackHigh(uint packed, uint byteIdx)
{
    uint byte = (packed >> (byteIdx * 8)) & 0xFF;
    int  nibH = int((byte >> 4) & 0xF) - 8;
    return float(nibH) * g_scale + g_zero;
}

float unpackLow(uint packed, uint byteIdx)
{
    uint byte = (packed >> (byteIdx * 8)) & 0xFF;
    int  nibL = int(byte & 0xF) - 8;
    return float(nibL) * g_scale + g_zero;
}

// ── Decode kernel — 2 floats per thread ──────────────────────────────────────

[numthreads(64, 1, 1)]
void CSDecodeInt4(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= g_numPacked) return;

    uint packed = IN[i];

    // Each uint32 encodes 8 INT4 values (4 bytes × 2 nibbles).
    // Unpack all 8 floats and write to OUT.
    uint base = g_dstOffset + i * 8;

    [unroll]
    for (uint b = 0; b < 4; ++b)
    {
        OUT[base + b * 2    ] = unpackHigh(packed, b);
        OUT[base + b * 2 + 1] = unpackLow (packed, b);
    }
}

// ── Fused decode + dot-product accumulation (optional fast path) ──────────────
// Useful when you want decode → matmul without staging to a separate buffer.
// Bind input activations as t1, accumulator as u1.

StructuredBuffer<float>   ACT : register(t1);  // input activations [hiddenDim]
RWStructuredBuffer<float> ACC : register(u1);  // accumulator [outDim]

cbuffer ScxFusedCB : register(b1)
{
    uint g_hiddenDim;
    uint g_outDim;
    uint g_fusedPacked;  // number of packed uint per output row
    uint _pad;
};

[numthreads(64, 1, 1)]
void CSFusedDecodeMatmul(uint3 id : SV_DispatchThreadID)
{
    uint outRow = id.x;
    if (outRow >= g_outDim) return;

    float acc = 0.0f;
    uint  rowBase = outRow * g_fusedPacked;

    for (uint i = 0; i < g_fusedPacked; ++i)
    {
        uint packed = IN[rowBase + i];

        [unroll]
        for (uint b = 0; b < 4; ++b)
        {
            uint actIdx = i * 8 + b * 2;
            if (actIdx + 1 >= g_hiddenDim) break;

            float w0 = unpackHigh(packed, b);
            float w1 = unpackLow (packed, b);
            acc += ACT[actIdx    ] * w0;
            acc += ACT[actIdx + 1] * w1;
        }
    }

    ACC[outRow] += acc;
}
