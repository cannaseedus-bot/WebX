cbuffer ExpertBlockInt4Params : register(b0)
{
    uint batchSize;
    uint hiddenSize;
    uint blockRows;
    uint outputOffset;
    float scale;
    float zeroPoint;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float> hiddenIn : register(t0);
ByteAddressBuffer weightBlock : register(t1);
RWStructuredBuffer<float> outputBlock : register(u0);

float decodeWeight(uint elementIndex)
{
    uint byteIndex = elementIndex >> 1;
    uint word = weightBlock.Load(byteIndex & ~3u);
    uint nibble = (word >> ((byteIndex & 3u) * 8u +
                            (elementIndex & 1u) * 4u)) & 0xFu;
    int signedValue = (nibble >= 8u) ? int(nibble) - 16 : int(nibble);
    return float(signedValue) * scale + zeroPoint;
}

[numthreads(64, 1, 1)]
void CSExpertBlockInt4Gemm(uint3 id : SV_DispatchThreadID)
{
    uint flat = id.x;
    uint total = batchSize * blockRows;
    if (flat >= total) return;

    uint token = flat / blockRows;
    uint row = flat % blockRows;
    uint inputBase = token * hiddenSize;
    uint weightBase = row * hiddenSize;
    float value = 0.0f;
    for (uint col = 0; col < hiddenSize; ++col)
        value += hiddenIn[inputBase + col] *
                 decodeWeight(weightBase + col);
    outputBlock[token * blockRows + row] = value;
}
