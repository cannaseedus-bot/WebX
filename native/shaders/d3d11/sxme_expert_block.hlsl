cbuffer ExpertBlockParams : register(b0)
{
    uint batchSize;
    uint hiddenSize;
    uint blockRows;
    uint outputOffset;
};

StructuredBuffer<float> hiddenIn : register(t0);
StructuredBuffer<float> weightBlock : register(t1);
RWStructuredBuffer<float> outputBlock : register(u0);

[numthreads(64, 1, 1)]
void CSExpertBlockGemm(uint3 id : SV_DispatchThreadID)
{
    uint flat = id.x;
    uint total = batchSize * blockRows;
    if (flat >= total) return;

    uint token = flat / blockRows;
    uint row = flat % blockRows;
    float value = 0.0f;
    uint inputBase = token * hiddenSize;
    uint weightBase = row * hiddenSize;
    for (uint col = 0; col < hiddenSize; ++col)
        value += hiddenIn[inputBase + col] * weightBlock[weightBase + col];
    outputBlock[token * blockRows + row] = value;
}
