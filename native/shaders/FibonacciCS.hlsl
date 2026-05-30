//================================================================================
// FibonacciCS.hlsl - Batched Fibonacci Compute Shader
// For KXML Sek compute nodes with Win2D backend
// Compile: dxc.exe -T cs_6_0 -E CSMain FibonacciCS.hlsl -Fo FibonacciCS.cso
//================================================================================

cbuffer FibonacciParams : register(b0)
{
    uint startIndex;
    uint count;
    uint mode;      // 0=iterative, 1=matrix, 2=fast_doubling
    uint stride;
    float4 reserved;
};

RWStructuredBuffer<uint> outputBuffer   : register(u0);
RWStructuredBuffer<uint> gradientBuffer : register(u1);

groupshared uint sharedFibCache[256];

// ─── Helper: 2x2 matrix multiply ─────────────────────────────────────────────

uint2x2 MatrixMultiply(uint2x2 a, uint2x2 b)
{
    uint2x2 r;
    r[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0];
    r[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1];
    r[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0];
    r[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1];
    return r;
}

// ─── Fibonacci implementations ────────────────────────────────────────────────

uint FibonacciIterative(uint n)
{
    if (n == 0) return 0;
    if (n == 1) return 1;
    uint a = 0, b = 1;
    for (uint i = 2; i <= n; i++) { uint t = a+b; a = b; b = t; }
    return b;
}

uint FibonacciMatrix(uint n)
{
    if (n == 0) return 0;
    uint2x2 result = {{1,0},{0,1}};
    uint2x2 base   = {{1,1},{1,0}};
    uint exp = n;
    while (exp > 0) {
        if (exp & 1) result = MatrixMultiply(result, base);
        base = MatrixMultiply(base, base);
        exp >>= 1;
    }
    return result[0][1];
}

uint FibonacciFastDoubling(uint n)
{
    if (n == 0) return 0;
    uint a = 0, b = 1;
    uint bit = 1u << 30;
    while (bit > n) bit >>= 1;
    while (bit > 0) {
        uint c = a * (2*b - a);
        uint d = a*a + b*b;
        a = c; b = d;
        if (n & bit) { uint t = a; a = b; b = t + b; }
        bit >>= 1;
    }
    return a;
}

// ─── Main compute kernel ──────────────────────────────────────────────────────

[numthreads(256, 1, 1)]
void CSMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint  groupIndex        : SV_GroupIndex
) {
    uint globalIndex = dispatchThreadId.x;
    if (globalIndex >= count) return;

    uint n = startIndex + globalIndex * stride;
    uint result = 0;

    switch (mode)
    {
        case 0:  result = FibonacciIterative(n);   break;
        case 1:  result = FibonacciMatrix(n);      break;
        default: result = FibonacciFastDoubling(n);break;
    }

    outputBuffer[globalIndex]   = result;
    gradientBuffer[globalIndex] = 0;  // Fibonacci is deterministic — no gradient

    sharedFibCache[groupIndex] = result;
    GroupMemoryBarrierWithGroupSync();
}
