// SH-Wave-Lattice: Simplified GPU Wave VM (Phase 3)
// Minimal compute kernel for testing

#define SH_L 9

struct SHNode
{
    float2 sh[SH_L];
    uint neighborCount;
    uint neighbors[6];
};

struct Instr32
{
    uint op;
    uint flags;
    uint a, b, c;
    float f0, f1, f2, f3;
};

StructuredBuffer<Instr32> Program : register(t0);
StructuredBuffer<SHNode> NodesIn : register(t1);
RWStructuredBuffer<SHNode> NodesOut : register(u0);

float2 crot(float2 v, float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return float2(v.x*c - v.y*s, v.x*s + v.y*c);
}

float clen(float2 v)
{
    return sqrt(v.x*v.x + v.y*v.y);
}

[numthreads(256, 1, 1)]
void WaveVM_Main(uint id : SV_DispatchThreadID)
{
    if (id >= 2048) return;

    SHNode node = NodesIn[id];

    // Simple propagation step
    float2 newSH[SH_L];

    for (uint i = 0; i < SH_L; i++)
    {
        float2 acc = node.sh[i];

        // Neighbor coupling
        for (uint k = 0; k < node.neighborCount; k++)
        {
            uint nIdx = node.neighbors[k];
            acc += NodesIn[nIdx].sh[i] * 0.33f;
        }

        // Phase rotation
        float l = floor(sqrt((float)i));
        acc = crot(acc, l * 0.4f);

        // Cross-band
        uint next_i = (i + 1) % SH_L;
        acc += node.sh[next_i] * 0.05f;

        // Normalize + inject
        float len = clen(acc);
        acc = (len < 1e-5f) ? float2(0,0) : (acc / len);
        acc += float2(0.005f, 0.003f);

        // Decay
        newSH[i] = acc * 0.99f;
    }

    // Write back
    for (uint j = 0; j < SH_L; j++)
        node.sh[j] = newSH[j];

    NodesOut[id] = node;
}
