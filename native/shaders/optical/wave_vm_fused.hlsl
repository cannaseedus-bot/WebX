// SH-Wave-Lattice: Fused GPU Wave VM + SH Propagation Kernel
// Phase 3: Parallel field execution on GPU
// 
// Design: Each thread = one node executor
//         All threads share instruction stream
//         VM transforms SH field in-place, fused with wave dynamics
//
// Advantages over separate VM + kernel:
//   - No CPU roundtrip
//   - Single pass execution
//   - Full node parallelism (256 threads/group)
//   - GPU cache efficient (coalesced reads)

#define SH_L 9
#define MAX_INSTR 32  // Further reduced for shader compilation
#define NEIGHBORHOOD 6

// Match C++ structure exactly
struct SHNode
{
    float2 sh[SH_L];        // Phase state: (cos, sin) per mode
    uint neighborCount;     // Actual neighbor count
    uint neighbors[6];      // Neighbor node indices
};

struct Instr32
{
    uint op;                // OpCode (1=INJECT, 2=PROPAGATE, etc.)
    uint flags;             // Reserved
    uint a, b, c;           // Arguments
    float f0, f1, f2, f3;   // Float arguments
};

// Buffers (set by CPU)
StructuredBuffer<Instr32> Program : register(t0);
StructuredBuffer<SHNode> NodesIn : register(t1);
RWStructuredBuffer<SHNode> NodesOut : register(u0);

// Helpers
float2 cmul(float2 a, float2 b)
{
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

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

float2 cnorm(float2 v)
{
    float len = clen(v);
    if (len < 1e-6) return float2(0,0);
    return v / len;
}

// Opcode handlers
void op_inject(inout SHNode node, uint nodeId, uint band, float amp, float phase)
{
    // Only inject at specified node
    // (in real system, could be broadcast or filtered)
    float c = cos(phase);
    float s = sin(phase);
    node.sh[band] += float2(amp*c, amp*s);
}

void op_propagate(inout SHNode node, in StructuredBuffer<SHNode> allNodes)
{
    float2 newSH[SH_L];

    for (uint i = 0; i < SH_L; i++)
    {
        // Start with own value
        float2 acc = node.sh[i];

        // Neighbor coupling (uniform, non-directional)
        for (uint k = 0; k < node.neighborCount; k++)
        {
            uint nIdx = node.neighbors[k];
            acc += allNodes[nIdx].sh[i] * 0.33f;
        }

        // Harmonic phase rotation (frequency-dependent)
        float l = floor(sqrt((float)i));
        float phaseShift = l * (l + 1.0f) * 0.4f;
        acc = crot(acc, phaseShift);

        // Cross-band coupling (energy transfer)
        uint next_i = (i + 1) % SH_L;
        acc += node.sh[next_i] * 0.05f;

        // Normalize with energy injection
        float len = clen(acc);
        acc = (len < 1e-5f) ? float2(0,0) : (acc / len);
        acc += float2(0.005f, 0.003f);  // Energy injection

        // Decay
        newSH[i] = acc * 0.99f;
    }

    // Commit
    for (uint j = 0; j < SH_L; j++)
        node.sh[j] = newSH[j];
}

void op_collapse(inout float coherence, in SHNode node)
{
    // Measure field coherence (for diagnostics / output)
    float sum = 0.0f;
    for (uint k = 0; k < SH_L; k++)
    {
        sum += clen(node.sh[k]);
    }
    coherence = sum / (float)SH_L;
}

// Main kernel
[numthreads(256, 1, 1)]
void WaveVM_Main(uint id : SV_DispatchThreadID)
{
    // Bounds check
    if (id >= 2048) return;  // Max 2048 nodes per dispatch

    SHNode node = NodesIn[id];

    // Execute program
    float coherence = 0.0f;
    uint ip = 0;

    while (ip < MAX_INSTR)
    {
        Instr32 ins = Program[ip];
        ip++;  // Increment immediately

        switch (ins.op)
        {
            case 1:  // OP_INJECT
                op_inject(node, id, ins.b, ins.f0, ins.f1);
                break;

            case 2:  // OP_PROPAGATE
            {
                // Repeat propagation N times
                for (uint rep = 0; rep < ins.a; rep++)
                {
                    op_propagate(node, NodesIn);
                    
                    // Barrier to ensure all threads updated before next iteration
                    GroupMemoryBarrierWithGroupSync();
                }
            }
            break;

            case 3:  // OP_COLLAPSE
                op_collapse(coherence, node);
                break;

            case 5:  // OP_COMPARE (no-op in wave VM, used for control flow)
                // Placeholder for pattern matching
                break;

            case 7:  // OP_HALT
                ip = MAX_INSTR;  // Exit loop
                break;

            default:
                break;
        }
    }

    // Write result back
    NodesOut[id] = node;
}
