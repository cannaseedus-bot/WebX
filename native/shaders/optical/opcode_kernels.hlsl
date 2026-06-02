// GPU Opcode Kernels - ISA execution via separate dispatch calls
// Design: Each opcode is a separate compute shader entry point
// CPU dispatcher calls them sequentially (no dynamic loops in shader)

#define SH_L 9
#define THREAD_GROUP_SIZE 256

// Instruction structure (matches Instr32 from gpu_wave_dispatcher.h)
struct Instruction {
    uint op;
    uint flags;
    uint a, b, c;
    float f0, f1, f2, f3;
};

// GPU node state (matches GPUSHNode)
struct SHNode {
    float sh[SH_L * 2];    // phase state (cos, sin) x 9 bands
    uint neighborCount;
    uint neighbors[6];
    uint pad[2];
};

// ============================================================================
// INPUT BUFFERS
// ============================================================================

StructuredBuffer<Instruction> Program : register(t0);
StructuredBuffer<SHNode> NodesIn : register(t1);

// ============================================================================
// OUTPUT BUFFERS
// ============================================================================

RWStructuredBuffer<SHNode> NodesOut : register(u0);

// ============================================================================
// OP_INJECT KERNEL
// Each thread injects energy into its node if node == instruction.a
// ============================================================================

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void OP_INJECT_Main(uint id : SV_DispatchThreadID)
{
    // Read current state
    SHNode node = NodesIn[id];
    
    // Read instruction from constant buffer (single instruction per dispatch)
    Instruction ins = Program[0];
    
    // If this thread's node matches the target node
    if (id == ins.a)
    {
        uint band = ins.b;
        if (band < SH_L)
        {
            float amp = ins.f0;
            float phase = ins.f1;
            
            // Inject: add (amp*cos(phase), amp*sin(phase)) to band
            float c = cos(phase);
            float s = sin(phase);
            
            node.sh[band * 2 + 0] += amp * c;  // cos component
            node.sh[band * 2 + 1] += amp * s;  // sin component
        }
    }
    
    // Write state
    NodesOut[id] = node;
}

// ============================================================================
// OP_PROPAGATE KERNEL
// Wave physics: each node mixes with neighbors + applies decay + coupling
// ============================================================================

float2 rot(float2 v, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void OP_PROPAGATE_Main(uint id : SV_DispatchThreadID)
{
    SHNode node = NodesIn[id];
    SHNode result = (SHNode)0;
    
    // Propagate each band independently
    for (uint band = 0; band < SH_L; band++)
    {
        float2 state = float2(node.sh[band * 2], node.sh[band * 2 + 1]);
        float2 mixed = state * 0.7f;  // Self-persistence
        
        // Accumulate from neighbors
        for (uint k = 0; k < node.neighborCount; k++)
        {
            uint neighbor_id = node.neighbors[k];
            SHNode neighbor = NodesIn[neighbor_id];
            
            float2 neighbor_state = float2(
                neighbor.sh[band * 2],
                neighbor.sh[band * 2 + 1]
            );
            
            // Weighted neighbor coupling
            mixed += neighbor_state * 0.1f;
        }
        
        // Apply harmonic rotation (l-dependent phase shift)
        float l = floor(sqrt((float)band));
        mixed = rot(mixed, l * 0.1f);
        
        // Cross-band coupling (weak)
        uint next_band = (band + 1) % SH_L;
        float2 next_state = float2(
            node.sh[next_band * 2],
            node.sh[next_band * 2 + 1]
        );
        mixed += next_state * 0.02f;
        
        // Normalize and apply energy decay
        float len = length(mixed);
        if (len > 1e-5f)
        {
            mixed = normalize(mixed) * min(len, 1.0f);
        }
        
        // Add small energy injection
        mixed += float2(0.002f, 0.001f);
        
        // Decay
        mixed *= 0.99f;
        
        result.sh[band * 2] = mixed.x;
        result.sh[band * 2 + 1] = mixed.y;
    }
    
    // Copy topology
    result.neighborCount = node.neighborCount;
    for (uint i = 0; i < 6; i++)
        result.neighbors[i] = node.neighbors[i];
    
    NodesOut[id] = result;
}

// ============================================================================
// OP_COLLAPSE KERNEL
// Extract attractor signature (optional for now - just copy state)
// ============================================================================

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void OP_COLLAPSE_Main(uint id : SV_DispatchThreadID)
{
    // For now, just preserve state
    // In future: extract pattern signature, compute coherence, etc.
    SHNode node = NodesIn[id];
    NodesOut[id] = node;
}

// ============================================================================
// OP_NOP KERNEL (No-op for safety)
// ============================================================================

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void OP_NOP_Main(uint id : SV_DispatchThreadID)
{
    SHNode node = NodesIn[id];
    NodesOut[id] = node;
}
