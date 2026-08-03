#pragma once

#include <DirectXMath.h>
#include <cstdint>
#include <vector>
#include <cmath>

using namespace DirectX;

#define SH_BANDS 9
#define MAX_PATTERNS 256

// Opcode set for wave VM
enum OpCode : uint8_t
{
    OP_INJECT,      // Inject signal into node/band
    OP_PROPAGATE,   // Execute wave propagation step
    OP_COLLAPSE,    // Measure node coherence
    OP_COMPARE,     // Compare node vs pattern
    OP_ROUTE,       // Conditional branch
    OP_MEMORY,      // Store/retrieve pattern
    OP_HALT         // Stop execution
};

// Input signal structure
struct InputSignal
{
    uint32_t node_id;
    uint32_t band;
    float amplitude;
    float phase;
};

// VM Instruction (32 bytes)
#pragma pack(push, 1)
struct Instruction
{
    OpCode op;
    uint8_t flags;
    uint16_t pad0;
    
    uint32_t a;         // arg1 (node_id, pattern_id, etc.)
    uint32_t b;         // arg2 (band, node_id, etc.)
    uint32_t c;         // arg3 (jump target, etc.)
    
    float f0, f1;       // float args (amplitude, phase, threshold)
};
#pragma pack(pop)

// Pattern/Attractor signature
struct Pattern
{
    float signature[SH_BANDS];
    float stability;
    uint32_t discovered_frame;
};

// Node state in wave field
struct SHNodeCPU
{
    XMFLOAT2 sh[SH_BANDS];
};

// VM execution context
struct VMContext
{
    std::vector<SHNodeCPU>* field;          // Pointer to wave field
    std::vector<Pattern>* patterns;         // Attractor library
    std::vector<Instruction>* program;      // Instruction stream
    
    uint32_t ip;                            // Instruction pointer
    float last_result;                      // Last measurement result
    uint32_t frame_count;
};

// Feature extraction
inline float read_band_energy(const SHNodeCPU& node, int band)
{
    float x = node.sh[band].x;
    float y = node.sh[band].y;
    return sqrtf(x * x + y * y);
}

// Distance metric (SH space)
inline float pattern_distance(const Pattern& p, const SHNodeCPU& node)
{
    float d = 0.0f;
    for (int i = 0; i < SH_BANDS; i++)
    {
        float e = read_band_energy(node, i);
        float diff = e - p.signature[i];
        d += diff * diff;
    }
    return sqrtf(d);
}

// Extract signature from entire field
inline void extract_signature(const std::vector<SHNodeCPU>& nodes, Pattern& p)
{
    for (int i = 0; i < SH_BANDS; i++)
        p.signature[i] = 0.0f;
    
    for (const auto& n : nodes)
    {
        for (int i = 0; i < SH_BANDS; i++)
        {
            p.signature[i] += read_band_energy(n, i);
        }
    }
    
    float norm = 1.0f / nodes.size();
    for (int i = 0; i < SH_BANDS; i++)
        p.signature[i] *= norm;
}

// Coherence metric (global)
inline float compute_coherence(const std::vector<SHNodeCPU>& nodes)
{
    float total = 0.0f;
    for (const auto& n : nodes)
    {
        for (int i = 0; i < SH_BANDS; i++)
        {
            total += read_band_energy(n, i);
        }
    }
    return total / (nodes.size() * SH_BANDS);
}
