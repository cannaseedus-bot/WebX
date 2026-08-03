// SH Wave Propagation Kernel
// GPU compute shader for spherical harmonic wave evolution

#define SH_BANDS 9

struct OpticalNode {
    float2 sh[SH_BANDS];    // Phase state: (cos, sin) per mode
    uint3 neighbors;         // Geodesic lattice connectivity
};

StructuredBuffer<OpticalNode> Nodes : register(t0);
RWStructuredBuffer<OpticalNode> OutNodes : register(u0);

// Rotate phase vector by angle
float2 rotate_phase(float2 v, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return float2(v.x*c - v.y*s, v.x*s + v.y*c);
}

[numthreads(256,1,1)]
void CS_WavePropagation(uint id : SV_DispatchThreadID)
{
    OpticalNode node = Nodes[id];
    OpticalNode outNode;

    // Process each SH band
    for (int i = 0; i < SH_BANDS; i++)
    {
        float2 accum = node.sh[i];

        // Neighbor coupling (wave propagation on geodesic lattice)
        [unroll]
        for (int n = 0; n < 3; n++)
        {
            uint neighbor_id = node.neighbors[n];
            if (neighbor_id > 0)
            {
                accum += Nodes[neighbor_id].sh[i] * 0.333;
            }
        }

        // Harmonic-dependent phase rotation (frequency mixing)
        float l = floor(sqrt(float(i)));  // rough harmonic order
        float phase_freq = l * 0.5;       // frequency scaling per band

        accum = rotate_phase(accum, phase_freq);

        // Normalize to maintain phase coherence
        float len = max(length(accum), 1e-4);
        accum /= len;

        outNode.sh[i] = accum;
    }

    outNode.neighbors = node.neighbors;
    OutNodes[id] = outNode;
}
