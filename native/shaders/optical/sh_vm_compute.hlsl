// GPU Wave VM Compute Shader
// Executes programs on GPU directly on wave field regions

#define SH_BANDS 9
#define REGION_SIZE 64

struct SHNode {
    float2 sh[SH_BANDS];
};

struct Instruction {
    uint op_flags;
    uint a, b, c;
    int f0, f1;  // fixed-point
};

StructuredBuffer<SHNode> Nodes : register(t0);
StructuredBuffer<Instruction> Program : register(t1);
RWStructuredBuffer<SHNode> OutNodes : register(u0);

float fixed_to_float(int f) {
    return float(f) / 32767.0f;
}

float2 rotate_phase(float2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

[numthreads(64, 1, 1)]
void CS_WaveVM(uint id : SV_DispatchThreadID)
{
    if (id >= REGION_SIZE)
        return;

    SHNode node = Nodes[id];
    uint ip = 0;
    float last_result = 0.0f;

    // Fetch program size from constant buffer (simplified: assume 32 max)
    uint prog_size = 32;

    while (ip < prog_size)
    {
        Instruction ins = Program[ip];
        uint op = ins.op_flags & 0x1F;

        switch (op)
        {
            case 0: // OP_INJECT
            {
                uint node_id = ins.a;
                uint band = ins.b;
                float amp = fixed_to_float(ins.f0);
                float phase = fixed_to_float(ins.f1);

                if (node_id == id && band < SH_BANDS)
                {
                    float c = cos(phase);
                    float s = sin(phase);
                    node.sh[band].x += amp * c;
                    node.sh[band].y += amp * s;
                }
                break;
            }

            case 1: // OP_PROPAGATE
            {
                uint steps = ins.a;
                for (uint step = 0; step < steps; step++)
                {
                    SHNode temp = node;

                    for (int j = 0; j < SH_BANDS; j++)
                    {
                        float x = node.sh[j].x;
                        float y = node.sh[j].y;

                        // Cross-band coupling
                        if (j + 1 < SH_BANDS)
                        {
                            x += node.sh[(j + 1) % SH_BANDS].x * 0.05f;
                            y += node.sh[(j + 1) % SH_BANDS].y * 0.05f;
                        }

                        // Phase rotation
                        float angle = (j + 1) * 0.02f;
                        float2 rot = rotate_phase(float2(x, y), angle);

                        // Normalize
                        float len = length(rot);
                        if (len > 1e-4f)
                            rot = rot / len;

                        // Energy injection
                        rot.x += 0.005f;
                        rot.y += 0.003f;

                        // Decay
                        rot *= 0.99f;

                        temp.sh[j] = rot;
                    }

                    node = temp;
                }
                break;
            }

            case 2: // OP_COLLAPSE
            {
                float total = 0.0f;
                for (int i = 0; i < SH_BANDS; i++)
                {
                    total += length(node.sh[i]);
                }
                last_result = total / SH_BANDS;
                break;
            }

            case 5: // OP_HALT
                ip = prog_size;  // Exit loop
                break;

            default:
                break;
        }

        ip++;
    }

    OutNodes[id] = node;
}

// Directional Wave Propagation Shader
// Includes propagation vectors for guided routing

struct NodeWithDir {
    float2 sh[SH_BANDS];
    float3 pos;
    float3 propagation_dir;
};

StructuredBuffer<NodeWithDir> DirectedNodes : register(t0);
RWStructuredBuffer<NodeWithDir> OutDirected : register(u0);

[numthreads(64, 1, 1)]
void CS_DirectedPropagation(uint id : SV_DispatchThreadID)
{
    if (id >= REGION_SIZE)
        return;

    NodeWithDir node = DirectedNodes[id];
    NodeWithDir out_node = node;

    for (int j = 0; j < SH_BANDS; j++)
    {
        float x = node.sh[j].x;
        float y = node.sh[j].y;

        // **Directional neighbor weighting** (key insight)
        // Instead of uniform 0.33 coupling, weight by propagation direction
        // (In real implementation: access neighbor nodes and compute direction)

        float3 dir = normalize(node.propagation_dir);

        // Placeholder: uniform coupling (GPU version would fetch neighbors)
        float coupling_strength = 0.05f;
        x += x * coupling_strength;  // Simplified for example
        y += y * coupling_strength;

        // Apply directional bias (preferentially rotate toward propagation_dir)
        float angle = (j + 1) * 0.02f;
        // Weight by dot product with propagation direction (0..1 range)
        // angle *= saturate(dot(dir, float3(1, 0, 0)));  // Simplified

        float2 rot = rotate_phase(float2(x, y), angle);

        // Normalize & inject energy
        float len = max(length(rot), 1e-5f);
        rot = rot / len;
        rot.x += 0.005f;
        rot.y += 0.003f;
        rot *= 0.99f;

        out_node.sh[j] = rot;
    }

    OutDirected[id] = out_node;
}
