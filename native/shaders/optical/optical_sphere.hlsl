// optical_sphere.hlsl — Sphere shader for optical compute nodes
//
// Each optical node (SH wave lattice vertex) renders as a lit sphere.
// The sphere is NOT decorative — it encodes compute state:
//   radius    = node energy (SH coherence)
//   color     = π-phase hue mapped to SH band dominance
//   brightness = wave amplitude at the dominant band
//   rim glow  = neighbour coupling strength
//
// Vertex shader: places the sphere billboard at node position
// Pixel shader:  ray-sphere intersection + SH-modulated lighting
//
// K'UHUL phase: Ch'en (result visualization after Sek compute dispatch)
// Maps to:
//   OpticalNode.pos       → sphere center
//   OpticalNode.sh[9×2]  → color/brightness/glow
//   OpticalNode.energy()  → sphere radius scale

// ─── Constant buffer ─────────────────────────────────────────────────────────

cbuffer FrameConstants : register(b0) {
    float4x4 view;
    float4x4 proj;
    float3   cameraPos;
    float    time;          // current time in seconds (for animation)
    float2   viewport;      // screen size
    float    globalEnergy;  // mean SH coherence across all nodes
    float    piPhase;       // current π-phase (0→2π)
};

// ─── Per-node data (structured buffer from optical mesh) ──────────────────────

struct OpticalNodeData {
    float3 position;        // sphere center on S²
    float  energy;          // mean SH amplitude
    float  phase;           // π-phase of this node
    float  sh_band0;        // Y00 coefficient (isotropic baseline)
    float  sh_band1;        // Y10 coefficient (vertical gradient)
    float  sh_band2;        // Y11 coefficient (horizontal gradient)
    float  neighbourEnergy; // mean energy of k nearest neighbours
    float  pad;
};

StructuredBuffer<OpticalNodeData> Nodes : register(t0);

// ─── Vertex shader ────────────────────────────────────────────────────────────
// Billboard: expand each node into a screen-aligned quad

struct VS_INPUT {
    float3 localPos   : POSITION;   // quad corner [-1,1] × [-1,1]
    uint   nodeId     : SV_InstanceID;
};

struct VS_OUTPUT {
    float4 svPos      : SV_Position;
    float2 localUV    : TEXCOORD0;  // quad UV for ray-sphere
    float3 nodeCenter : TEXCOORD1;  // world-space sphere center
    float  radius     : TEXCOORD2;
    float  energy     : TEXCOORD3;
    float  phase      : TEXCOORD4;
    float3 shColor    : TEXCOORD5;  // SH-derived color
    float  glowStrength: TEXCOORD6;
};

// Map SH phase → hue (same scale as visual music entropy)
float3 PhaseToColor(float phase, float energy) {
    // 12 musical intervals mapped to hue (from kuhul-visual-music-entropy.kuhul)
    float hue = fmod(phase / (2.0 * 3.14159265), 1.0);
    // HSV → RGB (saturation = energy, value = 1)
    float h6 = hue * 6.0;
    float c  = energy;
    float x  = c * (1.0 - abs(fmod(h6, 2.0) - 1.0));
    float3 rgb;
    if      (h6 < 1) rgb = float3(c, x, 0);
    else if (h6 < 2) rgb = float3(x, c, 0);
    else if (h6 < 3) rgb = float3(0, c, x);
    else if (h6 < 4) rgb = float3(0, x, c);
    else if (h6 < 5) rgb = float3(x, 0, c);
    else             rgb = float3(c, 0, x);
    return rgb;
}

VS_OUTPUT VS_OpticalSphere(VS_INPUT input) {
    OpticalNodeData node = Nodes[input.nodeId];

    // Sphere radius scales with energy (min 0.01, max 0.05 world units)
    float r = max(0.01, node.energy * 0.05);

    // Billboard: expand quad in view space
    float4 worldCenter = float4(node.position, 1.0);
    float4 viewCenter  = mul(view, worldCenter);
    float4 cornerView  = viewCenter + float4(input.localPos.xy * r, 0, 0);

    VS_OUTPUT o;
    o.svPos       = mul(proj, cornerView);
    o.localUV     = input.localPos.xy;
    o.nodeCenter  = node.position;
    o.radius      = r;
    o.energy      = node.energy;
    o.phase       = node.phase;
    o.shColor     = PhaseToColor(node.phase, saturate(node.energy * 2.0));
    o.glowStrength = saturate(node.neighbourEnergy * 3.0);
    return o;
}

// ─── Pixel shader ─────────────────────────────────────────────────────────────

struct PS_OUTPUT {
    float4 color : SV_Target;
};

PS_OUTPUT PS_OpticalSphere(VS_OUTPUT input) {
    PS_OUTPUT o;

    // Ray-sphere intersection in billboard space
    float2 uv = input.localUV;
    float  d2 = dot(uv, uv);
    if (d2 > 1.0) discard;   // outside sphere silhouette

    // Surface normal (sphere in view space)
    float3 N = normalize(float3(uv, sqrt(1.0 - d2)));

    // Simple directional light + SH ambient
    float3 L       = normalize(float3(0.5, 1.0, 0.8));
    float  diffuse = saturate(dot(N, L));
    float3 ambient = input.shColor * 0.3;

    // Core color from SH state
    float3 col = ambient + input.shColor * diffuse;

    // Rim glow — neighbour coupling visible as edge emission
    float rim = pow(1.0 - saturate(dot(N, float3(0,0,1))), 3.0);
    col += input.glowStrength * float3(0.0, 0.9, 1.0) * rim * 0.5;

    // Energy pulse — subtle breathing at π-frequency
    float pulse = 0.9 + 0.1 * sin(time * 3.14159 + input.phase);
    col *= pulse;

    // Fresnel edge softening
    float alpha = 1.0 - smoothstep(0.85, 1.0, d2);

    o.color = float4(col, alpha);
    return o;
}
