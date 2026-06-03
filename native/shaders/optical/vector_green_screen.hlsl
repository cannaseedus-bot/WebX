// vector_green_screen.hlsl — Vector green screen shader
//
// The green screen IS the spherical manifold S² — an infinite coordinate plane
// where every point (θ,φ) is mathematically addressable.
//
// This is NOT chroma key removal. The green screen is the EXECUTION SURFACE:
//   green   = S² coordinate grid (the manifold itself)
//   vectors = geodesic paths between optical nodes (data flows)
//   arcs    = replayable ARC trajectories (compute history)
//
// Pixel shader outputs:
//   RGB = geodesic grid lines on S² (green = empty, color = compute activity)
//   A   = 1.0 always (the surface is always present — it IS the compute fabric)
//
// K'UHUL phase: Wo (the surface is declared/bound before execution)
// The green screen is the [Wo] declaration — it declares the coordinate space
// within which all subsequent [Sek] compute operations take place.
//
// cs_5_0 DXBC — HD 4600 compatible, no DXIL

cbuffer ScreenConstants : register(b0) {
    float4x4 viewProj;
    float3   cameraPos;
    float    time;
    float    piPhase;       // current K'UHUL π-phase (0→2π)
    float    globalEnergy;  // mean SH coherence (fog/clarity of the grid)
    float2   viewport;
    float3   sphereCenter;  // center of the manifold S²
    float    sphereRadius;  // R = 1/√κ
    float    pad;
};

// Geodesic arc buffer (from ReplayableArc / ARC weight system)
struct ArcData {
    float3 start;    // start point on S²
    float3 end;      // end point on S²
    float  quality;  // arc replay quality [0,1]
    float  entropy;  // mean entropy along arc [0,1]
    float  age;      // how many ticks since last replay
    float  pad;
};
StructuredBuffer<ArcData> Arcs : register(t0);
uint arcCount;   // in cbuffer or root constant

// ─── Utility ─────────────────────────────────────────────────────────────────

#define PI 3.14159265359

// Point → spherical coordinates (θ, φ)
float2 CartesianToSpherical(float3 p) {
    float3 n = normalize(p);
    float theta = acos(clamp(n.y, -1.0, 1.0));
    float phi   = atan2(n.x, -n.z);
    return float2(theta, phi);
}

// Geodesic distance on S² (great circle)
float GeodesicDist(float3 a, float3 b) {
    float d = dot(normalize(a), normalize(b));
    return acos(clamp(d, -1.0 + 1e-6, 1.0 - 1e-6));
}

// Slerp between two points on S²
float3 Slerp(float3 a, float3 b, float t) {
    float cosA = dot(normalize(a), normalize(b));
    cosA = clamp(cosA, -1.0, 1.0);
    float A = acos(cosA);
    if (A < 1e-6) return a;
    return (sin((1.0-t)*A)*a + sin(t*A)*b) / sin(A);
}

// ─── Vertex shader (full-screen quad for the green screen surface) ────────────

struct VS_INPUT  { float4 pos : POSITION; float2 uv : TEXCOORD; };
struct VS_OUTPUT { float4 svPos : SV_Position; float2 uv : TEXCOORD0;
                   float3 worldRay : TEXCOORD1; };

VS_OUTPUT VS_GreenScreen(VS_INPUT i) {
    VS_OUTPUT o;
    o.svPos    = i.pos;
    o.uv       = i.uv;
    // Reconstruct world ray from clip-space position
    float4x4 invVP = /* pass inverse as constant */ viewProj; // placeholder
    float4 clip    = float4(i.pos.xy, 0.0, 1.0);
    float4 world   = mul(invVP, clip);
    o.worldRay = normalize(world.xyz / world.w - cameraPos);
    return o;
}

// ─── Pixel shader ─────────────────────────────────────────────────────────────

float4 PS_GreenScreen(VS_OUTPUT input) : SV_Target {
    // Ray-sphere intersection (S² at sphereCenter, radius sphereRadius)
    float3 ro = cameraPos - sphereCenter;
    float3 rd = normalize(input.worldRay);
    float  b  = dot(ro, rd);
    float  c  = dot(ro, ro) - sphereRadius * sphereRadius;
    float  h  = b*b - c;

    // Off-sphere: show faint dark green (the manifold edge)
    if (h < 0.0) return float4(0.0, 0.05, 0.02, 1.0);

    float  t   = -b - sqrt(h);
    float3 hit = ro + rd * t;   // hit point on S² (in sphere-local coords)
    float2 sph = CartesianToSpherical(hit);  // (θ, φ)

    // ── Green screen coordinate grid ──────────────────────────────────────────
    // Draw geodesic grid lines at multiples of π/6 (30°)
    float  gridTheta = fmod(abs(sph.x), PI/6.0);
    float  gridPhi   = fmod(abs(sph.y) + PI, PI/6.0);
    float  lineW     = 0.004;   // line width in radians
    bool   onGrid    = gridTheta < lineW || gridPhi < lineW;

    // Base: pure green (the "green screen" — the empty coordinate plane)
    float3 baseColor = float3(0.0, 0.15, 0.05);
    float3 gridColor = float3(0.0, 0.35, 0.12);   // brighter grid lines

    float3 col = onGrid ? gridColor : baseColor;

    // ── Equator and prime meridian (stronger lines) ───────────────────────────
    bool equator  = abs(sph.x - PI*0.5) < lineW * 2.0;
    bool meridian = abs(sph.y) < lineW * 2.0;
    if (equator || meridian) col = float3(0.0, 0.6, 0.2);

    // ── ARC trajectories on the green screen ──────────────────────────────────
    // High-quality arcs glow cyan; high-entropy arcs glow dim orange
    float arcGlow = 0.0;
    float3 arcCol = float3(0,0,0);

    // (In production: loop over GPU buffer; here simplified to first 32 arcs)
    for (uint a = 0; a < min(arcCount, 32u); a++) {
        ArcData arc = Arcs[a];
        // Minimum geodesic distance from hit point to the arc
        float minDist = 9999.0;
        for (float tp = 0.0; tp <= 1.0; tp += 0.05) {
            float3 arcPt = Slerp(arc.start, arc.end, tp);
            float  d = GeodesicDist(hit, arcPt * sphereRadius);
            minDist = min(minDist, d);
        }
        float arcWidth = 0.03 * arc.quality;
        if (minDist < arcWidth) {
            float intensity = (1.0 - minDist / arcWidth) * arc.quality;
            // Cyan for high quality, orange for high entropy
            float3 hq = float3(0.0, 0.9, 1.0);   // cyan = practiced arc
            float3 he = float3(1.0, 0.4, 0.0);   // orange = foggy arc
            arcCol  = lerp(hq, he, arc.entropy) * intensity;
            arcGlow = max(arcGlow, intensity);
        }
    }
    col = lerp(col, arcCol, arcGlow * 0.8);

    // ── π-phase pulse ─────────────────────────────────────────────────────────
    // The grid brightens at multiples of π — marks compute barriers
    float phaseBright = 0.5 + 0.5 * sin(piPhase * 2.0);
    col *= (0.85 + 0.15 * phaseBright);

    // ── Global energy fog ─────────────────────────────────────────────────────
    // Low coherence = foggy green screen (uncertainty); high = crisp
    float fog = 1.0 - globalEnergy * 0.5;
    col = lerp(col, float3(0.0, 0.08, 0.03), fog * 0.4);

    // ── Sphere edge darkening ─────────────────────────────────────────────────
    float3 N   = normalize(hit);
    float  rim = dot(N, -rd);
    col *= (0.5 + 0.5 * rim);

    return float4(col, 1.0);
}
