// GeometryFieldMap.h — DirectX Geometry → π-KUHUL Field Topology Isomorphism
//
// EVERY geometry function in Geometry.cpp maps to a distinct π-KUHUL field topology.
// One geometry. One field. One inference engine.
//
// Complete isomorphism table:
//
// ┌──────────────────────┬──────────────────────────────────────────────────────┐
// │ Geometry.cpp fn      │ π-KUHUL field topology                               │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeGeoSphere     │ Icosahedral subdivision manifold                     │
// │                      │ tokens = vertices (uniform sphere distribution)       │
// │                      │ faces  = card regions (52 shards)                    │
// │                      │ normals = field gradient directions                  │
// │                      │ UV = Maya Long Count (u=longitude/2π, v=latitude/π)  │
// │                      │ tessellation = field resolution                      │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeSphere        │ UV sphere — latitude ring field                      │
// │                      │ rings = K'ayab' phase iterations                     │
// │                      │ each ring = one Maya uinal (18 segments)             │
// │                      │ poles = singularity tokens (special carry nodes)     │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeTorus         │ Memory hierarchy field topology                      │
// │                      │ major-radius = capacity / bandwidth                  │
// │                      │ minor-radius = latency / access pattern              │
// │                      │ outerAngle = phase [0, 2π) progression               │
// │                      │ innerAngle = token carry field rotation              │
// │                      │ stride = tokens per uinal                            │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeIcosahedron   │ Mayan vigesimal field (20 faces = one uinal)         │
// │                      │ 12 vertices = 12 Haab months                         │
// │                      │ 20 faces   = base-20 digit space                     │
// │                      │ 30 edges   = 30-day uinal + bond connections         │
// │                      │ Golden ratio φ = pi_mod per card                     │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeDodecahedron  │ Dual of icosahedron (12 faces = Haab calendar)       │
// │                      │ 20 vertices = 20 Maya base digits                    │
// │                      │ 12 faces    = 12 months in Haab 365-day year         │
// │                      │ 5-sided face = 5 Wayeb' days (year end)              │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeTetrahedron   │ Minimal phase field (4 faces = Pop/Wo/Sek/Ch'en)    │
// │                      │ 4 vertices = 4 K'uhul phase states                  │
// │                      │ √2, √3, √6 coords = phase offset constants          │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeOctahedron    │ 8-card field (8 faces = 8 bit byte / 8D tensor axis) │
// │                      │ Starting topology for GeoSphere subdivision          │
// │                      │ North/south poles = singularity carry nodes          │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeBox           │ 6-face orthogonal field (attention + FFN layers)     │
// │                      │ 6 faces = 6 transformer layers (GPT-2 small)         │
// │                      │ face normals = attention head directions             │
// │                      │ half-size = key-value dimension                      │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeCylinder      │ Sequential pipeline field (belt of K'ayab' loops)   │
// │                      │ height = temporal depth (sequence length)            │
// │                      │ diameter = feature dimension                         │
// │                      │ caps = BOS/EOS token singularities                   │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeCone          │ Attention pyramid field (wide base → apex)           │
// │                      │ base = input token distribution                      │
// │                      │ apex = focused output token                          │
// │                      │ cone normal = attention weighting gradient           │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeTeapot        │ Bezier field evolution trajectory                    │
// │                      │ control points = token path waypoints                │
// │                      │ Bezier patches = smooth field interpolation          │
// │                      │ mirrorZ = bidirectional phase propagation            │
// │                      │ handle/spout = asymmetric feature pathways           │
// └──────────────────────┴──────────────────────────────────────────────────────┘
//
// Maya Calendar ↔ Platonic solid isomorphism:
//   Tzolk'in (260-day):  13 × 20 = 13 baktun × 20-base digits (icosahedron dual)
//   Haab'    (365-day):  18 × 20 + 5 = 12 dodecahedron faces × 5-gon + 5 Wayeb'
//   Long Count:          5-level hierarchy = 5 nested topology subdivisions
//
// Tessellation = field resolution:
//   tessellation=1  → 80  vertices  (raw octahedron subdivision)
//   tessellation=2  → 320 vertices
//   tessellation=3  → 1280 vertices
//   tessellation=4  → 2562 vertices (~2560 = 128 × 20 vigesimal blocks)
//   tessellation=5  → 5120 vertices  (5120 = 256 × 20)
//
// K'uhul phase mapping per geometry step:
//   Pop:   create empty vertex/index collections
//   Wo:    compute vertex position (XMVector3Normalize + scale to radius)
//   Sek:   compute normal (XMVector3Cross + Normalize) and texture coords
//   Ch'en: push_back to vertex/index collections
//   Xul:   apply ReverseWinding (LH↔RH) and/or InvertNormals

#pragma once
#ifndef GEOMETRY_FIELD_MAP_H
#define GEOMETRY_FIELD_MAP_H

#include "pi_kuhul_bridge.h"
#include "Geometry.h"

using namespace DirectX;
using namespace PiKuhul;

namespace GeoField {

// ─── Field topology tags ──────────────────────────────────────────────────────

enum class FieldTopology : uint32_t {
    GeoSphere    = 0,  // icosahedral subdivision — uniform sphere distribution
    UVSphere     = 1,  // latitude ring field
    Torus        = 2,  // memory hierarchy — major/minor radius
    Icosahedron  = 3,  // Mayan vigesimal (20 faces = 1 uinal)
    Dodecahedron = 4,  // Haab calendar (12 faces = 12 months)
    Tetrahedron  = 5,  // minimal phase (4 faces = Pop/Wo/Sek/Ch'en)
    Octahedron   = 6,  // 8-face field (8D tensor / byte)
    Box          = 7,  // orthogonal attention/FFN layer field
    Cylinder     = 8,  // sequential pipeline belt
    Cone         = 9,  // attention pyramid
    Teapot       = 10, // Bezier trajectory field
};

// ─── Dispatch: geometry → tokens + cards ─────────────────────────────────────

inline void GeometryToField(
    FieldTopology               topology,
    const VertexCollection&     vertices,
    const IndexCollection&      indices,
    std::vector<Token>&         tokens,
    std::vector<CardField>&     cards,
    float                       scale = 1.0f)
{
    switch (topology) {
    case FieldTopology::GeoSphere:
    case FieldTopology::UVSphere:
    case FieldTopology::Icosahedron:
    case FieldTopology::Dodecahedron:
    case FieldTopology::Tetrahedron:
    case FieldTopology::Octahedron:
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        break;

    case FieldTopology::Torus:
        // Torus: major/minor radius encodes memory bandwidth/latency
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        // Tag tokens with torus phase (outerAngle = phase, innerAngle = carry rotation)
        for (size_t i = 0; i < tokens.size(); i++) {
            tokens[i].phase = tokens[i].phase * 2.0f; // double wrap for torus dual cycle
        }
        break;

    case FieldTopology::Box:
        // Box: 6 faces = 6 attention/FFN layers
        // Normal direction = attention head orientation
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        for (size_t i = 0; i < cards.size() && i < 6; i++) {
            cards[i].maya_digit = static_cast<uint32_t>(i); // face 0-5 = layer 0-5
        }
        break;

    case FieldTopology::Cylinder:
        // Cylinder: sequential belt — height = sequence length
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        for (size_t i = 0; i < tokens.size(); i++) {
            // remap phase to [0, 2π] along height axis (y-coordinate)
            tokens[i].phase = (tokens[i].position.y / scale + 1.0f) * static_cast<float>(TAU) * 0.5f;
        }
        break;

    case FieldTopology::Cone:
        // Cone: attention pyramid — apex = focus token
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        for (auto& t : tokens) {
            // coherence inversely proportional to distance from apex (y=height/2)
            t.coherence = 1.0f - (t.position.y / scale + 0.5f);
        }
        break;

    case FieldTopology::Teapot:
        // Teapot: Bezier trajectory — control points as waypoints
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        // mirrorZ patches get reversed phase (bidirectional propagation)
        for (size_t i = tokens.size() / 2; i < tokens.size(); i++) {
            tokens[i].phase_velocity = -tokens[i].phase_velocity;
        }
        break;

    default:
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        break;
    }
}

// ─── Topology metadata ────────────────────────────────────────────────────────

struct TopologyInfo {
    FieldTopology topology;
    const char*   name;
    uint32_t      faces;
    uint32_t      vertices;
    uint32_t      edges;
    float         pi_mod_default;  // default pi_mod for cards
    const char*   maya_analogue;
};

static constexpr TopologyInfo TOPOLOGY_TABLE[] = {
    { FieldTopology::GeoSphere,    "GeoSphere",    0,  0,  0, 1.0f,  "icosahedral subdivision" },
    { FieldTopology::UVSphere,     "UVSphere",     0,  0,  0, 1.0f,  "latitude rings = uinals" },
    { FieldTopology::Torus,        "Torus",        0,  0,  0, 2.0f,  "dual-cycle memory hierarchy" },
    { FieldTopology::Icosahedron,  "Icosahedron",  20, 12, 30, 1.618f, "20 faces = 1 uinal (base-20)" },
    { FieldTopology::Dodecahedron, "Dodecahedron", 12, 20, 30, 1.0f,  "12 faces = 12 Haab months" },
    { FieldTopology::Tetrahedron,  "Tetrahedron",  4,  4,  6, 1.0f,  "4 faces = Pop/Wo/Sek/Ch'en" },
    { FieldTopology::Octahedron,   "Octahedron",   8,  6,  12, 1.0f, "8 faces = 8D tensor axes" },
    { FieldTopology::Box,          "Box",          6,  8,  12, 1.0f, "6 faces = 6 transformer layers" },
    { FieldTopology::Cylinder,     "Cylinder",     0,  0,  0,  1.0f, "belt = sequential pipeline" },
    { FieldTopology::Cone,         "Cone",         0,  0,  0,  1.0f, "pyramid = attention focus" },
    { FieldTopology::Teapot,       "Teapot",       0,  0,  0,  1.0f, "Bezier = trajectory field" },
};

inline const TopologyInfo* GetTopologyInfo(FieldTopology t) {
    for (const auto& info : TOPOLOGY_TABLE)
        if (info.topology == t) return &info;
    return nullptr;
}

// ─── π-KUHUL shape → tessellation recommendation ────────────────────────────

inline size_t RecommendedTessellation(FieldTopology t, uint32_t desired_tokens) {
    // GeoSphere: tokens ≈ 10 * 4^tess + 2
    // Aim for desired_tokens
    if (t == FieldTopology::GeoSphere || t == FieldTopology::UVSphere) {
        size_t tess = 1;
        size_t count = 80;
        while (count < desired_tokens && tess < 8) { tess++; count *= 4; }
        return tess;
    }
    return 3; // default for other topologies
}

} // namespace GeoField

#endif // GEOMETRY_FIELD_MAP_H
