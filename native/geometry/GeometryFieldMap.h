// GeometryFieldMap.h — Neural Runtime Tokenization Guide
//
// EVERY geometry function in Geometry.cpp maps to a distinct π-KUHUL field topology.
// One geometry. One field. One inference engine.
//
// ARCHITECTURAL PRINCIPLE:
//   Geometry is not symbolic analogy — it is SEMANTIC EXECUTION FIELD.
//   Each face/vertex/edge represents an execution domain, not just a shape.
//
// CANONICAL PROGRESSION:
//   Source (Geometry)
//       ↓
//   Field          ← Semantic region (owns Cards)
//       ↓
//   Card           ← Schedulable partition (what runtime schedules)
//       ↓
//   Token          ← Atomic element (owned by Card)
//       ↓
//   Fiber          ← Execution context (runs over Cards)
//       ↓
//   Execution      ← XVM/GPU/CPU processing
//
// GeometryFieldMap is the REFERENCE IMPLEMENTATION of an NNC tokenizer for geometric data.
// It demonstrates how one input domain (geometry) is normalized into the common
// Field/Card/Token model that every runtime component consumes:
//   - NeuralGrammar.Core (K'UHUL execution)
//   - asx_ram (memory residency)
//   - asx_gemm (compute dispatch)
//   - optical_processor (vision tokenization)
//   - XVM (fiber scheduling)
//   - SCXQ2 (serialization)
//
// Execution Domains:
//   Geometry   → Spatial structure (where concepts live)
//   Vision     → Image → geometry → semantics (optical_processor)
//   Memory     → Hierarchy, caching, paging (asx_ram)
//   Compute    → GPU kernels, XVM execution (asx_gemm)
//   Attention  → QKV, focus, weighting (fused attention shaders)
//   Semantic   → Token relationships, graphs (SCXQ2)
//   Temporal   → Sequence, pipeline, evolution (evolution shader)
//   Execution  → Fiber state, phase transitions (XVM fibers)
//   Evolution  → Selection, recombination, mutation (evolution kernel)
//
// Complete isomorphism table:
//
// ┌──────────────────────┬──────────────────────────────────────────────────────┐
// │ Geometry.cpp fn      │ π-KUHUL field topology                               │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeBox           │ 6-face orthogonal execution field                    │
// │                      │ Domain: Compute                                      │
// │                      │ ExecClass: Attention                                 │
// │                      │ 6 faces = 6 transformer layers (GPT-2 small)         │
// │                      │ face normals = attention head directions             │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeCone          │ Attention pyramid field (wide base → apex)           │
// │                      │ Domain: Attention                                    │
// │                      │ ExecClass: Attention                                 │
// │                      │ base = input token distribution                      │
// │                      │ apex = focused output token                          │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeCylinder      │ Sequential pipeline field (belt of K'ayab' loops)   │
// │                      │ Domain: Temporal                                     │
// │                      │ ExecClass: Fiber                                     │
// │                      │ height = temporal depth (sequence length)            │
// │                      │ caps = BOS/EOS token singularities                   │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeTorus         │ Memory hierarchy field topology                      │
// │                      │ Domain: Memory                                       │
// │                      │ ExecClass: Fiber                                     │
// │                      │ major-radius = capacity / bandwidth                  │
// │                      │ minor-radius = latency / access pattern              │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeGeoSphere     │ Icosahedral subdivision manifold                     │
// │                      │ Domain: Geometry                                     │
// │                      │ ExecClass: Wave                                      │
// │                      │ tokens = vertices (uniform sphere distribution)       │
// │                      │ tessellation = field resolution                      │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeIcosahedron   │ Mayan vigesimal field (20 faces = one uinal)         │
// │                      │ Domain: Semantic                                     │
// │                      │ ExecClass: Scalar                                    │
// │                      │ 12 vertices = 12 Haab months                         │
// │                      │ 20 faces   = base-20 digit space                     │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeDodecahedron  │ Dual of icosahedron (12 faces = Haab calendar)       │
// │                      │ Domain: Semantic                                     │
// │                      │ ExecClass: Scalar                                    │
// │                      │ 12 faces    = 12 months in Haab 365-day year         │
// │                      │ 5-sided face = 5 Wayeb' days (year end)              │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeTetrahedron   │ Minimal phase field (4 faces = Pop/Wo/Sek/Ch'en)    │
// │                      │ Domain: Execution                                    │
// │                      │ ExecClass: Fiber                                     │
// │                      │ 4 vertices = 4 K'uhul phase states                  │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeOctahedron    │ 8-card field (8 faces = 8 bit byte / 8D tensor axis) │
// │                      │ Domain: Geometry                                     │
// │                      │ ExecClass: SIMD                                      │
// │                      │ Starting topology for GeoSphere subdivision          │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ ComputeTeapot        │ Bezier field evolution trajectory                    │
// │                      │ Domain: Evolution                                    │
// │                      │ ExecClass: Evolution                                 │
// │                      │ control points = token path waypoints                │
// │                      │ mirrorZ = bidirectional phase propagation            │
// ├──────────────────────┼──────────────────────────────────────────────────────┤
// │ OpticalProcessor     │ Vision pipeline (image → geometry → edges → freq)    │
// │                      │ Domain: Vision                                       │
// │                      │ ExecClass: Optical                                   │
// │                      │ Emits FieldTopology::OpticalField                    │
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
    GeoSphere    = 0,   // Icosahedral subdivision — uniform sphere distribution
    UVSphere     = 1,   // Latitude ring field
    Torus        = 2,   // Memory hierarchy — major/minor radius
    Icosahedron  = 3,   // Mayan vigesimal (20 faces = 1 uinal)
    Dodecahedron  = 4,  // Haab calendar (12 faces = 12 months)
    Tetrahedron  = 5,   // Minimal phase (4 faces = Pop/Wo/Sek/Ch'en)
    Octahedron   = 6,   // 8-face field (8D tensor / byte)
    Box          = 7,   // Orthogonal attention/FFN layer field
    Cylinder     = 8,   // Sequential pipeline belt
    Cone         = 9,   // Attention pyramid
    Teapot       = 10,  // Bezier trajectory field
    OpticalField = 11,  // Vision: image → geometry → edges → frequency → semantics
};

// ─── Execution domain ─────────────────────────────────────────────────────────
// Defines WHAT semantic region this field represents

enum class FieldDomain : uint32_t {
    Geometry   = 0,   // Spatial structure (where concepts live)
    Vision     = 1,   // Image → geometry → semantics (optical_processor)
    Memory     = 2,   // Hierarchy, caching, paging (asx_ram)
    Compute    = 3,   // GPU kernels, XVM execution (asx_gemm)
    Attention  = 4,   // QKV, focus, weighting (fused attention shaders)
    Semantic   = 5,   // Token relationships, graphs (SCXQ2)
    Temporal   = 6,   // Sequence, pipeline, evolution (evolution shader)
    Execution  = 7,   // Fiber state, phase transitions (XVM fibers)
    Evolution  = 8,   // Selection, recombination, mutation (evolution kernel)
};

// ─── Execution class (XVM dispatch hint) ─────────────────────────────────────
// Defines WHICH GPU kernel to dispatch for this field

enum class ExecutionClass : uint32_t {
    Scalar     = 0,   // Scalar operations
    SIMD       = 1,   // Vector operations
    Attention  = 2,   // QKV attention kernel
    Wave       = 3,   // Wave propagation (SH bands)
    Fiber      = 4,   // XVM fiber execution
    Evolution  = 5,   // Evolution kernel
    Optical    = 6,   // Optical processor pipeline
};

// ─── Topology metadata ────────────────────────────────────────────────────────
// Rich metadata for runtime scheduling, residency management, and dispatch

struct TopologyInfo {
    FieldTopology topology;         // Which topology
    FieldDomain   domain;           // Execution domain (what this field represents)
    ExecutionClass exec_class;      // XVM dispatch hint (which kernel to use)
    const char*   name;             // Human-readable name
    uint32_t      faces;            // Number of faces (0 for parameterized)
    uint32_t      vertices;         // Number of vertices (0 for parameterized)
    uint32_t      edges;            // Number of edges (0 for parameterized)
    float         pi_mod_default;   // Default π-modulus for cards
    const char*   maya_analogue;    // Mayan calendar analogy
    
    // Execution capabilities (for runtime dispatch)
    uint32_t      preferred_threads;       // Preferred thread count
    uint32_t      preferred_group_size;    // Preferred thread group size
    bool          supports_attention;      // Can dispatch attention kernel
    bool          supports_optical;        // Can dispatch optical processor
    bool          supports_xvm;            // Can dispatch XVM fibers
    bool          supports_evolution;      // Can dispatch evolution kernel
    
    // Residency hints (for asx_ram memory management)
    bool          gpu_resident;      // Keep permanently in GPU memory
    bool          cacheable;         // Can be cached
    bool          pageable;          // Can be evicted when memory pressure
    uint32_t      priority;          // Eviction priority (0=highest, 3=lowest)
};

// ─── Topology table ───────────────────────────────────────────────────────────

static constexpr TopologyInfo TOPOLOGY_TABLE[] = {
    // topology, domain, exec_class, name, faces, verts, edges, pi_mod, analogue,
    // threads, group, attn, optical, xvm, evo, gpu_res, cache, page, priority
    
    { FieldTopology::GeoSphere,    FieldDomain::Geometry,  ExecutionClass::Wave,
      "GeoSphere",    0,  0,  0, 1.0f,  "icosahedral subdivision",
      256, 64, true,  true,  true,  false, true,  true,  false, 0 },
      
    { FieldTopology::UVSphere,     FieldDomain::Geometry,  ExecutionClass::Wave,
      "UVSphere",     0,  0,  0, 1.0f,  "latitude rings = uinals",
      256, 64, true,  false, true,  false, true,  true,  false, 0 },
      
    { FieldTopology::Torus,        FieldDomain::Memory,    ExecutionClass::Fiber,
      "Torus",        0,  0,  0, 2.0f,  "dual-cycle memory hierarchy",
      128, 64, false, false, true,  false, true,  true,  true,  1 },
      
    { FieldTopology::Icosahedron,  FieldDomain::Semantic,  ExecutionClass::Scalar,
      "Icosahedron",  20, 12, 30, 1.618f, "20 faces = 1 uinal (base-20)",
      64,  64, false, false, true,  true,  true,  true,  false, 0 },
      
    { FieldTopology::Dodecahedron, FieldDomain::Semantic,  ExecutionClass::Scalar,
      "Dodecahedron", 12, 20, 30, 1.0f,  "12 faces = 12 Haab months",
      64,  64, false, false, true,  true,  true,  true,  false, 0 },
      
    { FieldTopology::Tetrahedron,  FieldDomain::Execution, ExecutionClass::Fiber,
      "Tetrahedron",  4,  4,  6, 1.0f,  "4 faces = Pop/Wo/Sek/Ch'en",
      64,  64, false, false, true,  false, true,  true,  false, 0 },
      
    { FieldTopology::Octahedron,   FieldDomain::Geometry,  ExecutionClass::SIMD,
      "Octahedron",   8,  6,  12, 1.0f, "8 faces = 8D tensor axes",
      128, 64, true,  false, true,  false, true,  true,  false, 0 },
      
    { FieldTopology::Box,          FieldDomain::Compute,   ExecutionClass::Attention,
      "Box",          6,  8,  12, 1.0f, "6 faces = 6 transformer layers",
      256, 64, true,  false, true,  false, true,  true,  false, 0 },
      
    { FieldTopology::Cylinder,     FieldDomain::Temporal,  ExecutionClass::Fiber,
      "Cylinder",     0,  0,  0,  1.0f, "belt = sequential pipeline",
      128, 64, false, false, true,  true,  true,  true,  true,  2 },
      
    { FieldTopology::Cone,         FieldDomain::Attention, ExecutionClass::Attention,
      "Cone",         0,  0,  0,  1.0f, "pyramid = attention focus",
      256, 64, true,  false, true,  false, true,  true,  false, 0 },
      
    { FieldTopology::Teapot,       FieldDomain::Evolution, ExecutionClass::Evolution,
      "Teapot",       0,  0,  0, 1.0f, "Bezier = trajectory field",
      128, 64, false, false, false, true,  true,  true,  false, 1 },
      
    // NEW: OpticalField for vision pipeline
    { FieldTopology::OpticalField, FieldDomain::Vision,    ExecutionClass::Optical,
      "OpticalField", 0,  0,  0,  1.0f, "image → geometry → edges → frequency",
      512, 64, true,  true,  true,  false, true,  true,  false, 0 },
};

// ─── Get topology info ────────────────────────────────────────────────────────

inline const TopologyInfo* GetTopologyInfo(FieldTopology t) {
    for (const auto& info : TOPOLOGY_TABLE)
        if (info.topology == t) return &info;
    return nullptr;
}

// ─── Dispatch: geometry → tokens + cards ─────────────────────────────────────
// CRITICAL FUNCTION: This is the tokenization point where geometry becomes NNC.
// After this call, geometry no longer exists—only NNC tokens remain.

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

    case FieldTopology::OpticalField:
        // OpticalField: vision pipeline — image → geometry → edges → frequency → semantics
        // Emits directly into field ecosystem (no special vision path needed)
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        for (auto& t : tokens) {
            // Optical processor sets coherence based on edge strength / frequency magnitude
            // Phase encodes spatial frequency orientation
            t.coherence = t.coherence * 0.9f; // Slightly lower confidence (raw sensor data)
        }
        break;

    default:
        GeoSphereToTokens(vertices, indices, tokens, cards, scale);
        break;
    }
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

// ─── Domain name helper ───────────────────────────────────────────────────────

inline const char* GetDomainName(FieldDomain d) {
    switch (d) {
        case FieldDomain::Geometry:   return "Geometry";
        case FieldDomain::Vision:     return "Vision";
        case FieldDomain::Memory:     return "Memory";
        case FieldDomain::Compute:    return "Compute";
        case FieldDomain::Attention:  return "Attention";
        case FieldDomain::Semantic:   return "Semantic";
        case FieldDomain::Temporal:   return "Temporal";
        case FieldDomain::Execution:  return "Execution";
        case FieldDomain::Evolution:  return "Evolution";
        default: return "Unknown";
    }
}

// ─── Execution class name helper ─────────────────────────────────────────────

inline const char* GetExecutionClassName(ExecutionClass e) {
    switch (e) {
        case ExecutionClass::Scalar:     return "Scalar";
        case ExecutionClass::SIMD:       return "SIMD";
        case ExecutionClass::Attention:  return "Attention";
        case ExecutionClass::Wave:       return "Wave";
        case ExecutionClass::Fiber:      return "Fiber";
        case ExecutionClass::Evolution:  return "Evolution";
        case ExecutionClass::Optical:    return "Optical";
        default: return "Unknown";
    }
}

} // namespace GeoField

#endif // GEOMETRY_FIELD_MAP_H
