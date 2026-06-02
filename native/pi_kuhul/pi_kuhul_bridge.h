// pi_kuhul_bridge.h — DirectX Geometry → π-KUHUL Field Bridge
//
// DirectX Geometry IS π-KUHUL field generation.
// A geodesic sphere IS a 100D field manifold projected to 3D.
//
// Mapping (one geometry, one field, one inference engine):
//   Vertex position    → Token position (3D projection of 100D field)
//   Vertex normal      → Field gradient direction
//   Texture UV         → Maya calendar coords (u=longitude/2π, v=latitude/π)
//   Index triangle     → Token adjacency (shard connection graph)
//   Tessellation level → Field resolution (more = more tokens)
//   Face ID % 52       → Card/shard assignment (52-card deck)
//
// K'uhul phase mapping:
//   Pop   load tokens from mesh vertices  (GeoSphereToTokens)
//   Wo    allocate CardField array         (52 cards)
//   Sek   propagate field step             (PropagateField)
//   Ch'en advance Maya Long Count         (token.maya_kin++, ...)
//   Xul   read coherence / emit output   (ComputeGlobalCoherence)
//
// Requires: Geometry.h (DirectX toolkit geometry generators)
//           DirectXMath.h

#pragma once
#ifndef PI_KUHUL_BRIDGE_H
#define PI_KUHUL_BRIDGE_H

#include "Geometry.h"
#include <cmath>
#include <vector>
#include <cstdint>

using namespace DirectX;

namespace PiKuhul {

// ─── Constants ───────────────────────────────────────────────────────────────

constexpr float VIGESIMAL    = 20.0f;
constexpr float TAU          = 6.283185307179586f;  // 2π
constexpr float PI_CONST     = 3.141592653589793f;
constexpr uint32_t CARDS     = 52;
constexpr uint32_t ADJ_COUNT = 8;

// ─── Token ───────────────────────────────────────────────────────────────────
// One field node. Corresponds to one vertex of the geodesic mesh.
// Position is the 3D projection of the 100D field vector.

struct Token {
    XMFLOAT3 position;       // 3D projection of 100D position in M
    XMFLOAT3 velocity;       // Field velocity
    float    phase;          // π-phase [0, 2π)
    float    phase_velocity; // Phase change rate
    uint32_t shard_id;       // Current card/shard (0–51)
    float    coherence;      // Interference coherence [0,1]
    // Maya Long Count (base-20 calendar)
    uint32_t maya_baktun;
    uint32_t maya_katun;
    uint32_t maya_tun;
    uint32_t maya_uinal;
    uint32_t maya_kin;
};

// ─── CardField ────────────────────────────────────────────────────────────────
// One of 52 cards (shard regions). Tokens belong to one card at a time.
// Corresponds to one face group in the geodesic mesh.

struct CardField {
    XMFLOAT4 amplitude;                    // Field magnitude
    XMFLOAT4 gradient;                     // Force field direction
    XMFLOAT4 curvature;                    // Geometric curvature
    float    pi_mod;                       // π-phase modulation rate
    uint32_t maya_digit;                   // This card's vigesimal digit (0–19)
    uint32_t adjacency[ADJ_COUNT];         // Connected card indices
    float    adjacency_strength[ADJ_COUNT];
};

// ─── GeoSphereToTokens ────────────────────────────────────────────────────────
// Convert a DirectX geodesic sphere mesh to π-KUHUL field tokens + cards.
// Each vertex → Token. Each face group → CardField.

inline void GeoSphereToTokens(
    const VertexCollection& vertices,
    const IndexCollection&  indices,
    std::vector<Token>&     tokens,
    std::vector<CardField>& cards,
    float radius = 1.0f)
{
    tokens.clear();
    tokens.reserve(vertices.size());

    for (size_t i = 0; i < vertices.size(); i++) {
        const auto& v = vertices[i];
        Token t;

        // Position: 3D projection of 100D field
        t.position = v.position;

        // Velocity: initialized from normal (field gradient direction)
        t.velocity = { v.normal.x * .1f, v.normal.y * .1f, v.normal.z * .1f };

        // Phase: texture u-coord maps to [0, 2π]
        t.phase          = v.textureCoordinate.x * TAU;
        t.phase_velocity = .1f;

        // Shard: 52-card deck by vertex index
        t.shard_id  = static_cast<uint32_t>(i) % CARDS;
        t.coherence = 0.0f;

        // Maya calendar from spherical coords
        float theta = atan2f(v.position.z, v.position.x);
        float phi   = acosf(v.position.y / radius);

        t.maya_baktun = 13u;
        t.maya_katun  = static_cast<uint32_t>(theta / TAU * 20.0f) % 20u;
        t.maya_tun    = static_cast<uint32_t>(phi   / PI_CONST * 20.0f) % 20u;
        t.maya_uinal  = static_cast<uint32_t>(v.textureCoordinate.x * 18.0f) % 18u;
        t.maya_kin    = static_cast<uint32_t>(v.textureCoordinate.y * 20.0f) % 20u;

        tokens.push_back(t);
    }

    // Build card fields from face adjacency
    cards.assign(CARDS, CardField{});
    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t face_id = static_cast<uint32_t>(i / 3) % CARDS;
        auto& card = cards[face_id];

        XMVECTOR v0 = XMLoadFloat3(&tokens[indices[i    ]].position);
        XMVECTOR v1 = XMLoadFloat3(&tokens[indices[i + 1]].position);
        XMVECTOR v2 = XMLoadFloat3(&tokens[indices[i + 2]].position);
        XMVECTOR n  = XMVector3Normalize(XMVector3Cross(
            XMVectorSubtract(v1, v0), XMVectorSubtract(v2, v0)));

        XMStoreFloat4(&card.amplitude, n);
        card.gradient   = { 0, 0, 0, 0 };
        card.curvature  = { .1f, .1f, .1f, .1f };
        card.pi_mod     = 1.0f;
        card.maya_digit = face_id % 20u;

        for (int a = 0; a < static_cast<int>(ADJ_COUNT); a++) {
            card.adjacency[a]          = (face_id + a + 1u) % CARDS;
            card.adjacency_strength[a] = 1.0f / static_cast<float>(a + 1);
        }
    }
}

// ─── PropagateField ───────────────────────────────────────────────────────────
// One evolution step: advance phase, update velocity, advance Maya calendar.
// K'uhul phase: Sek (execute) + Ch'en (store to field).

inline void PropagateField(
    std::vector<Token>&           tokens,
    const std::vector<CardField>& cards,
    float dt          = 0.01f,
    float damping     = 0.95f)
{
    for (auto& t : tokens) {
        const auto& card = cards[t.shard_id];

        // Sek: π-phase evolution
        t.phase += dt * card.pi_mod;
        if (t.phase >= TAU) t.phase -= TAU;

        // Sek: Maya sync at uinal boundary
        if (t.maya_uinal == 0u && t.maya_kin == 0u) t.phase = 0.0f;

        // Sek: damped velocity integration
        t.velocity.x *= damping;
        t.velocity.y *= damping;
        t.velocity.z *= damping;
        t.position.x += t.velocity.x * dt;
        t.position.y += t.velocity.y * dt;
        t.position.z += t.velocity.z * dt;

        // Ch'en: advance Maya Long Count
        if (++t.maya_kin >= 20u) {
            t.maya_kin = 0u;
            if (++t.maya_uinal >= 18u) {
                t.maya_uinal = 0u;
                if (++t.maya_tun >= 20u) {
                    t.maya_tun = 0u;
                    if (++t.maya_katun >= 20u) {
                        t.maya_katun = 0u;
                        ++t.maya_baktun;
                    }
                }
            }
        }

        // Coherence decay
        t.coherence = t.coherence * .99f;
    }
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

inline float ComputeGlobalCoherence(const std::vector<Token>& tokens) {
    if (tokens.empty()) return 0.0f;
    float sum = 0.0f;
    for (const auto& t : tokens) sum += t.coherence;
    return sum / static_cast<float>(tokens.size());
}

inline uint32_t LongCountToDays(
    uint32_t baktun, uint32_t katun, uint32_t tun, uint32_t uinal, uint32_t kin) {
    return baktun * 144000u + katun * 7200u + tun * 360u + uinal * 20u + kin;
}

inline void DaysToLongCount(
    uint32_t days,
    uint32_t& baktun, uint32_t& katun, uint32_t& tun, uint32_t& uinal, uint32_t& kin) {
    baktun = days / 144000u; days %= 144000u;
    katun  = days /   7200u; days %=   7200u;
    tun    = days /    360u; days %=    360u;
    uinal  = days /     20u;
    kin    = days %     20u;
}

} // namespace PiKuhul

#endif // PI_KUHUL_BRIDGE_H
