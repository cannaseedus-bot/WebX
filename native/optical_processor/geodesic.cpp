#include "geodesic.h"
#include <map>
#include <algorithm>
#include <array>
#include <cmath>

using namespace DirectX;

static const float KUHUL_PI = 3.14159265358979323846f;

static XMFLOAT3 normalize_float3(XMFLOAT3 v)
{
    XMVECTOR vec = XMLoadFloat3(&v);
    vec = XMVector3Normalize(vec);
    XMStoreFloat3(&v, vec);
    return v;
}

struct EdgeKey {
    uint32_t a, b;
    bool operator<(const EdgeKey& o) const {
        if (a != o.a) return a < o.a;
        return b < o.b;
    }
};

static uint32_t midpoint(
    uint32_t a, uint32_t b,
    std::vector<XMFLOAT3>& verts,
    std::map<EdgeKey, uint32_t>& cache)
{
    uint32_t lo = (a < b) ? a : b;
    uint32_t hi = (a < b) ? b : a;
    EdgeKey key = { lo, hi };
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    XMFLOAT3 v1 = verts[a];
    XMFLOAT3 v2 = verts[b];
    XMFLOAT3 mid = {
        (v1.x + v2.x) * 0.5f,
        (v1.y + v2.y) * 0.5f,
        (v1.z + v2.z) * 0.5f
    };
    mid = normalize_float3(mid);

    uint32_t idx = (uint32_t)verts.size();
    verts.push_back(mid);
    cache[key] = idx;
    return idx;
}

// ─── generateIcosphere ───────────────────────────────────────────────────────
// Builds the icosahedral geodesic lattice.
// Now stores pos in each OpticalNode so the mesh can be recovered later.

void generateIcosphere(std::vector<OpticalNode>& nodes, int subdivisions)
{
    std::vector<XMFLOAT3> verts;
    std::vector<std::array<uint32_t, 3>> faces;

    const float t = (1.0f + sqrtf(5.0f)) / 2.0f;

    // 12 base vertices (golden icosahedron)
    verts = {
        { -1,  t,  0 }, {  1,  t,  0 }, { -1, -t,  0 }, {  1, -t,  0 },
        {  0, -1,  t }, {  0,  1,  t }, {  0, -1, -t }, {  0,  1, -t },
        {  t,  0, -1 }, {  t,  0,  1 }, { -t,  0, -1 }, { -t,  0,  1 }
    };

    for (auto& v : verts)
        v = normalize_float3(v);

    // 20 triangular faces
    faces = {
        { 0,11, 5 }, { 0, 5, 1 }, { 0, 1, 7 }, { 0, 7,10 }, { 0,10,11 },
        { 1, 5, 9 }, { 5,11, 4 }, {11,10, 2 }, {10, 7, 6 }, { 7, 1, 8 },
        { 3, 9, 4 }, { 3, 4, 2 }, { 3, 2, 6 }, { 3, 6, 8 }, { 3, 8, 9 },
        { 4, 9, 5 }, { 2, 4,11 }, { 6, 2,10 }, { 8, 6, 7 }, { 9, 8, 1 }
    };

    // Subdivide
    for (int i = 0; i < subdivisions; i++)
    {
        std::map<EdgeKey, uint32_t> cache;
        std::vector<std::array<uint32_t, 3>> newFaces;

        for (auto& f : faces)
        {
            uint32_t a = f[0], b = f[1], c = f[2];
            uint32_t ab = midpoint(a, b, verts, cache);
            uint32_t bc = midpoint(b, c, verts, cache);
            uint32_t ca = midpoint(c, a, verts, cache);

            newFaces.push_back({ a, ab, ca });
            newFaces.push_back({ b, bc, ab });
            newFaces.push_back({ c, ca, bc });
            newFaces.push_back({ ab, bc, ca });
        }

        faces = newFaces;
    }

    // Build nodes — store positions so ComputeOpticalMesh can recover them
    nodes.resize(verts.size());
    for (size_t i = 0; i < verts.size(); i++)
    {
        nodes[i].pos = verts[i];          // ← store the 3D vertex position
        for (int j = 0; j < SH_BANDS; j++)
        {
            nodes[i].sh[j].x = 0.0f;
            nodes[i].sh[j].y = 0.0f;
        }
        for (int j = 0; j < 6; j++)
            nodes[i].neighbors[j] = 0;
        nodes[i].neighborCount = 0;
        nodes[i].pad = 0;
    }

    // Build neighbor connectivity from faces
    for (const auto& f : faces)
    {
        for (int i = 0; i < 3; i++)
        {
            uint32_t a = f[i];
            uint32_t b = f[(i + 1) % 3];

            bool found = false;
            for (uint32_t j = 0; j < nodes[a].neighborCount; j++)
            {
                if (nodes[a].neighbors[j] == b)
                {
                    found = true;
                    break;
                }
            }
            if (!found && nodes[a].neighborCount < 6)
            {
                nodes[a].neighbors[nodes[a].neighborCount++] = b;
            }
        }
    }
}

// ─── ComputeOpticalMesh ───────────────────────────────────────────────────────
// Converts the OpticalNode lattice to a renderable polygon mesh.
// Output format matches ComputeGeoSphere() from Geometry.cpp:
//   vertices  = OpticalVertex (position + normal + uv)
//   indices   = uint16_t triangle list
// normal  = position on unit sphere (same as normalized position)
// uv      = (longitude/2π, latitude/π) — same mapping as ComputeGeoSphere

void ComputeOpticalMesh(
    OpticalVertexCollection& vertices,
    OpticalIndexCollection&  indices,
    const std::vector<OpticalNode>& nodes,
    float diameter,
    bool  rhcoords)
{
    vertices.clear();
    indices.clear();

    if (nodes.empty()) return;

    const float radius = diameter * 0.5f;

    // One vertex per node — normal == position on unit sphere
    vertices.reserve(nodes.size());
    for (const auto& node : nodes)
    {
        const XMFLOAT3& p = node.pos;

        // longitude / latitude from unit-sphere position
        float longitude = atan2f(p.x, -p.z);               // [-π, π]
        float latitude  = acosf(p.y);                       // [0, π]
        float u = longitude / (2.0f * KUHUL_PI) + 0.5f;
        float v = latitude  /         KUHUL_PI;

        OpticalVertex ov;
        ov.position          = { p.x * radius, p.y * radius, p.z * radius };
        ov.normal            = p;          // unit normal = position on sphere
        ov.textureCoordinate = { 1.0f - u, v };

        vertices.push_back(ov);
    }

    // Rebuild triangle indices from neighbor connectivity.
    // Each undirected triangle (a, b, c) is found by: for each node a,
    // for each pair of its neighbours (b, c) where c is also a neighbour of b.
    // To avoid duplicates we only emit when a < b < c.
    // This is O(n * 6 * 6) which is fine for typical subdivision levels.

    size_t n = nodes.size();
    for (uint32_t a = 0; a < (uint32_t)n; a++)
    {
        const OpticalNode& na = nodes[a];
        for (uint32_t bi = 0; bi < na.neighborCount; bi++)
        {
            uint32_t b = na.neighbors[bi];
            if (b <= a) continue;

            const OpticalNode& nb = nodes[b];
            for (uint32_t ci = 0; ci < nb.neighborCount; ci++)
            {
                uint32_t c = nb.neighbors[ci];
                if (c <= b) continue;

                // Verify c is also a neighbour of a
                bool ca = false;
                for (uint32_t k = 0; k < na.neighborCount; k++)
                {
                    if (na.neighbors[k] == c) { ca = true; break; }
                }
                if (!ca) continue;

                // Emit triangle (a, b, c) — winding matches rhcoords flag
                if (rhcoords)
                {
                    indices.push_back((uint16_t)a);
                    indices.push_back((uint16_t)b);
                    indices.push_back((uint16_t)c);
                }
                else
                {
                    indices.push_back((uint16_t)a);
                    indices.push_back((uint16_t)c);
                    indices.push_back((uint16_t)b);
                }
            }
        }
    }
}
