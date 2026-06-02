#pragma once

#include <DirectXMath.h>
#include <vector>
#include <cstdint>
#include "optical_processor.h"

using namespace DirectX;

// VertexPositionNormalTexture: position + normal + uv
// Matches DirectXTK Geometry.h convention so the mesh plugs straight
// into the same pipeline as ComputeGeoSphere / ComputeTorus etc.
struct OpticalVertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
    XMFLOAT2 textureCoordinate;
};

using OpticalVertexCollection = std::vector<OpticalVertex>;
using OpticalIndexCollection  = std::vector<uint16_t>;

// Generate icosahedral geodesic sphere and store positions in OpticalNode.pos
void generateIcosphere(std::vector<OpticalNode>& nodes, int subdivisions);

// Output the same icosphere as a renderable polygon mesh
// (VertexCollection + IndexCollection in the same format as Geometry.cpp).
// diameter: sphere diameter, rhcoords: right-handed coords
void ComputeOpticalMesh(
    OpticalVertexCollection& vertices,
    OpticalIndexCollection&  indices,
    const std::vector<OpticalNode>& nodes,
    float diameter = 2.0f,
    bool  rhcoords = true);
