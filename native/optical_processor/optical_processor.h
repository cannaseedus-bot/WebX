#pragma once

#include <DirectXMath.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <vector>
#include <string>
#include <memory>
#include "wave_vm.h"
#include "svg_compiler.h"

using namespace DirectX;

#define SH_BANDS 9  // L=9: l=0..2 (9 coefficients)

struct OpticalNode
{
    XMFLOAT3 pos;                // 3D position on unit sphere — THE POLYGON VERTEX
    XMFLOAT2 sh[SH_BANDS];      // Phase state: (cos, sin) per mode
    uint32_t neighbors[6];       // Geodesic lattice connectivity
    uint32_t neighborCount;      // How many neighbors are valid (up to 6)
    uint32_t pad;                // alignment
};

class Runtime;

class OpticalProcessor
{
public:
    OpticalProcessor(Runtime* rt);
    ~OpticalProcessor();

    // Initialization
    bool projectSVG(const std::string& svg_path);
    bool buildGeodesic(int subdivision_level);

    // Computation
    void stepWavePropagation();
    float readCoherence();

    // VM interface
    WaveVM* getVM() { return vm_.get(); }
    void initVM();
    bool loadSVG3DProgram(const std::string& filename);

    // Debug
    const std::vector<OpticalNode>& getNodes() const { return nodes_cpu; }
    const std::vector<SHNodeCPU>& getSHNodes() const { return sh_nodes_; }
    size_t getNodeCount() const { return nodes_cpu.size(); }

private:
    Runtime* runtime;

    // CPU-side state
    std::vector<OpticalNode> nodes_cpu;
    std::vector<SHNodeCPU> sh_nodes_;      // For VM
    std::unique_ptr<WaveVM> vm_;
};
