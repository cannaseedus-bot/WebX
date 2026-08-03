#include "optical_processor.h"
#include "runtime.h"
#include "geodesic.h"
#include "projection.h"
#include "sh_kernel.h"
#include "collapse.h"
#include "buffer_manager.h"
#include <d3d11.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

OpticalProcessor::OpticalProcessor(Runtime* rt)
    : runtime(rt), vm_(nullptr)
{
}

OpticalProcessor::~OpticalProcessor()
{
}

bool OpticalProcessor::projectSVG(const std::string& svg_path)
{
    printf("Projecting SVG: %s\n", svg_path.c_str());

    // Initialize nodes with wave-seeded SH coefficients
    if (nodes_cpu.empty()) return true;
    
    for (size_t i = 0; i < nodes_cpu.size(); i++)
    {
        // Derive theta/phi from the stored vertex position (spherical coords)
        const XMFLOAT3& p = nodes_cpu[i].pos;
        float theta = acosf(p.y);                          // latitude  [0, π]
        float phi   = atan2f(p.x, -p.z) + 3.14159f;       // longitude [0, 2π]
        
        for (int j = 0; j < SH_BANDS; j++)
        {
            // Evaluate SH basis at node position
            float Y_val = evaluateSH(j, theta, phi);
            
            // Create phase offset per band (CRITICAL for dynamics)
            float phase = phi * (j + 1) * 0.5f;
            
            float cos_phase = cosf(phase);
            float sin_phase = sinf(phase);
            
            // Initialize with both cos and sin components (real wave)
            nodes_cpu[i].sh[j].x = Y_val * cos_phase * 0.1f;
            nodes_cpu[i].sh[j].y = Y_val * sin_phase * 0.1f;
        }
    }
    
    printf("✓ Initialized wave field with phase offsets\n");
    return true;
}

bool OpticalProcessor::buildGeodesic(int level)
{
    printf("Building geodesic sphere (level %d)...\n", level);

    generateIcosphere(nodes_cpu, level);
    printf("✓ Generated %zu nodes\n", nodes_cpu.size());

    return true;
}

void OpticalProcessor::stepWavePropagation()
{
    // Deprecated: Wave propagation now runs on CPU via Wave VM
}

float OpticalProcessor::readCoherence()
{
    if (nodes_cpu.empty())
        return 0.0f;

    float total = 0.0f;

    for (const auto& node : nodes_cpu)
    {
        float energy = 0.0f;
        for (int j = 0; j < SH_BANDS; j++)
        {
            float len = sqrtf(node.sh[j].x * node.sh[j].x +
                             node.sh[j].y * node.sh[j].y);
            energy += len;
        }
        total += energy;
    }

    return total / (nodes_cpu.size() * SH_BANDS);
}

void OpticalProcessor::initVM()
{
    // Convert OpticalNode to SHNodeCPU for VM (store as member)
    sh_nodes_.resize(nodes_cpu.size());
    
    for (size_t i = 0; i < nodes_cpu.size(); i++)
    {
        for (int j = 0; j < SH_BANDS; j++)
        {
            sh_nodes_[i].sh[j] = nodes_cpu[i].sh[j];
        }
    }
    
    // Create VM and set field reference
    vm_ = std::make_unique<WaveVM>();
    vm_->set_field(&sh_nodes_);
    
    printf("✓ Wave VM initialized (%zu nodes, %d SH bands)\n", sh_nodes_.size(), SH_BANDS);
}

bool OpticalProcessor::loadSVG3DProgram(const std::string& filename)
{
    SVG3DCompiler compiler;
    
    if (!compiler.parse(filename))
    {
        printf("✗ SVG parse failed: %s\n", compiler.error().c_str());
        return false;
    }
    
    std::vector<Instruction> program;
    if (!compiler.compile(program))
    {
        printf("✗ SVG compile failed: %s\n", compiler.error().c_str());
        return false;
    }
    
    printf("✓ Loaded SVG-3D program (%zu instructions)\n", program.size());
    
    if (!vm_)
        initVM();
    
    return vm_->execute(program);
}
