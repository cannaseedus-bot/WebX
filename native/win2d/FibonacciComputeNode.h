//================================================================================
// FibonacciComputeNode.h
// KXML Sek-phase compute node — batched Fibonacci via Win2D GPU compute.
//
// Phase lifecycle: Pop → Wo → Sek → Ch'en → Xul
// Bind: <node id="fib_n" phase="Sek" domain="compute" device="gpu">
//         <bind from="startIndex,count" to="results" transform="fibonacci" />
//       </node>
//================================================================================
#pragma once

#include <cstdint>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <string>
#include <windows.h>
#include <d3d11.h>

namespace KXML { namespace Win2D {

// ─── KXML Phase enum ─────────────────────────────────────────────────────────

enum class Phase : uint32_t { Pop=0, Wo=1, Sek=2, Chen=3, Xul=4 };

// ─── Constant buffer (matches FibonacciCS.hlsl cbuffer) ──────────────────────

struct FibonacciParams {
    uint32_t startIndex;
    uint32_t count;
    uint32_t mode;    // 0=iterative, 1=matrix, 2=fast_doubling
    uint32_t stride;
    float    reserved[4];
};

// ─── FibonacciComputeNode ─────────────────────────────────────────────────────

class FibonacciComputeNode {
public:
    FibonacciComputeNode(ID3D11Device* dev, ID3D11DeviceContext* ctx,
                         const wchar_t* csoPath);
    ~FibonacciComputeNode();

    // Full Pop→Wo→Sek→Ch'en→Xul pipeline
    std::vector<uint32_t> ExecuteBatch(uint32_t startIndex, uint32_t count,
                                       uint32_t mode = 2, uint32_t stride = 1);

    // Benchmark: returns ms/call average over `iterations` runs
    double Benchmark(uint32_t count, uint32_t iterations = 10);

    Phase CurrentPhase() const { return m_phase; }

private:
    ID3D11Device*           m_dev;
    ID3D11DeviceContext*    m_ctx;
    ID3D11ComputeShader*    m_cs        = nullptr;
    ID3D11Buffer*           m_cbuf      = nullptr;
    ID3D11Buffer*           m_outBuf    = nullptr;
    ID3D11Buffer*           m_gradBuf   = nullptr;
    ID3D11Buffer*           m_stageBuf  = nullptr;
    ID3D11UnorderedAccessView* m_outUAV = nullptr;
    ID3D11UnorderedAccessView* m_gradUAV= nullptr;
    uint32_t                m_capacity  = 0;
    Phase                   m_phase     = Phase::Pop;

    void LoadShader(const wchar_t* path);
    void EnsureCapacity(uint32_t n);
    void UpdateCBuf(uint32_t start, uint32_t count, uint32_t mode, uint32_t stride);
    void ReleaseDynamic();

    bool PopPhase();
    bool WoPhase(uint32_t start, uint32_t count, uint32_t mode, uint32_t stride);
    bool SekPhase(uint32_t count);
    std::vector<uint32_t> ChenPhase(uint32_t count);
    bool XulPhase();
};

}} // namespace KXML::Win2D
