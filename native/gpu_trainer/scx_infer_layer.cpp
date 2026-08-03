/**
 * scx_infer_layer.cpp
 * Minimal working 1-layer streaming inference loop (D3D11, cs_5_0).
 *
 * What it does:
 *   Thread 1 (stream)  → writes INT4 weight chunks into weightBuf_ (upload heap)
 *   Thread 2 (infer)   → waits until weights ready, dispatches CSQProj, reads Y
 *
 * Compile and run standalone:
 *   cl scx_infer_layer.cpp /EHsc /std:c++17 d3d11.lib dxgi.lib d3dcompiler.lib
 *
 * Or #include "scx_infer_layer.h" and call from your existing trainer.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <string>
#include <functional>
#include <cassert>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

// ── Dimensions (change here to scale up) ──────────────────────────────────────

static constexpr uint32_t BATCH     =   1;   // tokens in this batch
static constexpr uint32_t K_DIM     =  64;   // input hidden dim
static constexpr uint32_t N_DIM     =  64;   // output dim (Q projection)
static constexpr uint32_t NUM_CHUNKS =   4;  // how many chunks the "stream" sends

// INT4 packing: 8 weights per uint32
static constexpr uint32_t PACKED_PER_ROW = (N_DIM + 7) / 8;
static constexpr uint32_t TOTAL_PACKED   = K_DIM * PACKED_PER_ROW;

// ── Constant buffer (mirrors InferParams in HLSL) ─────────────────────────────

struct InferParams {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t use_bias;
    float    w_scale;
    float    w_zero;
    uint32_t numPacked;
    uint32_t dstOffset;
};

// ── State shared between stream thread and infer thread ───────────────────────

struct LayerSync {
    std::mutex              mu;
    std::condition_variable cv;
    std::atomic<int>        chunksDelivered{0};
    std::atomic<bool>       weightsReady{false};
    std::atomic<bool>       done{false};
};

// ── D3D11 helpers ──────────────────────────────────────────────────────────────

static ComPtr<ID3D11Device>        gDev;
static ComPtr<ID3D11DeviceContext> gCtx;

static bool InitD3D11() {
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1 };
    D3D_FEATURE_LEVEL got;
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
        &gDev, &got, &gCtx);
    if (FAILED(hr)) {
        // Fallback to WARP (always works, useful for CI / no-GPU)
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0,
            levels, ARRAYSIZE(levels), D3D11_SDK_VERSION,
            &gDev, &got, &gCtx);
    }
    return SUCCEEDED(hr);
}

// Create a GPU-only structured buffer (DEFAULT usage, SRV + UAV)
static ComPtr<ID3D11Buffer> MakeStructBuf(uint32_t elemSz, uint32_t count,
                                           const void* initData = nullptr)
{
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth           = elemSz * count;
    bd.Usage               = D3D11_USAGE_DEFAULT;
    bd.BindFlags           = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = elemSz;

    D3D11_SUBRESOURCE_DATA sd{ initData, 0, 0 };
    ComPtr<ID3D11Buffer> buf;
    gDev->CreateBuffer(&bd, initData ? &sd : nullptr, &buf);
    return buf;
}

// Create staging buffer for CPU→GPU upload and GPU→CPU readback
static ComPtr<ID3D11Buffer> MakeStagingBuf(uint32_t bytes) {
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = bytes;
    bd.Usage     = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    ComPtr<ID3D11Buffer> buf;
    gDev->CreateBuffer(&bd, nullptr, &buf);
    return buf;
}

// Create SRV over a structured buffer
static ComPtr<ID3D11ShaderResourceView> MakeSRV(ID3D11Buffer* buf, uint32_t count) {
    D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Format              = DXGI_FORMAT_UNKNOWN;
    sd.ViewDimension       = D3D11_SRV_DIMENSION_BUFFEREX;
    sd.BufferEx.NumElements = count;
    ComPtr<ID3D11ShaderResourceView> srv;
    gDev->CreateShaderResourceView(buf, &sd, &srv);
    return srv;
}

// Create UAV over a structured buffer
static ComPtr<ID3D11UnorderedAccessView> MakeUAV(ID3D11Buffer* buf, uint32_t count) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format              = DXGI_FORMAT_UNKNOWN;
    ud.ViewDimension       = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements  = count;
    ComPtr<ID3D11UnorderedAccessView> uav;
    gDev->CreateUnorderedAccessView(buf, &ud, &uav);
    return uav;
}

// Upload CPU data → GPU DEFAULT buffer via UpdateSubresource
static void Upload(ID3D11Buffer* dst, const void* data, uint32_t bytes) {
    D3D11_BOX box{ 0, 0, 0, bytes, 1, 1 };
    gCtx->UpdateSubresource(dst, 0, &box, data, bytes, 0);
}

// Upload a sub-region at byteOffset
static void UploadAt(ID3D11Buffer* dst, const void* data,
                     uint32_t bytes, uint32_t byteOffset)
{
    D3D11_BOX box{ byteOffset, 0, 0, byteOffset + bytes, 1, 1 };
    gCtx->UpdateSubresource(dst, 0, &box, data, bytes, 0);
}

// Readback from GPU → CPU via staging buffer
static void Readback(ID3D11Buffer* src, void* dst, uint32_t bytes) {
    auto staging = MakeStagingBuf(bytes);
    gCtx->CopyResource(staging.Get(), src);
    D3D11_MAPPED_SUBRESOURCE ms{};
    gCtx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &ms);
    std::memcpy(dst, ms.pData, bytes);
    gCtx->Unmap(staging.Get(), 0);
}

// Compile HLSL from file, returns shader bytecode blob
static ComPtr<ID3DBlob> CompileShader(const wchar_t* path, const char* entry) {
    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3DCompileFromFile(
        path, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry, "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
        &blob, &err);
    if (FAILED(hr)) {
        if (err) OutputDebugStringA((char*)err->GetBufferPointer());
        return nullptr;
    }
    return blob;
}

// ── INT4 weight encoder (CPU-side, for generating test data) ──────────────────

// Pack 8 signed INT4 values (range -8..7) into one uint32
static uint32_t PackInt4x8(const int8_t w[8]) {
    uint32_t out = 0;
    for (int i = 0; i < 8; ++i)
        out |= uint32_t(uint8_t(w[i] + 8) & 0xF) << (i * 4);
    return out;
}

// Fill packed weight array with a simple test pattern
// W[row, col] = (row + col) % 7 - 3  (range -3..3)
static void MakeTestWeights(std::vector<uint32_t>& packed) {
    packed.resize(TOTAL_PACKED);
    for (uint32_t r = 0; r < K_DIM; ++r) {
        for (uint32_t g = 0; g < PACKED_PER_ROW; ++g) {
            int8_t w[8]{};
            for (int j = 0; j < 8; ++j) {
                uint32_t col = g * 8 + j;
                w[j] = (col < N_DIM) ? int8_t((r + col) % 7 - 3) : 0;
            }
            packed[r * PACKED_PER_ROW + g] = PackInt4x8(w);
        }
    }
}

// ── Constant buffer helper ────────────────────────────────────────────────────

static ComPtr<ID3D11Buffer> MakeCB(const void* data, uint32_t bytes) {
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth      = (bytes + 15) & ~15u;  // 16-byte aligned
    bd.Usage          = D3D11_USAGE_DYNAMIC;
    bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    D3D11_SUBRESOURCE_DATA sd{ data, 0, 0 };
    ComPtr<ID3D11Buffer> cb;
    gDev->CreateBuffer(&bd, data ? &sd : nullptr, &cb);
    return cb;
}

static void UpdateCB(ID3D11Buffer* cb, const void* data, uint32_t bytes) {
    D3D11_MAPPED_SUBRESOURCE ms{};
    gCtx->Map(cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
    std::memcpy(ms.pData, data, bytes);
    gCtx->Unmap(cb, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// 1-Layer Inference Demo
// ═════════════════════════════════════════════════════════════════════════════

static void RunOneLayerDemo()
{
    // ── 1. D3D11 init ────────────────────────────────────────────────────────
    if (!InitD3D11()) { puts("[FAIL] D3D11 init"); return; }
    puts("[OK]  D3D11 ready");

    // ── 2. Compile shader ────────────────────────────────────────────────────
    auto blob = CompileShader(
        L"C:\\Users\\canna\\.gpu_trainer\\shaders\\scxq2_infer_layer.hlsl",
        "CSQProj");
    if (!blob) { puts("[FAIL] CSQProj compile"); return; }

    ComPtr<ID3D11ComputeShader> cs;
    gDev->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                              nullptr, &cs);
    puts("[OK]  CSQProj compiled");

    // ── 3. GPU buffers ───────────────────────────────────────────────────────

    // Input activations X [BATCH × K_DIM]
    std::vector<float> hostX(BATCH * K_DIM, 1.0f);  // all-ones test vector
    auto xBuf  = MakeStructBuf(sizeof(float), BATCH * K_DIM, hostX.data());
    auto xSRV  = MakeSRV(xBuf.Get(), BATCH * K_DIM);

    // INT4 packed weights [TOTAL_PACKED uint32] — allocated empty, filled by stream thread
    auto wBuf  = MakeStructBuf(sizeof(uint32_t), TOTAL_PACKED);
    auto wSRV  = MakeSRV(wBuf.Get(), TOTAL_PACKED);

    // Bias [N_DIM] — zeros for this demo
    std::vector<float> hostBias(N_DIM, 0.0f);
    auto biasBuf = MakeStructBuf(sizeof(float), N_DIM, hostBias.data());
    auto biasSRV = MakeSRV(biasBuf.Get(), N_DIM);

    // Output Y [BATCH × N_DIM]
    auto yBuf  = MakeStructBuf(sizeof(float), BATCH * N_DIM);
    auto yUAV  = MakeUAV(yBuf.Get(), BATCH * N_DIM);

    // Constant buffer
    InferParams params{ BATCH, K_DIM, N_DIM, 0, 1.0f/8.0f, 0.0f, 0, 0 };
    auto cb = MakeCB(&params, sizeof(params));

    puts("[OK]  GPU buffers allocated");

    // ── 4. Generate test weight data ─────────────────────────────────────────
    std::vector<uint32_t> allWeights;
    MakeTestWeights(allWeights);

    // Split into NUM_CHUNKS for streaming simulation
    uint32_t chunkPacked = TOTAL_PACKED / NUM_CHUNKS;

    // ── 5. Sync object ───────────────────────────────────────────────────────
    LayerSync sync;

    // ── 6. Stream thread ─────────────────────────────────────────────────────
    // Simulates SCXQ2 chunks arriving over the network.
    // In production: replace the loop body with reading from a socket/DataChannel.
    std::thread streamThread([&]{
        for (uint32_t c = 0; c < NUM_CHUNKS; ++c) {
            // Simulate network latency
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            uint32_t byteOffset = c * chunkPacked * sizeof(uint32_t);
            const void* src = allWeights.data() + c * chunkPacked;
            uint32_t byteCount = chunkPacked * sizeof(uint32_t);

            // Upload this chunk into the weight buffer at its offset
            // (D3D11 UpdateSubresource is thread-safe for DEFAULT buffers
            //  when the immediate context is used by one thread at a time;
            //  in production use a deferred context or lock here)
            UploadAt(wBuf.Get(), src, byteCount, byteOffset);

            int delivered = sync.chunksDelivered.fetch_add(1) + 1;
            printf("[stream] chunk %u/%u uploaded (%u bytes @ +%u)\n",
                   delivered, NUM_CHUNKS, byteCount, byteOffset);

            if (delivered == int(NUM_CHUNKS)) {
                sync.weightsReady.store(true);
                sync.cv.notify_all();
            }
        }
    });

    // ── 7. Inference thread ───────────────────────────────────────────────────
    // Waits until all weight chunks have arrived, then dispatches compute.
    // In production: run this in a loop, dispatching per-token as each layer
    // becomes ready (see scx_stream_engine.h layerReady() pattern).
    std::thread inferThread([&]{
        puts("[infer] waiting for weights...");
        {
            std::unique_lock<std::mutex> lk(sync.mu);
            sync.cv.wait(lk, [&]{ return sync.weightsReady.load(); });
        }
        puts("[infer] weights ready — dispatching CSQProj");

        // Bind compute shader
        gCtx->CSSetShader(cs.Get(), nullptr, 0);

        // Bind constant buffer
        ID3D11Buffer* cbs[] = { cb.Get() };
        gCtx->CSSetConstantBuffers(0, 1, cbs);

        // Bind SRVs: t0=X, t1=W_int4, t2=bias
        ID3D11ShaderResourceView* srvs[] = { xSRV.Get(), wSRV.Get(), biasSRV.Get() };
        gCtx->CSSetShaderResources(0, 3, srvs);

        // Bind UAV: u0=Y
        ID3D11UnorderedAccessView* uavs[] = { yUAV.Get() };
        gCtx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);

        // Dispatch: ceil(N/16) × ceil(M/16) groups
        uint32_t gx = (N_DIM + 15) / 16;
        uint32_t gy = (BATCH + 15) / 16;
        gCtx->Dispatch(gx, gy, 1);

        // Unbind UAV before readback
        ID3D11UnorderedAccessView* nullUAV = nullptr;
        gCtx->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);

        // Readback Y
        std::vector<float> hostY(BATCH * N_DIM, 0.f);
        Readback(yBuf.Get(), hostY.data(), uint32_t(hostY.size() * sizeof(float)));

        // Print first 8 outputs
        printf("[infer] Y[0..7] = ");
        for (int i = 0; i < 8 && i < int(N_DIM); ++i)
            printf("%.4f ", hostY[i]);
        printf("...\n");

        // CPU reference for Y[0] = sum_k X[k] * W[k,0] / 8.0
        float ref = 0.f;
        for (uint32_t k = 0; k < K_DIM; ++k) {
            int8_t w = int8_t((k + 0) % 7 - 3);  // col=0
            ref += 1.0f * float(w) / 8.0f;        // X[k]=1.0
        }
        printf("[check] CPU ref Y[0] = %.4f  GPU = %.4f  diff = %.6f\n",
               ref, hostY[0], std::fabsf(ref - hostY[0]));

        sync.done.store(true);
    });

    streamThread.join();
    inferThread.join();

    if (sync.done.load())
        puts("[PASS] 1-layer inference loop complete");
    else
        puts("[FAIL] inference thread did not finish");
}

// ── main (standalone demo entry) ─────────────────────────────────────────────

int main() {
    printf("=== SCXQ2 1-Layer Inference Demo ===\n");
    printf("  Input:   BATCH=%u  K=%u  N=%u\n", BATCH, K_DIM, N_DIM);
    printf("  Weights: %u uint32 (%u KB)  INT4-packed\n",
           TOTAL_PACKED, TOTAL_PACKED * 4 / 1024);
    printf("  Chunks:  %u × %u uint32\n\n", NUM_CHUNKS, TOTAL_PACKED/NUM_CHUNKS);

    RunOneLayerDemo();
    return 0;
}
