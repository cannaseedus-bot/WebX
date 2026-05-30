// d3d11_engine.cpp — D3D11 device init for GPT-2 GPU trainer
// Targets Intel HD 4600 iGPU via hardware adapter (forceWarp=false).
// The GPT2Trainer uses rawDevice()/rawCtx() directly for compute dispatch.

#include "d3d11_engine.h"
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <cstring>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

bool D3D11Engine::init(bool forceWarp, bool verboseLog) {
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    // Feature levels — HD 4600 supports up to 11_0
    D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    UINT levelCount = ARRAYSIZE(levels);

    if (!forceWarp) {
        // Enumerate adapters to find the iGPU name
        ComPtr<IDXGIFactory1> factory;
        HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)factory.GetAddressOf());
        if (SUCCEEDED(hr)) {
            ComPtr<IDXGIAdapter1> adapter;
            if (SUCCEEDED(factory->EnumAdapters1(0, adapter.GetAddressOf()))) {
                DXGI_ADAPTER_DESC1 desc;
                if (SUCCEEDED(adapter->GetDesc1(&desc))) {
                    // Convert WCHAR → std::string
                    char buf[256] = {};
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, buf, sizeof(buf)-1, nullptr, nullptr);
                    adapterName_ = buf;
                }
            }
        }

        hr = D3D11CreateDevice(
            nullptr,                  // default adapter (iGPU on this rig)
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr, flags,
            levels, levelCount,
            D3D11_SDK_VERSION,
            device_.GetAddressOf(),
            &featureLevel_,
            ctx_.GetAddressOf()
        );
        if (SUCCEEDED(hr)) {
            if (verboseLog)
                OutputDebugStringA(("[D3D11Engine] Hardware device OK: " + adapterName_ + "\n").c_str());
            return true;
        }
        initReason_ = "Hardware device failed hr=" + std::to_string((unsigned)hr);
        if (verboseLog)
            OutputDebugStringA(("[D3D11Engine] Fallback to WARP: " + initReason_ + "\n").c_str());
    }

    // Fallback: WARP software rasterizer (always works)
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_WARP,
        nullptr, flags,
        levels, levelCount,
        D3D11_SDK_VERSION,
        device_.GetAddressOf(),
        &featureLevel_,
        ctx_.GetAddressOf()
    );
    if (SUCCEEDED(hr)) {
        usedWarp_ = true;
        adapterName_ = "WARP (software)";
        initReason_ = "fell back to WARP";
        return true;
    }
    initReason_ = "D3D11 init failed entirely hr=" + std::to_string((unsigned)hr);
    return false;
}

// ── XVM stubs (not used by GPT2Trainer, required by header) ──────────────────

bool D3D11Engine::uploadVM(XVMState& vm) { (void)vm; return false; }
bool D3D11Engine::dispatch(XVMState& vm) { (void)vm; return false; }
bool D3D11Engine::dumpTrace(std::uint32_t) { return false; }

bool D3D11Engine::loadShader() { return false; }
bool D3D11Engine::createBuffers(XVMState&) { return false; }
bool D3D11Engine::createStaging(std::uint32_t) { return false; }
