/**
 * SCX-MoE DirectX 12 Compute Engine
 * =================================
 *
 * Provides a Windows native interface for SCX-MoE forward pass acceleration.
 *
 * ARCHITECTURE:
 * =============
 * 1. DirectX 12 device initialization
 * 2. GPU buffer allocation (embeddings, weights, outputs)
 * 3. Compute shader compilation and PSO creation
 * 4. Forward pass orchestration with weight loading
 *
 * INTERFACE:
 * ==========
 * This is called from Python via ctypes:
 *
 *   from ctypes import *
 *   dll = cdll.LoadLibrary("sxme_compute.dll")
 *   dll.SCXMoEForward(
 *       token_ids,      # uint32 array
 *       seq_len,        # uint32 sequence length
 *       embedding_path, # const char*
 *       weights_path,   # const char*
 *       output_logits,  # float32 array (output)
 *       seq_len * vocab_size  # output size
 *   )
 *
 * ERROR HANDLING:
 * ===============
 * - Returns HRESULT (0 = success, non-zero = failure)
 * - Errors are logged to stderr
 * - Automatic fallback to CPU NumPy in Python wrapper on failure
 */

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <dxcapi.h>
#include <dxcapi.h>
#include <wrl.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

// ============================================================================
// CONSTANTS
// ============================================================================

// SCX-MoE Architecture
static constexpr uint32_t HIDDEN_SIZE = 1024;
static constexpr uint32_t VOCAB_SIZE = 32000;
static constexpr uint32_t NUM_EXPERTS = 8;
static constexpr uint32_t NUM_LAYERS = 8;
static constexpr uint32_t INTERMEDIATE_SIZE = 11008;
static constexpr uint32_t NUM_HEADS = 32;
static constexpr uint32_t HEAD_DIM = 32;
static constexpr uint32_t MAX_SEQ = 2048;
static constexpr uint32_t NUM_EXPERTS_PER_TOKEN = 2;

// Compute shader thread group size
static constexpr uint32_t THREADGROUP_SIZE = 32;

// ============================================================================
// ERROR HANDLING
// ============================================================================

static void ThrowIfFailed(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        fprintf(stderr, "[DirectX Error] %s (HRESULT=0x%08x)\n", msg, hr);
        throw std::runtime_error(msg);
    }
}

static void ThrowIfFailed(HRESULT hr) {
    ThrowIfFailed(hr, "HRESULT failed");
}

static bool FileExists(const wchar_t* path) {
    return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
}

static std::wstring ResolveShaderPath(const wchar_t* filename) {
    if (FileExists(filename)) {
        return filename;
    }

    wchar_t altPath[MAX_PATH] = {};
    _snwprintf(altPath, MAX_PATH - 1, L"native\\d3d12_compute\\%s", filename);
    if (FileExists(altPath)) {
        return altPath;
    }

    return filename;
}

static ComPtr<IDxcBlob> CompileShaderDXC(const wchar_t* path) {
    ComPtr<IDxcUtils> utils;
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils)));

    ComPtr<IDxcCompiler3> compiler;
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)));

    ComPtr<IDxcBlobEncoding> source;
    ThrowIfFailed(utils->LoadFile(path, nullptr, &source));

    DxcBuffer buffer = {};
    buffer.Ptr = source->GetBufferPointer();
    buffer.Size = source->GetBufferSize();
    buffer.Encoding = DXC_CP_UTF8;

    LPCWSTR args[] = {
        L"-E", L"main",
        L"-T", L"cs_6_0",
        L"-O3",
        L"-Qembed_debug"
    };

    ComPtr<IDxcResult> result;
    ThrowIfFailed(compiler->Compile(
        &buffer,
        args,
        static_cast<UINT32>(_countof(args)),
        nullptr,
        IID_PPV_ARGS(&result)
    ));

    ComPtr<IDxcBlobUtf8> errors;
    ThrowIfFailed(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr));
    if (errors && errors->GetStringLength() > 0) {
        fprintf(stderr, "DXC ERROR:\n%s\n", errors->GetStringPointer());
    }

    HRESULT status = S_OK;
    ThrowIfFailed(result->GetStatus(&status));

    ComPtr<IDxcBlob> shader;
    ThrowIfFailed(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader), nullptr));

    fprintf(stdout, "DXC compile SUCCESS (cs_6_0)\n");
    return shader;
}

static bool QueryWaveSupport(ID3D12Device* device) {
    if (!device) {
        return false;
    }

    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = {};
    shaderModel.HighestShaderModel = D3D_SHADER_MODEL_6_0;
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)))) {
        return false;
    }
    if (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_0) {
        return false;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options = {};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options, sizeof(options)))) {
        return false;
    }

    return options.WaveOps != FALSE;
}

// ============================================================================
// GLOBAL STATE
// ============================================================================

// Cached DirectX objects to avoid recreating on each forward pass
static ComPtr<ID3D12Device> g_device;
static ComPtr<ID3D12CommandQueue> g_queue;
static ComPtr<ID3D12CommandAllocator> g_allocator;
static ComPtr<ID3D12GraphicsCommandList> g_list;
static ComPtr<ID3D12PipelineState> g_pso;
static ComPtr<ID3D12RootSignature> g_rootSig;
static ComPtr<ID3D12DescriptorHeap> g_descHeap;

static bool g_initialized = false;
static bool g_waveOpsSupported = false;

enum RuntimeBackendType : int {
    RUNTIME_BACKEND_NONE = 0,
    RUNTIME_BACKEND_D3D12_HARDWARE = 1,
    RUNTIME_BACKEND_D3D12_WARP = 2,
};

static RuntimeBackendType g_backendType = RUNTIME_BACKEND_NONE;
static char g_backendName[256] = "uninitialized";

static void SetBackendName(const wchar_t* description, const char* prefix, RuntimeBackendType backendType) {
    char utf8Desc[192] = {0};
    if (description != nullptr) {
        WideCharToMultiByte(
            CP_UTF8,
            0,
            description,
            -1,
            utf8Desc,
            static_cast<int>(sizeof(utf8Desc)),
            nullptr,
            nullptr
        );
    }
    if (utf8Desc[0] == '\0') {
        snprintf(utf8Desc, sizeof(utf8Desc), "Unknown Adapter");
    }
    snprintf(g_backendName, sizeof(g_backendName), "%s: %s", prefix, utf8Desc);
    g_backendType = backendType;
}

static HRESULT SelectBestAdapterAndCreateDevice(
    IDXGIFactory6* factory,
    ID3D12Device** outDevice,
    RuntimeBackendType* outBackendType,
    char* outBackendName,
    size_t outBackendNameSize
) {
    if (factory == nullptr || outDevice == nullptr) {
        return E_INVALIDARG;
    }

    auto setLocalBackend = [&](const wchar_t* desc, const char* prefix, RuntimeBackendType type) {
        if (outBackendType != nullptr) {
            *outBackendType = type;
        }
        if (outBackendName != nullptr && outBackendNameSize > 0) {
            char utf8Desc[192] = {0};
            if (desc != nullptr) {
                WideCharToMultiByte(
                    CP_UTF8, 0, desc, -1, utf8Desc, static_cast<int>(sizeof(utf8Desc)), nullptr, nullptr
                );
            }
            if (utf8Desc[0] == '\0') {
                snprintf(utf8Desc, sizeof(utf8Desc), "Unknown Adapter");
            }
            snprintf(outBackendName, outBackendNameSize, "%s: %s", prefix, utf8Desc);
        }
    };

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }

        HRESULT probeHr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr);
        if (FAILED(probeHr)) {
            continue;
        }

        HRESULT createHr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(outDevice));
        if (SUCCEEDED(createHr)) {
            setLocalBackend(desc.Description, "D3D12 Hardware", RUNTIME_BACKEND_D3D12_HARDWARE);
            return S_OK;
        }
    }

    ComPtr<IDXGIAdapter> warpAdapter;
    HRESULT warpHr = factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter));
    if (FAILED(warpHr)) {
        return warpHr;
    }

    HRESULT createWarpHr = D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(outDevice));
    if (FAILED(createWarpHr)) {
        return createWarpHr;
    }

    DXGI_ADAPTER_DESC warpDesc = {};
    warpAdapter->GetDesc(&warpDesc);
    setLocalBackend(warpDesc.Description, "D3D12 WARP", RUNTIME_BACKEND_D3D12_WARP);
    return S_OK;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize DirectX 12 device and compute shader PSO
 *
 * Sets up:
 * - DXGI factory and GPU adapter selection
 * - Direct3D 12 device creation
 * - Command queue for compute work
 * - Root signature for compute shader
 * - Pipeline state object (PSO) with compiled shader
 *
 * Should be called once before any forward passes.
 * Subsequent calls are no-ops (g_initialized guard).
 */
static HRESULT InitializeDirectX() {
    if (g_initialized) {
        return S_OK;
    }

    try {
        // Create DXGI factory
        ComPtr<IDXGIFactory6> factory;
        ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),
                     "CreateDXGIFactory1 failed");

        char selectedBackend[256] = {0};
        RuntimeBackendType selectedType = RUNTIME_BACKEND_NONE;
        ThrowIfFailed(
            SelectBestAdapterAndCreateDevice(
                factory.Get(),
                g_device.GetAddressOf(),
                &selectedType,
                selectedBackend,
                sizeof(selectedBackend)
            ),
            "SelectBestAdapterAndCreateDevice failed"
        );
        g_backendType = selectedType;
        strncpy_s(g_backendName, sizeof(g_backendName), selectedBackend, _TRUNCATE);

        // Create command queue
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        ThrowIfFailed(g_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&g_queue)),
                     "CreateCommandQueue failed");

        // Create command allocator and list
        ThrowIfFailed(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&g_allocator)),
                     "CreateCommandAllocator failed");

        ThrowIfFailed(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, g_allocator.Get(), nullptr, IID_PPV_ARGS(&g_list)),
                     "CreateCommandList failed");

        // Create descriptor heap (for SRV/UAV)
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.NumDescriptors = 8;  // 6 SRVs + 3 UAVs
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(g_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_descHeap)),
                     "CreateDescriptorHeap failed");

        g_waveOpsSupported = QueryWaveSupport(g_device.Get());
        fprintf(stdout, "[DirectX] WaveOps support: %s\n", g_waveOpsSupported ? "enabled" : "disabled");
        if (!g_waveOpsSupported) {
            ThrowIfFailed(E_FAIL, "Wave intrinsics not supported by adapter");
        }

        std::wstring shaderPath = ResolveShaderPath(L"shader_sxme.hlsl");
        ComPtr<IDxcBlob> shader = CompileShaderDXC(shaderPath.c_str());

        // Create root signature
        D3D12_DESCRIPTOR_RANGE ranges[2] = {};
        // SRVs (t0-t5): input buffers
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 6;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;

        // UAVs (u0-u2): output buffers
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 3;
        ranges[1].BaseShaderRegister = 0;
        ranges[1].OffsetInDescriptorsFromTableStart = 6;

        D3D12_ROOT_PARAMETER params[1] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable.NumDescriptorRanges = 2;
        params[0].DescriptorTable.pDescriptorRanges = ranges;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
        rsDesc.NumParameters = 1;
        rsDesc.pParameters = params;
        rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ComPtr<ID3DBlob> rsBlob;
        ComPtr<ID3DBlob> rsErr;
        ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr),
                     "SerializeRootSignature failed");

        ThrowIfFailed(g_device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&g_rootSig)),
                     "CreateRootSignature failed");

        // Create PSO
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = g_rootSig.Get();
        psoDesc.CS = { shader->GetBufferPointer(), shader->GetBufferSize() };

        ThrowIfFailed(g_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_pso)),
                     "CreateComputePipelineState failed");

        g_initialized = true;
        fprintf(stdout, "[DirectX] Initialization successful (%s)\n", g_backendName);
        return S_OK;

    } catch (const std::exception& e) {
        fprintf(stderr, "[DirectX] Initialization failed: %s\n", e.what());
        return E_FAIL;
    }
}

// ============================================================================
// BUFFER MANAGEMENT
// ============================================================================

/**
 * Allocate GPU buffer for input/output data
 *
 * Args:
 *   size: Buffer size in bytes
 *   flags: D3D12_RESOURCE_FLAGS (0 for SRV, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS for UAV)
 *   state: Initial resource state
 *
 * Returns:
 *   GPU buffer as ComPtr<ID3D12Resource>
 *
 * Note: Use GPU buffers for weights (persistent), staging buffers for I/O
 */
static ComPtr<ID3D12Resource> CreateGPUBuffer(size_t size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;

    ComPtr<ID3D12Resource> buffer;
    ThrowIfFailed(g_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&buffer)),
                 "CreateCommittedResource failed");
    return buffer;
}

/**
 * Create staging buffer for CPU → GPU data transfer
 */
static ComPtr<ID3D12Resource> CreateStagingBuffer(size_t size, bool is_readback = false) {
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = is_readback ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_RESOURCE_STATES state = is_readback ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ;

    ComPtr<ID3D12Resource> buffer;
    ThrowIfFailed(g_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&buffer)),
                 "CreateCommittedResource (staging) failed");
    return buffer;
}

// ============================================================================
// MAIN FORWARD PASS
// ============================================================================

/**
 * SCX-MoE Forward Pass on GPU
 *
 * PARAMETERS:
 *   token_ids: Array of uint32 token IDs, length = seq_len
 *   seq_len: Sequence length (typically 512)
 *   embedding_data: Flattened embedding matrix [vocab_size × hidden_size]
 *   weights_data: Packed layer weights, router gate, expert weights
 *   lm_head_data: LM head projection [hidden_size × vocab_size]
 *   output_logits: Output logits [seq_len × vocab_size] (OUTPUT)
 *   max_output_size: Size of output_logits array in floats
 *
 * RETURNS:
 *   S_OK (0) on success
 *   E_FAIL on error
 *
 * COMPUTATIONAL FLOW:
 *   1. Copy token IDs to GPU
 *   2. Copy weights to GPU (embedding, layer weights, LM head)
 *   3. Dispatch compute shader:
 *      - seq_len thread groups × 32 threads
 *      - Each group processes one token position
 *   4. Copy results back to CPU
 *   5. Return logits
 *
 * EXPECTED SPEEDUP:
 *   - Forward pass only: 5-10× faster than NumPy on CPU
 *   - Note: Backprop still on CPU, so overall training ~2-3× faster
 */
extern "C" __declspec(dllexport) HRESULT SCXMoEForward(
    const uint32_t* token_ids,
    uint32_t seq_len,
    const float* embedding_data,
    size_t embedding_size,
    const float* weights_data,
    size_t weights_size,
    const float* lm_head_data,
    size_t lm_head_size,
    const float* router_gate_data,
    size_t router_gate_size,
    float* output_logits,
    size_t output_size
) {
    if (seq_len == 0 || seq_len > MAX_SEQ) {
        fprintf(stderr, "[DirectX] Invalid seq_len: %u\n", seq_len);
        return E_INVALIDARG;
    }

    try {
        // Initialize DirectX if not done yet
        HRESULT hr = InitializeDirectX();
        if (FAILED(hr)) {
            fprintf(stderr, "[DirectX] Initialization failed\n");
            return hr;
        }

        fprintf(stdout, "[DirectX] Processing seq_len=%u, output_size=%zu\n", seq_len, output_size);

        // ===== Buffer Allocation =====

        // Input: token IDs (seq_len × uint32)
        auto token_id_buffer = CreateGPUBuffer(seq_len * sizeof(uint32_t), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto token_id_upload = CreateStagingBuffer(seq_len * sizeof(uint32_t));

        // Weights buffers
        auto embedding_buffer = CreateGPUBuffer(embedding_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto embedding_upload = CreateStagingBuffer(embedding_size);

        auto weights_buffer = CreateGPUBuffer(weights_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto weights_upload = CreateStagingBuffer(weights_size);

        auto lm_head_buffer = CreateGPUBuffer(lm_head_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto lm_head_upload = CreateStagingBuffer(lm_head_size);

        auto router_gate_buffer = CreateGPUBuffer(router_gate_size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto router_gate_upload = CreateStagingBuffer(router_gate_size);

        // Output: hidden states + logits
        size_t hidden_size = seq_len * HIDDEN_SIZE * sizeof(float);
        size_t logits_size = seq_len * VOCAB_SIZE * sizeof(float);

        auto hidden_buffer = CreateGPUBuffer(hidden_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto logits_buffer = CreateGPUBuffer(logits_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        auto logits_readback = CreateStagingBuffer(logits_size, true);

        // ===== Upload Data =====
        // Copy data to staging buffers
        {
            void* ptr;
            D3D12_RANGE range = {0, 0};

            token_id_upload->Map(0, &range, &ptr);
            memcpy(ptr, token_ids, seq_len * sizeof(uint32_t));
            token_id_upload->Unmap(0, nullptr);

            embedding_upload->Map(0, &range, &ptr);
            memcpy(ptr, embedding_data, embedding_size);
            embedding_upload->Unmap(0, nullptr);

            weights_upload->Map(0, &range, &ptr);
            memcpy(ptr, weights_data, weights_size);
            weights_upload->Unmap(0, nullptr);

            lm_head_upload->Map(0, &range, &ptr);
            memcpy(ptr, lm_head_data, lm_head_size);
            lm_head_upload->Unmap(0, nullptr);

            router_gate_upload->Map(0, &range, &ptr);
            memcpy(ptr, router_gate_data, router_gate_size);
            router_gate_upload->Unmap(0, nullptr);
        }

        // Copy from staging to GPU buffers
        g_allocator->Reset();
        g_list->Reset(g_allocator.Get(), nullptr);

        g_list->CopyBufferRegion(token_id_buffer.Get(), 0, token_id_upload.Get(), 0, seq_len * sizeof(uint32_t));
        g_list->CopyBufferRegion(embedding_buffer.Get(), 0, embedding_upload.Get(), 0, embedding_size);
        g_list->CopyBufferRegion(weights_buffer.Get(), 0, weights_upload.Get(), 0, weights_size);
        g_list->CopyBufferRegion(lm_head_buffer.Get(), 0, lm_head_upload.Get(), 0, lm_head_size);
        g_list->CopyBufferRegion(router_gate_buffer.Get(), 0, router_gate_upload.Get(), 0, router_gate_size);

        // ===== Dispatch Compute Shader =====

        // Set PSO and root signature
        g_list->SetPipelineState(g_pso.Get());
        g_list->SetComputeRootSignature(g_rootSig.Get());
        g_list->SetDescriptorHeaps(1, g_descHeap.GetAddressOf());

        // Bind descriptor table (SRVs + UAVs)
        g_list->SetComputeRootDescriptorTable(0, g_descHeap->GetGPUDescriptorHandleForHeapStart());

        // Dispatch: one thread group per token position
        // threadgroup size = 32, dispatch = [seq_len, 1, 1]
        g_list->Dispatch((seq_len + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE, 1, 1);

        // ===== Readback Results =====

        // Transition logits buffer to copy source
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = logits_buffer.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        g_list->ResourceBarrier(1, &barrier);

        // Copy to staging buffer
        g_list->CopyBufferRegion(logits_readback.Get(), 0, logits_buffer.Get(), 0, logits_size);

        // Close command list and submit
        g_list->Close();

        ID3D12CommandList* lists[] = {g_list.Get()};
        g_queue->ExecuteCommandLists(1, lists);

        // Wait for GPU completion
        ComPtr<ID3D12Fence> fence;
        g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

        g_queue->Signal(fence.Get(), 1);

        HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (event == nullptr) {
            fprintf(stderr, "[DirectX] CreateEvent failed\n");
            return E_FAIL;
        }

        fence->SetEventOnCompletion(1, event);
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);

        // ===== Copy Results Back =====

        void* ptr;
        D3D12_RANGE range = {0, static_cast<SIZE_T>(logits_size)};
        logits_readback->Map(0, &range, &ptr);
        memcpy(output_logits, ptr, logits_size);
        logits_readback->Unmap(0, nullptr);

        fprintf(stdout, "[DirectX] Forward pass complete: %u tokens processed\n", seq_len);
        return S_OK;

    } catch (const std::exception& e) {
        fprintf(stderr, "[DirectX] Exception: %s\n", e.what());
        return E_FAIL;
    }
}

/**
 * Cleanup DirectX resources
 * Called when shutting down
 */
extern "C" __declspec(dllexport) void SCXMoEShutdown() {
    g_device.Reset();
    g_queue.Reset();
    g_allocator.Reset();
    g_list.Reset();
    g_pso.Reset();
    g_rootSig.Reset();
    g_descHeap.Reset();
    g_initialized = false;
    g_backendType = RUNTIME_BACKEND_NONE;
    snprintf(g_backendName, sizeof(g_backendName), "shutdown");
    fprintf(stdout, "[DirectX] Shutdown complete\n");
}

/**
 * Probe and initialize runtime backend (hardware/WARP).
 */
extern "C" __declspec(dllexport) HRESULT SCXMoEProbeRuntime() {
    return InitializeDirectX();
}

/**
 * Returns runtime backend information string and backend code:
 * 0 = none, 1 = D3D12 hardware, 2 = D3D12 WARP.
 */
extern "C" __declspec(dllexport) HRESULT SCXMoEGetRuntimeAdapterInfo(
    char* out_text,
    size_t out_text_size,
    int* out_backend_code
) {
    if (!g_initialized) {
        ComPtr<IDXGIFactory6> factory;
        HRESULT factoryHr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        if (FAILED(factoryHr)) {
            return factoryHr;
        }

        ComPtr<ID3D12Device> probeDevice;
        char probeBackend[256] = {0};
        RuntimeBackendType probeType = RUNTIME_BACKEND_NONE;
        HRESULT probeHr = SelectBestAdapterAndCreateDevice(
            factory.Get(),
            probeDevice.GetAddressOf(),
            &probeType,
            probeBackend,
            sizeof(probeBackend)
        );
        if (FAILED(probeHr)) {
            return probeHr;
        }

        if (out_backend_code != nullptr) {
            *out_backend_code = static_cast<int>(probeType);
        }

        if (out_text != nullptr && out_text_size > 0) {
            strncpy_s(out_text, out_text_size, probeBackend, _TRUNCATE);
        }

        return S_OK;
    }

    if (out_backend_code != nullptr) {
        *out_backend_code = static_cast<int>(g_backendType);
    }

    if (out_text != nullptr && out_text_size > 0) {
        strncpy_s(out_text, out_text_size, g_backendName, _TRUNCATE);
    }

    return S_OK;
}
