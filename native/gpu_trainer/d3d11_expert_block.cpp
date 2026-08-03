#include "d3d11_expert_block.h"

#include <d3dcompiler.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace {

bool diagnosticsEnabled() {
    return std::getenv("KUHUL_D3D11_DEBUG") == std::string("1");
}

struct ExpertBlockParams {
    uint32_t batchSize;
    uint32_t hiddenSize;
    uint32_t blockRows;
    uint32_t outputOffset;
};

Microsoft::WRL::ComPtr<ID3D11Buffer> makeBuffer(
    ID3D11Device* device, uint32_t bytes, uint32_t bindFlags,
    D3D11_USAGE usage, uint32_t cpuAccess = 0) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = bytes;
    desc.BindFlags = bindFlags;
    desc.Usage = usage;
    desc.CPUAccessFlags = cpuAccess;
    if ((bindFlags & (D3D11_BIND_SHADER_RESOURCE |
                      D3D11_BIND_UNORDERED_ACCESS)) != 0 ||
        usage == D3D11_USAGE_STAGING) {
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
    }
    Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
    const HRESULT hr = device->CreateBuffer(&desc, nullptr, &buffer);
    if (FAILED(hr) && diagnosticsEnabled())
        std::cerr << "D3D11 CreateBuffer failed hr=0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << "\n";
    if (FAILED(hr)) return nullptr;
    return buffer;
}

Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> makeSrv(
    ID3D11Device* device, ID3D11Buffer* buffer, uint32_t elements) {
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    desc.Buffer.NumElements = elements;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
    const HRESULT hr = device->CreateShaderResourceView(buffer, &desc, &view);
    if (FAILED(hr) && diagnosticsEnabled())
        std::cerr << "D3D11 CreateSRV failed hr=0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << "\n";
    if (FAILED(hr))
        return nullptr;
    return view;
}

Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> makeRawSrv(
    ID3D11Device* device, ID3D11Buffer* buffer, uint32_t bytes) {
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    desc.BufferEx.NumElements = bytes / 4;
    desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
    if (FAILED(device->CreateShaderResourceView(buffer, &desc, &view)))
        return nullptr;
    return view;
}

Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> makeUav(
    ID3D11Device* device, ID3D11Buffer* buffer, uint32_t elements) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.NumElements = elements;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> view;
    const HRESULT hr = device->CreateUnorderedAccessView(buffer, &desc, &view);
    if (FAILED(hr) && diagnosticsEnabled())
        std::cerr << "D3D11 CreateUAV failed hr=0x" << std::hex
                  << static_cast<unsigned long>(hr) << std::dec << "\n";
    if (FAILED(hr))
        return nullptr;
    return view;
}

} // namespace

bool D3D11ExpertBlockGemm::init(D3D11Engine& engine,
                                const std::string& csoPath) {
    error_.clear();
    device_ = engine.rawDevice();
    context_ = engine.rawCtx();
    if (!device_ || !context_) {
        error_ = "D3D11 device is not initialized";
        return false;
    }

    std::ifstream file(csoPath, std::ios::binary | std::ios::ate);
    if (!file) {
        error_ = "cannot open expert block CSO: " + csoPath;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        error_ = "expert block CSO is empty";
        return false;
    }
    std::vector<uint8_t> bytecode(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytecode.data()), size)) {
        error_ = "cannot read expert block CSO";
        return false;
    }
    const HRESULT shaderHr = device_->CreateComputeShader(
        bytecode.data(), bytecode.size(), nullptr, &shader_);
    if (FAILED(shaderHr)) {
        if (diagnosticsEnabled())
            std::cerr << "D3D11 CreateComputeShader failed hr=0x" << std::hex
                      << static_cast<unsigned long>(shaderHr) << std::dec << "\n";
        error_ = "cannot create expert block compute shader";
        return false;
    }
    return true;
}

bool D3D11ExpertBlockGemm::run(const std::vector<float>& hidden,
                               const std::vector<float>& weights,
                               uint32_t batchSize, uint32_t hiddenSize,
                               uint32_t blockRows,
                               std::vector<float>& output) {
    error_.clear();
    if (!device_ || !context_ || !shader_) {
        error_ = "expert block adapter is not initialized";
        return false;
    }
    const uint64_t hiddenElements =
        static_cast<uint64_t>(batchSize) * hiddenSize;
    const uint64_t weightElements =
        static_cast<uint64_t>(blockRows) * hiddenSize;
    const uint64_t outputElements =
        static_cast<uint64_t>(batchSize) * blockRows;
    if (hidden.size() != hiddenElements || weights.size() != weightElements ||
        outputElements > std::numeric_limits<uint32_t>::max()) {
        error_ = "expert block tensor shape mismatch";
        return false;
    }

    const uint32_t hiddenBytes = static_cast<uint32_t>(hiddenElements * 4);
    const uint32_t weightBytes = static_cast<uint32_t>(weightElements * 4);
    const uint32_t outputBytes = static_cast<uint32_t>(outputElements * 4);
    auto hiddenBuffer = makeBuffer(
        device_, hiddenBytes, D3D11_BIND_SHADER_RESOURCE,
        D3D11_USAGE_DEFAULT);
    auto weightBuffer = makeBuffer(
        device_, weightBytes, D3D11_BIND_SHADER_RESOURCE,
        D3D11_USAGE_DEFAULT);
    auto outputBuffer = makeBuffer(
        device_, outputBytes, D3D11_BIND_UNORDERED_ACCESS,
        D3D11_USAGE_DEFAULT);
    auto staging = makeBuffer(
        device_, outputBytes, 0, D3D11_USAGE_STAGING,
        D3D11_CPU_ACCESS_READ);
    auto hiddenSrv = makeSrv(
        device_, hiddenBuffer.Get(), static_cast<uint32_t>(hiddenElements));
    auto weightSrv = makeSrv(
        device_, weightBuffer.Get(), static_cast<uint32_t>(weightElements));
    auto outputUav = makeUav(
        device_, outputBuffer.Get(), static_cast<uint32_t>(outputElements));
    if (!hiddenBuffer || !weightBuffer || !outputBuffer || !staging ||
        !hiddenSrv || !weightSrv || !outputUav) {
        error_ = "cannot allocate expert block buffers";
        return false;
    }

    context_->UpdateSubresource(
        hiddenBuffer.Get(), 0, nullptr, hidden.data(), 0, 0);
    context_->UpdateSubresource(
        weightBuffer.Get(), 0, nullptr, weights.data(), 0, 0);

    ExpertBlockParams params{batchSize, hiddenSize, blockRows, 0};
    auto constantBuffer = makeBuffer(
        device_, sizeof(params), D3D11_BIND_CONSTANT_BUFFER,
        D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE);
    if (!constantBuffer) {
        error_ = "cannot allocate expert block parameters";
        return false;
    }
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(context_->Map(
            constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        error_ = "cannot map expert block parameters";
        return false;
    }
    std::memcpy(mapped.pData, &params, sizeof(params));
    context_->Unmap(constantBuffer.Get(), 0);

    ID3D11ShaderResourceView* srvs[] = {hiddenSrv.Get(), weightSrv.Get()};
    ID3D11UnorderedAccessView* uavs[] = {outputUav.Get()};
    ID3D11Buffer* constants[] = {constantBuffer.Get()};
    context_->CSSetShader(shader_.Get(), nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, constants);
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    if (diagnosticsEnabled()) std::cerr << "D3D11 Dispatch begin\n";
    context_->Dispatch(
        (static_cast<uint32_t>(outputElements) + 63) / 64, 1, 1);
    if (diagnosticsEnabled()) std::cerr << "D3D11 Dispatch complete\n";
    context_->CopyResource(staging.Get(), outputBuffer.Get());

    ID3D11ShaderResourceView* nullSrvs[] = {nullptr, nullptr};
    ID3D11UnorderedAccessView* nullUavs[] = {nullptr};
    context_->CSSetShaderResources(0, 2, nullSrvs);
    context_->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
    context_->CSSetShader(nullptr, nullptr, 0);

    D3D11_MAPPED_SUBRESOURCE result{};
    if (diagnosticsEnabled()) std::cerr << "D3D11 Map begin\n";
    const HRESULT mapHr = context_->Map(
        staging.Get(), 0, D3D11_MAP_READ, 0, &result);
    if (FAILED(mapHr)) {
        if (diagnosticsEnabled())
            std::cerr << "D3D11 Map failed hr=0x" << std::hex
                      << static_cast<unsigned long>(mapHr) << std::dec << "\n";
        error_ = "cannot map expert block result";
        return false;
    }
    output.resize(static_cast<size_t>(outputElements));
    std::memcpy(output.data(), result.pData, outputBytes);
    context_->Unmap(staging.Get(), 0);
    return true;
}

bool D3D11ExpertBlockInt8Gemm::init(D3D11Engine& engine,
                                    const std::string& csoPath) {
    error_.clear();
    device_ = engine.rawDevice();
    context_ = engine.rawCtx();
    if (!device_ || !context_) {
        error_ = "D3D11 device is not initialized";
        return false;
    }
    std::ifstream file(csoPath, std::ios::binary | std::ios::ate);
    if (!file) {
        error_ = "cannot open INT8 expert block CSO: " + csoPath;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        error_ = "INT8 expert block CSO is empty";
        return false;
    }
    std::vector<uint8_t> bytecode(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytecode.data()), size)) {
        error_ = "cannot read INT8 expert block CSO";
        return false;
    }
    if (FAILED(device_->CreateComputeShader(
            bytecode.data(), bytecode.size(), nullptr, &shader_))) {
        error_ = "cannot create INT8 expert block compute shader";
        return false;
    }
    return true;
}

bool D3D11ExpertBlockInt8Gemm::run(
    const std::vector<float>& hidden, const std::vector<uint8_t>& weights,
    uint32_t batchSize, uint32_t hiddenSize, uint32_t blockRows, float scale,
    float zeroPoint, std::vector<float>& output) {
    error_.clear();
    if (!device_ || !context_ || !shader_) {
        error_ = "INT8 expert block adapter is not initialized";
        return false;
    }
    const uint64_t hiddenElements =
        static_cast<uint64_t>(batchSize) * hiddenSize;
    const uint64_t weightElements =
        static_cast<uint64_t>(blockRows) * hiddenSize;
    const uint64_t outputElements =
        static_cast<uint64_t>(batchSize) * blockRows;
    if (hidden.size() != hiddenElements || weights.size() != weightElements ||
        outputElements > std::numeric_limits<uint32_t>::max()) {
        error_ = "INT8 expert block tensor shape mismatch";
        return false;
    }
    const uint32_t hiddenBytes = static_cast<uint32_t>(hiddenElements * 4);
    const uint32_t rawWeightBytes =
        (static_cast<uint32_t>(weights.size()) + 3u) & ~3u;
    const uint32_t outputBytes = static_cast<uint32_t>(outputElements * 4);
    auto hiddenBuffer = makeBuffer(
        device_, hiddenBytes, D3D11_BIND_SHADER_RESOURCE,
        D3D11_USAGE_DEFAULT);
    D3D11_BUFFER_DESC weightDesc{};
    weightDesc.ByteWidth = rawWeightBytes;
    weightDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    weightDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    Microsoft::WRL::ComPtr<ID3D11Buffer> weightBuffer;
    if (FAILED(device_->CreateBuffer(&weightDesc, nullptr, &weightBuffer)))
        weightBuffer = nullptr;
    auto outputBuffer = makeBuffer(
        device_, outputBytes, D3D11_BIND_UNORDERED_ACCESS,
        D3D11_USAGE_DEFAULT);
    auto staging = makeBuffer(
        device_, outputBytes, 0, D3D11_USAGE_STAGING,
        D3D11_CPU_ACCESS_READ);
    auto hiddenSrv = makeSrv(
        device_, hiddenBuffer.Get(), static_cast<uint32_t>(hiddenElements));
    auto weightSrv = makeRawSrv(device_, weightBuffer.Get(), rawWeightBytes);
    auto outputUav = makeUav(
        device_, outputBuffer.Get(), static_cast<uint32_t>(outputElements));
    if (!hiddenBuffer || !weightBuffer || !outputBuffer || !staging ||
        !hiddenSrv || !weightSrv || !outputUav) {
        error_ = "cannot allocate INT8 expert block buffers";
        return false;
    }
    std::vector<uint8_t> paddedWeights(rawWeightBytes, 0);
    std::memcpy(paddedWeights.data(), weights.data(), weights.size());
    context_->UpdateSubresource(
        hiddenBuffer.Get(), 0, nullptr, hidden.data(), 0, 0);
    context_->UpdateSubresource(
        weightBuffer.Get(), 0, nullptr, paddedWeights.data(), 0, 0);

    struct Int8Params {
        uint32_t batchSize;
        uint32_t hiddenSize;
        uint32_t blockRows;
        uint32_t outputOffset;
        float scale;
        float zeroPoint;
        uint32_t pad0;
        uint32_t pad1;
    } params{batchSize, hiddenSize, blockRows, 0, scale, zeroPoint, 0, 0};
    auto constantBuffer = makeBuffer(
        device_, sizeof(params), D3D11_BIND_CONSTANT_BUFFER,
        D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE);
    if (!constantBuffer) {
        error_ = "cannot allocate INT8 expert block parameters";
        return false;
    }
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(context_->Map(
            constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        error_ = "cannot map INT8 expert block parameters";
        return false;
    }
    std::memcpy(mapped.pData, &params, sizeof(params));
    context_->Unmap(constantBuffer.Get(), 0);

    ID3D11ShaderResourceView* srvs[] = {hiddenSrv.Get(), weightSrv.Get()};
    ID3D11UnorderedAccessView* uavs[] = {outputUav.Get()};
    ID3D11Buffer* constants[] = {constantBuffer.Get()};
    context_->CSSetShader(shader_.Get(), nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, constants);
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    context_->Dispatch(
        (static_cast<uint32_t>(outputElements) + 63) / 64, 1, 1);
    context_->CopyResource(staging.Get(), outputBuffer.Get());
    ID3D11ShaderResourceView* nullSrvs[] = {nullptr, nullptr};
    ID3D11UnorderedAccessView* nullUavs[] = {nullptr};
    context_->CSSetShaderResources(0, 2, nullSrvs);
    context_->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
    context_->CSSetShader(nullptr, nullptr, 0);

    D3D11_MAPPED_SUBRESOURCE result{};
    if (FAILED(context_->Map(
            staging.Get(), 0, D3D11_MAP_READ, 0, &result))) {
        error_ = "cannot map INT8 expert block result";
        return false;
    }
    output.resize(static_cast<size_t>(outputElements));
    std::memcpy(output.data(), result.pData, outputBytes);
    context_->Unmap(staging.Get(), 0);
    return true;
}

bool D3D11ExpertBlockInt4Gemm::init(D3D11Engine& engine,
                                    const std::string& csoPath) {
    error_.clear();
    device_ = engine.rawDevice();
    context_ = engine.rawCtx();
    if (!device_ || !context_) {
        error_ = "D3D11 device is not initialized";
        return false;
    }
    std::ifstream file(csoPath, std::ios::binary | std::ios::ate);
    if (!file) {
        error_ = "cannot open INT4 expert block CSO: " + csoPath;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        error_ = "INT4 expert block CSO is empty";
        return false;
    }
    std::vector<uint8_t> bytecode(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytecode.data()), size)) {
        error_ = "cannot read INT4 expert block CSO";
        return false;
    }
    if (FAILED(device_->CreateComputeShader(
            bytecode.data(), bytecode.size(), nullptr, &shader_))) {
        error_ = "cannot create INT4 expert block compute shader";
        return false;
    }
    return true;
}

bool D3D11ExpertBlockInt4Gemm::run(
    const std::vector<float>& hidden,
    const std::vector<uint8_t>& packedWeights, uint32_t batchSize,
    uint32_t hiddenSize, uint32_t blockRows, float scale, float zeroPoint,
    std::vector<float>& output) {
    error_.clear();
    if (!device_ || !context_ || !shader_) {
        error_ = "INT4 expert block adapter is not initialized";
        return false;
    }
    const uint64_t hiddenElements =
        static_cast<uint64_t>(batchSize) * hiddenSize;
    const uint64_t weightElements =
        static_cast<uint64_t>(blockRows) * hiddenSize;
    const uint64_t packedBytes = (weightElements + 1) / 2;
    const uint64_t outputElements =
        static_cast<uint64_t>(batchSize) * blockRows;
    if (hidden.size() != hiddenElements ||
        packedWeights.size() != packedBytes ||
        outputElements > std::numeric_limits<uint32_t>::max()) {
        error_ = "INT4 expert block tensor shape mismatch";
        return false;
    }
    const uint32_t hiddenBytes = static_cast<uint32_t>(hiddenElements * 4);
    const uint32_t rawWeightBytes =
        (static_cast<uint32_t>(packedWeights.size()) + 3u) & ~3u;
    const uint32_t outputBytes = static_cast<uint32_t>(outputElements * 4);
    auto hiddenBuffer = makeBuffer(
        device_, hiddenBytes, D3D11_BIND_SHADER_RESOURCE,
        D3D11_USAGE_DEFAULT);
    D3D11_BUFFER_DESC weightDesc{};
    weightDesc.ByteWidth = rawWeightBytes;
    weightDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    weightDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    Microsoft::WRL::ComPtr<ID3D11Buffer> weightBuffer;
    if (FAILED(device_->CreateBuffer(&weightDesc, nullptr, &weightBuffer)))
        weightBuffer = nullptr;
    auto outputBuffer = makeBuffer(
        device_, outputBytes, D3D11_BIND_UNORDERED_ACCESS,
        D3D11_USAGE_DEFAULT);
    auto staging = makeBuffer(
        device_, outputBytes, 0, D3D11_USAGE_STAGING,
        D3D11_CPU_ACCESS_READ);
    auto hiddenSrv = makeSrv(
        device_, hiddenBuffer.Get(), static_cast<uint32_t>(hiddenElements));
    auto weightSrv = makeRawSrv(device_, weightBuffer.Get(), rawWeightBytes);
    auto outputUav = makeUav(
        device_, outputBuffer.Get(), static_cast<uint32_t>(outputElements));
    if (!hiddenBuffer || !weightBuffer || !outputBuffer || !staging ||
        !hiddenSrv || !weightSrv || !outputUav) {
        error_ = "cannot allocate INT4 expert block buffers";
        return false;
    }
    std::vector<uint8_t> paddedWeights(rawWeightBytes, uint8_t{0});
    std::memcpy(paddedWeights.data(), packedWeights.data(),
                packedWeights.size());
    context_->UpdateSubresource(
        hiddenBuffer.Get(), 0, nullptr, hidden.data(), 0, 0);
    context_->UpdateSubresource(
        weightBuffer.Get(), 0, nullptr, paddedWeights.data(), 0, 0);

    struct Int4Params {
        uint32_t batchSize;
        uint32_t hiddenSize;
        uint32_t blockRows;
        uint32_t outputOffset;
        float scale;
        float zeroPoint;
        uint32_t pad0;
        uint32_t pad1;
    } params{batchSize, hiddenSize, blockRows, 0, scale, zeroPoint, 0, 0};
    auto constantBuffer = makeBuffer(
        device_, sizeof(params), D3D11_BIND_CONSTANT_BUFFER,
        D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE);
    if (!constantBuffer) {
        error_ = "cannot allocate INT4 expert block parameters";
        return false;
    }
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(context_->Map(
            constantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        error_ = "cannot map INT4 expert block parameters";
        return false;
    }
    std::memcpy(mapped.pData, &params, sizeof(params));
    context_->Unmap(constantBuffer.Get(), 0);

    ID3D11ShaderResourceView* srvs[] = {hiddenSrv.Get(), weightSrv.Get()};
    ID3D11UnorderedAccessView* uavs[] = {outputUav.Get()};
    ID3D11Buffer* constants[] = {constantBuffer.Get()};
    context_->CSSetShader(shader_.Get(), nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, constants);
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
    context_->Dispatch(
        (static_cast<uint32_t>(outputElements) + 63) / 64, 1, 1);
    context_->CopyResource(staging.Get(), outputBuffer.Get());
    ID3D11ShaderResourceView* nullSrvs[] = {nullptr, nullptr};
    ID3D11UnorderedAccessView* nullUavs[] = {nullptr};
    context_->CSSetShaderResources(0, 2, nullSrvs);
    context_->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
    context_->CSSetShader(nullptr, nullptr, 0);

    D3D11_MAPPED_SUBRESOURCE result{};
    if (FAILED(context_->Map(
            staging.Get(), 0, D3D11_MAP_READ, 0, &result))) {
        error_ = "cannot map INT4 expert block result";
        return false;
    }
    output.resize(static_cast<size_t>(outputElements));
    std::memcpy(output.data(), result.pData, outputBytes);
    context_->Unmap(staging.Get(), 0);
    return true;
}
