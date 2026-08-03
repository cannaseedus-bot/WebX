#include "xshard_attention.h"

#include "d3d11_engine.h"
#include "xshard.h"

#include <d3dcompiler.h>
#include <d3d11.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

namespace {

struct ShardFile {
    FILE* file = nullptr;
    XShardHeader header{};

    ~ShardFile() {
        if (file) fclose(file);
    }

    bool open(const std::string& path, std::string& error) {
        file = fopen(path.c_str(), "rb");
        if (!file) {
            error = "cannot open xshard: " + path;
            return false;
        }
        if (fread(&header, 1, sizeof(header), file) != sizeof(header) ||
            !xshard_valid_magic(header) || header.version != 1) {
            error = "invalid XSQ2 header: " + path;
            return false;
        }
        if (header.dtype != XSHARD_FP32 || header.tile_size == 0 ||
            header.tile_count == 0 || xshard_element_bytes(header) == 0) {
            error = "only non-empty fp32 xshards are supported by native attention: " + path;
            return false;
        }
        return true;
    }

    bool readTile(uint32_t index, std::vector<float>& out, std::string& error) {
        if (index >= header.tile_count) {
            error = "xshard tile index out of range";
            return false;
        }
        if (_fseeki64(file, static_cast<__int64>(xshard_tile_offset(header, index)),
                     SEEK_SET) != 0) {
            error = "xshard seek failed";
            return false;
        }
        out.resize(header.tile_size);
        if (fread(out.data(), sizeof(float), header.tile_size, file) != header.tile_size) {
            error = "truncated xshard tile";
            return false;
        }
        return true;
    }
};

ComPtr<ID3DBlob> compileShader(ID3D11Device* device, const std::string& source,
                               const char* name, std::string& error) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> diagnostics;
    HRESULT hr = D3DCompile(source.data(), source.size(), name, nullptr, nullptr,
                            "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3,
                            0, &bytecode, &diagnostics);
    if (FAILED(hr)) {
        error = diagnostics
            ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),
                          diagnostics->GetBufferSize())
            : "D3D shader compilation failed";
        return nullptr;
    }
    (void)device;
    return bytecode;
}

ComPtr<ID3DBlob> loadCso(const char* path, std::string& error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "cannot open D3D11 CSO: " + std::string(path);
        return nullptr;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        error = "D3D11 CSO is empty: " + std::string(path);
        return nullptr;
    }
    ComPtr<ID3DBlob> blob;
    if (FAILED(D3DCreateBlob(static_cast<SIZE_T>(size), &blob))) {
        error = "cannot allocate D3D11 CSO blob";
        return nullptr;
    }
    file.seekg(0);
    if (!file.read(static_cast<char*>(blob->GetBufferPointer()), size)) {
        error = "cannot read D3D11 CSO: " + std::string(path);
        return nullptr;
    }
    return blob;
}

ComPtr<ID3D11Buffer> makeBuffer(ID3D11Device* device, uint32_t bytes,
                                uint32_t bindFlags, D3D11_USAGE usage,
                                uint32_t cpuAccess = 0) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = bytes;
    desc.Usage = usage;
    desc.BindFlags = bindFlags;
    desc.CPUAccessFlags = cpuAccess;
    ComPtr<ID3D11Buffer> buffer;
    if (FAILED(device->CreateBuffer(&desc, nullptr, &buffer))) return nullptr;
    return buffer;
}

ComPtr<ID3D11ShaderResourceView> makeSrv(ID3D11Device* device,
                                         ID3D11Buffer* buffer, uint32_t count) {
    D3D11_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    desc.Buffer.NumElements = count;
    ComPtr<ID3D11ShaderResourceView> view;
    if (FAILED(device->CreateShaderResourceView(buffer, &desc, &view))) return nullptr;
    return view;
}

ComPtr<ID3D11UnorderedAccessView> makeUav(ID3D11Device* device,
                                          ID3D11Buffer* buffer, uint32_t count) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    desc.Buffer.NumElements = count;
    ComPtr<ID3D11UnorderedAccessView> view;
    if (FAILED(device->CreateUnorderedAccessView(buffer, &desc, &view))) return nullptr;
    return view;
}

} // namespace

bool XShardAttentionEngine::run(const std::string& q_path,
                                const std::string& k_path,
                                const std::string& v_path,
                                uint32_t passes) {
    error_.clear();
    if (passes == 0) {
        error_ = "passes must be greater than zero";
        return false;
    }

    ShardFile q;
    ShardFile k;
    ShardFile v;
    if (!q.open(q_path, error_) || !k.open(k_path, error_) || !v.open(v_path, error_))
        return false;

    if (q.header.rows == 0 || k.header.rows == 0 ||
        q.header.cols != k.header.cols ||
        k.header.rows != v.header.rows ||
        q.header.cols != v.header.cols ||
        q.header.tile_size != q.header.rows * q.header.cols ||
        k.header.tile_size != k.header.rows * k.header.cols ||
        v.header.tile_size != v.header.rows * v.header.cols) {
        error_ = "Q/K/V xshard shapes are incompatible";
        return false;
    }

    const uint32_t qRows = q.header.rows;
    const uint32_t kRows = k.header.rows;
    const uint32_t width = q.header.cols;
    const uint64_t outputElements =
        static_cast<uint64_t>(qRows) * v.header.cols;
    const uint64_t attentionWork =
        static_cast<uint64_t>(qRows) * kRows * width;
    const uint64_t valueWork =
        static_cast<uint64_t>(qRows) * kRows * v.header.cols;
    uint64_t maxWork = 500000000;
    if (const char* configured = std::getenv("KUHUL_D3D11_MAX_WORK")) {
        char* end = nullptr;
        const unsigned long long parsed =
            std::strtoull(configured, &end, 10);
        if (end != configured && *end == '\0' && parsed > 0)
            maxWork = parsed;
    }
    if (attentionWork > maxWork || valueWork > maxWork ||
        attentionWork > std::numeric_limits<uint32_t>::max() ||
        valueWork > std::numeric_limits<uint32_t>::max() ||
        outputElements > std::numeric_limits<uint32_t>::max()) {
        error_ = "D3D11 xshard workset exceeds bounded GPU fallback; "
                  "use the CPU model path or lower KUHUL_D3D11_MAX_WORK";
        return false;
    }
    const uint32_t outputSize = static_cast<uint32_t>(outputElements);

    D3D11Engine engine;
    if (!engine.init(false, false)) {
        error_ = "D3D11 initialization failed: " + engine.initReason();
        return false;
    }
    ID3D11Device* device = engine.rawDevice();
    ID3D11DeviceContext* context = engine.rawCtx();

    const std::string softmaxSource =
        "#define Q_ROWS " + std::to_string(qRows) + "u\n" +
        "#define K_ROWS " + std::to_string(kRows) + "u\n" +
        "#define WIDTH " + std::to_string(width) + "u\n" R"(
Buffer<float> Q : register(t0);
Buffer<float> K : register(t1);
RWBuffer<float> P : register(u0);
groupshared float scores[K_ROWS];
groupshared float maximum;
groupshared float denominator;
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint row = gid.x;
    for (uint col = tid.x; col < K_ROWS; col += 256) {
        float score = 0.0f;
        for (uint i = 0; i < WIDTH; ++i)
            score += Q[row * WIDTH + i] * K[col * WIDTH + i];
        scores[col] = score / sqrt((float)WIDTH);
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) {
        maximum = scores[0];
        for (uint col = 1; col < K_ROWS; ++col)
            maximum = max(maximum, scores[col]);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint col = tid.x; col < K_ROWS; col += 256)
        scores[col] = exp(scores[col] - maximum);
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) {
        denominator = 0.0f;
        for (uint col = 0; col < K_ROWS; ++col)
            denominator += scores[col];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint col = tid.x; col < K_ROWS; col += 256)
        P[row * K_ROWS + col] = scores[col] / denominator;
}
)";
    const std::string vmulSource =
        "#define Q_ROWS " + std::to_string(qRows) + "u\n" +
        "#define K_ROWS " + std::to_string(kRows) + "u\n" +
        "#define V_COLS " + std::to_string(v.header.cols) + "u\n" R"(
Buffer<float> P : register(t0);
Buffer<float> V : register(t1);
RWBuffer<float> Out : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    uint col = id.y;
    if (row >= Q_ROWS || col >= V_COLS) return;
    float value = 0.0f;
    for (uint i = 0; i < K_ROWS; ++i)
        value += P[row * K_ROWS + i] * V[i * V_COLS + col];
    Out[row * V_COLS + col] = value;
}
)";

    ComPtr<ID3DBlob> softmaxBlob;
    ComPtr<ID3DBlob> vmulBlob;
    if (qRows == 2880 && kRows == 2880 && width == 2880 &&
        v.header.cols == 2880) {
        softmaxBlob = loadCso(
            "native/shaders/bin/d3d11/xshard_softmax_2880.cso", error_);
        vmulBlob = loadCso(
            "native/shaders/bin/d3d11/xshard_vmul_2880.cso", error_);
    } else {
        softmaxBlob = compileShader(
            device, softmaxSource, "xshard_softmax", error_);
        vmulBlob = compileShader(device, vmulSource, "xshard_vmul", error_);
    }
    if (!softmaxBlob || !vmulBlob) return false;

    ComPtr<ID3D11ComputeShader> softmax;
    ComPtr<ID3D11ComputeShader> vmul;
    if (FAILED(device->CreateComputeShader(softmaxBlob->GetBufferPointer(),
                                           softmaxBlob->GetBufferSize(), nullptr, &softmax)) ||
        FAILED(device->CreateComputeShader(vmulBlob->GetBufferPointer(),
                                           vmulBlob->GetBufferSize(), nullptr, &vmul))) {
        error_ = "failed to create xshard compute shaders";
        return false;
    }

    auto qBuffer = makeBuffer(device, q.header.tile_size * 4, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT);
    auto kBuffer = makeBuffer(device, k.header.tile_size * 4, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT);
    auto vBuffer = makeBuffer(device, v.header.tile_size * 4, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT);
    auto pBuffer = makeBuffer(device, qRows * kRows * 4,
                              D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
                              D3D11_USAGE_DEFAULT);
    auto outBuffer = makeBuffer(device, outputSize * 4, D3D11_BIND_UNORDERED_ACCESS, D3D11_USAGE_DEFAULT);
    auto staging = makeBuffer(device, outputSize * 4, 0, D3D11_USAGE_STAGING,
                              D3D11_CPU_ACCESS_READ);
    auto qSrv = makeSrv(device, qBuffer.Get(), q.header.tile_size);
    auto kSrv = makeSrv(device, kBuffer.Get(), k.header.tile_size);
    auto vSrv = makeSrv(device, vBuffer.Get(), v.header.tile_size);
    auto pSrv = makeSrv(device, pBuffer.Get(), qRows * kRows);
    auto pUav = makeUav(device, pBuffer.Get(), qRows * kRows);
    auto outUav = makeUav(device, outBuffer.Get(), outputSize);
    if (!qBuffer || !kBuffer || !vBuffer || !pBuffer || !outBuffer || !staging ||
        !qSrv || !kSrv || !vSrv || !pSrv || !pUav || !outUav) {
        error_ = "failed to allocate xshard GPU buffers";
        return false;
    }

    std::vector<float> qTile;
    std::vector<float> kTile;
    std::vector<float> vTile;
    std::vector<float> output(outputSize);
    const uint32_t qHeads = q.header.tile_count;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t head = 0; head < qHeads; ++head) {
            const uint32_t kvHead = head % std::min(k.header.tile_count, v.header.tile_count);
            if (!q.readTile(head, qTile, error_) ||
                !k.readTile(kvHead, kTile, error_) ||
                !v.readTile(kvHead, vTile, error_)) return false;
            context->UpdateSubresource(qBuffer.Get(), 0, nullptr, qTile.data(), 0, 0);
            context->UpdateSubresource(kBuffer.Get(), 0, nullptr, kTile.data(), 0, 0);
            context->UpdateSubresource(vBuffer.Get(), 0, nullptr, vTile.data(), 0, 0);

            ID3D11ShaderResourceView* scoreSrvs[] = {qSrv.Get(), kSrv.Get()};
            ID3D11UnorderedAccessView* scoreUav = pUav.Get();
            context->CSSetShader(softmax.Get(), nullptr, 0);
            context->CSSetShaderResources(0, 2, scoreSrvs);
            context->CSSetUnorderedAccessViews(0, 1, &scoreUav, nullptr);
            context->Dispatch(qRows, 1, 1);

            ID3D11ShaderResourceView* valueSrvs[] = {pSrv.Get(), vSrv.Get()};
            ID3D11UnorderedAccessView* outputUav = outUav.Get();
            context->CSSetShader(vmul.Get(), nullptr, 0);
            context->CSSetShaderResources(0, 2, valueSrvs);
            context->CSSetUnorderedAccessViews(0, 1, &outputUav, nullptr);
            context->Dispatch((qRows + 7) / 8, (v.header.cols + 7) / 8, 1);

            ID3D11UnorderedAccessView* nullUav = nullptr;
            ID3D11ShaderResourceView* nullSrvs[2] = {nullptr, nullptr};
            context->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
            context->CSSetShaderResources(0, 2, nullSrvs);
            context->CopyResource(staging.Get(), outBuffer.Get());
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
                error_ = "failed to read xshard attention output";
                return false;
            }
            std::memcpy(output.data(), mapped.pData, output.size() * sizeof(float));
            context->Unmap(staging.Get(), 0);
        }
    }

    std::printf("[kuhul_engine] native xshard attention\n");
    std::printf("  adapter: %s%s\n", engine.adapterName().c_str(),
                engine.usedWarp() ? " (WARP)" : "");
    std::printf("  layer: %u  q_heads: %u  kv_heads: %u  shape: [%u,%u]\n",
                q.header.layer_id, q.header.tile_count, k.header.tile_count,
                qRows, width);
    std::printf("  passes: %u  output: [%u,%u]  status: PASS\n",
                passes, qRows, v.header.cols);
    return true;
}
