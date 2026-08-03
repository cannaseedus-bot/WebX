#pragma once

#include "d3d11_engine.h"

#include <cstdint>
#include <string>
#include <vector>

class D3D11ExpertBlockGemm {
public:
    bool init(D3D11Engine& engine,
              const std::string& csoPath =
                  "native/shaders/bin/d3d11/sxme_expert_block.cso");
    bool run(const std::vector<float>& hidden,
             const std::vector<float>& weights,
             uint32_t batchSize, uint32_t hiddenSize, uint32_t blockRows,
             std::vector<float>& output);

    const std::string& error() const { return error_; }

private:
    ID3D11Device* device_ = nullptr;
    ID3D11DeviceContext* context_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader_;
    std::string error_;
};

class D3D11ExpertBlockInt8Gemm {
public:
    bool init(D3D11Engine& engine,
              const std::string& csoPath =
                  "native/shaders/bin/d3d11/sxme_expert_block_int8.cso");
    bool run(const std::vector<float>& hidden,
             const std::vector<uint8_t>& weights,
             uint32_t batchSize, uint32_t hiddenSize, uint32_t blockRows,
             float scale, float zeroPoint, std::vector<float>& output);

    const std::string& error() const { return error_; }

private:
    ID3D11Device* device_ = nullptr;
    ID3D11DeviceContext* context_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader_;
    std::string error_;
};

class D3D11ExpertBlockInt4Gemm {
public:
    bool init(D3D11Engine& engine,
              const std::string& csoPath =
                  "native/shaders/bin/d3d11/sxme_expert_block_int4.cso");
    bool run(const std::vector<float>& hidden,
             const std::vector<uint8_t>& packedWeights,
             uint32_t batchSize, uint32_t hiddenSize, uint32_t blockRows,
             float scale, float zeroPoint, std::vector<float>& output);

    const std::string& error() const { return error_; }

private:
    ID3D11Device* device_ = nullptr;
    ID3D11DeviceContext* context_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader_;
    std::string error_;
};
