#pragma once

#include <cstdint>
#include <string>

#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "xvm_core.h"

using Microsoft::WRL::ComPtr;

class D3D12Engine {
 public:
  bool init(bool verboseLog = false);
  bool uploadVM(XVMState& vm);
  bool dispatch(XVMState& vm);
  bool dumpTrace(std::uint32_t maxEntries = 32);
  bool streamSCXQ2ToCodeBuffer(const wchar_t* path, std::uint64_t fileSize);
  bool streamFileSliceToCodeBuffer(const wchar_t* path, std::uint64_t fileOffset, std::uint64_t bytes);

  D3D_FEATURE_LEVEL featureLevel() const { return featureLevel_; }
  const std::string& adapterName() const { return adapterName_; }
  const std::string& initReason() const { return initReason_; }
  const std::string& capabilitySummary() const { return capabilitySummary_; }
  const std::string& attentionKernel() const { return attentionKernel_; }

  // Tensor layer access — expose device/queue for GPT2 model layer
  ID3D12Device*       rawDevice() const { return device_.Get(); }
  ID3D12CommandQueue* rawQueue()  const { return queue_.Get(); }

 private:
  ComPtr<ID3D12Device> device_;
  ComPtr<ID3D12CommandQueue> queue_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> cmdList_;

  ComPtr<ID3D12Fence> fence_;
  HANDLE fenceEvent_ = nullptr;
  std::uint64_t fenceValue_ = 0;

  ComPtr<ID3D12DescriptorHeap> heap_;
  UINT descriptorSize_ = 0;

  ComPtr<ID3D12RootSignature> rootSig_;
  ComPtr<ID3D12PipelineState> pso_;
  D3D_FEATURE_LEVEL featureLevel_ = D3D_FEATURE_LEVEL_1_0_CORE;
  std::string adapterName_;
  std::string initReason_;
  std::string capabilitySummary_;
  bool waveOpsSupported_ = false;
  std::string attentionKernel_ = "xvm_fused_qkv_attention.hlsl";

  ComPtr<ID3D12Resource> codeBuffer_;
  ComPtr<ID3D12Resource> fiberBuffer_;
  ComPtr<ID3D12Resource> sharedBuffer_;
  ComPtr<ID3D12Resource> stackBuffer_;
  ComPtr<ID3D12Resource> traceBuffer_;
  ComPtr<ID3D12Resource> traceIndexBuffer_;
  ComPtr<ID3D12Resource> traceReadback_;
  ComPtr<ID3D12Resource> traceIndexReadback_;

  std::uint32_t traceCapacityWords_ = 131072;

  bool createDescriptorHeap();
  bool createRootSignatureAndPSO();
  bool createViews(std::uint32_t codeWordCount, std::uint32_t fiberCount, std::uint32_t sharedCount, std::uint32_t stackCount);

  void waitGPU();
};
