#include <windows.h>
#undef min
#undef max
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl.h>

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

static ID3D12Device* g_device = nullptr;

static void ThrowIfFailed(HRESULT hr, const char* msg) {
  if (FAILED(hr)) {
    std::cerr << msg << " (HRESULT=0x" << std::hex << hr << ")";
    if (g_device) {
      HRESULT removed = g_device->GetDeviceRemovedReason();
      if (FAILED(removed)) {
        std::cerr << " | DeviceRemovedReason=0x" << std::hex << removed;
      }
    }
    std::cerr << std::endl;
    ExitProcess(1);
  }
}

static ComPtr<ID3DBlob> CompileCS(const wchar_t* path, const char* entry, const char* target) {
  UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
  flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
  ComPtr<ID3DBlob> shader;
  ComPtr<ID3DBlob> errors;
  HRESULT hr = D3DCompileFromFile(
    path, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
    entry, target, flags, 0, &shader, &errors);
  if (FAILED(hr)) {
    if (errors) {
      std::cerr << (const char*)errors->GetBufferPointer() << std::endl;
    }
    ThrowIfFailed(hr, "D3DCompileFromFile failed");
  }
  return shader;
}

static ComPtr<ID3DBlob> CompileCSWithFallback(const wchar_t* filename, const char* entry, const char* target) {
  if (GetFileAttributesW(filename) != INVALID_FILE_ATTRIBUTES) {
    return CompileCS(filename, entry, target);
  }

  wchar_t altPath[260] = {};
  _snwprintf(altPath, 259, L"native\\d3d12_compute\\%s", filename);
  if (GetFileAttributesW(altPath) != INVALID_FILE_ATTRIBUTES) {
    return CompileCS(altPath, entry, target);
  }

  return CompileCS(filename, entry, target);
}

static D3D12_RESOURCE_BARRIER TransitionBarrier(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Transition.pResource = res;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  return barrier;
}

struct D3DContext {
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> fence;
  HANDLE fenceEvent = nullptr;
  UINT64 fenceValue = 0;
};

static D3DContext CreateContext() {
  D3DContext ctx;

  ComPtr<IDXGIFactory6> factory;
  ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1 failed");

  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      continue;
    }
    break;
  }

  ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx.device)),
                "D3D12CreateDevice failed");
  g_device = ctx.device.Get();

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  ThrowIfFailed(ctx.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&ctx.queue)),
                "CreateCommandQueue failed");

  ThrowIfFailed(ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&ctx.allocator)),
                "CreateCommandAllocator failed");

  ThrowIfFailed(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, ctx.allocator.Get(), nullptr,
                                               IID_PPV_ARGS(&ctx.list)),
                "CreateCommandList failed");
  // A freshly created command list starts in recording state.
  // Close it once so the first ExecuteAndWait() can Reset allocator/list safely.
  ThrowIfFailed(ctx.list->Close(), "Initial command list close failed");

  ThrowIfFailed(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx.fence)), "CreateFence failed");
  ctx.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!ctx.fenceEvent) {
    ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), "CreateEvent failed");
  }

  return ctx;
}

static double ExecuteAndWait(D3DContext& ctx, ID3D12PipelineState* pso, const std::function<void(ID3D12GraphicsCommandList*)>& record) {
  ThrowIfFailed(ctx.allocator->Reset(), "CommandAllocator reset failed");
  ThrowIfFailed(ctx.list->Reset(ctx.allocator.Get(), pso), "CommandList reset failed");

  record(ctx.list.Get());

  ThrowIfFailed(ctx.list->Close(), "CommandList close failed");
  ID3D12CommandList* lists[] = { ctx.list.Get() };

  auto start = std::chrono::high_resolution_clock::now();
  ctx.queue->ExecuteCommandLists(1, lists);

  ctx.fenceValue++;
  ThrowIfFailed(ctx.queue->Signal(ctx.fence.Get(), ctx.fenceValue), "Queue signal failed");
  if (ctx.fence->GetCompletedValue() < ctx.fenceValue) {
    ThrowIfFailed(ctx.fence->SetEventOnCompletion(ctx.fenceValue, ctx.fenceEvent), "Fence SetEventOnCompletion failed");
    WaitForSingleObject(ctx.fenceEvent, INFINITE);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count();
}

static ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device, UINT64 size, D3D12_HEAP_TYPE heapType,
                                           D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = heapType;
  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = flags;

  ComPtr<ID3D12Resource> res;
  ThrowIfFailed(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&res)),
                "CreateCommittedResource failed");
  return res;
}

static void UploadBuffer(D3DContext& ctx, ID3D12Resource* dst, const void* data, UINT64 size, D3D12_RESOURCE_STATES afterState) {
  auto upload = CreateBuffer(ctx.device.Get(), size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ);

  void* mapped = nullptr;
  ThrowIfFailed(upload->Map(0, nullptr, &mapped), "Upload buffer map failed");
  memcpy(mapped, data, size);
  upload->Unmap(0, nullptr);

  ExecuteAndWait(ctx, nullptr, [&](ID3D12GraphicsCommandList* list) {
    list->CopyBufferRegion(dst, 0, upload.Get(), 0, size);
    auto barrier = TransitionBarrier(dst, D3D12_RESOURCE_STATE_COPY_DEST, afterState);
    list->ResourceBarrier(1, &barrier);
  });
}

static std::vector<float> ReadbackFloats(D3DContext& ctx, ID3D12Resource* src, UINT64 size) {
  auto readback = CreateBuffer(ctx.device.Get(), size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST);

  ExecuteAndWait(ctx, nullptr, [&](ID3D12GraphicsCommandList* list) {
    auto barrier = TransitionBarrier(src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback.Get(), 0, src, 0, size);
  });

  void* mapped = nullptr;
  ThrowIfFailed(readback->Map(0, nullptr, &mapped), "Readback map failed");
  std::vector<float> out(size / sizeof(float));
  memcpy(out.data(), mapped, size);
  readback->Unmap(0, nullptr);
  return out;
}

static ComPtr<ID3D12RootSignature> CreateRootSignature(ID3D12Device* device, const std::vector<D3D12_DESCRIPTOR_RANGE>& ranges) {
  D3D12_ROOT_PARAMETER param = {};
  param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  param.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(ranges.size());
  param.DescriptorTable.pDescriptorRanges = ranges.data();
  param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

  D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
  rsDesc.NumParameters = 1;
  rsDesc.pParameters = &param;
  rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ComPtr<ID3DBlob> rsBlob;
  ComPtr<ID3DBlob> rsErr;
  ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr),
                "SerializeRootSignature failed");

  ComPtr<ID3D12RootSignature> rootSig;
  ThrowIfFailed(device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)),
                "CreateRootSignature failed");
  return rootSig;
}

struct TriangleData {
  uint32_t v0;
  uint32_t v1;
  uint32_t v2;
  uint32_t pad0;
  float w0;
  float w1;
  float w2;
  float pad1;
};

struct ParsedMesh {
  std::vector<float> vertices;
  std::vector<TriangleData> triangles;
  std::vector<int32_t> neighbors;
  uint32_t vertexCount = 0;
  uint32_t triangleCount = 0;
};

#pragma pack(push, 1)
struct MeshXHeader {
  char magic[6];
  uint16_t version;
  uint16_t flags;
  uint32_t vertexCount;
  uint32_t triangleCount;
  uint32_t maxNeighbors;
  uint64_t verticesOffset;
  uint64_t trianglesOffset;
  uint64_t neighborsOffset;
};
#pragma pack(pop)

struct TensorBinHeader {
  uint32_t magic;
  uint32_t format;
  uint32_t count;
};

static bool ReadBinary(std::ifstream& file, void* dst, size_t size);

static bool LoadTensorBinFp32(const std::string& path, const char* targetName, std::vector<float>& out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;

  TensorBinHeader header = {};
  if (!ReadBinary(file, &header, sizeof(header))) return false;
  if (header.magic != 0x54424E31 || header.format != 0) return false;

  for (uint32_t i = 0; i < header.count; i++) {
    uint32_t nameLen = 0;
    if (!ReadBinary(file, &nameLen, sizeof(nameLen))) return false;
    std::string name(nameLen, '\0');
    if (!ReadBinary(file, name.data(), nameLen)) return false;
    uint32_t rank = 0;
    if (!ReadBinary(file, &rank, sizeof(rank))) return false;
    uint32_t shape[8] = {};
    if (rank > 8) return false;
    if (!ReadBinary(file, shape, sizeof(uint32_t) * rank)) return false;
    uint64_t count = 1;
    for (uint32_t r = 0; r < rank; r++) count *= shape[r];

    if (strcmp(name.c_str(), targetName) == 0) {
      out.resize(static_cast<size_t>(count));
      if (!ReadBinary(file, out.data(), sizeof(float) * out.size())) return false;
      return true;
    } else {
      file.seekg(static_cast<std::streamoff>(sizeof(float) * count), std::ios::cur);
      if (!file.good()) return false;
    }
  }
  return false;
}

static bool ReadBinary(std::ifstream& file, void* dst, size_t size) {
  file.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size));
  return file.good();
}

static bool LoadMeshX(const std::string& path, ParsedMesh& out, uint32_t& maxNeighbors) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }

  MeshXHeader header = {};
  if (!ReadBinary(file, &header, sizeof(header))) {
    return false;
  }

  if (strncmp(header.magic, "MESHX1", 6) != 0) {
    return false;
  }

  if (header.vertexCount == 0 || header.triangleCount == 0) {
    return false;
  }

  maxNeighbors = header.maxNeighbors > 0 ? header.maxNeighbors : 6;
  out.vertexCount = header.vertexCount;
  out.triangleCount = header.triangleCount;

  out.vertices.resize(static_cast<size_t>(out.vertexCount) * 4);
  out.triangles.resize(out.triangleCount);

  file.seekg(static_cast<std::streamoff>(header.verticesOffset), std::ios::beg);
  if (!ReadBinary(file, out.vertices.data(), out.vertices.size() * sizeof(float))) {
    return false;
  }

  file.seekg(static_cast<std::streamoff>(header.trianglesOffset), std::ios::beg);
  if (!ReadBinary(file, out.triangles.data(), out.triangleCount * sizeof(TriangleData))) {
    return false;
  }

  if (header.neighborsOffset != 0) {
    out.neighbors.resize(static_cast<size_t>(out.triangleCount) * maxNeighbors);
    file.seekg(static_cast<std::streamoff>(header.neighborsOffset), std::ios::beg);
    if (!ReadBinary(file, out.neighbors.data(), out.neighbors.size() * sizeof(int32_t))) {
      return false;
    }
  }

  return true;
}

static bool FileExistsA(const std::string& path) {
  DWORD attr = GetFileAttributesA(path.c_str());
  return (attr != INVALID_FILE_ATTRIBUTES) && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

struct DdsPixelData {
  std::vector<float> data;
  uint32_t width = 0;
  uint32_t height = 0;
};

static bool LoadDdsR32Float(const std::string& path, DdsPixelData& out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }

  uint32_t magic = 0;
  if (!ReadBinary(file, &magic, sizeof(magic))) {
    return false;
  }
  if (magic != 0x20534444) {
    return false;
  }

  uint32_t header[31] = {};
  if (!ReadBinary(file, header, sizeof(header))) {
    return false;
  }

  uint32_t height = header[2];
  uint32_t width = header[3];
  uint32_t pfFlags = header[19];
  uint32_t fourCC = header[20];

  if (pfFlags != 0x00000004 || fourCC != 0x30315844) {
    return false;
  }

  uint32_t dx10[5] = {};
  if (!ReadBinary(file, dx10, sizeof(dx10))) {
    return false;
  }

  uint32_t dxgiFormat = dx10[0];
  if (dxgiFormat != 41) {
    return false;
  }

  uint64_t count = static_cast<uint64_t>(width) * height;
  if (count == 0) {
    return false;
  }

  out.data.resize(static_cast<size_t>(count));
  if (!ReadBinary(file, out.data.data(), out.data.size() * sizeof(float))) {
    return false;
  }

  out.width = width;
  out.height = height;
  return true;
}

static bool ReadTextFile(const std::string& path, std::string& out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  out = ss.str();
  return true;
}

static bool ExtractAttrInTag(const std::string& s, size_t tagPos, const char* attr, std::string& out) {
  size_t tagEnd = s.find('>', tagPos);
  if (tagEnd == std::string::npos) {
    return false;
  }

  size_t attrPos = s.find(attr, tagPos);
  if (attrPos == std::string::npos || attrPos > tagEnd) {
    return false;
  }

  size_t eqPos = s.find('=', attrPos);
  if (eqPos == std::string::npos || eqPos > tagEnd) {
    return false;
  }

  size_t quotePos = s.find_first_of("\"'", eqPos + 1);
  if (quotePos == std::string::npos || quotePos > tagEnd) {
    return false;
  }
  char quote = s[quotePos];
  size_t endQuote = s.find(quote, quotePos + 1);
  if (endQuote == std::string::npos || endQuote > tagEnd) {
    return false;
  }

  out = s.substr(quotePos + 1, endQuote - quotePos - 1);
  return true;
}

static void AddNeighbor(std::vector<int32_t>& neighbors, uint32_t tri, int32_t other, uint32_t maxNeighbors) {
  uint32_t base = tri * maxNeighbors;
  for (uint32_t i = 0; i < maxNeighbors; i++) {
    if (neighbors[base + i] == other) {
      return;
    }
    if (neighbors[base + i] < 0) {
      neighbors[base + i] = other;
      return;
    }
  }
}

static void BuildNeighbors(uint32_t triangleCount, const std::vector<TriangleData>& tris,
                           std::vector<int32_t>& neighbors, uint32_t maxNeighbors) {
  neighbors.assign(triangleCount * maxNeighbors, -1);
  std::unordered_map<uint64_t, uint32_t> edgeOwner;
  edgeOwner.reserve(triangleCount * 3);

  auto edgeKey = [](uint32_t a, uint32_t b) -> uint64_t {
    uint32_t lo = (a < b) ? a : b;
    uint32_t hi = (a < b) ? b : a;
    return (static_cast<uint64_t>(lo) << 32) | hi;
  };

  for (uint32_t i = 0; i < triangleCount; i++) {
    const TriangleData& t = tris[i];
    uint32_t v[3] = { t.v0, t.v1, t.v2 };
    for (uint32_t e = 0; e < 3; e++) {
      uint32_t a = v[e];
      uint32_t b = v[(e + 1) % 3];
      uint64_t key = edgeKey(a, b);
      auto it = edgeOwner.find(key);
      if (it == edgeOwner.end()) {
        edgeOwner[key] = i;
      } else {
        uint32_t other = it->second;
        AddNeighbor(neighbors, i, static_cast<int32_t>(other), maxNeighbors);
        AddNeighbor(neighbors, other, static_cast<int32_t>(i), maxNeighbors);
      }
    }
  }
}

static bool ParseMeshXml(const std::string& path, ParsedMesh& out, bool& hasPrecomputedNeighbors) {
  std::string xml;
  if (!ReadTextFile(path, xml)) {
    return false;
  }

  std::unordered_map<std::string, uint32_t> idToIndex;
  idToIndex.reserve(4096);
  hasPrecomputedNeighbors = false;

  struct TriRef {
    TriangleData t;
    std::string id;
    std::string n0;
    std::string n1;
    std::string n2;
  };
  std::vector<TriRef> triRefs;
  triRefs.reserve(1024);

  size_t pos = 0;
  while ((pos = xml.find("<vertex", pos)) != std::string::npos) {
    std::string id, xs, ys, zs;
    if (ExtractAttrInTag(xml, pos, "id", id) &&
        ExtractAttrInTag(xml, pos, "x", xs) &&
        ExtractAttrInTag(xml, pos, "y", ys) &&
        ExtractAttrInTag(xml, pos, "z", zs)) {
      float x = std::strtof(xs.c_str(), nullptr);
      float y = std::strtof(ys.c_str(), nullptr);
      float z = std::strtof(zs.c_str(), nullptr);
      uint32_t index = static_cast<uint32_t>(out.vertexCount);
      idToIndex[id] = index;
      out.vertices.push_back(x);
      out.vertices.push_back(y);
      out.vertices.push_back(z);
      out.vertices.push_back(1.0f);
      out.vertexCount++;
    }
    pos++;
  }

  pos = 0;
  while ((pos = xml.find("<triangle", pos)) != std::string::npos) {
    std::string as, bs, cs, id, n0, n1, n2;
    if (ExtractAttrInTag(xml, pos, "a", as) &&
        ExtractAttrInTag(xml, pos, "b", bs) &&
        ExtractAttrInTag(xml, pos, "c", cs)) {
      auto itA = idToIndex.find(as);
      auto itB = idToIndex.find(bs);
      auto itC = idToIndex.find(cs);
      if (itA != idToIndex.end() && itB != idToIndex.end() && itC != idToIndex.end()) {
        TriangleData t = {};
        t.v0 = itA->second;
        t.v1 = itB->second;
        t.v2 = itC->second;
        t.w0 = 0.6f;
        t.w1 = 0.2f;
        t.w2 = 0.2f;

        TriRef ref = {};
        ref.t = t;
        ExtractAttrInTag(xml, pos, "id", id);
        ExtractAttrInTag(xml, pos, "n0", n0);
        ExtractAttrInTag(xml, pos, "n1", n1);
        ExtractAttrInTag(xml, pos, "n2", n2);
        ref.id = id;
        ref.n0 = n0;
        ref.n1 = n1;
        ref.n2 = n2;

        if (!ref.n0.empty() || !ref.n1.empty() || !ref.n2.empty()) {
          hasPrecomputedNeighbors = true;
        }

        triRefs.push_back(ref);
      }
    }
    pos++;
  }

  out.triangleCount = static_cast<uint32_t>(triRefs.size());
  out.triangles.reserve(out.triangleCount);
  for (const auto& ref : triRefs) {
    out.triangles.push_back(ref.t);
  }

  if (hasPrecomputedNeighbors) {
    std::unordered_map<std::string, uint32_t> triIdToIndex;
    triIdToIndex.reserve(triRefs.size());
    for (uint32_t i = 0; i < triRefs.size(); i++) {
      if (!triRefs[i].id.empty()) {
        triIdToIndex[triRefs[i].id] = i;
      }
    }

    out.neighbors.assign(out.triangleCount * 6, -1);
    auto resolveIndex = [&](const std::string& token) -> int32_t {
      if (token.empty()) {
        return -1;
      }
      auto it = triIdToIndex.find(token);
      if (it != triIdToIndex.end()) {
        return static_cast<int32_t>(it->second);
      }
      char* endptr = nullptr;
      long val = strtol(token.c_str(), &endptr, 10);
      if (endptr && *endptr == '\0' && val >= 0 && val < static_cast<long>(out.triangleCount)) {
        return static_cast<int32_t>(val);
      }
      return -1;
    };

    for (uint32_t i = 0; i < triRefs.size(); i++) {
      int32_t n0i = resolveIndex(triRefs[i].n0);
      int32_t n1i = resolveIndex(triRefs[i].n1);
      int32_t n2i = resolveIndex(triRefs[i].n2);
      if (n0i >= 0) AddNeighbor(out.neighbors, i, n0i, 6);
      if (n1i >= 0) AddNeighbor(out.neighbors, i, n1i, 6);
      if (n2i >= 0) AddNeighbor(out.neighbors, i, n2i, 6);
    }
  }

  return out.vertexCount > 0 && out.triangleCount > 0;
}

struct KuhulBuffer {
  ComPtr<ID3D12Resource> resource;
  UINT64 size = 0;
};

struct KuhulRuntime {
  D3DContext* ctx = nullptr;
  ComPtr<ID3D12RootSignature> vecRoot;
  ComPtr<ID3D12PipelineState> vecPso;
  ComPtr<ID3D12RootSignature> triRoot;
  ComPtr<ID3D12PipelineState> triPso;
  ComPtr<ID3D12Resource> vertexBuffer;
  ComPtr<ID3D12Resource> neighborBuffer;
  ComPtr<ID3D12Resource> triangleBuffer;
  UINT vertexCount = 0;
  UINT triangleCount = 0;
  UINT maxNeighbors = 0;
  std::vector<KuhulBuffer> buffers;
};

struct KuhulInst {
  const char* glyph = nullptr;
  UINT64 size = 0;
  const void* initData = nullptr;
  D3D12_RESOURCE_STATES afterState = D3D12_RESOURCE_STATE_COMMON;
  int a = -1;
  int b = -1;
  int out = -1;
  UINT count = 0;
  int phaseIn = -1;
  int outIn = -1;
  int phaseOut = -1;
  int outOut = -1;
  UINT triangleCount = 0;
  int* resultId = nullptr;
};

static int KuhulAlloc(KuhulRuntime& rt, UINT64 size, const void* initData, D3D12_RESOURCE_STATES afterState) {
  auto buffer = CreateBuffer(rt.ctx->device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
  if (initData) {
    UploadBuffer(*rt.ctx, buffer.Get(), initData, size, afterState);
  } else {
    std::vector<uint8_t> zero(size, 0);
    UploadBuffer(*rt.ctx, buffer.Get(), zero.data(), size, afterState);
  }

  rt.buffers.push_back({ buffer, size });
  return static_cast<int>(rt.buffers.size() - 1);
}

static ComPtr<ID3D12DescriptorHeap> CreateTriangleHeap(ID3D12Device* device, ID3D12Resource* vertexBuffer,
                                                       ID3D12Resource* neighborBuffer, ID3D12Resource* triangleBuffer,
                                                       ID3D12Resource* tensorBuffer, UINT tensorCount,
                                                       UINT vertexCount, UINT triangleCount, UINT maxNeighbors,
                                                       ID3D12Resource* phaseIn, ID3D12Resource* outIn,
                                                       ID3D12Resource* phaseOut, ID3D12Resource* outOut) {
  D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
  heapDesc.NumDescriptors = 8;
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ComPtr<ID3D12DescriptorHeap> heap;
  ThrowIfFailed(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap failed");

  D3D12_SHADER_RESOURCE_VIEW_DESC vertexSrv = {};
  vertexSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  vertexSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  vertexSrv.Buffer.NumElements = vertexCount;
  vertexSrv.Buffer.StructureByteStride = sizeof(float) * 4;
  vertexSrv.Format = DXGI_FORMAT_UNKNOWN;

  D3D12_SHADER_RESOURCE_VIEW_DESC neighborSrv = {};
  neighborSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  neighborSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  neighborSrv.Buffer.NumElements = triangleCount * maxNeighbors;
  neighborSrv.Buffer.StructureByteStride = sizeof(int32_t);
  neighborSrv.Format = DXGI_FORMAT_UNKNOWN;

  D3D12_SHADER_RESOURCE_VIEW_DESC phaseSrv = {};
  phaseSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  phaseSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  phaseSrv.Buffer.NumElements = triangleCount;
  phaseSrv.Buffer.StructureByteStride = sizeof(float);
  phaseSrv.Format = DXGI_FORMAT_UNKNOWN;

  D3D12_SHADER_RESOURCE_VIEW_DESC outSrv = phaseSrv;

  D3D12_SHADER_RESOURCE_VIEW_DESC tensorSrv = {};
  tensorSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  tensorSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  tensorSrv.Buffer.NumElements = tensorCount;
  tensorSrv.Buffer.StructureByteStride = sizeof(float);
  tensorSrv.Format = DXGI_FORMAT_UNKNOWN;

  D3D12_UNORDERED_ACCESS_VIEW_DESC triUav = {};
  triUav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  triUav.Buffer.NumElements = triangleCount;
  triUav.Buffer.StructureByteStride = sizeof(TriangleData);
  triUav.Format = DXGI_FORMAT_UNKNOWN;

  D3D12_UNORDERED_ACCESS_VIEW_DESC phaseUav = {};
  phaseUav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  phaseUav.Buffer.NumElements = triangleCount;
  phaseUav.Buffer.StructureByteStride = sizeof(float);
  phaseUav.Format = DXGI_FORMAT_UNKNOWN;

  D3D12_UNORDERED_ACCESS_VIEW_DESC outUav = phaseUav;

  const UINT handleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto cpuHandle = heap->GetCPUDescriptorHandleForHeapStart();
  device->CreateShaderResourceView(vertexBuffer, &vertexSrv, cpuHandle);
  cpuHandle.ptr += handleSize;
  device->CreateShaderResourceView(neighborBuffer, &neighborSrv, cpuHandle);
  cpuHandle.ptr += handleSize;
  device->CreateShaderResourceView(phaseIn, &phaseSrv, cpuHandle);
  cpuHandle.ptr += handleSize;
  device->CreateShaderResourceView(outIn, &outSrv, cpuHandle);
  cpuHandle.ptr += handleSize;
  device->CreateShaderResourceView(tensorBuffer, &tensorSrv, cpuHandle);
  cpuHandle.ptr += handleSize;
  device->CreateUnorderedAccessView(triangleBuffer, nullptr, &triUav, cpuHandle);
  cpuHandle.ptr += handleSize;
  device->CreateUnorderedAccessView(phaseOut, nullptr, &phaseUav, cpuHandle);
  cpuHandle.ptr += handleSize;
  device->CreateUnorderedAccessView(outOut, nullptr, &outUav, cpuHandle);

  return heap;
}

static void DispatchVectorAdd(D3DContext& ctx, ID3D12RootSignature* rootSig, ID3D12PipelineState* pso,
                              ID3D12Resource* a, ID3D12Resource* b, ID3D12Resource* out, UINT elementCount) {
  D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
  heapDesc.NumDescriptors = 3;
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ComPtr<ID3D12DescriptorHeap> heap;
  ThrowIfFailed(ctx.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap failed");

  D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
  srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
  srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srvDesc.Buffer.NumElements = elementCount;
  srvDesc.Buffer.StructureByteStride = sizeof(float);
  srvDesc.Format = DXGI_FORMAT_UNKNOWN;

  D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
  uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  uavDesc.Buffer.NumElements = elementCount;
  uavDesc.Buffer.StructureByteStride = sizeof(float);
  uavDesc.Format = DXGI_FORMAT_UNKNOWN;

  const UINT handleSize = ctx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  auto cpuHandle = heap->GetCPUDescriptorHandleForHeapStart();
  ctx.device->CreateShaderResourceView(a, &srvDesc, cpuHandle);
  cpuHandle.ptr += handleSize;
  ctx.device->CreateShaderResourceView(b, &srvDesc, cpuHandle);
  cpuHandle.ptr += handleSize;
  ctx.device->CreateUnorderedAccessView(out, nullptr, &uavDesc, cpuHandle);

  ExecuteAndWait(ctx, pso, [&](ID3D12GraphicsCommandList* list) {
    list->SetDescriptorHeaps(1, heap.GetAddressOf());
    list->SetComputeRootSignature(rootSig);
    list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    list->Dispatch((elementCount + 63) / 64, 1, 1);
  });
}

static void RunKuhulProgram(KuhulRuntime& rt, const std::vector<KuhulInst>& program) {
  for (const auto& inst : program) {
    if (!inst.glyph) {
      continue;
    }

    if (strcmp(inst.glyph, "Wo") == 0) {
      int id = KuhulAlloc(rt, inst.size, inst.initData, inst.afterState);
      if (inst.resultId) {
        *inst.resultId = id;
      }
      continue;
    }

    if (strcmp(inst.glyph, "Sek.vector_add") == 0) {
      DispatchVectorAdd(*rt.ctx, rt.vecRoot.Get(), rt.vecPso.Get(),
                        rt.buffers[inst.a].resource.Get(),
                        rt.buffers[inst.b].resource.Get(),
                        rt.buffers[inst.out].resource.Get(),
                        inst.count);
      continue;
    }

    if (strcmp(inst.glyph, "Sek.triangle_step") == 0) {
      auto heap = CreateTriangleHeap(rt.ctx->device.Get(),
                                     rt.vertexBuffer.Get(),
                                     rt.neighborBuffer.Get(),
                                     rt.triangleBuffer.Get(),
                                     rt.buffers[inst.outIn].resource.Get(),
                                     inst.triangleCount,
                                     rt.vertexCount,
                                     inst.triangleCount,
                                     rt.maxNeighbors,
                                     rt.buffers[inst.phaseIn].resource.Get(),
                                     rt.buffers[inst.outIn].resource.Get(),
                                     rt.buffers[inst.phaseOut].resource.Get(),
                                     rt.buffers[inst.outOut].resource.Get());

      ExecuteAndWait(*rt.ctx, rt.triPso.Get(), [&](ID3D12GraphicsCommandList* list) {
        list->SetDescriptorHeaps(1, heap.GetAddressOf());
        list->SetComputeRootSignature(rt.triRoot.Get());
        list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        list->Dispatch((inst.triangleCount + 63) / 64, 1, 1);
      });
      continue;
    }

    std::cerr << "Unknown KUHUL glyph: " << inst.glyph << std::endl;
  }
}

int main(int argc, char** argv) {
  const UINT elementCount = 1024;
  const UINT bufferSize = elementCount * sizeof(float);

  D3DContext ctx = CreateContext();

  std::string gsnrDir;
  UINT steps = 8;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--gsnr") == 0 && i + 1 < argc) {
      gsnrDir = argv[++i];
    } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
      steps = static_cast<UINT>(atoi(argv[++i]));
    }
  }

  auto vectorShader = CompileCSWithFallback(L"shader.hlsl", "main", "cs_5_0");

  D3D12_DESCRIPTOR_RANGE vectorRanges[2] = {};
  vectorRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  vectorRanges[0].NumDescriptors = 2;
  vectorRanges[0].BaseShaderRegister = 0;
  vectorRanges[0].OffsetInDescriptorsFromTableStart = 0;

  vectorRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  vectorRanges[1].NumDescriptors = 1;
  vectorRanges[1].BaseShaderRegister = 0;
  vectorRanges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

  std::vector<D3D12_DESCRIPTOR_RANGE> vecRanges = { vectorRanges[0], vectorRanges[1] };
  auto vecRootSig = CreateRootSignature(ctx.device.Get(), vecRanges);

  D3D12_COMPUTE_PIPELINE_STATE_DESC vecPsoDesc = {};
  vecPsoDesc.pRootSignature = vecRootSig.Get();
  vecPsoDesc.CS = { vectorShader->GetBufferPointer(), vectorShader->GetBufferSize() };

  ComPtr<ID3D12PipelineState> vecPso;
  ThrowIfFailed(ctx.device->CreateComputePipelineState(&vecPsoDesc, IID_PPV_ARGS(&vecPso)),
                "CreateComputePipelineState failed");

  auto bufferA = CreateBuffer(ctx.device.Get(), bufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                              D3D12_RESOURCE_STATE_COPY_DEST);
  auto bufferB = CreateBuffer(ctx.device.Get(), bufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                              D3D12_RESOURCE_STATE_COPY_DEST);
  auto bufferOut = CreateBuffer(ctx.device.Get(), bufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  std::vector<float> a(elementCount), b(elementCount);
  for (UINT i = 0; i < elementCount; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(elementCount - i);
  }

  UploadBuffer(ctx, bufferA.Get(), a.data(), bufferSize, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  UploadBuffer(ctx, bufferB.Get(), b.data(), bufferSize, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

  DispatchVectorAdd(ctx, vecRootSig.Get(), vecPso.Get(), bufferA.Get(), bufferB.Get(), bufferOut.Get(), elementCount);
  auto vecOut = ReadbackFloats(ctx, bufferOut.Get(), bufferSize);
  std::cout << "D3D12 result: " << vecOut[0] << ", " << vecOut[1] << ", " << vecOut[2] << ", " << vecOut[3] << std::endl;

  auto triangleShader = CompileCSWithFallback(L"triangle.hlsl", "main", "cs_5_0");

  D3D12_DESCRIPTOR_RANGE triRanges[2] = {};
  triRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  triRanges[0].NumDescriptors = 5;
  triRanges[0].BaseShaderRegister = 0;
  triRanges[0].OffsetInDescriptorsFromTableStart = 0;

  triRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  triRanges[1].NumDescriptors = 3;
  triRanges[1].BaseShaderRegister = 0;
  triRanges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

  std::vector<D3D12_DESCRIPTOR_RANGE> triRangesVec = { triRanges[0], triRanges[1] };
  auto triRootSig = CreateRootSignature(ctx.device.Get(), triRangesVec);

  D3D12_COMPUTE_PIPELINE_STATE_DESC triPsoDesc = {};
  triPsoDesc.pRootSignature = triRootSig.Get();
  triPsoDesc.CS = { triangleShader->GetBufferPointer(), triangleShader->GetBufferSize() };

  ComPtr<ID3D12PipelineState> triPso;
  ThrowIfFailed(ctx.device->CreateComputePipelineState(&triPsoDesc, IID_PPV_ARGS(&triPso)),
                "CreateComputePipelineState failed");

  UINT triangleCount = 32;
  UINT vertexCount = 16;
  const UINT maxNeighbors = 6;
  ParsedMesh parsedMesh;
  bool hasGsnrMesh = false;

  std::vector<float> vertexData;
  std::vector<TriangleData> triangles;
  std::vector<int32_t> neighbors;

  std::vector<float> tensorWeights;
  UINT tensorCount = 1;

  if (!gsnrDir.empty()) {
    std::string meshxPath = gsnrDir;
    char last = meshxPath.empty() ? 0 : meshxPath.back();
    if (last != '\\' && last != '/') {
      meshxPath += "\\";
    }
    meshxPath += "meshx.bin";

    uint32_t meshxNeighbors = maxNeighbors;
    if (FileExistsA(meshxPath) && LoadMeshX(meshxPath, parsedMesh, meshxNeighbors)) {
      vertexCount = parsedMesh.vertexCount;
      triangleCount = parsedMesh.triangleCount;
      vertexData = std::move(parsedMesh.vertices);
      triangles = std::move(parsedMesh.triangles);
      neighbors = std::move(parsedMesh.neighbors);
      if (meshxNeighbors > 0) {
        if (meshxNeighbors != maxNeighbors) {
          std::cout << "meshx maxNeighbors=" << meshxNeighbors << " (expected " << maxNeighbors << ")" << std::endl;
        }
      }
      if (neighbors.empty()) {
        BuildNeighbors(triangleCount, triangles, neighbors, maxNeighbors);
      }
      hasGsnrMesh = true;
      std::cout << "MeshX loaded: " << meshxPath
                << " | vertices=" << vertexCount
                << " triangles=" << triangleCount << std::endl;
    }
  }

  if (!gsnrDir.empty() && !hasGsnrMesh) {
    std::string meshPath = gsnrDir;
    char last = meshPath.empty() ? 0 : meshPath.back();
    if (last != '\\' && last != '/') {
      meshPath += "\\";
    }
    meshPath += "mesh.xml";

    bool hasPrecomputedNeighbors = false;
    if (ParseMeshXml(meshPath, parsedMesh, hasPrecomputedNeighbors)) {
      vertexCount = parsedMesh.vertexCount;
      triangleCount = parsedMesh.triangleCount;
      vertexData = std::move(parsedMesh.vertices);
      triangles = std::move(parsedMesh.triangles);
      neighbors = std::move(parsedMesh.neighbors);
      if (!hasPrecomputedNeighbors || neighbors.empty()) {
        BuildNeighbors(triangleCount, triangles, neighbors, maxNeighbors);
      }
      hasGsnrMesh = true;
      std::cout << "GSNR mesh loaded: " << meshPath
                << " | vertices=" << vertexCount
                << " triangles=" << triangleCount << std::endl;
    } else {
      std::cout << "GSNR mesh load failed: " << meshPath << " (using demo mesh)" << std::endl;
    }
  }

  if (!gsnrDir.empty()) {
    std::string tensorPath = gsnrDir;
    char last = tensorPath.empty() ? 0 : tensorPath.back();
    if (last != '\\' && last != '/') {
      tensorPath += "\\";
    }
    tensorPath += "tensors_fp32.tbin";

    if (FileExistsA(tensorPath) && LoadTensorBinFp32(tensorPath, "triangle_weights", tensorWeights)) {
      tensorCount = static_cast<UINT>(tensorWeights.size());
      std::cout << "Loaded tensor weights: " << tensorPath << " (" << tensorCount << " floats)" << std::endl;
    }
  }

  if (!hasGsnrMesh) {
    vertexData.assign(vertexCount * 4, 0.0f);
    for (UINT i = 0; i < vertexCount; i++) {
      vertexData[i * 4] = 0.25f + 0.01f * static_cast<float>(i);
    }

    triangles.assign(triangleCount, {});
    for (UINT i = 0; i < triangleCount; i++) {
      triangles[i].v0 = i % vertexCount;
      triangles[i].v1 = (i + 1) % vertexCount;
      triangles[i].v2 = (i + 2) % vertexCount;
      triangles[i].w0 = 0.6f;
      triangles[i].w1 = 0.2f;
      triangles[i].w2 = 0.2f;
    }

    neighbors.assign(triangleCount * maxNeighbors, -1);
    for (UINT i = 0; i < triangleCount; i++) {
      neighbors[i * maxNeighbors + 0] = static_cast<int32_t>((i + triangleCount - 1) % triangleCount);
      neighbors[i * maxNeighbors + 1] = static_cast<int32_t>((i + 1) % triangleCount);
    }
  }

  const UINT phaseCount = triangleCount;
  std::vector<float> phaseA(phaseCount, 0.1f);
  std::vector<float> phaseB(phaseCount, 0.0f);
  std::vector<float> outA(phaseCount, 0.0f);
  std::vector<float> outB(phaseCount, 0.0f);

  if (tensorWeights.empty()) {
    tensorWeights.push_back(1.0f);
    tensorCount = 1;
  }

  if (!gsnrDir.empty()) {
    std::string phasePath = gsnrDir;
    std::string outPath = gsnrDir;
    char last = phasePath.empty() ? 0 : phasePath.back();
    if (last != '\\' && last != '/') {
      phasePath += "\\";
      outPath += "\\";
    }
    phasePath += "phase.dds";
    outPath += "output.dds";

    DdsPixelData phaseDds;
    if (FileExistsA(phasePath) && LoadDdsR32Float(phasePath, phaseDds)) {
      size_t count = phaseDds.data.size();
      size_t copyCount = std::min(count, phaseA.size());
      for (size_t i = 0; i < copyCount; i++) {
        phaseA[i] = phaseDds.data[i];
      }
      std::cout << "Loaded phase.dds (" << phaseDds.width << "x" << phaseDds.height << ")" << std::endl;
    }

    DdsPixelData outDds;
    if (FileExistsA(outPath) && LoadDdsR32Float(outPath, outDds)) {
      size_t count = outDds.data.size();
      size_t copyCount = std::min(count, outA.size());
      for (size_t i = 0; i < copyCount; i++) {
        outA[i] = outDds.data[i];
      }
      std::cout << "Loaded output.dds (" << outDds.width << "x" << outDds.height << ")" << std::endl;
    }
  }

  auto vertexBuffer = CreateBuffer(ctx.device.Get(), vertexData.size() * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  auto neighborBuffer = CreateBuffer(ctx.device.Get(), neighbors.size() * sizeof(int32_t), D3D12_HEAP_TYPE_DEFAULT,
                                     D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  auto triangleBuffer = CreateBuffer(ctx.device.Get(), triangles.size() * sizeof(TriangleData), D3D12_HEAP_TYPE_DEFAULT,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
  auto tensorBuffer = CreateBuffer(ctx.device.Get(), tensorWeights.size() * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  auto phaseBufferA = CreateBuffer(ctx.device.Get(), phaseA.size() * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
  auto phaseBufferB = CreateBuffer(ctx.device.Get(), phaseB.size() * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
  auto outBufferA = CreateBuffer(ctx.device.Get(), outA.size() * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
  auto outBufferB = CreateBuffer(ctx.device.Get(), outB.size() * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);

  UploadBuffer(ctx, vertexBuffer.Get(), vertexData.data(), vertexData.size() * sizeof(float),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  UploadBuffer(ctx, neighborBuffer.Get(), neighbors.data(), neighbors.size() * sizeof(int32_t),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  UploadBuffer(ctx, triangleBuffer.Get(), triangles.data(), triangles.size() * sizeof(TriangleData),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  UploadBuffer(ctx, tensorBuffer.Get(), tensorWeights.data(), tensorWeights.size() * sizeof(float),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  UploadBuffer(ctx, phaseBufferA.Get(), phaseA.data(), phaseA.size() * sizeof(float),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  UploadBuffer(ctx, phaseBufferB.Get(), phaseB.data(), phaseB.size() * sizeof(float),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  UploadBuffer(ctx, outBufferA.Get(), outA.data(), outA.size() * sizeof(float),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  UploadBuffer(ctx, outBufferB.Get(), outB.data(), outB.size() * sizeof(float),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

  auto heapPing0 = CreateTriangleHeap(ctx.device.Get(), vertexBuffer.Get(), neighborBuffer.Get(), triangleBuffer.Get(),
                                      tensorBuffer.Get(), tensorCount, vertexCount, triangleCount, maxNeighbors,
                                      phaseBufferA.Get(), outBufferA.Get(), phaseBufferB.Get(), outBufferB.Get());
  auto heapPing1 = CreateTriangleHeap(ctx.device.Get(), vertexBuffer.Get(), neighborBuffer.Get(), triangleBuffer.Get(),
                                      tensorBuffer.Get(), tensorCount, vertexCount, triangleCount, maxNeighbors,
                                      phaseBufferB.Get(), outBufferB.Get(), phaseBufferA.Get(), outBufferA.Get());

  ID3D12Resource* phaseIn = phaseBufferA.Get();
  ID3D12Resource* phaseOut = phaseBufferB.Get();
  ID3D12Resource* outIn = outBufferA.Get();
  ID3D12Resource* outOut = outBufferB.Get();

  double triangleMs = ExecuteAndWait(ctx, triPso.Get(), [&](ID3D12GraphicsCommandList* list) {
    list->SetComputeRootSignature(triRootSig.Get());

    for (UINT step = 0; step < steps; step++) {
      if (step > 0) {
        D3D12_RESOURCE_BARRIER barriers[4] = {
          TransitionBarrier(phaseIn, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
          TransitionBarrier(outIn, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
          TransitionBarrier(phaseOut, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
          TransitionBarrier(outOut, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        list->ResourceBarrier(4, barriers);
      }

      ID3D12DescriptorHeap* heap = (step % 2 == 0) ? heapPing0.Get() : heapPing1.Get();
      list->SetDescriptorHeaps(1, &heap);
      list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
      list->Dispatch((triangleCount + 63) / 64, 1, 1);

      std::swap(phaseIn, phaseOut);
      std::swap(outIn, outOut);
    }
  });

  auto finalOut = (steps % 2 == 0) ? outBufferA.Get() : outBufferB.Get();
  auto triOut = ReadbackFloats(ctx, finalOut, triangleCount * sizeof(float));

  double stepsPerSec = (steps / (triangleMs / 1000.0));
  double trisPerSec = (stepsPerSec * triangleCount);

  std::cout << "Triangle mesh output: " << triOut[0] << ", " << triOut[1] << ", " << triOut[2] << ", " << triOut[3] << std::endl;
  std::cout << "Triangle steps: " << steps << " in " << triangleMs << " ms"
            << " | steps/sec=" << stepsPerSec << " | tris/sec=" << trisPerSec
            << " | fence=" << ctx.fenceValue << std::endl;

  KuhulRuntime rt;
  rt.ctx = &ctx;
  rt.vecRoot = vecRootSig;
  rt.vecPso = vecPso;
  rt.triRoot = triRootSig;
  rt.triPso = triPso;
  rt.vertexBuffer = vertexBuffer;
  rt.neighborBuffer = neighborBuffer;
  rt.triangleBuffer = triangleBuffer;
  rt.vertexCount = vertexCount;
  rt.triangleCount = triangleCount;
  rt.maxNeighbors = maxNeighbors;

  int bufA = -1;
  int bufB = -1;
  int bufOut = -1;
  int phaseInId = -1;
  int outInId = -1;
  int phaseOutId = -1;
  int outOutId = -1;

  std::vector<float> kuhulPhaseIn(phaseCount, 0.15f);
  std::vector<float> kuhulOutIn(phaseCount, 0.0f);
  std::vector<float> kuhulPhaseOut(phaseCount, 0.0f);
  std::vector<float> kuhulOutOut(phaseCount, 0.0f);

  std::vector<KuhulInst> allocProgram;
  KuhulInst inst;

  inst = {};
  inst.glyph = "Wo";
  inst.size = bufferSize;
  inst.initData = a.data();
  inst.afterState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  inst.resultId = &bufA;
  allocProgram.push_back(inst);

  inst = {};
  inst.glyph = "Wo";
  inst.size = bufferSize;
  inst.initData = b.data();
  inst.afterState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  inst.resultId = &bufB;
  allocProgram.push_back(inst);

  inst = {};
  inst.glyph = "Wo";
  inst.size = bufferSize;
  inst.initData = nullptr;
  inst.afterState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  inst.resultId = &bufOut;
  allocProgram.push_back(inst);

  inst = {};
  inst.glyph = "Wo";
  inst.size = phaseCount * sizeof(float);
  inst.initData = kuhulPhaseIn.data();
  inst.afterState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  inst.resultId = &phaseInId;
  allocProgram.push_back(inst);

  inst = {};
  inst.glyph = "Wo";
  inst.size = phaseCount * sizeof(float);
  inst.initData = kuhulOutIn.data();
  inst.afterState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  inst.resultId = &outInId;
  allocProgram.push_back(inst);

  inst = {};
  inst.glyph = "Wo";
  inst.size = phaseCount * sizeof(float);
  inst.initData = kuhulPhaseOut.data();
  inst.afterState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  inst.resultId = &phaseOutId;
  allocProgram.push_back(inst);

  inst = {};
  inst.glyph = "Wo";
  inst.size = phaseCount * sizeof(float);
  inst.initData = kuhulOutOut.data();
  inst.afterState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  inst.resultId = &outOutId;
  allocProgram.push_back(inst);

  std::cout << "KUHUL backend: D3D12" << std::endl;
  RunKuhulProgram(rt, allocProgram);

  std::vector<KuhulInst> computeProgram;

  inst = {};
  inst.glyph = "Sek.vector_add";
  inst.a = bufA;
  inst.b = bufB;
  inst.out = bufOut;
  inst.count = elementCount;
  computeProgram.push_back(inst);

  inst = {};
  inst.glyph = "Sek.triangle_step";
  inst.phaseIn = phaseInId;
  inst.outIn = outInId;
  inst.phaseOut = phaseOutId;
  inst.outOut = outOutId;
  inst.triangleCount = triangleCount;
  computeProgram.push_back(inst);

  RunKuhulProgram(rt, computeProgram);

  auto kuhulVecOut = ReadbackFloats(ctx, rt.buffers[bufOut].resource.Get(), bufferSize);
  std::cout << "KUHUL vector add: " << kuhulVecOut[0] << ", " << kuhulVecOut[1] << ", " << kuhulVecOut[2] << ", " << kuhulVecOut[3] << std::endl;

  auto kuhulTriOut = ReadbackFloats(ctx, rt.buffers[outOutId].resource.Get(), phaseCount * sizeof(float));
  std::cout << "KUHUL triangle step: " << kuhulTriOut[0] << ", " << kuhulTriOut[1] << ", " << kuhulTriOut[2] << ", " << kuhulTriOut[3] << std::endl;

  return 0;
}
