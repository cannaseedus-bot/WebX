/**
 * scx_stream_engine.cpp
 * SCXQ2 streaming inference engine — DX12 implementation.
 *
 * Thread model:
 *   submitFrame()  → IO caller thread       (enqueues ChunkWork)
 *   decodeWorker() → N decode threads       (upload + GPU dispatch)
 *   inferBatch()   → inference caller thread (waits on layer ready, runs attn)
 */

#include "scx_stream_engine.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>

// ── HLSL bytecode stubs ───────────────────────────────────────────────────────
// In production, compile int4_decode + attention shaders offline (fxc/dxc)
// and embed as byte arrays.  Stubs keep the skeleton buildable without them.
extern "C" { extern const uint8_t g_CSDecodeInt4[];       extern size_t g_CSDecodeInt4_sz; }
extern "C" { extern const uint8_t g_CSFusedDecodeMatmul[]; extern size_t g_CSFusedDecodeMatmul_sz; }
extern "C" { extern const uint8_t g_CSAttention[];        extern size_t g_CSAttention_sz; }

// ── Constant buffer structs (mirror HLSL) ─────────────────────────────────────

struct ScxDecodeCB {
    uint32_t numPacked;
    uint32_t dstOffset;
    float    scale;
    float    zero;
};

struct ScxFusedCB {
    uint32_t hiddenDim;
    uint32_t outDim;
    uint32_t fusedPacked;
    uint32_t _pad;
};

// ── init ──────────────────────────────────────────────────────────────────────

bool ScxStreamEngine::init(ID3D12Device* dev, ID3D12CommandQueue* queue,
                           uint32_t numLayers, uint32_t hiddenDim)
{
    dev_       = dev;
    queue_     = queue;
    numLayers_ = numLayers;
    hiddenDim_ = hiddenDim;

    layers_.resize(numLayers);
    writeOffset_.resize(numLayers);
    for (auto& v : writeOffset_) v.store(0);

    if (!allocWeightBuffers())    return false;
    if (!allocUploadRing())       return false;
    if (!createDecodeRootSigAndPso()) return false;
    if (!createAttnRootSigAndPso())   return false;

    // Command infrastructure (single allocator — GPU worker serialises)
    dev_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(&cmdAlloc_));
    dev_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
        cmdAlloc_.Get(), nullptr, IID_PPV_ARGS(&cmdList_));
    cmdList_->Close();

    dev_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    running_.store(true);

    // Spawn N-1 decode threads (keep one core for inference)
    uint32_t nDecode = std::max(1u, std::thread::hardware_concurrency() - 2u);
    for (uint32_t i = 0; i < nDecode; ++i)
        decodeThreads_.emplace_back([this]{ decodeWorker(); });

    emitOp("STREAM_ENGINE_READY",
        "{\"layers\":" + std::to_string(numLayers) +
        ",\"hidden\":" + std::to_string(hiddenDim) +
        ",\"decodeThreads\":" + std::to_string(nDecode) + "}");
    return true;
}

void ScxStreamEngine::shutdown()
{
    running_.store(false);
    workCv_.notify_all();
    for (auto& t : decodeThreads_) if (t.joinable()) t.join();
    decodeThreads_.clear();
    waitGPU();
    if (fenceEvent_) { CloseHandle(fenceEvent_); fenceEvent_ = nullptr; }
}

// ── GPU buffer allocation ─────────────────────────────────────────────────────

bool ScxStreamEngine::allocWeightBuffers()
{
    // Each layer: Q K V O FFN1 FFN2  (6 matrices × hiddenDim² × sizeof(float))
    // We over-allocate and let decode write at tracked offsets.
    const uint64_t bytesPerLayer =
        alignUp(6ull * hiddenDim_ * hiddenDim_ * sizeof(float), SCX_TILE_ALIGN);

    D3D12_HEAP_PROPERTIES heapDefault{};
    heapDefault.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = bytesPerLayer;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    weightBuf_.resize(numLayers_);
    for (uint32_t i = 0; i < numLayers_; ++i) {
        HRESULT hr = dev_->CreateCommittedResource(
            &heapDefault, D3D12_HEAP_FLAG_NONE,
            &rd, D3D12_RESOURCE_STATE_COMMON,
            nullptr, IID_PPV_ARGS(&weightBuf_[i]));
        if (FAILED(hr)) return false;
    }
    return true;
}

bool ScxStreamEngine::allocUploadRing()
{
    D3D12_HEAP_PROPERTIES heapUpload{};
    heapUpload.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = SCX_CHUNK_SIZE * 2;  // headroom for large tiles
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    uploadRing_.resize(SCX_UPLOAD_RING);
    for (auto& slot : uploadRing_) {
        HRESULT hr = dev_->CreateCommittedResource(
            &heapUpload, D3D12_HEAP_FLAG_NONE,
            &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&slot.buf));
        if (FAILED(hr)) return false;
        slot.buf->Map(0, nullptr, &slot.mapped);
        slot.free.store(true);
    }
    return true;
}

// ── PSO creation ──────────────────────────────────────────────────────────────

bool ScxStreamEngine::createDecodeRootSigAndPso()
{
    // Root signature: CBV b0, SRV t0, UAV u0
    D3D12_DESCRIPTOR_RANGE ranges[3]{};
    ranges[0].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    ranges[0].NumDescriptors     = 1;
    ranges[0].BaseShaderRegister = 0;

    ranges[1].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors     = 1;
    ranges[1].BaseShaderRegister = 0;

    ranges[2].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[2].NumDescriptors     = 1;
    ranges[2].BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER params[3]{};
    for (int i = 0; i < 3; ++i) {
        params[i].ParameterType    = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable  = {1, &ranges[i]};
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 3;
    rsd.pParameters   = params;

    ComPtr<ID3DBlob> blob, err;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                &blob, &err);
    dev_->CreateRootSignature(0, blob->GetBufferPointer(),
                              blob->GetBufferSize(), IID_PPV_ARGS(&decodeSig_));

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = decodeSig_.Get();
    psd.CS             = {g_CSDecodeInt4, g_CSDecodeInt4_sz};
    return SUCCEEDED(dev_->CreateComputePipelineState(&psd, IID_PPV_ARGS(&decodePso_)));
}

bool ScxStreamEngine::createAttnRootSigAndPso()
{
    // Minimal attention PSO — same root sig shape, different shader
    D3D12_DESCRIPTOR_RANGE ranges[3]{};
    ranges[0].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;  ranges[0].NumDescriptors=1; ranges[0].BaseShaderRegister=0;
    ranges[1].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  ranges[1].NumDescriptors=4; ranges[1].BaseShaderRegister=0; // Q K V weights + activations
    ranges[2].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;  ranges[2].NumDescriptors=1; ranges[2].BaseShaderRegister=0;

    D3D12_ROOT_PARAMETER params[3]{};
    for (int i=0;i<3;++i){params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;params[i].DescriptorTable={1,&ranges[i]};params[i].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;}

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters=3; rsd.pParameters=params;
    ComPtr<ID3DBlob> blob, err;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
    dev_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&attnSig_));

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = attnSig_.Get();
    psd.CS             = {g_CSAttention, g_CSAttention_sz};
    return SUCCEEDED(dev_->CreateComputePipelineState(&psd, IID_PPV_ARGS(&attnPso_)));
}

// ── Stream input path ─────────────────────────────────────────────────────────

void ScxStreamEngine::submitFrame(const uint8_t* frame, size_t frameBytes)
{
    if (frameBytes < sizeof(ScxChunkHeader)) return;

    const auto* hdr = reinterpret_cast<const ScxChunkHeader*>(frame);
    if (hdr->type != 0x02) return;  // only CHUNK frames carry tile data

    // Payload follows header
    const uint8_t* payload = frame + sizeof(ScxChunkHeader);
    size_t payloadLen = frameBytes - sizeof(ScxChunkHeader);
    if (payloadLen < sizeof(ScxTileMeta)) return;

    const auto* meta = reinterpret_cast<const ScxTileMeta*>(payload);
    const uint8_t* packed = payload + sizeof(ScxTileMeta);
    size_t packedLen = payloadLen - sizeof(ScxTileMeta);

    bool last = (hdr->flags & 0x04) != 0;

    submitChunk(meta->layer_id, TileType(meta->tile_type),
                meta->tile_row, meta->tile_col,
                meta->tile_rows, meta->tile_cols,
                packed, packedLen, last);
}

void ScxStreamEngine::submitChunk(uint32_t layer, TileType t,
                                  uint32_t tileRow, uint32_t tileCol,
                                  uint32_t tileRows, uint32_t tileCols,
                                  const uint8_t* packed, size_t bytes,
                                  bool lastTile)
{
    if (layer >= numLayers_) return;

    ChunkWork work;
    work.layer    = layer;
    work.tileType = t;
    work.tileRow  = tileRow;
    work.tileCol  = tileCol;
    work.tileRows = tileRows;
    work.tileCols = tileCols;
    work.packed.assign(packed, packed + bytes);
    work.last     = lastTile;

    {
        std::lock_guard<std::mutex> lk(workMu_);
        workQ_.push(std::move(work));
    }
    workCv_.notify_one();
}

// ── Decode worker thread ──────────────────────────────────────────────────────

void ScxStreamEngine::decodeWorker()
{
    while (true) {
        ChunkWork work;
        {
            std::unique_lock<std::mutex> lk(workMu_);
            workCv_.wait(lk, [this]{
                return !workQ_.empty() || !running_.load();
            });
            if (!running_.load() && workQ_.empty()) return;
            work = std::move(workQ_.front());
            workQ_.pop();
        }
        uploadAndDecode(work);
    }
}

// ── Upload + GPU dispatch ─────────────────────────────────────────────────────

void ScxStreamEngine::uploadAndDecode(const ChunkWork& work)
{
    // 1. Acquire an upload ring slot
    int slot = acquireUploadSlot();

    // 2. Write packed INT4 data into the upload heap (CPU-side)
    size_t packedBytes = work.packed.size();
    std::memcpy(uploadRing_[slot].mapped, work.packed.data(), packedBytes);

    // 3. Compute destination offset in the layer's weight buffer
    uint64_t numFloats   = uint64_t(work.tileRows) * work.tileCols;
    uint64_t dstByteOff  = writeOffset_[work.layer].fetch_add(
                               numFloats * sizeof(float));
    uint64_t dstElemOff  = dstByteOff / sizeof(float);

    // 4. Record: CopyBufferRegion → Dispatch decode kernel
    cmdAlloc_->Reset();
    cmdList_->Reset(cmdAlloc_.Get(), decodePso_.Get());

    // Barrier: upload → copy source
    D3D12_RESOURCE_BARRIER barUpload{};
    barUpload.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barUpload.Transition.pResource   = uploadRing_[slot].buf.Get();
    barUpload.Transition.StateBefore = D3D12_RESOURCE_STATE_GENERIC_READ;
    barUpload.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cmdList_->ResourceBarrier(1, &barUpload);

    // Barrier: weight buffer → copy dest
    D3D12_RESOURCE_BARRIER barW{};
    barW.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barW.Transition.pResource   = weightBuf_[work.layer].Get();
    barW.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barW.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    cmdList_->ResourceBarrier(1, &barW);

    // Copy: upload → weight buffer (still INT4, pre-decode)
    cmdList_->CopyBufferRegion(
        weightBuf_[work.layer].Get(), dstByteOff,
        uploadRing_[slot].buf.Get(), 0,
        packedBytes);

    // Barrier: weight buffer copy_dest → UAV (decode writes here too)
    barW.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barW.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList_->ResourceBarrier(1, &barW);

    // Set decode PSO + root sig
    cmdList_->SetPipelineState(decodePso_.Get());
    cmdList_->SetComputeRootSignature(decodeSig_.Get());

    // Constant buffer (inline root constant alternative — kept simple here)
    ScxDecodeCB cb{};
    cb.numPacked  = uint32_t(packedBytes / 4);  // uint32 elements in upload buf
    cb.dstOffset  = uint32_t(dstElemOff);
    cb.scale      = 1.f / 8.f;
    cb.zero       = 0.f;
    // (In production: write CB into a small upload buffer and bind as CBV)

    uint32_t groups = (cb.numPacked + SCX_DECODE_GROUPS - 1) / SCX_DECODE_GROUPS;
    cmdList_->Dispatch(groups, 1, 1);

    // Barrier: UAV → SRV for subsequent inference reads
    barW.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barW.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    cmdList_->ResourceBarrier(1, &barW);

    cmdList_->Close();

    // 5. Execute and signal fence
    ID3D12CommandList* lists[] = {cmdList_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    waitGPU();

    // 6. Release upload slot
    releaseUploadSlot(slot);

    // 7. Update tile counter; mark layer ready if all tiles are in
    auto& ls = layers_[work.layer];
    int loaded = ls.tilesLoaded.fetch_add(1) + 1;
    if (work.last) {
        ls.tilesTotal.store(loaded);
    }
    if (ls.tilesTotal.load() >= 0 && loaded >= ls.tilesTotal.load()) {
        ls.ready.store(true);
        emitOp("PLUGIN_REGISTER",
            "{\"layer\":" + std::to_string(work.layer) +
            ",\"tiles\":" + std::to_string(loaded) + "}");
    }

    emitOp("GPU_DECODE",
        "{\"layer\":" + std::to_string(work.layer) +
        ",\"type\":" + std::to_string(uint8_t(work.tileType)) +
        ",\"floats\":" + std::to_string(numFloats) + "}");
}

// ── Inference path ────────────────────────────────────────────────────────────

bool ScxStreamEngine::inferBatch(uint32_t layer,
                                  const float* inputTokens,
                                  float* output,
                                  uint32_t batchSz)
{
    // Block until this layer's weights are fully decoded into VRAM
    while (!layerReady(layer)) {
        if (!running_.load()) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // Dispatch attention kernel for this layer + batch
    cmdAlloc_->Reset();
    cmdList_->Reset(cmdAlloc_.Get(), attnPso_.Get());

    cmdList_->SetPipelineState(attnPso_.Get());
    cmdList_->SetComputeRootSignature(attnSig_.Get());

    // (Bind SRVs for weight buffer slices, UAV for output, CB with dims)
    // Simplified: one group per output row
    uint32_t groups = (batchSz * hiddenDim_ + SCX_DECODE_GROUPS - 1) / SCX_DECODE_GROUPS;
    cmdList_->Dispatch(groups, 1, 1);

    cmdList_->Close();
    ID3D12CommandList* lists[] = {cmdList_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    waitGPU();

    emitOp("GPU_RENDER",
        "{\"layer\":" + std::to_string(layer) +
        ",\"batch\":" + std::to_string(batchSz) + "}");
    return true;
}

bool ScxStreamEngine::generate(const uint32_t* promptTokens, uint32_t promptLen,
                                uint32_t* outTokens, uint32_t genLen)
{
    std::vector<float> hidden(hiddenDim_ * promptLen, 0.f);
    std::vector<float> next(hiddenDim_ * promptLen, 0.f);

    for (uint32_t step = 0; step < genLen; ++step) {
        for (uint32_t layer = 0; layer < numLayers_; ++layer) {
            if (!inferBatch(layer, hidden.data(), next.data(), promptLen))
                return false;
            std::swap(hidden, next);
        }
        // Greedy argmax over lm_head — placeholder
        outTokens[step] = 0;
    }
    return true;
}

// ── Upload ring ───────────────────────────────────────────────────────────────

int ScxStreamEngine::acquireUploadSlot()
{
    while (true) {
        for (int i = 0; i < int(uploadRing_.size()); ++i) {
            bool expected = true;
            if (uploadRing_[i].free.compare_exchange_strong(expected, false))
                return i;
        }
        std::this_thread::yield();
    }
}

void ScxStreamEngine::releaseUploadSlot(int slot)
{
    uploadRing_[slot].free.store(true);
}

// ── GPU sync ──────────────────────────────────────────────────────────────────

void ScxStreamEngine::waitGPU()
{
    ++fenceVal_;
    queue_->Signal(fence_.Get(), fenceVal_);
    if (fence_->GetCompletedValue() < fenceVal_) {
        fence_->SetEventOnCompletion(fenceVal_, fenceEvent_);
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

// ── Opcode callback ───────────────────────────────────────────────────────────

void ScxStreamEngine::emitOp(const std::string& op, const std::string& json) const
{
    if (opcodeCallback_)
        opcodeCallback_(op, json);
}
