/**
 * scx_stream_engine.cpp
 * SCX streaming inference engine — DX12 implementation.
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
#include <fstream>

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

static float scxHalfToFloat(uint16_t value)
{
    const uint32_t sign = (value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t fraction = value & 0x03ffu;
    uint32_t bits = sign;

    if (exponent == 0) {
        if (fraction != 0) {
            uint32_t normalized = fraction;
            int32_t exponentValue = -14;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                --exponentValue;
            }
            bits |= static_cast<uint32_t>(exponentValue + 127) << 23;
            bits |= (normalized & 0x03ffu) << 13;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (fraction << 13);
    } else {
        bits |= (exponent + 112u) << 23;
        bits |= fraction << 13;
    }

    float result = 0.f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// ── init ──────────────────────────────────────────────────────────────────────

bool ScxStreamEngine::init(ID3D12Device* dev, ID3D12CommandQueue* queue,
                           uint32_t numLayers, uint32_t hiddenDim)
{
    initError_.clear();
    dev_       = dev;
    queue_     = queue;
    numLayers_ = numLayers;
    hiddenDim_ = hiddenDim;

    layers_.resize(numLayers);
    writeOffset_.resize(numLayers);
    for (auto& v : writeOffset_)
        v = std::make_unique<std::atomic<uint64_t>>(0);

    if (!allocWeightBuffers()) {
        initError_ = "allocWeightBuffers";
        return false;
    }
    if (!allocUploadRing()) {
        initError_ = "allocUploadRing";
        return false;
    }
    if (!createDecodeRootSigAndPso()) {
        initError_ = "createDecodeRootSigAndPso";
        return false;
    }
    if (!createAttnRootSigAndPso()) {
        initError_ = "createAttnRootSigAndPso";
        return false;
    }

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
    uint32_t nDecode = 1;
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
    weightStates_.assign(numLayers_, D3D12_RESOURCE_STATE_COMMON);
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
    const uint64_t modelTileBytes =
        uint64_t(hiddenDim_) * hiddenDim_ * sizeof(float);
    rd.Width            = std::max<uint64_t>(
        SCX_CHUNK_SIZE * 2, modelTileBytes);
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    uploadRing_.resize(SCX_UPLOAD_RING);
    uploadStates_.assign(SCX_UPLOAD_RING,
                         D3D12_RESOURCE_STATE_GENERIC_READ);
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

bool ScxStreamEngine::loadShaderBytecode(const char* name,
                                         std::vector<uint8_t>& bytecode)
{
    const std::string relative = std::string("native/shaders/bin/cso/") + name;
    std::ifstream file(relative, std::ios::binary | std::ios::ate);
    if (!file) return false;

    const std::streamsize size = file.tellg();
    if (size <= 0) return false;
    bytecode.resize(static_cast<size_t>(size));
    file.seekg(0);
    return file.read(reinterpret_cast<char*>(bytecode.data()), size).good();
}

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

    if (!loadShaderBytecode("scxq2_int4_decode.cso", decodeShader_))
        return false;

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = decodeSig_.Get();
    psd.CS             = {decodeShader_.data(), decodeShader_.size()};
    return SUCCEEDED(dev_->CreateComputePipelineState(&psd, IID_PPV_ARGS(&decodePso_)));
}

bool ScxStreamEngine::createAttnRootSigAndPso()
{
    // Minimal attention PSO — same root sig shape, different shader
    D3D12_DESCRIPTOR_RANGE ranges[3]{};
    ranges[0].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;  ranges[0].NumDescriptors=1; ranges[0].BaseShaderRegister=1;
    ranges[1].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  ranges[1].NumDescriptors=2; ranges[1].BaseShaderRegister=3;
    ranges[2].RangeType          = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;  ranges[2].NumDescriptors=1; ranges[2].BaseShaderRegister=2;

    D3D12_ROOT_PARAMETER params[3]{};
    for (int i = 0; i < 3; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable = {1, &ranges[i]};
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters=3; rsd.pParameters=params;
    ComPtr<ID3DBlob> blob, err;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
    dev_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&attnSig_));

    if (!loadShaderBytecode("scxq2_infer_layer.cso", attentionShader_))
        return false;

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd{};
    psd.pRootSignature = attnSig_.Get();
    psd.CS             = {attentionShader_.data(), attentionShader_.size()};
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
                                  bool lastTile, uint32_t dtype)
{
    if (layer >= numLayers_) return;

    ChunkWork work;
    work.layer    = layer;
    work.tileType = t;
    work.dtype    = dtype;
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
    std::lock_guard<std::mutex> gpuLock(gpuMu_);

    // 1. Acquire an upload ring slot
    int slot = acquireUploadSlot();

    // 2. Normalize non-INT4 tiles to the float storage format.
    uint64_t numFloats   = uint64_t(work.tileRows) * work.tileCols;
    const bool decodeInt4 = work.dtype == 3;
    std::vector<uint8_t> normalized;
    const uint8_t* uploadData = work.packed.data();
    size_t uploadBytes = work.packed.size();
    if (!decodeInt4) {
        normalized.resize(static_cast<size_t>(numFloats) * sizeof(float));
        float* output = reinterpret_cast<float*>(normalized.data());
        if (work.dtype == 1) {
            const auto* input =
                reinterpret_cast<const uint16_t*>(work.packed.data());
            for (uint64_t i = 0; i < numFloats; ++i)
                output[i] = scxHalfToFloat(input[i]);
        } else if (work.dtype == 0) {
            const size_t sourceBytes =
                static_cast<size_t>(numFloats) * sizeof(float);
            if (work.packed.size() < sourceBytes) {
                releaseUploadSlot(slot);
                return;
            }
            std::memcpy(output, work.packed.data(), sourceBytes);
        } else if (work.dtype == 2) {
            const auto* input =
                reinterpret_cast<const int8_t*>(work.packed.data());
            for (uint64_t i = 0; i < numFloats; ++i)
                output[i] = static_cast<float>(input[i]);
        } else {
            releaseUploadSlot(slot);
            return;
        }
        uploadData = normalized.data();
        uploadBytes = normalized.size();
    }
    // 3. Compute destination offset in the layer's weight buffer
    const uint64_t outputBytes = decodeInt4
        ? numFloats * sizeof(float)
        : uploadBytes;
    uint64_t dstByteOff  = writeOffset_[work.layer]->fetch_add(outputBytes);
    uint64_t dstElemOff  = dstByteOff / sizeof(float);
    std::memcpy(uploadRing_[slot].mapped, uploadData, uploadBytes);

    // 4. Record: CopyBufferRegion → Dispatch decode kernel
    cmdAlloc_->Reset();
    cmdList_->Reset(cmdAlloc_.Get(), decodePso_.Get());

    // Barrier: weight buffer → copy dest
    D3D12_RESOURCE_BARRIER barW{};
    barW.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barW.Transition.pResource   = weightBuf_[work.layer].Get();
    barW.Transition.StateBefore = weightStates_[work.layer];
    barW.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    if (barW.Transition.StateBefore != barW.Transition.StateAfter)
        cmdList_->ResourceBarrier(1, &barW);
    weightStates_[work.layer] = D3D12_RESOURCE_STATE_COPY_DEST;

    // Copy: upload → weight buffer (still INT4, pre-decode)
    cmdList_->CopyBufferRegion(
        weightBuf_[work.layer].Get(), dstByteOff,
        uploadRing_[slot].buf.Get(), 0,
    uploadBytes);

    if (decodeInt4) {
        barW.Transition.StateBefore = weightStates_[work.layer];
        barW.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        cmdList_->ResourceBarrier(1, &barW);
        weightStates_[work.layer] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        cmdList_->SetPipelineState(decodePso_.Get());
        cmdList_->SetComputeRootSignature(decodeSig_.Get());

        ScxDecodeCB cb{};
        cb.numPacked  = uint32_t(uploadBytes / 4);
        cb.dstOffset  = uint32_t(dstElemOff);
        cb.scale      = 1.f / 8.f;
        cb.zero       = 0.f;
        uint32_t groups = (cb.numPacked + SCX_DECODE_GROUPS - 1) / SCX_DECODE_GROUPS;
        cmdList_->Dispatch(groups, 1, 1);

        barW.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    } else {
        barW.Transition.StateBefore = weightStates_[work.layer];
    }
    barW.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    cmdList_->ResourceBarrier(1, &barW);
    weightStates_[work.layer] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

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
