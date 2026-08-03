#include "xshard_hot_swap.h"

#include "xshard_model_loader.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace {

using TensorSignature =
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
               uint32_t>;

std::vector<TensorSignature> tensorSignatures(
        const XShardModelManifest& manifest) {
    std::vector<TensorSignature> signatures;
    signatures.reserve(manifest.tensors.size());
    for (const auto& tensor : manifest.tensors) {
        const auto& header = tensor.header;
        signatures.emplace_back(
            header.layer_id, header.tensor_type, header.rows, header.cols,
            header.tile_size, header.tile_count, header.dtype);
    }
    std::sort(signatures.begin(), signatures.end());
    return signatures;
}

void observe(const XShardPhaseObserver& observer, const char* phase) {
    if (observer) observer(phase);
}

} // namespace

bool XShardHotSwapRuntime::activate(
        const std::string& path,
        const XShardHotSwapBounds& bounds,
        const ScxTileSubmitter& stagingSubmitter,
        XShardHotSwapReceipt& receipt,
        std::string& error,
        const XShardPhaseObserver& phaseObserver) {
    receipt = {};
    error.clear();

    observe(phaseObserver, "Pop");
    XShardModelManifest candidate;
    XShardModelLoader loader;
    if (!loader.validate(path, candidate)) {
        error = "Pop validation failed: " + loader.error();
        return false;
    }

    observe(phaseObserver, "Wo");
    ScxManifestPlan candidatePlan;
    if (!buildScxManifestPlan(candidate, candidatePlan)) {
        error = "Wo stream planning failed";
        if (!candidatePlan.errors.empty())
            error += ": " + candidatePlan.errors.front();
        return false;
    }

    observe(phaseObserver, "Yax");
    if (bounds.max_tiles == 0) {
        error = "Yax admission failed: max_tiles must be bounded and non-zero";
        return false;
    }
    if (!stagingSubmitter) {
        error = "Yax admission failed: staging submitter is not configured";
        return false;
    }
    XShardModelManifest activeSnapshot;
    uint64_t observedGeneration = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        activeSnapshot = active_manifest_;
        observedGeneration = generation_;
    }
    if (observedGeneration != 0 &&
        !compatible(activeSnapshot, candidate, error)) {
        error = "Yax admission failed: " + error;
        return false;
    }

    observe(phaseObserver, "Sek");
    uint64_t streamedBytes = 0;
    uint64_t fingerprint = 1469598103934665603ull;
    uint32_t streamedTiles = 0;
    std::string streamError;
    const bool streamed = streamXShardTiles(
        candidate, bounds.max_layer, bounds.max_tiles,
        [&](uint32_t layer, uint8_t tileType, uint32_t tileRow,
            uint32_t tileCol, uint32_t tileRows, uint32_t tileCols,
            uint32_t dtype, const uint8_t* payload, size_t payloadBytes,
            bool lastTile) {
            if (payload == nullptr || payloadBytes == 0 ||
                payloadBytes >
                    std::numeric_limits<uint64_t>::max() - streamedBytes)
                return false;
            for (size_t i = 0; i < payloadBytes; ++i) {
                fingerprint ^= payload[i];
                fingerprint *= 1099511628211ull;
            }
            streamedBytes += static_cast<uint64_t>(payloadBytes);
            return stagingSubmitter(
                layer, tileType, tileRow, tileCol, tileRows, tileCols, dtype,
                payload, payloadBytes, lastTile);
        },
        streamedTiles, streamError, bounds.tensor_type);
    if (!streamed) {
        error = "Sek staging failed: " + streamError;
        return false;
    }
    if (streamedTiles == 0) {
        error = "Sek staging failed: bounds selected no xshard tiles";
        return false;
    }

    observe(phaseObserver, "Chen");
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (generation_ != observedGeneration) {
            error = "Chen commit failed: active model changed during staging";
            return false;
        }
        receipt.previous_root = active_manifest_.root;
        active_manifest_ = std::move(candidate);
        active_plan_ = std::move(candidatePlan);
        ++generation_;
        receipt.active_root = active_manifest_.root;
        receipt.generation = generation_;
    }
    receipt.streamed_tiles = streamedTiles;
    receipt.streamed_bytes = streamedBytes;
    receipt.payload_fingerprint = fingerprint;

    observe(phaseObserver, "Xul");
    return true;
}

bool XShardHotSwapRuntime::compatible(
        const XShardModelManifest& active,
        const XShardModelManifest& candidate,
        std::string& error) const {
    if (active.layer_count != candidate.layer_count) {
        error = "candidate layer count differs from the active model";
        return false;
    }
    if (tensorSignatures(active) != tensorSignatures(candidate)) {
        error = "candidate tensor topology differs from the active model";
        return false;
    }
    return true;
}

bool XShardHotSwapRuntime::hasActiveModel() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return generation_ != 0;
}

uint64_t XShardHotSwapRuntime::generation() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return generation_;
}

XShardModelManifest XShardHotSwapRuntime::activeManifest() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return active_manifest_;
}

ScxManifestPlan XShardHotSwapRuntime::activePlan() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return active_plan_;
}
