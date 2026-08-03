#pragma once

#include "scx_manifest_bridge.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

struct XShardHotSwapBounds {
    uint32_t max_layer = 0;
    uint32_t max_tiles = 1;
    uint32_t tensor_type = UINT32_MAX;
};

struct XShardHotSwapReceipt {
    std::string previous_root;
    std::string active_root;
    uint64_t generation = 0;
    uint32_t streamed_tiles = 0;
    uint64_t streamed_bytes = 0;
    uint64_t payload_fingerprint = 0;
};

using XShardPhaseObserver = std::function<void(const char* phase)>;

class XShardHotSwapRuntime {
public:
    bool activate(const std::string& path,
                  const XShardHotSwapBounds& bounds,
                  const ScxTileSubmitter& staging_submitter,
                  XShardHotSwapReceipt& receipt,
                  std::string& error,
                  const XShardPhaseObserver& phase_observer = {});

    bool hasActiveModel() const;
    uint64_t generation() const;
    XShardModelManifest activeManifest() const;
    ScxManifestPlan activePlan() const;

private:
    bool compatible(const XShardModelManifest& active,
                    const XShardModelManifest& candidate,
                    std::string& error) const;

    mutable std::mutex state_mutex_;
    XShardModelManifest active_manifest_;
    ScxManifestPlan active_plan_;
    uint64_t generation_ = 0;
};
