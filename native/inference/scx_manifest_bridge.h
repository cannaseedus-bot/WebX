#pragma once

#include "model_manifest.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class ScxStreamEngine;

struct ScxManifestFamily {
    uint32_t tensor_type = 0;
    uint32_t tensor_count = 0;
    uint32_t tile_count = 0;
    uint64_t payload_bytes = 0;
    uint32_t dtype_mask = 0;
};

struct ScxManifestLayer {
    uint32_t layer_id = 0;
    uint32_t tensor_count = 0;
    uint32_t tile_count = 0;
    uint64_t payload_bytes = 0;
    bool ready = false;
    std::vector<ScxManifestFamily> families;
};

struct ScxManifestPlan {
    std::string root;
    uint32_t layer_count = 0;
    uint32_t total_tiles = 0;
    uint64_t total_payload_bytes = 0;
    std::vector<ScxManifestLayer> layers;
    std::vector<std::string> errors;

    bool valid() const { return errors.empty() && !layers.empty(); }
};

bool buildScxManifestPlan(const XShardModelManifest& manifest,
                          ScxManifestPlan& plan);

using ScxTileSubmitter = std::function<bool(
    uint32_t layer, uint8_t tileType, uint32_t tileRow, uint32_t tileCol,
    uint32_t tileRows, uint32_t tileCols, uint32_t dtype,
    const uint8_t* payload, size_t payloadBytes, bool lastTile)>;

bool streamXShardTiles(const XShardModelManifest& manifest,
                       uint32_t maxLayer,
                       uint32_t maxTiles,
                       const ScxTileSubmitter& submit,
                       uint32_t& streamedTiles,
                       std::string& error,
                       uint32_t tensorType = UINT32_MAX);

bool streamXShardTilesToEngine(const XShardModelManifest& manifest,
                               ScxStreamEngine& engine,
                               uint32_t maxLayer,
                               uint32_t maxTiles,
                               uint32_t& streamedTiles,
                               std::string& error,
                               uint32_t tensorType = UINT32_MAX);
