#include "scx_manifest_bridge.h"
#include "../gpu_trainer/scx_stream_engine.h"

#include <algorithm>
#include <fstream>
#include <map>

namespace {

bool mapTensorType(uint32_t tensorType, TileType& tileType) {
    switch (tensorType) {
        case 0: tileType = TileType::Q;    return true;
        case 1: tileType = TileType::K;    return true;
        case 2: tileType = TileType::V;    return true;
        case 3: tileType = TileType::O;    return true;
        case 4:
        case 5: tileType = TileType::FFN1; return true;
        case 6: tileType = TileType::FFN2; return true;
        default: return false;
    }
}

} // namespace

bool buildScxManifestPlan(const XShardModelManifest& manifest,
                          ScxManifestPlan& plan) {
    plan = {};
    plan.root = manifest.root;
    if (manifest.tensors.empty()) {
        plan.errors.push_back("xshard manifest contains no tensors");
        return false;
    }

    std::map<uint32_t, ScxManifestLayer> layers;
    std::map<std::pair<uint32_t, uint32_t>, ScxManifestFamily> families;
    for (const auto& tensor : manifest.tensors) {
        auto& layer = layers[tensor.header.layer_id];
        layer.layer_id = tensor.header.layer_id;
        layer.tensor_count++;
        layer.tile_count += tensor.header.tile_count;
        layer.payload_bytes += tensor.required_bytes;
        plan.total_tiles += tensor.header.tile_count;
        plan.total_payload_bytes += tensor.required_bytes;

        auto& family = families[{tensor.header.layer_id,
                                 tensor.header.tensor_type}];
        family.tensor_type = tensor.header.tensor_type;
        family.tensor_count++;
        family.tile_count += tensor.header.tile_count;
        family.payload_bytes += tensor.required_bytes;
        family.dtype_mask |= 1u << tensor.header.dtype;
    }

    for (const auto& entry : families)
        layers[entry.first.first].families.push_back(entry.second);

    for (auto& entry : layers) {
        entry.second.ready = entry.second.tile_count > 0;
        plan.layers.push_back(entry.second);
    }
    plan.layer_count = static_cast<uint32_t>(plan.layers.size());
    return plan.valid();
}

bool streamXShardTiles(const XShardModelManifest& manifest,
                       uint32_t maxLayer,
                       uint32_t maxTiles,
                       const ScxTileSubmitter& submit,
                       uint32_t& streamedTiles,
                       std::string& error,
                       uint32_t tensorType) {
    streamedTiles = 0;
    error.clear();
    if (!submit) {
        error = "xshard tile submitter is not configured";
        return false;
    }

    for (const auto& tensor : manifest.tensors) {
        if (tensor.header.layer_id > maxLayer) continue;
        if (tensorType != UINT32_MAX &&
            tensor.header.tensor_type != tensorType)
            continue;
        if (maxTiles != 0 && streamedTiles >= maxTiles) break;

        const uint64_t tileBytes = xshard_tile_bytes(tensor.header);
        if (tileBytes == 0 || tileBytes > static_cast<uint64_t>(SIZE_MAX)) {
            error = "xshard tile size is invalid: " + tensor.path;
            return false;
        }

        const uint32_t tilesAcross =
            (tensor.header.cols + tensor.header.tile_size - 1) /
            tensor.header.tile_size;
        if (tilesAcross == 0) {
            error = "xshard tile geometry is invalid: " + tensor.path;
            return false;
        }

        std::ifstream input(tensor.path, std::ios::binary);
        if (!input) {
            error = "cannot open xshard payload: " + tensor.path;
            return false;
        }

        std::vector<uint8_t> payload(static_cast<size_t>(tileBytes));
        for (uint32_t tile = 0; tile < tensor.header.tile_count; ++tile) {
            if (maxTiles != 0 && streamedTiles >= maxTiles) break;

            input.seekg(static_cast<std::streamoff>(
                xshard_tile_offset(tensor.header, tile)));
            if (!input.read(reinterpret_cast<char*>(payload.data()),
                            static_cast<std::streamsize>(payload.size()))) {
                error = "cannot read xshard tile: " + tensor.path;
                return false;
            }

            const uint32_t row = tile / tilesAcross;
            const uint32_t col = tile % tilesAcross;
            const uint32_t rows = std::min(
                tensor.header.tile_size,
                tensor.header.rows - row * tensor.header.tile_size);
            const uint32_t cols = std::min(
                tensor.header.tile_size,
                tensor.header.cols - col * tensor.header.tile_size);
            const bool lastTile =
                tile + 1 == tensor.header.tile_count &&
                (&tensor == &manifest.tensors.back() ||
                 manifest.tensors[&tensor - manifest.tensors.data() + 1]
                         .header.layer_id != tensor.header.layer_id);
            const bool boundedLast =
                maxTiles != 0 && streamedTiles + 1 == maxTiles;

            if (!submit(tensor.header.layer_id,
                        static_cast<uint8_t>(tensor.header.tensor_type),
                        row, col, rows, cols, tensor.header.dtype,
                        payload.data(), payload.size(),
                        lastTile || boundedLast)) {
                error = "xshard tile submitter rejected tile: " + tensor.path;
                return false;
            }
            ++streamedTiles;
        }
    }
    return true;
}

bool streamXShardTilesToEngine(const XShardModelManifest& manifest,
                               ScxStreamEngine& engine,
                               uint32_t maxLayer,
                               uint32_t maxTiles,
                               uint32_t& streamedTiles,
                               std::string& error,
                               uint32_t tensorType) {
    return streamXShardTiles(
        manifest, maxLayer, maxTiles,
        [&engine](uint32_t layer, uint8_t tileType, uint32_t tileRow,
                  uint32_t tileCol, uint32_t tileRows, uint32_t tileCols,
                  uint32_t dtype, const uint8_t* payload, size_t payloadBytes,
                  bool lastTile) {
            TileType mappedType;
            if (!mapTensorType(tileType, mappedType))
                return false;
            engine.submitChunk(
                layer, mappedType, tileRow, tileCol,
                tileRows, tileCols, payload, payloadBytes, lastTile, dtype);
            return true;
        },
        streamedTiles, error, tensorType);
}
