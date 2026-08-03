#pragma once

#include "../gpu_trainer/xshard.h"

#include <cstdint>
#include <string>
#include <vector>

struct XShardTensorRecord {
    std::string path;
    XShardHeader header{};
    uint64_t file_bytes = 0;
    uint64_t required_bytes = 0;
};

struct XShardModelManifest {
    std::string root;
    uint32_t layer_count = 0;
    uint32_t max_layer_id = 0;
    uint64_t total_bytes = 0;
    std::vector<XShardTensorRecord> tensors;
    std::vector<std::string> errors;
};
