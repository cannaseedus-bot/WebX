#pragma once

#include "model_manifest.h"

#include <cstdint>
#include <string>
#include <vector>

class XShardBlockReader {
public:
    bool open(const std::string& path);
    bool readBlock(uint32_t expert, uint32_t rowStart, uint32_t rowCount,
                   uint32_t colStart, uint32_t colCount,
                   std::vector<float>& values);
    bool readRawBlock(uint32_t expert, uint32_t rowStart, uint32_t rowCount,
                      uint32_t colStart, uint32_t colCount,
                      std::vector<uint8_t>& values);
    bool readPackedInt4Block(uint32_t expert, uint32_t rowStart,
                             uint32_t rowCount, uint32_t colStart,
                             uint32_t colCount, std::vector<uint8_t>& values);

    const XShardHeader& header() const { return header_; }
    const std::string& error() const { return error_; }

private:
    std::string path_;
    XShardHeader header_{};
    std::string error_;
};
