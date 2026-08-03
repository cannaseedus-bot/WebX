#include "xshard_int8_quantizer.h"

#include "xshard_block_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

bool XShardInt8Quantizer::convert(const std::string& inputPath,
                                  const std::string& outputPath) {
    error_.clear();
    XShardBlockReader reader;
    if (!reader.open(inputPath)) {
        error_ = reader.error();
        return false;
    }
    const XShardHeader& inputHeader = reader.header();
    if (inputHeader.dtype != XSHARD_FP32 &&
        inputHeader.dtype != XSHARD_FP16) {
        error_ = "INT8 quantizer requires FP32 or FP16 input";
        return false;
    }
    if (inputHeader.tile_size != inputHeader.rows * inputHeader.cols) {
        error_ = "INT8 quantizer requires full-matrix records";
        return false;
    }

    float maxAbs = 0.0f;
    std::vector<float> values;
    for (uint32_t expert = 0; expert < inputHeader.tile_count; ++expert) {
        if (!reader.readBlock(expert, 0, inputHeader.rows, 0,
                              inputHeader.cols, values)) {
            error_ = reader.error();
            return false;
        }
        for (float value : values)
            if (std::isfinite(value))
                maxAbs = std::max(maxAbs, std::fabs(value));
    }
    const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;

    XShardHeader outputHeader = inputHeader;
    outputHeader.dtype = XSHARD_INT8;
    std::memset(outputHeader.reserved, 0, sizeof(outputHeader.reserved));
    outputHeader.reserved[0] = 'I';
    outputHeader.reserved[1] = '8';
    outputHeader.reserved[2] = 'Q';
    outputHeader.reserved[3] = '1';
    const float zeroPoint = 0.0f;
    std::memcpy(outputHeader.reserved + 4, &scale, sizeof(scale));
    std::memcpy(outputHeader.reserved + 8, &zeroPoint, sizeof(zeroPoint));

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output ||
        !output.write(reinterpret_cast<const char*>(&outputHeader),
                      sizeof(outputHeader))) {
        error_ = "cannot create INT8 xshard";
        return false;
    }
    const uint64_t tileBytes = xshard_tile_bytes(outputHeader);
    std::vector<uint8_t> packed(static_cast<size_t>(tileBytes), 0);
    for (uint32_t expert = 0; expert < inputHeader.tile_count; ++expert) {
        if (!reader.readBlock(expert, 0, inputHeader.rows, 0,
                              inputHeader.cols, values)) {
            error_ = reader.error();
            return false;
        }
        std::fill(packed.begin(), packed.end(), uint8_t{0});
        for (size_t i = 0; i < values.size(); ++i) {
            const float value = std::isfinite(values[i]) ? values[i] : 0.0f;
            const int quantized = static_cast<int>(
                std::lround(std::clamp(value / scale, -127.0f, 127.0f)));
            packed[i] = static_cast<uint8_t>(
                static_cast<int8_t>(quantized));
        }
        if (!output.write(reinterpret_cast<const char*>(packed.data()),
                          static_cast<std::streamsize>(packed.size()))) {
            error_ = "cannot write INT8 xshard payload";
            return false;
        }
    }
    return true;
}

bool XShardInt8Quantizer::convertInt4(const std::string& inputPath,
                                      const std::string& outputPath) {
    error_.clear();
    XShardBlockReader reader;
    if (!reader.open(inputPath)) {
        error_ = reader.error();
        return false;
    }
    const XShardHeader& inputHeader = reader.header();
    if (inputHeader.dtype != XSHARD_FP32 &&
        inputHeader.dtype != XSHARD_FP16) {
        error_ = "INT4 quantizer requires FP32 or FP16 input";
        return false;
    }
    if (inputHeader.tile_size != inputHeader.rows * inputHeader.cols) {
        error_ = "INT4 quantizer requires full-matrix records";
        return false;
    }
    float maxAbs = 0.0f;
    std::vector<float> values;
    for (uint32_t expert = 0; expert < inputHeader.tile_count; ++expert) {
        if (!reader.readBlock(expert, 0, inputHeader.rows, 0,
                              inputHeader.cols, values)) {
            error_ = reader.error();
            return false;
        }
        for (float value : values)
            if (std::isfinite(value))
                maxAbs = std::max(maxAbs, std::fabs(value));
    }
    const float scale = maxAbs > 0.0f ? maxAbs / 7.0f : 1.0f;
    XShardHeader outputHeader = inputHeader;
    outputHeader.dtype = XSHARD_INT4;
    std::memset(outputHeader.reserved, 0, sizeof(outputHeader.reserved));
    outputHeader.reserved[0] = 'I';
    outputHeader.reserved[1] = '4';
    outputHeader.reserved[2] = 'Q';
    outputHeader.reserved[3] = '1';
    const float zeroPoint = 0.0f;
    std::memcpy(outputHeader.reserved + 4, &scale, sizeof(scale));
    std::memcpy(outputHeader.reserved + 8, &zeroPoint, sizeof(zeroPoint));

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output ||
        !output.write(reinterpret_cast<const char*>(&outputHeader),
                      sizeof(outputHeader))) {
        error_ = "cannot create INT4 xshard";
        return false;
    }
    const uint64_t tileBytes = xshard_tile_bytes(outputHeader);
    std::vector<uint8_t> packed(static_cast<size_t>(tileBytes), 0);
    for (uint32_t expert = 0; expert < inputHeader.tile_count; ++expert) {
        if (!reader.readBlock(expert, 0, inputHeader.rows, 0,
                              inputHeader.cols, values)) {
            error_ = reader.error();
            return false;
        }
        std::fill(packed.begin(), packed.end(), uint8_t{0});
        for (size_t i = 0; i < values.size(); ++i) {
            const float value = std::isfinite(values[i]) ? values[i] : 0.0f;
            const int quantized = static_cast<int>(
                std::lround(std::clamp(value / scale, -8.0f, 7.0f)));
            const uint8_t nibble = static_cast<uint8_t>(quantized) & 0xfu;
            packed[i >> 1] |=
                static_cast<uint8_t>(nibble << ((i & 1) * 4));
        }
        if (!output.write(reinterpret_cast<const char*>(packed.data()),
                          static_cast<std::streamsize>(packed.size()))) {
            error_ = "cannot write INT4 xshard payload";
            return false;
        }
    }
    return true;
}
