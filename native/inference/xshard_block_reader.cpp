#include "xshard_block_reader.h"

#include <cstring>
#include <fstream>
#include <limits>

namespace {

float halfToFloat(uint16_t value) {
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
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool rangeFits(uint32_t start, uint32_t count, uint32_t limit) {
    return start <= limit && count <= limit - start;
}

} // namespace

bool XShardBlockReader::open(const std::string& path) {
    error_.clear();
    path_ = path;
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(&header_), sizeof(header_))) {
        error_ = "cannot read xshard header";
        return false;
    }
    if (!xshard_valid_magic(header_) || header_.version != 1) {
        error_ = "invalid or unsupported xshard header";
        return false;
    }
    if (header_.tile_size != header_.rows * header_.cols) {
        error_ = "block reader requires full-matrix expert records";
        return false;
    }
    if (header_.dtype != XSHARD_FP32 && header_.dtype != XSHARD_FP16 &&
        header_.dtype != XSHARD_INT8 && header_.dtype != XSHARD_INT4) {
        error_ = "block reader supports FP32, FP16, INT8, and packed INT4 records";
        return false;
    }
    return true;
}

bool XShardBlockReader::readBlock(uint32_t expert, uint32_t rowStart,
                                  uint32_t rowCount, uint32_t colStart,
                                  uint32_t colCount,
                                  std::vector<float>& values) {
    values.clear();
    if (path_.empty()) {
        error_ = "xshard is not open";
        return false;
    }
    if (expert >= header_.tile_count ||
        !rangeFits(rowStart, rowCount, header_.rows) ||
        !rangeFits(colStart, colCount, header_.cols)) {
        error_ = "requested expert block is out of bounds";
        return false;
    }
    const uint64_t elementCount =
        static_cast<uint64_t>(rowCount) * colCount;
    if (elementCount > std::numeric_limits<size_t>::max()) {
        error_ = "requested expert block is too large";
        return false;
    }
    values.resize(static_cast<size_t>(elementCount));
    if (header_.dtype == XSHARD_INT8) {
        error_ = "use readRawBlock for INT8 expert records";
        values.clear();
        return false;
    }

    std::ifstream input(path_, std::ios::binary);
    if (!input) {
        error_ = "cannot open xshard payload";
        values.clear();
        return false;
    }
    const uint64_t base = xshard_tile_offset(header_, expert);
    const uint32_t elementBytes = xshard_element_bytes(header_);
    std::vector<uint8_t> rowBytes(
        static_cast<size_t>(colCount) * elementBytes);
    for (uint32_t row = 0; row < rowCount; ++row) {
        const uint64_t offset =
            base + (static_cast<uint64_t>(rowStart) * header_.cols +
                    colStart) * elementBytes;
        input.seekg(static_cast<std::streamoff>(offset));
        if (!input.read(reinterpret_cast<char*>(rowBytes.data()),
                        static_cast<std::streamsize>(rowBytes.size()))) {
            error_ = "cannot read expert block";
            values.clear();
            return false;
        }
        for (uint32_t col = 0; col < colCount; ++col) {
            if (header_.dtype == XSHARD_FP32) {
                std::memcpy(&values[static_cast<size_t>(row) * colCount + col],
                            rowBytes.data() + col * sizeof(float),
                            sizeof(float));
            } else {
                uint16_t half = 0;
                std::memcpy(&half, rowBytes.data() + col * sizeof(uint16_t),
                            sizeof(half));
                values[static_cast<size_t>(row) * colCount + col] =
                    halfToFloat(half);
            }
        }
    }
    return true;
}

bool XShardBlockReader::readRawBlock(uint32_t expert, uint32_t rowStart,
                                     uint32_t rowCount, uint32_t colStart,
                                     uint32_t colCount,
                                     std::vector<uint8_t>& values) {
    values.clear();
    if (path_.empty()) {
        error_ = "xshard is not open";
        return false;
    }
    if (header_.dtype != XSHARD_INT8 ||
        expert >= header_.tile_count ||
        !rangeFits(rowStart, rowCount, header_.rows) ||
        !rangeFits(colStart, colCount, header_.cols)) {
        error_ = "requested raw INT8 block is invalid";
        return false;
    }
    const uint64_t elementCount =
        static_cast<uint64_t>(rowCount) * colCount;
    if (elementCount > std::numeric_limits<size_t>::max()) {
        error_ = "requested raw INT8 block is too large";
        return false;
    }
    values.resize(static_cast<size_t>(elementCount));
    std::ifstream input(path_, std::ios::binary);
    if (!input) {
        error_ = "cannot open xshard payload";
        values.clear();
        return false;
    }
    const uint64_t base = xshard_tile_offset(header_, expert);
    for (uint32_t row = 0; row < rowCount; ++row) {
        const uint64_t offset =
            base + (static_cast<uint64_t>(rowStart) * header_.cols +
                    colStart) + row * header_.cols;
        input.seekg(static_cast<std::streamoff>(offset));
        if (!input.read(reinterpret_cast<char*>(
                            values.data() + static_cast<size_t>(row) * colCount),
                        static_cast<std::streamsize>(colCount))) {
            error_ = "cannot read raw INT8 expert block";
            values.clear();
            return false;
        }
    }
    return true;
}

bool XShardBlockReader::readPackedInt4Block(
    uint32_t expert, uint32_t rowStart, uint32_t rowCount,
    uint32_t colStart, uint32_t colCount, std::vector<uint8_t>& values) {
    values.clear();
    if (path_.empty()) {
        error_ = "xshard is not open";
        return false;
    }
    if (header_.dtype != XSHARD_INT4 ||
        header_.reserved[0] != 'I' || header_.reserved[1] != '4' ||
        header_.reserved[2] != 'Q' || header_.reserved[3] != '1' ||
        expert >= header_.tile_count ||
        !rangeFits(rowStart, rowCount, header_.rows) ||
        !rangeFits(colStart, colCount, header_.cols)) {
        error_ = "requested packed INT4 block is invalid";
        return false;
    }
    const uint64_t elementCount =
        static_cast<uint64_t>(rowCount) * colCount;
    if (elementCount > (std::numeric_limits<size_t>::max() - 1) * 2) {
        error_ = "requested packed INT4 block is too large";
        return false;
    }
    values.assign(static_cast<size_t>((elementCount + 1) / 2), 0);
    std::ifstream input(path_, std::ios::binary);
    if (!input) {
        error_ = "cannot open xshard payload";
        values.clear();
        return false;
    }
    const uint64_t base = xshard_tile_offset(header_, expert);
    for (uint32_t row = 0; row < rowCount; ++row) {
        for (uint32_t col = 0; col < colCount; ++col) {
            const uint64_t element =
                (static_cast<uint64_t>(rowStart + row) * header_.cols) +
                colStart + col;
            input.seekg(static_cast<std::streamoff>(base + (element >> 1)));
            uint8_t encoded = 0;
            if (!input.read(reinterpret_cast<char*>(&encoded), 1)) {
                error_ = "cannot read packed INT4 expert block";
                values.clear();
                return false;
            }
            const uint8_t nibble = (encoded >> ((element & 1) * 4)) & 0xf;
            const size_t local = static_cast<size_t>(row) * colCount + col;
            values[local >> 1] |=
                static_cast<uint8_t>(nibble << ((local & 1) * 4));
        }
    }
    return true;
}
