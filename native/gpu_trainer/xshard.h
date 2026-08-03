#pragma once

#include <cstdint>
#include <cstring>

// NNC-K XSQ2 shard container. The on-disk layout is shared with the
// HuggingFace safetensors -> xshard exporter.
#pragma pack(push, 1)
struct XShardHeader {
    char magic[4];
    uint32_t version;
    uint32_t layer_id;
    uint32_t tensor_type;
    uint32_t rows;
    uint32_t cols;
    uint32_t tile_size;
    uint32_t tile_count;
    uint32_t dtype;
    uint32_t shard_class;
    uint8_t reserved[24];
};
#pragma pack(pop)

static_assert(sizeof(XShardHeader) == 64, "XSQ2 header must be 64 bytes");

enum XShardTensorType : uint32_t {
    XSHARD_Q = 0,
    XSHARD_K = 1,
    XSHARD_V = 2,
    XSHARD_O = 3
};

enum XShardDType : uint32_t {
    XSHARD_FP32 = 0,
    XSHARD_FP16 = 1,
    XSHARD_INT8 = 2,
    XSHARD_INT4 = 3
};

inline bool xshard_valid_magic(const XShardHeader& header) {
    return header.magic[0] == 'X' && header.magic[1] == 'S' &&
           header.magic[2] == 'Q' && header.magic[3] == '2';
}

inline uint32_t xshard_element_bytes(const XShardHeader& header) {
    switch (header.dtype) {
        case XSHARD_FP32: return 4;
        case XSHARD_FP16: return 2;
        case XSHARD_INT8:
        case XSHARD_INT4: return 1;
        default: return 0;
    }
}

inline uint64_t xshard_tile_bytes(const XShardHeader& header) {
    const bool packedInt4 = header.dtype == XSHARD_INT4 &&
        header.reserved[0] == 'I' && header.reserved[1] == '4' &&
        header.reserved[2] == 'Q' && header.reserved[3] == '1';
    const uint64_t raw = packedInt4
        ? (static_cast<uint64_t>(header.tile_size) + 1) / 2
        : static_cast<uint64_t>(header.tile_size) *
              xshard_element_bytes(header);
    return (raw + 4095u) & ~uint64_t(4095u);
}

inline uint64_t xshard_tile_offset(const XShardHeader& header,
                                   uint32_t tile_index) {
    return sizeof(XShardHeader) +
           static_cast<uint64_t>(tile_index) * xshard_tile_bytes(header);
}

inline bool xshard_has_int8_quantization(const XShardHeader& header) {
    return header.dtype == XSHARD_INT8 && header.reserved[0] == 'I' &&
           header.reserved[1] == '8' && header.reserved[2] == 'Q' &&
           header.reserved[3] == '1';
}

inline float xshard_int8_scale(const XShardHeader& header) {
    float value = 1.0f;
    std::memcpy(&value, header.reserved + 4, sizeof(value));
    return value > 0.0f ? value : 1.0f;
}

inline float xshard_int8_zero_point(const XShardHeader& header) {
    float value = 0.0f;
    std::memcpy(&value, header.reserved + 8, sizeof(value));
    return value;
}

inline float xshard_int4_scale(const XShardHeader& header) {
    float value = 1.0f;
    std::memcpy(&value, header.reserved + 4, sizeof(value));
    return value > 0.0f ? value : 1.0f;
}

inline float xshard_int4_zero_point(const XShardHeader& header) {
    float value = 0.0f;
    std::memcpy(&value, header.reserved + 8, sizeof(value));
    return value;
}
