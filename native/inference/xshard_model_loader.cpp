#include "xshard_model_loader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>

namespace {

namespace fs = std::filesystem;

bool hasShardExtension(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return extension == ".xshard" || extension == ".xsq2";
}

bool multiplyFits(uint64_t left, uint64_t right, uint64_t& result) {
    if (right != 0 &&
        left > std::numeric_limits<uint64_t>::max() / right) {
        return false;
    }
    result = left * right;
    return true;
}

} // namespace

bool XShardModelLoader::validate(const std::string& path,
                                 XShardModelManifest& manifest) {
    error_.clear();
    manifest = {};
    manifest.root = path;

    std::error_code ec;
    const fs::path input(path);
    if (fs::is_regular_file(input, ec)) {
        if (!validateFile(input.string(), manifest)) return false;
    } else if (fs::is_directory(input, ec)) {
        std::vector<fs::path> files;
        for (fs::recursive_directory_iterator it(input, ec), end; it != end;
             it.increment(ec)) {
            if (ec) break;
            if (it->is_regular_file(ec) && hasShardExtension(it->path()))
                files.push_back(it->path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            error_ = "no .xshard or .xsq2 files found: " + path;
            return false;
        }
        for (const auto& file : files) {
            if (!validateFile(file.string(), manifest)) return false;
        }
    } else {
        error_ = "xshard path does not exist or is not readable: " + path;
        return false;
    }

    std::set<uint32_t> layers;
    for (const auto& tensor : manifest.tensors)
        layers.insert(tensor.header.layer_id);
    manifest.layer_count = static_cast<uint32_t>(layers.size());
    return !manifest.tensors.empty();
}

bool XShardModelLoader::validateFile(const std::string& path,
                                     XShardModelManifest& manifest) {
    std::error_code ec;
    const uint64_t fileBytes = fs::file_size(path, ec);
    if (ec) {
        error_ = "cannot stat xshard: " + path;
        return false;
    }
    if (fileBytes < sizeof(XShardHeader)) {
        error_ = "xshard is smaller than its header: " + path;
        return false;
    }

    XShardHeader header{};
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        error_ = "cannot read xshard header: " + path;
        return false;
    }

    uint64_t requiredBytes = 0;
    std::string validationError;
    if (!validateHeader(header, fileBytes, requiredBytes, validationError)) {
        error_ = path + ": " + validationError;
        return false;
    }

    manifest.tensors.push_back(
        {path, header, fileBytes, requiredBytes});
    manifest.total_bytes += fileBytes;
    manifest.max_layer_id = std::max(manifest.max_layer_id, header.layer_id);
    return true;
}

bool XShardModelLoader::validateHeader(const XShardHeader& header,
                                       uint64_t fileBytes,
                                       uint64_t& requiredBytes,
                                       std::string& error) const {
    if (!xshard_valid_magic(header)) {
        error = "invalid XSQ2 magic";
        return false;
    }
    if (header.version != 1) {
        error = "unsupported XSQ2 version";
        return false;
    }
    if (header.rows == 0 || header.cols == 0 ||
        header.tile_size == 0 || header.tile_count == 0) {
        error = "shape, tile size, and tile count must be non-zero";
        return false;
    }
    if (header.dtype > XSHARD_INT4 || xshard_element_bytes(header) == 0) {
        error = "unsupported xshard dtype";
        return false;
    }

    uint64_t tileBytes = xshard_tile_bytes(header);
    uint64_t payloadBytes = 0;
    if (!multiplyFits(tileBytes, header.tile_count, payloadBytes) ||
        payloadBytes > std::numeric_limits<uint64_t>::max() -
                           sizeof(XShardHeader)) {
        error = "xshard payload size overflows";
        return false;
    }
    requiredBytes = sizeof(XShardHeader) + payloadBytes;
    if (requiredBytes > fileBytes) {
        error = "xshard payload is truncated";
        return false;
    }
    return true;
}
