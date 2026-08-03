#include "safetensors_reader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>

namespace {

namespace fs = std::filesystem;

bool parseUnsigned(const std::string& value, uint64_t& result) {
    if (value.empty() ||
        !std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        })) {
        return false;
    }
    try {
        result = std::stoull(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

size_t findObjectEnd(const std::string& text, size_t open) {
    int depth = 0;
    bool quoted = false;
    bool escaped = false;
    for (size_t i = open; i < text.size(); ++i) {
        const char ch = text[i];
        if (quoted) {
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == '"') quoted = false;
            continue;
        }
        if (ch == '"') {
            quoted = true;
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}' && --depth == 0) {
            return i;
        }
    }
    return std::string::npos;
}

} // namespace

bool SafeTensorsReader::validate(const std::string& path,
                                 SafeTensorManifest& manifest) {
    error_.clear();
    manifest = {};
    manifest.path = path;

    std::error_code ec;
    const uint64_t fileBytes = fs::file_size(path, ec);
    if (ec || fileBytes < sizeof(uint64_t)) {
        error_ = "SafeTensors file is missing or too small: " + path;
        return false;
    }

    std::ifstream input(path, std::ios::binary);
    uint64_t headerBytes = 0;
    if (!input.read(reinterpret_cast<char*>(&headerBytes), sizeof(headerBytes)) ||
        headerBytes > fileBytes - sizeof(uint64_t) ||
        headerBytes > 128ull * 1024ull * 1024ull) {
        error_ = "invalid SafeTensors header length: " + path;
        return false;
    }

    std::string header(static_cast<size_t>(headerBytes), '\0');
    if (!input.read(header.data(), static_cast<std::streamsize>(header.size()))) {
        error_ = "truncated SafeTensors header: " + path;
        return false;
    }
    if (header.empty() || header.front() != '{' || header.back() != '}') {
        error_ = "SafeTensors header is not a JSON object: " + path;
        return false;
    }

    manifest.header_bytes = headerBytes;
    manifest.file_bytes = fileBytes;
    size_t cursor = 0;
    while ((cursor = header.find('"', cursor)) != std::string::npos) {
        const size_t nameEnd = header.find('"', cursor + 1);
        if (nameEnd == std::string::npos) break;
        const std::string name = header.substr(cursor + 1, nameEnd - cursor - 1);
        const size_t colon = header.find(':', nameEnd + 1);
        if (colon == std::string::npos) break;
        const size_t objectStart = header.find('{', colon + 1);
        if (objectStart == std::string::npos) break;
        const size_t objectEnd = findObjectEnd(header, objectStart);
        if (objectEnd == std::string::npos) {
            error_ = "unterminated tensor object in SafeTensors header";
            return false;
        }
        if (name != "__metadata__") {
            if (!parseTensorObject(name,
                                   header.substr(objectStart,
                                                objectEnd - objectStart + 1),
                                   manifest)) {
                return false;
            }
        }
        cursor = objectEnd + 1;
    }

    if (manifest.tensors.empty()) {
        error_ = "SafeTensors header contains no tensor records";
        return false;
    }
    return true;
}

bool SafeTensorsReader::parseTensorObject(const std::string& name,
                                          const std::string& object,
                                          SafeTensorManifest& manifest) {
    static const std::regex dtypePattern(
        R"re("dtype"\s*:\s*"([A-Za-z0-9_]+)")re");
    static const std::regex shapePattern(
        R"re("shape"\s*:\s*\[([^\]]*)\])re");
    static const std::regex offsetsPattern(
        R"re("data_offsets"\s*:\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\])re");

    std::smatch match;
    SafeTensorRecord record;
    record.name = name;
    if (!std::regex_search(object, match, dtypePattern)) {
        error_ = "tensor is missing dtype: " + name;
        return false;
    }
    record.dtype = match[1].str();

    if (!std::regex_search(object, match, shapePattern)) {
        error_ = "tensor is missing shape: " + name;
        return false;
    }
    std::string shape = match[1].str();
    size_t start = 0;
    while (start < shape.size()) {
        const size_t comma = shape.find(',', start);
        const std::string value = shape.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        const size_t first = value.find_first_not_of(" \t\r\n");
        const size_t last = value.find_last_not_of(" \t\r\n");
        uint64_t dimension = 0;
        if (first == std::string::npos ||
            !parseUnsigned(value.substr(first, last - first + 1), dimension)) {
            error_ = "tensor has invalid shape: " + name;
            return false;
        }
        record.shape.push_back(dimension);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }

    if (!std::regex_search(object, match, offsetsPattern) ||
        !parseUnsigned(match[1].str(), record.data_begin) ||
        !parseUnsigned(match[2].str(), record.data_end) ||
        record.data_begin > record.data_end ||
        record.data_end > manifest.file_bytes - sizeof(uint64_t) -
                               manifest.header_bytes) {
        error_ = "tensor has invalid data offsets: " + name;
        return false;
    }
    manifest.tensors.push_back(std::move(record));
    return true;
}
