#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct SafeTensorRecord {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

struct SafeTensorManifest {
    std::string path;
    uint64_t header_bytes = 0;
    uint64_t file_bytes = 0;
    std::vector<SafeTensorRecord> tensors;
    std::vector<std::string> errors;
};

class SafeTensorsReader {
public:
    bool validate(const std::string& path, SafeTensorManifest& manifest);
    const std::string& error() const { return error_; }

private:
    bool parseTensorObject(const std::string& name,
                           const std::string& object,
                           SafeTensorManifest& manifest);
    std::string error_;
};
