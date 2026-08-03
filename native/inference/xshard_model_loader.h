#pragma once

#include "model_manifest.h"

#include <string>

class XShardModelLoader {
public:
    bool validate(const std::string& path, XShardModelManifest& manifest);
    const std::string& error() const { return error_; }

private:
    bool validateFile(const std::string& path, XShardModelManifest& manifest);
    bool validateHeader(const XShardHeader& header,
                        uint64_t fileBytes,
                        uint64_t& requiredBytes,
                        std::string& error) const;

    std::string error_;
};
