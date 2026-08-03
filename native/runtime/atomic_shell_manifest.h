#pragma once

#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct AtomicShellManifest {
    std::string id;
    std::string block;
    std::string version;
    std::string schema;
    std::string backend = "terminal";
    std::string feedParser;
    std::string feedParserArtifact;
    std::string assetFormat;
    std::string assetUri;
    bool executionGated = false;
    std::vector<std::string> blocks;
};

class AtomicShellManifestLoader {
public:
    bool load(const std::string& path, AtomicShellManifest& manifest);
    const std::string& error() const { return error_; }

private:
    std::string error_;
};

} // namespace Kuhul::Runtime
