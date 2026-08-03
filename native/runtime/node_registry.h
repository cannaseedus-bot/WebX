#pragma once

#include "../webx_compute.h"
#include "phase_runtime.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct NodeDescriptor {
    std::string name;
    NodeKind kind = NodeKind::Compute;
    std::string capability;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::filesystem::path contract;
    std::string provider;
    bool provider_available = false;
};

class NodeRegistry {
public:
    bool loadManifest(const std::filesystem::path& manifestPath);
    void resolveProviders(const WebX::ProviderManager& providers);
    const std::vector<NodeDescriptor>& nodes() const { return nodes_; }
    const std::string& error() const { return error_; }

private:
    bool loadNode(const std::filesystem::path& base,
                  const std::filesystem::path& contractPath);
    std::vector<NodeDescriptor> nodes_;
    std::string error_;
};

} // namespace Kuhul::Runtime
