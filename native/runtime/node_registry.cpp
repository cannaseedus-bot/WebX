#include "node_registry.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>

namespace Kuhul::Runtime {
namespace {

std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) return {};
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::string valueAfter(const std::string& source, const std::string& key) {
    const size_t start = source.find(key);
    if (start == std::string::npos) return {};
    const size_t valueStart = source.find_first_not_of(" \t", start + key.size());
    if (valueStart == std::string::npos) return {};
    const size_t end = source.find_first_of(" \t\r\n}", valueStart);
    return source.substr(valueStart, end == std::string::npos
        ? std::string::npos
        : end - valueStart);
}

std::vector<std::string> bracketValues(const std::string& source,
                                       const std::string& key) {
    const size_t start = source.find(key);
    if (start == std::string::npos) return {};
    const size_t open = source.find('[', start + key.size());
    const size_t close = source.find(']', open == std::string::npos ? open : open + 1);
    if (open == std::string::npos || close == std::string::npos) return {};
    std::vector<std::string> values;
    std::istringstream input(source.substr(open + 1, close - open - 1));
    std::string value;
    while (input >> value) values.push_back(value);
    return values;
}

bool parseKind(const std::string& value, NodeKind& kind) {
    const NodeKind kinds[] = {
        NodeKind::Parse, NodeKind::AST, NodeKind::Compute, NodeKind::Tensor,
        NodeKind::Graph, NodeKind::Shader, NodeKind::Compile, NodeKind::Memory,
        NodeKind::Storage, NodeKind::Network, NodeKind::Verification, NodeKind::Tool
    };
    const char* names[] = {
        "Parse", "AST", "Compute", "Tensor", "Graph", "Shader",
        "Compile", "Memory", "Storage", "Network", "Verification", "Tool"
    };
    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
        if (value == names[i]) {
            kind = kinds[i];
            return true;
        }
    }
    return false;
}

WebX::ProviderCapability capabilityFor(const std::string& capability) {
    if (capability == "TensorCompute" || capability == "Compute")
        return WebX::ProviderCapability::TensorCompute;
    if (capability == "Shader" || capability == "ShaderCompiler")
        return WebX::ProviderCapability::ShaderCompiler;
    if (capability == "Compiler" || capability == "Parser")
        return WebX::ProviderCapability::NativeCompiler;
    if (capability == "Storage" || capability == "Memory")
        return WebX::ProviderCapability::Cache;
    return WebX::ProviderCapability::Debug;
}

} // namespace

bool NodeRegistry::loadManifest(const std::filesystem::path& manifestPath) {
    nodes_.clear();
    error_.clear();
    std::filesystem::path resolved = manifestPath;
    if (resolved.is_relative() && !std::filesystem::exists(resolved)) {
        auto directory = std::filesystem::current_path();
        while (true) {
            const auto candidate = directory / resolved;
            if (std::filesystem::exists(candidate)) {
                resolved = candidate;
                break;
            }
            const auto parent = directory.parent_path();
            if (parent == directory) break;
            directory = parent;
        }
    }
    const std::string source = readText(resolved);
    if (source.empty()) {
        error_ = "manifest_unreadable";
        return false;
    }
    const size_t blockStart = source.find("nodes");
    if (blockStart == std::string::npos) {
        error_ = "nodes_missing";
        return false;
    }
    const size_t blockEnd = source.find(']', blockStart);
    const std::string block = source.substr(blockStart,
        blockEnd == std::string::npos ? std::string::npos : blockEnd - blockStart);
    const auto base = resolved.parent_path();
    size_t cursor = 0;
    while ((cursor = block.find('"', cursor)) != std::string::npos) {
        const size_t end = block.find('"', cursor + 1);
        if (end == std::string::npos) break;
        const auto relative = block.substr(cursor + 1, end - cursor - 1);
        if (!loadNode(base, base / relative)) return false;
        cursor = end + 1;
    }
    return !nodes_.empty();
}

bool NodeRegistry::loadNode(const std::filesystem::path&,
                            const std::filesystem::path& contractPath) {
    const std::string source = readText(contractPath);
    if (source.empty()) {
        error_ = "node_contract_unreadable:" + contractPath.string();
        return false;
    }
    NodeDescriptor node;
    node.name = contractPath.stem().string();
    const std::string kind = valueAfter(source, "kind");
    node.capability = valueAfter(source, "capability");
    if (!parseKind(kind, node.kind) || node.capability.empty()) {
        error_ = "node_contract_invalid:" + contractPath.string();
        return false;
    }
    node.inputs = bracketValues(source, "inputs");
    node.outputs = bracketValues(source, "outputs");
    node.contract = contractPath;
    nodes_.push_back(std::move(node));
    return true;
}

void NodeRegistry::resolveProviders(const WebX::ProviderManager& providers) {
    for (auto& node : nodes_) {
        const auto capability = capabilityFor(node.capability);
        const auto selected = std::find_if(
            providers.getProviders().begin(), providers.getProviders().end(),
            [capability](const WebX::Provider& provider) {
                return provider.capability == capability && provider.available;
            });
        if (selected != providers.getProviders().end()) {
            node.provider = selected->id;
            node.provider_available = true;
        } else {
            node.provider.clear();
            node.provider_available = false;
        }
    }
}

} // namespace Kuhul::Runtime
