#include "fold_registry.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <utility>

namespace Kuhul::Runtime {
namespace {

std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) return {};
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::vector<std::string> quotedValues(const std::string& source) {
    std::vector<std::string> values;
    size_t offset = 0;
    while ((offset = source.find('"', offset)) != std::string::npos) {
        const size_t end = source.find('"', offset + 1);
        if (end == std::string::npos) break;
        values.push_back(source.substr(offset + 1, end - offset - 1));
        offset = end + 1;
    }
    return values;
}

std::string fieldValue(const std::string& source, const std::string& field) {
    const std::string marker = field + ":";
    const size_t start = source.find(marker);
    if (start == std::string::npos) return {};
    const size_t valueStart = source.find_first_not_of(" \t", start + marker.size());
    if (valueStart == std::string::npos) return {};
    if (source[valueStart] == '"') {
        const size_t end = source.find('"', valueStart + 1);
        return end == std::string::npos
            ? std::string{}
            : source.substr(valueStart + 1, end - valueStart - 1);
    }
    const size_t end = source.find_first_of(" \t\r\n", valueStart);
    return source.substr(valueStart, end == std::string::npos
        ? std::string::npos
        : end - valueStart);
}

bool parsePhase(const std::string& value, Phase& phase) {
    if (value == "Pop") phase = Phase::Pop;
    else if (value == "Wo") phase = Phase::Wo;
    else if (value == "Yax") phase = Phase::Yax;
    else if (value == "Sek") phase = Phase::Sek;
    else if (value == "Chen" || value == "Ch'en") phase = Phase::Chen;
    else if (value == "Xul") phase = Phase::Xul;
    else return false;
    return true;
}

} // namespace

bool FoldRegistry::loadManifest(const std::filesystem::path& manifestPath) {
    folds_.clear();
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

    const std::filesystem::path base = resolved.parent_path();
    const size_t blockStart = source.find("runtime_folds");
    if (blockStart == std::string::npos) {
        error_ = "runtime_folds_missing";
        return false;
    }
    const size_t blockEnd = source.find(']', blockStart);
    const std::string block = source.substr(blockStart,
        blockEnd == std::string::npos ? std::string::npos : blockEnd - blockStart);
    for (const auto& relative : quotedValues(block)) {
        if (relative.size() < 6 ||
            relative.compare(relative.size() - 6, 6, ".kuhul") != 0)
            continue;
        if (!loadFold(base, base / relative)) return false;
    }
    return !folds_.empty();
}

bool FoldRegistry::loadFold(const std::filesystem::path& manifestBase,
                            const std::filesystem::path& contractPath) {
    const std::string source = readText(contractPath);
    if (source.empty()) {
        error_ = "fold_contract_unreadable:" + contractPath.string();
        return false;
    }

    FoldDescriptor fold;
    fold.name = contractPath.stem().string();
    fold.id = fieldValue(source, "id");
    fold.micronaut = fieldValue(source, "micronaut");
    fold.entry_point = fieldValue(source, "entry_point");
    fold.contract = contractPath;
    const std::string phase = fieldValue(source, "phase");
    const std::string artifact = fieldValue(source, "artifact");
    if (fold.id.empty() || fold.micronaut.empty() || fold.entry_point.empty() ||
        artifact.empty() || !parsePhase(phase, fold.phase)) {
        error_ = "fold_contract_invalid:" + contractPath.string();
        return false;
    }
    fold.artifact = manifestBase / artifact;
    if (!std::filesystem::exists(fold.artifact)) {
        const auto repositoryRoot = manifestBase.parent_path().parent_path();
        fold.artifact = repositoryRoot / artifact;
    }
    if (!std::filesystem::exists(fold.artifact)) {
        error_ = "fold_artifact_missing:" + fold.artifact.string();
        return false;
    }
    fold.observer = source.find("authority: observer") != std::string::npos;
    folds_.push_back(std::move(fold));
    return true;
}

std::vector<FoldDescriptor> FoldRegistry::admitted(
    Phase phase, const std::vector<std::string>& names) const {
    std::vector<FoldDescriptor> result;
    for (const auto& fold : folds_) {
        if (fold.phase != phase) continue;
        if (std::find(names.begin(), names.end(), fold.name) == names.end()) continue;
        result.push_back(fold);
    }
    return result;
}

} // namespace Kuhul::Runtime
