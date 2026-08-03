#pragma once

#include "phase_runtime.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct FoldDescriptor {
    std::string name;
    std::string id;
    Phase phase = Phase::Pop;
    std::string micronaut;
    std::string entry_point;
    std::filesystem::path contract;
    std::filesystem::path artifact;
    bool observer = false;
};

class FoldRegistry {
public:
    bool loadManifest(const std::filesystem::path& manifestPath);
    std::vector<FoldDescriptor> admitted(Phase phase,
                                         const std::vector<std::string>& names) const;
    const std::vector<FoldDescriptor>& folds() const { return folds_; }
    const std::string& error() const { return error_; }

private:
    bool loadFold(const std::filesystem::path& manifestBase,
                  const std::filesystem::path& contractPath);
    std::vector<FoldDescriptor> folds_;
    std::string error_;
};

} // namespace Kuhul::Runtime
