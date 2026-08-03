#pragma once

#include "dag.h"

#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct CodeGenNode {
    std::string id;
    std::string language;
    std::string targetLayer;
    std::vector<std::string> planDependsOn;
    std::vector<std::string> schematicInputs;
    std::vector<std::string> schematicOutputs;
    std::vector<std::string> schematicThrows;
    std::string astArtifact;
    std::string validationDetail;
    float schematicPressure = 0.0f;
    unsigned repairAttempts = 0;
    bool accepted = false;
};

struct CodeGenValidation {
    bool valid = false;
    std::string error;
    std::vector<std::string> ordered;
};

class CodeGenGraph {
public:
    bool addNode(CodeGenNode node, std::string& error);
    CodeGenValidation validate() const;
    bool recordValidation(const std::string& id, bool passed,
                          const std::string& detail);
    bool canRepair(const std::string& id, unsigned maxAttempts = 3) const;
    const std::vector<CodeGenNode>& nodes() const { return nodes_; }

private:
    std::vector<CodeGenNode> nodes_;
};

} // namespace Kuhul::Runtime
