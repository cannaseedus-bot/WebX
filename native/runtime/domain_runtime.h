#pragma once

#include "phase_runtime.h"

#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct DomainResult {
    std::string domain;
    bool admitted = false;
    bool executed = false;
    float pressure = 0.0f;
};

DomainResult executeDomain(RuntimeContext& context,
                           const std::string& domain,
                           const std::string& nodeId,
                           float pressure,
                           const std::vector<std::string>& semanticSteps);

DomainResult runLearning(RuntimeContext& context);
DomainResult runInference(RuntimeContext& context);
DomainResult runShader(RuntimeContext& context);
DomainResult runCompile(RuntimeContext& context);
DomainResult runProvider(RuntimeContext& context);
DomainResult runForge(RuntimeContext& context);
DomainResult runSCX(RuntimeContext& context);

std::vector<DomainResult> runAllDomains(RuntimeContext& context);

} // namespace Kuhul::Runtime
