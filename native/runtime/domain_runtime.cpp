#include "domain_runtime.h"

#include <algorithm>

namespace Kuhul::Runtime {

DomainResult executeDomain(RuntimeContext& context,
                           const std::string& domain,
                           const std::string& nodeId,
                           float pressure,
                           const std::vector<std::string>& semanticSteps) {
    context.nodes.clear();
    context.pressure.clear();
    context.queue.clear();
    context.trace.clear();
    context.tick = TickState{};
    context.external_capability_requested = false;
    context.nodes.push_back({nodeId, pressure});
    context.pressure[nodeId] = pressure;

    context.trace.push_back(domain + ".Pop");
    Pop(context);
    context.trace.push_back(domain + ".Wo");
    Wo(context);
    context.trace.push_back(domain + ".Yax");
    Yax(context);

    const auto admittedNode = std::find_if(
        context.nodes.begin(),
        context.nodes.end(),
        [&nodeId](const SemanticNode& node) { return node.id == nodeId; });
    if (admittedNode == context.nodes.end()) {
        return {domain, false, false, 0.0f};
    }

    const bool admitted = admittedNode->admitted;
    if (admitted) {
        context.trace.insert(context.trace.end(),
                             semanticSteps.begin(), semanticSteps.end());
        context.trace.push_back(domain + ".Sek");
        Sek(context);
        context.trace.push_back(domain + ".Chen");
        Chen(context);
    }

    const float finalPressure = admittedNode->pressure;
    context.trace.push_back(domain + ".Xul");
    Xul(context);
    return {domain, admitted, admitted, finalPressure};
}

std::vector<DomainResult> runAllDomains(RuntimeContext& context) {
    return {
        runLearning(context),
        runInference(context),
        runShader(context),
        runCompile(context),
        runProvider(context),
        runForge(context),
        runSCX(context)
    };
}

} // namespace Kuhul::Runtime
