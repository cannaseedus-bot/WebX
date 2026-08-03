#include "phase_runtime.h"
#include "confidence.h"

#include <algorithm>

namespace Kuhul::Runtime {

void Chen(RuntimeContext& runtime) {
    runtime.trace.push_back("Chen");
    ConfidenceEngine confidence;
    for (auto& node : runtime.nodes) {
        if (node.active) {
            node.pressure = std::min(1.0f, node.pressure + 0.05f);
            runtime.pressure[node.id] = node.pressure;
            const ConfidenceEvidence evidence{
                1.0f,
                node.history > 0 ? 1.0f : 0.5f,
                node.provider.empty() ? 0.5f : 1.0f,
                node.executions > 1 ? 1.0f : 0.75f,
                node.executions > 0 ? 1.0f : 0.5f,
                node.history > 0 ? 1.0f : 0.5f,
                1.0f};
            confidence.Update(node, confidence.Evaluate(node, evidence));
        }
    }
}

} // namespace Kuhul::Runtime
