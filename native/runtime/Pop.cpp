#include "phase_runtime.h"

#include <algorithm>

namespace Kuhul::Runtime {

void Pop(RuntimeContext& runtime) {
    runtime.trace.push_back("Pop");
    runtime.nodes.erase(
        std::remove_if(
            runtime.nodes.begin(),
            runtime.nodes.end(),
            [](const SemanticNode& node) { return node.pressure <= 0.0f; }),
        runtime.nodes.end());
    for (auto& node : runtime.nodes) {
        node.active = false;
        node.admitted = false;
    }
}

} // namespace Kuhul::Runtime
