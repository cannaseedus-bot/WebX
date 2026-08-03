#include "phase_runtime.h"

#include <algorithm>

namespace Kuhul::Runtime {

void Yax(RuntimeContext& runtime) {
    runtime.trace.push_back("Yax");
    runtime.queue.clear();
    for (auto& node : runtime.nodes) {
        if (node.id == "MicrosoftSDK" && !runtime.external_capability_requested) {
            node.pressure *= 0.5f;
            node.admitted = false;
            continue;
        }
        node.pressure = std::clamp(
            node.pressure * node.personality_priority, 0.0f, 1.0f);
        node.admitted = Admit(node.pressure);
        runtime.pressure[node.id] = node.pressure;
        if (node.admitted) {
            runtime.queue.push_back({node.id});
        }
    }
}

} // namespace Kuhul::Runtime
