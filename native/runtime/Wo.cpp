#include "phase_runtime.h"

namespace Kuhul::Runtime {

void ApplyPersonality(RuntimeContext& runtime) {
    if (!runtime.personality.enabled) return;

    for (auto& node : runtime.nodes) {
        float priority = 1.0f;
        priority += std::min(0.10f, std::max(0.0f, node.confidence) * 0.10f);
        if (node.residency == Residency::Hot) priority += 0.05f;
        else if (node.residency == Residency::Warm) priority += 0.02f;
        node.personality_priority = priority;
    }
    runtime.trace.push_back("Personality:PT-0001");
}

void Wo(RuntimeContext& runtime) {
    runtime.trace.push_back("Wo");
    runtime.cache.updates++;
    runtime.memory.paged = true;
    ApplyPersonality(runtime);
}

} // namespace Kuhul::Runtime
