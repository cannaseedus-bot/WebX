#include "phase_runtime.h"

namespace Kuhul::Runtime {

void Xul(RuntimeContext& runtime) {
    runtime.trace.push_back("Xul");
    runtime.queue.clear();
    runtime.tick.number++;
    for (auto& node : runtime.nodes) {
        node.active = false;
    }
}

} // namespace Kuhul::Runtime
