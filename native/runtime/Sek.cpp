#include "phase_runtime.h"

namespace Kuhul::Runtime {

void Sek(RuntimeContext& runtime) {
    runtime.trace.push_back("Sek");
    runtime.backends.selections++;
    for (auto& node : runtime.nodes) {
        node.active = node.admitted;
    }
}

} // namespace Kuhul::Runtime
