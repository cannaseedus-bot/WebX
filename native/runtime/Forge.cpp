#include "domain_runtime.h"

namespace Kuhul::Runtime {

DomainResult runForge(RuntimeContext& context) {
    return executeDomain(
        context, "Forge", "Forge", 0.63f,
        {"Forge.DiscoverNeed", "Forge.ConstructSpec",
         "Forge.AdmitRequirements", "Forge.Materialize",
         "Forge.Verify", "Forge.Deploy"});
}

} // namespace Kuhul::Runtime
