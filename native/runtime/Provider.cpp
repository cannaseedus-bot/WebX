#include "domain_runtime.h"

namespace Kuhul::Runtime {

DomainResult runProvider(RuntimeContext& context) {
    return executeDomain(
        context, "Provider", "Provider", 0.66f,
        {"Provider.DiscoverCapabilities", "Provider.ConstructGraph",
         "Provider.AdmitCapability", "Provider.Resolve",
         "Provider.Verify", "Provider.Cache"});
}

} // namespace Kuhul::Runtime
