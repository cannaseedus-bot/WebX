#include "domain_runtime.h"

namespace Kuhul::Runtime {

DomainResult runCompile(RuntimeContext& context) {
    return executeDomain(
        context, "Compile", "Compile", 0.71f,
        {"Compile.LoadSource", "Compile.ResolveDependencies",
         "Compile.AdmitDependencies", "Compile.EmitObject",
         "Compile.ValidateObject", "Compile.CacheObject"});
}

} // namespace Kuhul::Runtime
