#include "domain_runtime.h"

namespace Kuhul::Runtime {

DomainResult runShader(RuntimeContext& context) {
    return executeDomain(
        context, "Shader", "Shader", 0.68f,
        {"Shader.Load", "Shader.ResolveIncludes", "Shader.SelectCompiler",
         "Shader.Compile", "Shader.Validate", "Shader.CacheCSO"});
}

} // namespace Kuhul::Runtime
