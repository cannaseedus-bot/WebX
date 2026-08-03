#include "domain_runtime.h"

namespace Kuhul::Runtime {

DomainResult runSCX(RuntimeContext& context) {
    return executeDomain(
        context, "SCX", "SCX", 0.69f,
        {"SCX.InitializeWorkingSet", "SCX.BuildGraph",
         "SCX.AdmitPressure", "SCX.Schedule",
         "SCX.MeasurePressure", "SCX.PersistWorkingSet"});
}

} // namespace Kuhul::Runtime
