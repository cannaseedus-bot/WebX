#include "domain_runtime.h"

namespace Kuhul::Runtime {

DomainResult runInference(RuntimeContext& context) {
    return executeDomain(
        context, "Inference", "Inference", 0.74f,
        {"Inference.LoadPrompt", "Inference.ConstructWorkingSet",
         "Inference.AdmitContext", "Inference.ForwardPass",
         "Inference.MeasureConfidence", "Inference.PersistSession"});
}

} // namespace Kuhul::Runtime
