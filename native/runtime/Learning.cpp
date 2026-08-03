#include "domain_runtime.h"

namespace Kuhul::Runtime {

DomainResult runLearning(RuntimeContext& context) {
    return executeDomain(
        context, "Learning", "Learning", 0.78f,
        {"Learning.LoadDataset", "Learning.ConstructTensorGraph",
         "Learning.AdmitBatch", "Learning.ForwardBackward",
         "Learning.ComputeEvidence", "Learning.PersistCheckpoint"});
}

} // namespace Kuhul::Runtime
