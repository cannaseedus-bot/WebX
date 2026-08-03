#pragma once

#include "phase_runtime.h"
#include "chat_history.h"

namespace Kuhul::Runtime {

struct ConfidenceEvidence {
    float verification = 0.0f;
    float replay = 0.0f;
    float provider = 0.0f;
    float agreement = 0.0f;
    float semanticCoverage = 0.0f;
    float userFeedback = 0.0f;
    float gapPenalty = 1.0f;
};

class ConfidenceEngine {
public:
    float Evaluate(const SemanticNode& node,
                   const ConfidenceEvidence& evidence) const;
    void Update(SemanticNode& node, float confidence) const;
    void UpdateFromChat(SemanticNode& node, const ChatHistory& history) const;
};

} // namespace Kuhul::Runtime
