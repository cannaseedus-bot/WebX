#include "confidence.h"

#include <algorithm>

namespace Kuhul::Runtime {

namespace {

float clampUnit(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

} // namespace

float ConfidenceEngine::Evaluate(
    const SemanticNode& node,
    const ConfidenceEvidence& evidence) const {
    const float verification = clampUnit(evidence.verification);
    const float replay = clampUnit(evidence.replay);
    const float provider = clampUnit(evidence.provider);
    const float agreement = clampUnit(evidence.agreement);
    const float semanticCoverage = clampUnit(evidence.semanticCoverage);
    const float userFeedback = clampUnit(evidence.userFeedback);
    const float gapPenalty = clampUnit(evidence.gapPenalty);
    const float evidenceScore =
        verification * replay * provider * agreement *
        semanticCoverage * userFeedback * gapPenalty;
    return clampUnit(node.active ? evidenceScore : evidenceScore * 0.5f);
}

void ConfidenceEngine::Update(SemanticNode& node, float confidence) const {
    node.confidence = clampUnit(confidence);
}

void ConfidenceEngine::UpdateFromChat(SemanticNode& node,
                                       const ChatHistory& history) const {
    ConfidenceEvidence evidence;
    evidence.semanticCoverage = history.semanticCoverage(node.id);
    evidence.userFeedback = history.userFeedback(node.id);
    evidence.gapPenalty = history.gapPenalty(node.id);
    evidence.verification = node.active ? 1.0f : 0.5f;
    evidence.replay = node.history > 0 ? 1.0f : 0.5f;
    evidence.provider = node.provider.empty() ? 0.5f : 1.0f;
    evidence.agreement = node.executions > 1 ? 1.0f : 0.75f;
    Update(node, Evaluate(node, evidence));
}

} // namespace Kuhul::Runtime
