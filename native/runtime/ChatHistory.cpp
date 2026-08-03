#include "chat_history.h"

#include <algorithm>
#include <utility>

namespace Kuhul::Runtime {

ChatHistory::ChatHistory(std::size_t maxEvents)
    : maxEvents_(std::max<std::size_t>(1, maxEvents)) {}

void ChatHistory::record(ChatHistoryEvent event) {
    events_.push_back(std::move(event));
    if (events_.size() > maxEvents_)
        events_.erase(events_.begin());
}

float ChatHistory::semanticCoverage(const std::string& nodeId) const {
    std::size_t related = 0;
    std::size_t covered = 0;
    for (const auto& event : events_) {
        if (event.nodeId != nodeId) continue;
        related += event.topics.size();
        covered += event.topics.size() - std::min(
            event.topics.size(), event.unresolvedTopics.size());
    }
    return related == 0 ? 0.5f
                        : static_cast<float>(covered) /
                              static_cast<float>(related);
}

float ChatHistory::userFeedback(const std::string& nodeId) const {
    std::size_t feedback = 0;
    std::size_t positive = 0;
    for (const auto& event : events_) {
        if (event.nodeId != nodeId || event.role != "user") continue;
        ++feedback;
        if (event.acceptedExplanation && !event.userCorrection) ++positive;
    }
    return feedback == 0 ? 0.5f
                         : static_cast<float>(positive) /
                               static_cast<float>(feedback);
}

float ChatHistory::gapPenalty(const std::string& nodeId) const {
    std::size_t gaps = 0;
    for (const auto& event : events_) {
        if (event.nodeId == nodeId) gaps += event.unresolvedTopics.size();
    }
    return 1.0f / (1.0f + static_cast<float>(gaps));
}

} // namespace Kuhul::Runtime
