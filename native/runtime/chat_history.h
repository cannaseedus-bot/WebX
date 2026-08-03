#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct ChatHistoryEvent {
    std::string sessionId;
    std::string nodeId;
    std::string role;
    std::string text;
    std::vector<std::string> topics;
    std::vector<std::string> unresolvedTopics;
    bool userCorrection = false;
    bool acceptedExplanation = false;
};

class ChatHistory {
public:
    explicit ChatHistory(std::size_t maxEvents = 256);

    void record(ChatHistoryEvent event);
    const std::vector<ChatHistoryEvent>& events() const { return events_; }

    float semanticCoverage(const std::string& nodeId) const;
    float userFeedback(const std::string& nodeId) const;
    float gapPenalty(const std::string& nodeId) const;

private:
    std::size_t maxEvents_;
    std::vector<ChatHistoryEvent> events_;
};

} // namespace Kuhul::Runtime
