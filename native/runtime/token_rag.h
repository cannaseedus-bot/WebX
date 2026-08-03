#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct TokenRAGQuery {
    std::size_t tokenIndex = 0;
    std::size_t generationPosition = 0;
    std::string token;
    std::string query;
    float pressure = 0.0f;
    float confidence = 0.0f;
};

struct TokenRAGEvidence {
    std::string id;
    std::string source;
    std::string text;
    std::size_t generationPosition = 0;
    float score = 0.0f;
    float confidence = 0.0f;
    bool active = true;
};

class TokenRAGEngine {
public:
    explicit TokenRAGEngine(std::size_t maxBatch = 8,
                            float admissionPressure = 0.7f,
                            float retentionConfidence = 0.5f);

    void enqueue(TokenRAGQuery query);
    std::vector<TokenRAGQuery> admit();
    void record(const TokenRAGQuery& query,
                std::vector<TokenRAGEvidence> evidence);
    void prune();

    const std::vector<TokenRAGQuery>& pending() const { return pending_; }
    const std::vector<TokenRAGEvidence>& evidence() const {
        return evidence_;
    }

private:
    std::size_t maxBatch_;
    float admissionPressure_;
    float retentionConfidence_;
    std::vector<TokenRAGQuery> pending_;
    std::vector<TokenRAGEvidence> evidence_;
};

} // namespace Kuhul::Runtime
