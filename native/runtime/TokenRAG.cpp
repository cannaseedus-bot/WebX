#include "token_rag.h"

#include <algorithm>

namespace Kuhul::Runtime {

TokenRAGEngine::TokenRAGEngine(std::size_t maxBatch,
                               float admissionPressure,
                               float retentionConfidence)
    : maxBatch_(std::max<std::size_t>(1, maxBatch)),
      admissionPressure_(std::clamp(admissionPressure, 0.0f, 1.0f)),
      retentionConfidence_(std::clamp(retentionConfidence, 0.0f, 1.0f)) {}

void TokenRAGEngine::enqueue(TokenRAGQuery query) {
    pending_.push_back(std::move(query));
}

std::vector<TokenRAGQuery> TokenRAGEngine::admit() {
    std::stable_sort(
        pending_.begin(), pending_.end(),
        [](const TokenRAGQuery& left, const TokenRAGQuery& right) {
            return left.pressure > right.pressure;
        });

    std::vector<TokenRAGQuery> admitted;
    for (auto it = pending_.begin();
         it != pending_.end() && admitted.size() < maxBatch_;) {
        if (it->pressure >= admissionPressure_) {
            admitted.push_back(*it);
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
    return admitted;
}

void TokenRAGEngine::record(const TokenRAGQuery& query,
                            std::vector<TokenRAGEvidence> evidence) {
    for (auto& item : evidence) {
        item.generationPosition = query.generationPosition;
        item.active = item.confidence >= retentionConfidence_;
        const auto existing = std::find_if(
            evidence_.begin(), evidence_.end(),
            [&item](const TokenRAGEvidence& current) {
                return current.id == item.id &&
                       current.generationPosition == item.generationPosition;
            });
        if (existing == evidence_.end()) {
            evidence_.push_back(std::move(item));
        } else if (item.score > existing->score) {
            *existing = std::move(item);
        }
    }
    prune();
}

void TokenRAGEngine::prune() {
    evidence_.erase(
        std::remove_if(
            evidence_.begin(), evidence_.end(),
            [this](const TokenRAGEvidence& item) {
                return !item.active || item.confidence < retentionConfidence_;
            }),
        evidence_.end());
}

} // namespace Kuhul::Runtime
