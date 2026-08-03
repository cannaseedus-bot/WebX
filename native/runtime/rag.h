#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace Kuhul::Runtime {

struct RetrievalDocument {
    std::string id;
    std::string source;
    std::string text;
    float score = 0.0f;
};

struct RAGState {
    std::string query;
    std::string rewrittenQuery;
    std::vector<RetrievalDocument> documents;
    std::vector<std::string> unresolvedTopics;
    float coverage = 0.0f;
    float relevance = 0.0f;
    std::size_t retryCount = 0;
    bool terminal = false;
};

class RAGEngine {
public:
    explicit RAGEngine(std::size_t maxRetries = 2);

    RAGState analyze(const std::string& query) const;
    void merge(RAGState& state,
               const std::vector<RetrievalDocument>& documents) const;
    bool grade(RAGState& state) const;
    bool rewrite(RAGState& state) const;

private:
    std::size_t maxRetries_;
};

} // namespace Kuhul::Runtime
