#include "rag.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace Kuhul::Runtime {

namespace {

std::vector<std::string> terms(const std::string& value) {
    std::istringstream input(value);
    std::vector<std::string> result;
    std::string word;
    while (input >> word) {
        std::transform(word.begin(), word.end(), word.begin(),
                       [](unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        word.erase(std::remove_if(word.begin(), word.end(),
                                  [](unsigned char ch) {
                                      return !std::isalnum(ch);
                                  }),
                   word.end());
        if (word.size() > 2) result.push_back(std::move(word));
    }
    return result;
}

} // namespace

RAGEngine::RAGEngine(std::size_t maxRetries)
    : maxRetries_(maxRetries) {}

RAGState RAGEngine::analyze(const std::string& query) const {
    RAGState state;
    state.query = query;
    state.rewrittenQuery = query;
    state.unresolvedTopics = terms(query);
    return state;
}

void RAGEngine::merge(RAGState& state,
                      const std::vector<RetrievalDocument>& documents) const {
    for (const auto& document : documents) {
        const auto duplicate = std::find_if(
            state.documents.begin(), state.documents.end(),
            [&document](const RetrievalDocument& existing) {
                return existing.id == document.id ||
                       existing.text == document.text;
            });
        if (duplicate == state.documents.end())
            state.documents.push_back(document);
        else if (document.score > duplicate->score)
            *duplicate = document;
    }
    std::stable_sort(state.documents.begin(), state.documents.end(),
                     [](const RetrievalDocument& left,
                        const RetrievalDocument& right) {
                         return left.score > right.score;
                     });
}

bool RAGEngine::grade(RAGState& state) const {
    const auto queryTerms = terms(state.rewrittenQuery);
    if (queryTerms.empty() || state.documents.empty()) {
        state.coverage = 0.0f;
        state.relevance = 0.0f;
        state.terminal = state.retryCount >= maxRetries_;
        return false;
    }

    std::size_t covered = 0;
    for (const auto& queryTerm : queryTerms) {
        const bool found = std::any_of(
            state.documents.begin(), state.documents.end(),
            [&queryTerm](const RetrievalDocument& document) {
                std::string text = document.text;
                std::transform(text.begin(), text.end(), text.begin(),
                               [](unsigned char ch) {
                                   return static_cast<char>(std::tolower(ch));
                               });
                return text.find(queryTerm) != std::string::npos;
            });
        if (found) ++covered;
    }
    state.coverage = static_cast<float>(covered) /
                     static_cast<float>(queryTerms.size());
    state.relevance = state.documents.front().score;
    const bool valid = state.coverage >= 0.5f && state.relevance >= 0.5f;
    state.terminal = valid || state.retryCount >= maxRetries_;
    return valid;
}

bool RAGEngine::rewrite(RAGState& state) const {
    if (state.terminal || state.retryCount >= maxRetries_) {
        state.terminal = true;
        return false;
    }
    ++state.retryCount;
    std::ostringstream rewritten;
    rewritten << state.rewrittenQuery;
    for (const auto& topic : state.unresolvedTopics)
        rewritten << ' ' << topic;
    state.rewrittenQuery = rewritten.str();
    return true;
}

} // namespace Kuhul::Runtime
