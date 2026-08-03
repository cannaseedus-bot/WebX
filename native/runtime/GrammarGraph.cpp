#include "grammar_graph.h"

#include <algorithm>
#include <unordered_set>

namespace Kuhul::Runtime {

bool GrammarGraph::addNode(GrammarNode node, std::string& error) {
    if (node.id.empty() || node.symbol.empty()) {
        error = "grammar_node_missing_identity";
        return false;
    }
    const auto duplicate = std::find_if(
        nodes_.begin(), nodes_.end(),
        [&node](const GrammarNode& existing) { return existing.id == node.id; });
    if (duplicate != nodes_.end()) {
        error = "duplicate_grammar_node:" + node.id;
        return false;
    }
    nodes_.push_back(std::move(node));
    return true;
}

bool GrammarGraph::addEdge(GrammarEdge edge, std::string& error) {
    const auto hasNode = [this](const std::string& id) {
        return std::any_of(
            nodes_.begin(), nodes_.end(),
            [&id](const GrammarNode& node) { return node.id == id; });
    };
    if (!hasNode(edge.from) || !hasNode(edge.to)) {
        error = "grammar_edge_node_missing";
        return false;
    }
    edges_.push_back(std::move(edge));
    return true;
}

GrammarValidation GrammarGraph::validate() const {
    if (nodes_.empty()) return {false, "grammar_graph_empty"};

    size_t accepting = 0;
    for (const auto& node : nodes_) {
        if (node.accepting) ++accepting;
    }
    if (accepting == 0) return {false, "grammar_accepting_node_missing"};

    std::unordered_set<std::string> ids;
    for (const auto& node : nodes_) ids.insert(node.id);
    for (const auto& edge : edges_) {
        if (!ids.count(edge.from) || !ids.count(edge.to))
            return {false, "grammar_edge_node_missing"};
    }

    if (form_ == GrammarForm::PEG) {
        for (const auto& edge : edges_) {
            if (edge.priority < 0)
                return {false, "peg_edge_priority_invalid"};
        }
    }
    return {true, {}};
}

std::vector<GrammarEdge> GrammarGraph::outgoing(
    const std::string& nodeId) const {
    std::vector<GrammarEdge> result;
    for (const auto& edge : edges_) {
        if (edge.from == nodeId) result.push_back(edge);
    }
    if (form_ == GrammarForm::PEG) {
        std::stable_sort(
            result.begin(), result.end(),
            [](const GrammarEdge& left, const GrammarEdge& right) {
                return left.priority < right.priority;
            });
    }
    return result;
}

} // namespace Kuhul::Runtime
