#pragma once

#include <string>
#include <vector>

namespace Kuhul::Runtime {

enum class GrammarForm {
    EBNF,
    GBNF,
    PEG
};

struct GrammarNode {
    std::string id;
    std::string symbol;
    bool terminal = false;
    bool accepting = false;
};

struct GrammarEdge {
    std::string from;
    std::string to;
    std::string label;
    int priority = 0; // PEG ordered-choice priority; ignored by EBNF/GBNF.
    bool repeating = false;
};

struct GrammarValidation {
    bool valid = false;
    std::string error;
};

class GrammarGraph {
public:
    explicit GrammarGraph(GrammarForm form) : form_(form) {}

    bool addNode(GrammarNode node, std::string& error);
    bool addEdge(GrammarEdge edge, std::string& error);
    GrammarValidation validate() const;
    std::vector<GrammarEdge> outgoing(const std::string& nodeId) const;

    GrammarForm form() const { return form_; }
    const std::vector<GrammarNode>& nodes() const { return nodes_; }
    const std::vector<GrammarEdge>& edges() const { return edges_; }

private:
    GrammarForm form_;
    std::vector<GrammarNode> nodes_;
    std::vector<GrammarEdge> edges_;
};

} // namespace Kuhul::Runtime
