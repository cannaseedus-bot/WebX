#include "codegen_graph.h"

#include <algorithm>

namespace Kuhul::Runtime {

bool CodeGenGraph::addNode(CodeGenNode node, std::string& error) {
    if (node.id.empty()) {
        error = "codegen_node_missing_id";
        return false;
    }
    const auto duplicate = std::find_if(
        nodes_.begin(), nodes_.end(),
        [&node](const CodeGenNode& existing) { return existing.id == node.id; });
    if (duplicate != nodes_.end()) {
        error = "duplicate_codegen_node:" + node.id;
        return false;
    }
    nodes_.push_back(std::move(node));
    return true;
}

CodeGenValidation CodeGenGraph::validate() const {
    std::vector<DAGTask> tasks;
    tasks.reserve(nodes_.size());
    for (const auto& node : nodes_)
        tasks.push_back({node.id, node.planDependsOn});

    const DAGSchedule schedule = DAGScheduler{}.schedule(tasks);
    return {schedule.error.empty(), schedule.error, schedule.ordered};
}

bool CodeGenGraph::recordValidation(const std::string& id, bool passed,
                                    const std::string& detail) {
    const auto it = std::find_if(
        nodes_.begin(), nodes_.end(),
        [&id](const CodeGenNode& node) { return node.id == id; });
    if (it == nodes_.end()) return false;

    it->validationDetail = detail;
    it->accepted = passed;
    if (passed) {
        it->schematicPressure = std::max(0.0f, it->schematicPressure - 0.1f);
    } else {
        ++it->repairAttempts;
        it->schematicPressure = std::min(1.0f, it->schematicPressure + 0.2f);
    }
    return true;
}

bool CodeGenGraph::canRepair(const std::string& id,
                             unsigned maxAttempts) const {
    const auto it = std::find_if(
        nodes_.begin(), nodes_.end(),
        [&id](const CodeGenNode& node) { return node.id == id; });
    return it != nodes_.end() && !it->accepted &&
           it->repairAttempts < maxAttempts;
}

} // namespace Kuhul::Runtime
