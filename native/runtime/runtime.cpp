#include "phase_runtime.h"

#include <algorithm>

namespace Kuhul::Runtime {

namespace {
RuntimeSandbox instance;
}

bool Admit(float pressure, float threshold) {
    return pressure >= threshold;
}

const char* phaseName(Phase phase) {
    switch (phase) {
    case Phase::Pop: return "Pop";
    case Phase::Wo: return "Wo";
    case Phase::Yax: return "Yax";
    case Phase::Sek: return "Sek";
    case Phase::Chen: return "Chen";
    case Phase::Xul: return "Xul";
    }
    return "Unknown";
}

void RuntimeSandbox::reset() {
    context_ = RuntimeContext{};
}

void RuntimeSandbox::setExternalCapabilityRequested(bool requested) {
    context_.external_capability_requested = requested;
}

void RuntimeSandbox::seedNode(const std::string& id, float pressure) {
    context_.nodes.push_back(SemanticNode{id, pressure});
    context_.pressure[id] = pressure;
}

PhaseDecision RuntimeSandbox::decide(const std::string& nodeId, Phase phase) {
    const auto node = std::find_if(
        context_.nodes.begin(),
        context_.nodes.end(),
        [&nodeId](const SemanticNode& candidate) { return candidate.id == nodeId; });
    if (node == context_.nodes.end()) {
        return {phase, false, 0.0f, "node_not_registered"};
    }

    context_.phase = phase;
    switch (phase) {
    case Phase::Pop: Pop(context_); break;
    case Phase::Wo: Wo(context_); break;
    case Phase::Yax: Yax(context_); break;
    case Phase::Sek: Sek(context_); break;
    case Phase::Chen: Chen(context_); break;
    case Phase::Xul: Xul(context_); break;
    }

    const auto current = std::find_if(
        context_.nodes.begin(),
        context_.nodes.end(),
        [&nodeId](const SemanticNode& candidate) { return candidate.id == nodeId; });
    return {
        phase,
        current->admitted,
        current->pressure,
        current->active ? "Active" : (current->admitted ? "Admitted" : "Rejected")
    };
}

void RuntimeSandbox::Run() {
    Pop(context_);
    Wo(context_);
    Yax(context_);
    Sek(context_);
    Chen(context_);
    Xul(context_);
}

RuntimeSandbox& runtime() {
    return instance;
}

} // namespace Kuhul::Runtime
