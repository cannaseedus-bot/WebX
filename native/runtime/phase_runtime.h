#pragma once

#include "webx_compute.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace Kuhul::Runtime {

// Fold handlers are state operators. Providers and Micronauts may vary
// behind a handler, but the operator order remains invariant.
enum class Phase {
    Pop,
    Wo,
    Yax,
    Sek,
    Chen,
    Xul
};

enum class NodeKind {
    Parse,
    AST,
    Compute,
    Tensor,
    Graph,
    Shader,
    Compile,
    Memory,
    Storage,
    Network,
    Verification,
    Tool
};

enum class Residency {
    Streaming,
    Warm,
    Hot
};

struct SemanticNode {
    std::string id;
    float pressure = 0.0f;
    float personality_priority = 1.0f;
    float confidence = 0.0f;
    NodeKind kind = NodeKind::Compute;
    Residency residency = Residency::Streaming;
    std::string capability = "Compute";
    std::string provider;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    uint64_t executions = 0;
    uint64_t history = 0;
    bool admitted = false;
    bool active = false;
};

struct PersonalityPolicy {
    std::string id = "PT-0001";
    std::string style = "deterministic-collaborative";
    bool enabled = true;
    bool execution_owner = false;
    bool can_change_fold_order = false;
    bool can_change_node_operation = false;
};

using WorkingSet = std::vector<SemanticNode>;
using PressureField = std::unordered_map<std::string, float>;

struct MemoryState {
    bool paged = false;
};

struct ExecutionTask {
    std::string node_id;
};

using ExecutionQueue = std::vector<ExecutionTask>;

struct RuntimeCache {
    uint64_t updates = 0;
};

struct MicronautRegistry {
    uint64_t lookups = 0;
};

struct BackendRegistry {
    uint64_t selections = 0;
};

struct TickState {
    uint64_t number = 0;
};

struct Session {
    std::string id = "default";
};

struct RuntimeContext {
    WebX::FieldGraph* graph = nullptr;
    WorkingSet nodes;
    PressureField pressure;
    MemoryState memory;
    ExecutionQueue queue;
    RuntimeCache cache;
    MicronautRegistry micronauts;
    BackendRegistry backends;
    TickState tick;
    Session session;
    PersonalityPolicy personality;
    Phase phase = Phase::Pop;
    bool external_capability_requested = false;
    std::vector<std::string> trace;
};

struct PhaseDecision {
    Phase phase = Phase::Pop;
    bool admitted = false;
    float pressure = 0.0f;
    std::string reason;
};

bool Admit(float pressure, float threshold = 0.5f);
const char* phaseName(Phase phase);
void Pop(RuntimeContext& runtime);
void Wo(RuntimeContext& runtime);
void ApplyPersonality(RuntimeContext& runtime);
void Yax(RuntimeContext& runtime);
void Sek(RuntimeContext& runtime);
void Chen(RuntimeContext& runtime);
void Xul(RuntimeContext& runtime);

class RuntimeSandbox {
public:
    void reset();
    void setExternalCapabilityRequested(bool requested);
    void seedNode(const std::string& id, float pressure);
    PhaseDecision decide(const std::string& nodeId, Phase phase);
    // Compose the six fold operators without embedding provider policy.
    void Run();
    const RuntimeContext& context() const { return context_; }

private:
    RuntimeContext context_;
};

RuntimeSandbox& runtime();

} // namespace Kuhul::Runtime
