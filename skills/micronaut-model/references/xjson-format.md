# xjson Format Reference

Complete `@`-key semantics for the `atomic_block_fold` micronaut format.

---

## Top-Level Structure

Every micronaut `.xjson` file contains these top-level `@`-keyed sections:

| Key | Required | Purpose |
|---|---|---|
| `@meta` | ✅ | Identity, versioning, format declaration |
| `@lanes` | ✅ | Named data buses (5 standard lanes) |
| `@phases` | ✅ | Execution phase declarations (Pop/Wo/Sek/Ch'en) |
| `@variables` | ✅ | Mutable and immutable variable schemas |
| `@edges` | ✅ | Signal routing topology (skill→expert, router→experts, feedback) |
| `@agent.main` | ✅ | Primary agent cognitive block (lane 0) |
| `@model.core` | ✅ | Weight backend declaration |
| `@runtime.gpu` | ✅ | GPU/D3D12 hardware abstraction |
| `@moe.router` | ✅ | Expert router configuration |
| `@skills` | ✅ | Skill trigger definitions (at least 1) |
| `@experts` | ✅ | Expert compute endpoints (at least 1) |
| `@backend` | ⚠ API only | HTTP endpoint block (api_* backends only) |

---

## `@meta`

Identity and format declaration. All fields IMMUTABLE.

```json
"@meta": {
  "id": "SM-2",
  "name": "SummarizationMicronaut",
  "description": "Summarizes long documents into key points",
  "version": "1.0",
  "format": "atomic_block_fold",
  "spec": "docs/specs/horizontal-fold-linear-system.md"
}
```

- `id`: Short ID. Convention: 2-3 uppercase chars + number (e.g. CM-1, SM-2, RM-3)
- `format`: Always `"atomic_block_fold"` for micronauts
- `spec`: Path to the format spec doc (constant)

---

## `@lanes`

Named data buses. 5 standard lanes; do not add/remove without a schema migration.

```json
"@lanes": {
  "agent":   { "@dict": 0, "@role": "stateful cognitive tensor space", "@bus": "cognition" },
  "skills":  { "@dict": 1, "@role": "event routing triggers",          "@bus": "intent"    },
  "experts": { "@dict": 2, "@role": "compute endpoints (MoE)",         "@bus": "execution" },
  "runtime": { "@dict": 3, "@role": "hardware abstraction",            "@bus": "hardware"  },
  "router":  { "@dict": 4, "@role": "control plane",                   "@bus": "control",  "@kernel": "implicit" }
}
```

- `@dict`: Lane index. agent=0, skills=1, experts=2, runtime=3, router=4
- `@bus`: Bus name used in edge routing
- `@kernel: "implicit"`: The router lane is managed implicitly by the orchestrator

---

## `@phases`

Execution phases in order. Maps to the 4-phase cognitive cycle.

```json
"@phases": {
  "Pop":   { "@op": "load",    "@desc": "load state vector from BSON snapshot" },
  "Wo":    { "@op": "resolve", "@desc": "resolve lanes → build x blocks" },
  "Sek":   { "@op": "execute", "@desc": "x_next = A·x, top_k dispatch to experts" },
  "Ch'en": { "@op": "update",  "@desc": "A ← A + ΔA, persist learned_biases" }
}
```

- `Pop` — Load persisted state (BSON)
- `Wo` — Resolve lanes, construct activation vector
- `Sek` — Execute forward pass, dispatch to experts
- `Ch'en` — Update weights/biases, persist learned state

---

## `@variables`

Declares all model variables with mutability, update rules, persistence targets.

```json
"@variables": {
  "routing_bias": {
    "@mutability": "MUTABLE",
    "@update": "reinforce(success) - decay(failure)",
    "@range": [0, 1],
    "@persist": "@agent.main.learned_biases"
  },
  "learned": {
    "@mutability": "MUTABLE",
    "@update": "A + ΔA per turn",
    "@range": [-1, 1],
    "@persist": "models/<id>_state.bson"
  },
  "memory": {
    "@mutability": "MUTABLE",
    "@update": "last_topic, context_window",
    "@persist": "@agent.main.memory"
  },
  "model": {
    "@mutability": "IMMUTABLE",
    "@source": "@model.core",
    "@note": "vocab_size, num_layers schema-locked"
  },
  "experts": {
    "@mutability": "IMMUTABLE",
    "@source": "@experts",
    "@note": "structure schema-locked; routing_bias MUTABLE within"
  }
}
```

- `MUTABLE`: Changes during inference (routing biases, memory, learned state)
- `IMMUTABLE`: Schema-locked (model architecture, expert structure)
- `@persist`: Where the variable is saved (BSON path or JSON pointer)

---

## `@edges`

Signal routing topology. Three mandatory categories:

### Skill → Expert edges

One edge per skill/expert combination. These define which experts a skill can activate.

```json
{
  "@from": "skills.<skill_name>",
  "@to": "experts.<expert_name>",
  "@weight": 1.0,
  "@phase": "Sek"
}
```

### Router → Experts dispatch edge

Always present. The router dispatches to ALL experts (top_k selection happens inside the router).

```json
{
  "@from": "router.moe",
  "@to": "experts",
  "@op": "top_k_dispatch",
  "@phase": "Sek"
}
```

### Experts → Learned feedback edge

**CRITICAL** — This closes the learning loop. Without this edge, the model cannot adapt routing biases.

```json
{
  "@from": "experts",
  "@to": "learned",
  "@op": "reinforce",
  "@phase": "Ch'en",
  "@note": "closed-loop learning — DO NOT REMOVE"
}
```

---

## `@agent.main`

The primary agent cognitive block. Lives in lane 0, phase Pop.

```json
"@agent.main": {
  "@lane": 0,
  "@dict": 0,
  "@kind": "atomic_block",
  "@phase": "Pop",
  "id": "agent-main",
  "name": "SummaryBot",
  "persona": "Expert summarizer focused on clarity and brevity",
  "system_prompt": "You are SummaryBot, an expert at condensing complex documents...",
  "temperature": 0.7,
  "max_tokens": 2048,
  "interaction_count": 0,
  "learned_biases": {},
  "memory": {}
}
```

- `@kind: "atomic_block"`: Standard cognitive block type
- `@phase: "Pop"`: Loaded first (from BSON snapshot)
- `learned_biases`: Starts empty; populated by `routing_bias` variable updates
- `memory`: Starts empty; populated with `last_topic`, `context_window`, etc.

---

## `@model.core`

Backend weight declaration. `@xcfe: "IMMUTABLE"` locks the schema.

See `references/backends.md` for field schemas per backend type.

Common fields across all backends:
- `@xcfe: "IMMUTABLE"`: Required on all model.core blocks
- `backend`: One of `local_gguf | local_scxqdds | fetch_oss | api_openai | api_anthropic | api_ollama | api_custom`

---

## `@runtime.gpu`

Hardware abstraction layer. Lives in lane 3.

```json
"@runtime.gpu": {
  "@lane": 3,
  "@dict": 0,
  "@kind": "atomic_block",
  "@phase": "Pop",
  "enabled": true,
  "backend": "d3d12",
  "@condition_exports": ["gpu_enabled", "memory_available"]
}
```

- `backend: "d3d12"`: DirectX 12 compute (Windows). Use `"metal"` for macOS, `"vulkan"` for Linux.
- `@condition_exports`: Variables that other blocks can condition on at runtime

---

## `@moe.router`

Mixture-of-Experts router. Lives in lane 4 (control plane), kernel-implicit.

```json
"@moe.router": {
  "@lane": 4,
  "@dict": 0,
  "@kind": "atomic_block",
  "@kernel": "implicit",
  "top_k_experts": 2,
  "allow_self_modify": true,
  "persist_learning": true,
  "state_file": "models/<id>_state.bson"
}
```

- `top_k_experts`: How many experts to activate per token (2 is standard for MoE)
- `allow_self_modify`: Enables A ← A + ΔA weight updates
- `persist_learning`: Saves learned routing biases to `state_file`
- `state_file`: BSON file path for persisted router state

---

## `@skills`

Skill trigger definitions. Each skill maps to one or more experts.

```json
"@skills": {
  "@lane": 1,
  "summarize": {
    "@dict": 0,
    "@kind": "atomic_block",
    "@phase": "Wo",
    "id": "skill-summarize",
    "trigger": ["summarize", "summary", "condense", "tl;dr", "brief"],
    "routes_to": ["experts.summarization", "experts.extraction"],
    "priority": 1
  }
}
```

- `trigger`: List of keywords/phrases that activate this skill
- `routes_to`: Expert IDs this skill can activate (referenced in `@edges`)
- `priority`: Dispatch priority (lower = higher priority; 1 is highest)

---

## `@experts`

Expert compute endpoints. Each expert has a `routing_bias` initialized to 0.0.

```json
"@experts": {
  "@lane": 2,
  "summarization": {
    "@dict": 0,
    "@kind": "atomic_block",
    "@phase": "Sek",
    "id": "expert-summarization",
    "domain": "text summarization and compression",
    "routing_bias": 0.0,
    "dispatch_signal": "summarize_text"
  },
  "extraction": {
    "@dict": 1,
    "@kind": "atomic_block",
    "@phase": "Sek",
    "id": "expert-extraction",
    "domain": "key point and entity extraction",
    "routing_bias": 0.0,
    "dispatch_signal": "extract_points"
  }
}
```

- `routing_bias`: Always 0.0 for new experts — the router learns this from usage
- `dispatch_signal`: The signal name emitted when this expert is selected

---

## `@backend` (API backends only)

Required for all `api_*` backends. Declares the HTTP endpoint the bots.py will call.

```json
"@backend": {
  "type": "api_openai",
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "stream": true
}
```

Not needed for `local_gguf`, `local_scxqdds`, or `fetch_oss` backends.

---

## JSON Pointer Semantics

Within a `.xjson` file, `@`-prefixed cross-references follow JSON Pointer syntax:

| Reference | Points to |
|---|---|
| `@agent.main.learned_biases` | The `learned_biases` field in the `@agent.main` block |
| `@agent.main.memory` | The `memory` field in the `@agent.main` block |
| `@model.core` | The entire `@model.core` block |
| `@experts` | The entire `@experts` section |

---

## Naming Conventions

| Field | Convention | Examples |
|---|---|---|
| `id` (micronaut) | `[A-Z]{2,3}-[0-9]+` | `CM-1`, `SM-2`, `RM-3` |
| `id` (block) | `<type>-<name>` kebab-case | `agent-main`, `expert-code`, `skill-summarize` |
| `dispatch_signal` | `<verb>_<noun>` snake_case | `summarize_text`, `route_query`, `extract_entities` |
| `trigger` phrases | Lowercase user-facing strings | `["summarize", "condense", "tl;dr"]` |
| State file | `models/<id>_state.bson` | `models/SM-2_state.bson` |

---

## Validation Rules

1. `@edges` must contain exactly one `experts→learned` feedback edge
2. Every skill in `@skills.*.routes_to` must reference an expert defined in `@experts`
3. `routing_bias` must be `0.0` on initial creation (learned from use)
4. `@model.core.@xcfe` must be `"IMMUTABLE"` on all backends
5. API backends must have an `@backend` block; local backends must not
6. `state_file` in `@moe.router` must use the pattern `models/<id>_state.bson`
