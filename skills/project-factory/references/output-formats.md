# Output Formats Reference

Field-level documentation for all files the project-factory generates.

---

## CLAUDE.md

```markdown
# CLAUDE.md

## What This Project Is
# One paragraph. What the project does, who uses it, what technology powers it.
# Be specific — this is the first thing any agent reads in this project.

## Stack
# Bullet list: languages, frameworks, key libraries WITH versions.
# e.g. "- Node.js 20.x (ESM only)" not just "- Node.js"

## Commands
### Install
<exact command>
### Test
<exact command>
### Build
<exact command, if applicable>
### Run
<exact command, if applicable>

## Architecture
# Directory map with one-line descriptions.
# Format:
#   dir/           — what lives here
#   dir/subdir/    — more specific
# Only include dirs that an agent would need to navigate.

## Key Files by Responsibility
| File/Module | Responsibility | Size (approx) |
|---|---|---|
| path/to/file | What it does | N LOC |

## Conventions
# Rules that are NOT obvious from the code:
# - Naming: PascalCase for X, snake_case for Y
# - Imports: always use X style
# - Error handling: throw vs return
# - Testing: where tests live, what runner
# - Commit style: if non-standard

## Important Gotchas
# Non-obvious things that would trip up an agent:
# 1. "This project uses X but you should never do Y"
# 2. "config.toml must be edited before running tests"
# 3. "File X is generated — don't edit it directly"
```

---

## skill.matrix.toml

```toml
[skill]
name        = "my-skill"          # hyphen-case, matches folder name
version     = "1.0.0"             # semver, bump when actions change
description = "..."               # full description + trigger phrases
runtime     = "json.exe"          # always json.exe for spec skills
shard_checkpoint = false          # true only if skill reads .scxqdds
category    = "meta|domain|tool"  # meta=describes stack, domain=project area, tool=utility
excluded_agents = []              # list of agent names that should NOT load this skill

[actions.<name>]
entry       = "ClassName.method_name"   # must match method in *Actions.json
runtime     = "json.exe"
description = "What this action returns (used by agents to decide when to call it)"
resources   = { memory = "16MB|32MB", timeout = "2s|3s|5s", network = false|true, filesystem = "none|readonly|readwrite" }
```

**Action naming conventions:**
- `spec` — full spec dump (always present)
- `glossary` — file/doc index (always present)
- `stack` — tech stack details
- `agents` — agent roster
- `architecture` — directory and data-flow
- `models` — model contracts
- `<noun>_spec` — specific subsystem spec

---

## *Actions.json

```json
{
  "class": "ClassName",
  "methods": {
    "method_name": {
      "params": [],           // [] for no params, or [{"name":"x","type":"string"}]
      "body": [{
        "op": "json",         // always "json" for spec/glossary methods
        "value": { ... }      // the actual data — must be real project data, no placeholders
      }]
    }
  }
}
```

**op types:**
- `"json"` — return a JSON value directly
- `"read"` — read a file: `{"op":"read","path":"relative/path"}`
- `"exec"` — run a command: `{"op":"exec","cmd":"node script.js"}`

**value structure for `spec`:**
```json
{
  "ok": true,
  "project": "my-project",
  "version": "1.0.0",
  "purpose": "One sentence description",
  "stack": {
    "language": "Node.js 20",
    "framework": "Express 4"
  },
  "architecture": {
    "src/": "Application source",
    "tests/": "Test suite"
  },
  "agents": [
    {"name": "agent-name", "role": "what it does", "file": ".claude/agents/agent-name.md"}
  ],
  "key_files": {
    "src/index.js": "Entry point",
    "src/router.js": "Request routing"
  }
}
```

**value structure for `glossary`:**
```json
{
  "ok": true,
  "description": "All key files in this project indexed by category",
  "categories": {
    "source": [
      {"path": "src/index.js", "title": "Entry Point", "description": "Starts the server"}
    ],
    "config": [
      {"path": "config.toml", "title": "App Config", "description": "Runtime settings"}
    ],
    "docs": [
      {"path": "docs/API.md", "title": "API Reference", "description": "All endpoints"}
    ]
  },
  "quick_lookup": {
    "understand the routing": "src/router.js",
    "configure the app": "config.toml",
    "add a new endpoint": "src/router.js → docs/API.md"
  }
}
```

---

## .claude/agents/<name>.md

```markdown
---
name: agent-name              # hyphen-case, matches filename
description: "..."            # CRITICAL: this controls when the agent triggers
model: sonnet|opus|haiku      # sonnet for most, opus for deep reasoning, haiku for fast
color: blue|green|purple|orange|red|yellow
---
```

**description field format (most important field):**
```
"Use this agent when <condition>. Trigger on: '<phrase1>', '<phrase2>', '<phrase3>'

<example>
user: \"Realistic message a developer would type\"
assistant: agent-name does X, reads Y, produces Z.
</example>

<example>
user: \"Another realistic message\"
assistant: agent-name does A, B, C.
</example>"
```

Rules for description:
- Must include at least 3 distinct trigger phrases
- Examples must use actual file names and operations from the project
- Never generic — "helps with coding" is useless

---

## model/agents/<name>/model.json (xcfe-model-1 schema)

```json
{
  "schema": "xcfe-model-1",
  "id": "model-name",
  "displayName": "Human Name",
  "coordFrame": "grid",
  "zLayer": 0.5,              // 0.0-1.0, higher = closer to UI surface
  "fold": "COMPUTE_FOLD",     // COMPUTE|STORAGE|META|UI
  "micronaut": "CP-1",        // micronautId for this model
  "description": "...",
  "scxGraph": {
    "nodes": [],              // node definitions with id, type, zCoord, role
    "arcs": []                // geodesic-entropy-arcs between nodes/folds
  },
  "architecture": {
    "type": "dense_attention|moe_sparse",
    "moe": false,
    "hiddenDim": 2048,
    "numHeads": 16,
    "numLayers": 4,
    "contextWindow": 2048,
    "vocabSize": 32768,
    "dtype": "fp16|fp32|int8",
    "attention": "full_causal|sparse"
  },
  "training": {
    "phases": [],
    "datasets": []
  },
  "clearanceLayer": {
    "fold": "COMPUTE_FOLD",
    "description": "what this model is allowed to read/write"
  }
}
```

**zLayer by fold:**
- `COMPUTE_FOLD` → 0.5
- `STORAGE_FOLD` → 0.3
- `META_FOLD`    → 0.7
- `UI_FOLD`      → 0.9

---

## model/agents/<name>/model.runtime.json

```json
{
  "schema": "xcfe-runtime-1",
  "id": "model-name",
  "port": 3200,               // unique port, check existing models to avoid collision
  "lane": "PROMPT",           // PROMPT|COMPUTE|STORAGE|META
  "fold": "COMPUTE_FOLD",
  "computeProfile": {
    "dtype": "fp16",
    "contextWindow": 2048,
    "temperatureBase": 1.0,
    "topP": 0.9,
    "maxNewTokens": 1024,
    "hlslEntry": "CS_ComputeFold",
    "textureRegisters": "t0-t7",
    "cm1Gate": "0x0001"
  },
  "foldInputs": {
    "t0": "description of what arrives at t0",
    "t1": "description of what arrives at t1"
  },
  "foldOutputs": {
    "u0": "primary output description",
    "u1": "secondary output description"
  },
  "verifierRules": [
    "V0 — input validation rule",
    "V1 — output constraint"
  ]
}
```

**port assignment by fold (reference ranges):**
- `COMPUTE_FOLD` → 3100–3139
- `STORAGE_FOLD` → 3140–3149
- `META_FOLD`    → 3150–3159
- `UI_FOLD`      → 3160–3169

**hlslEntry by fold:**
- `COMPUTE_FOLD` → `CS_ComputeFold`
- `STORAGE_FOLD` → `CS_StorageFold`
- `META_FOLD`    → `CS_MetaFold`
- `UI_FOLD`      → `CS_UIFold`

---

## Codex SKILL.md

```markdown
---
name: skill-name              # hyphen-case
description: "Full description. Include: what it does, when to use it, trigger phrases."
metadata:
  short-description: Max 8 words    # shown in Codex skill chip UI
---

# Skill Title

## Purpose
<what this skill enables, project-specific>

## Operations
### <op name>
<instructions>

## Project Context
<project-specific file paths, schemas, conventions>
```

## Codex agents/openai.yaml

```yaml
interface:
  display_name: "Human Readable Name"       # shown in Codex UI
  short_description: "One sentence, 80 char max"
  default_prompt: "Example user message"    # pre-filled when user picks this skill
```
