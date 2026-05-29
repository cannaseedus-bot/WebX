# Release Audit: KUHUL.EXE.v3.0.0

**Path:** `releases/KUHUL.EXE.v3.0.0/`
**Audited:** 2026-05-28

---

## Artifacts

| File | Purpose |
|------|---------|
| kuhul_swarm_server.cpp | Swarmable micronaut HTTP server — 1276 lines C++ |
| kuhul_swarm_server.h | Header (PhaseArray, Agent, Skill, Swarm, SwarmManager) |
| agents/ | Markdown agent definitions (YAML frontmatter) |
| skills/ | Markdown skill definitions (YAML frontmatter) |
| swarms/ | Markdown swarm definitions (YAML frontmatter) |

---

## Core Innovations

### PhaseArray (π-geodesic signature)

16-byte agent fingerprint computed from content hash XOR:
- djb2 hash of markdown source
- XOR with raw content bytes at each byte position
- Euclidean distance metric across all 16 bytes
- Used for task routing: dispatch to nearest agent in phase space

```cpp
struct PhaseArray {
    uint8_t bytes[16];
    float distanceTo(const PhaseArray& other) const;
    void fromString(const std::string& content);
};
```

### Swarm Strategies

5 strategies (informational — selection is caller responsibility):
- **hierarchical** — sequential delegation top-down
- **mesh** — all agents receive task simultaneously
- **star** — single coordinator delegates
- **broadcast** — all agents receive, first response wins
- **consensus** — majority agreement required

### Swarm Coherence

```
coherence = 1 - (sum_pairwise_dist / pair_count / 255)
```

Range [0, 1]. Value of 1 = all agents have identical phase (homogeneous swarm). Value near 0 = maximum phase diversity (heterogeneous swarm). Coherence stored as a running metric per swarm; updated on `executeSwarm`.

### Agent/Skill/Swarm Instantiation from Markdown

YAML frontmatter in `.md` files defines agents/skills/swarms at runtime. Hot-reload on Windows via `ReadDirectoryChangesW`. Frontmatter schema:

**Agent:**
```yaml
---
name: MyAgent
role: coordinator
personality: methodical
skills:
  - skill-id-1
  - skill-id-2
swarms:
  - swarm-id-1
---
Agent description body...
```

**Skill:**
```yaml
---
name: SearchWeb
description: Searches the web for information
category: research
weight: 0.8
prerequisites:
  - network-access
triggers:
  - search
  - find
  - lookup
---
Skill action body / prompt template...
```

**Swarm:**
```yaml
---
name: ResearchSwarm
strategy: mesh
agents:
  - agent-id-1
  - agent-id-2
---
Swarm description...
```

### REST API Surface

| Route | Function |
|-------|---------|
| GET /api | API version + route listing |
| GET /api/agents | List all registered agents |
| GET /api/agents/{id} | Get agent detail + phase array |
| GET /api/skills | List all registered skills |
| GET /api/skills/{id} | Get skill detail + usage stats |
| GET /api/swarms | List all swarms |
| POST /api/swarms/{id} | Execute task across swarm |
| POST /api/execute | Execute single agent skill |
| POST /api/route | Route task by phase distance |
| POST /api/swarm/create | Create dynamic swarm from agent list |
| GET /api/stats | System stats (agents/skills/swarms/capabilities) |

### Hot-Reload (Windows-native, not ported)

`ReadDirectoryChangesW` watches agents/ skills/ swarms/ directories for `.md` file changes. On change, re-parses frontmatter, updates SwarmManager in place. This feature is Windows-specific and not included in the browser port.

---

## WebX-3D Port

**Target:** `src/micronaut/swarm.js`

Pure in-memory ES module — no HTTP server, no file system, no hot-reload (those remain Windows-native in the C++ binary).

Key changes from C++:
- `std::hash<std::string>` → djb2 (implementation-defined in C++, deterministic in JS)
- Euclidean distance: identical math
- `parseFrontmatter` replicates the C++ YAML parser (key:value + list blocks)
- `SwarmManager` exposes same logical API as the C++ class

**Merge target:** `src/micronaut/swarm.js`
**Exported from:** `src/index.js` (KUHUL.EXE.v3.0.0 block)

---

## Notes

- HTTP server (port 8080) and WebSocket (port 8081) ports are documented as constants but not implemented in the JS port
- `executSkill` validates agent ownership of skill before executing (matches C++ behavior)
- `createDynamicSwarm` does not persist to disk (browser/Node in-memory only)
- Swarm coherence is a heuristic metric, not a hard constraint on task dispatch
