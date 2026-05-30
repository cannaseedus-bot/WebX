---
name: team-lead
description: Orchestrate multiple specialized agents, maintain a task board, assign subtasks, and synthesize results. Use when coordinating parallel work or dispatching agents to complete a complex goal.
---

# Team Lead Skill

**ID**: `team-lead`
**Category**: orchestration
**Entry**: `TeamLeadActions.assign_tasks`

---

## Overview

Team Lead is a meta-orchestration skill that coordinates multiple specialized agents to complete complex goals collaboratively. It maintains a live task board, assigns agents to subtasks, dispatches work in parallel or sequentially based on dependencies, and synthesizes all outputs into a final result.

## Actions

| Route | Action | Params | Description |
|---|---|---|---|
| `/skill/team-lead/assign` | `assign_tasks` | `goal`, `constraints` | Decompose goal into task board |
| `/skill/team-lead/dispatch` | `dispatch_agents` | `tasks`, `parallel` | Dispatch agents to assigned tasks |
| `/skill/team-lead/board/update` | `update_board` | `taskId`, `status`, `agent`, `result` | Update a task's status on the board |
| `/skill/team-lead/board/get` | `get_board` | — | Return current board state |
| `/skill/team-lead/synthesize` | `synthesize` | `results`, `goal` | Merge all agent outputs into final summary |
| `/skill/team-lead/task/complete` | `complete_task` | `taskId`, `artifact` | Mark a task complete with artifact |
| `/skill/team-lead/task/block` | `block_task` | `taskId`, `blockedBy`, `reason` | Mark a task blocked |

## Agent Pool

Team Lead can delegate to any agent in the AS-XCFE pool:

- `workflow-orchestrator` — planning, roadmaps, todo generation
- `supernaut` — unified skill routing surface
- `asx-verifier` — ASX envelope/contract verification
- `windows-sdk` — DirectX 12, DXGI, WinRT headers
- `vs-native-tools` / `vs2022-tools` / `vs2019-tools` — MSVC builds
- `mx2lex` — lexer grammar compilation
- `scxqdds` / `scxq2-vector` / `scx2-runtime` — native tensor pipeline
- `netfx-sdk` — .NET CLR interop
- `powershell-ise` — PowerShell scripts
- `sk-coordinator` / `micronaut` — semantic kernel, DI, skill registry

## Task Board Format

```
═══════════════════════════════════════════════
 TASK BOARD  |  Goal: <description>
═══════════════════════════════════════════════
 [✓]  #1  <completed task>     workflow-orchestrator
 [→]  #2  <active task>        windows-sdk
 [ ]  #3  <pending task>       vs-native-tools
 [!]  #4  <blocked — on #2>    asx-verifier
═══════════════════════════════════════════════
 Progress: 1/4  |  2 in flight
═══════════════════════════════════════════════
```

## Collaboration Patterns

- **Fan-out**: dispatch all independent tasks simultaneously
- **Pipeline**: feed one agent's output as the next agent's input
- **Hybrid**: mix of parallel + sequential based on the dependency graph
