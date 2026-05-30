# Agent Trigger Guide

How to write `description` fields that actually trigger correctly.

---

## The Problem

The `description` field in a `.claude/agents/<name>.md` file is the ONLY thing Claude Code reads to decide whether to invoke this agent. If it's vague, the agent never fires. If it's wrong, the agent fires at the wrong time.

---

## The Formula

```
"Use this agent when <condition describing the work domain>.

Trigger on: '<phrase1>', '<phrase2>', '<phrase3>', '<phrase4>'

<example>
user: \"<realistic message a developer in this project would type>\"
assistant: <agent-name> <what it does — specific files, operations, outputs>.
</example>

<example>
user: \"<another realistic message>\"
assistant: <agent-name> <what it does>.
</example>"
```

---

## Trigger Phrase Rules

**Must be phrases a real developer would say**, not capability descriptions:

| Bad (capability description) | Good (developer phrase) |
|---|---|
| "SQL database queries" | "query the database", "write a SQL query", "check what's in the db" |
| "CSS styling" | "fix the layout", "change the button color", "make the sidebar responsive" |
| "error handling" | "the app is crashing", "trace this error", "why is it throwing" |
| "HLSL shader files" | "write a compute shader", "edit kuhul.hlsl", "add a GPU kernel" |

**Include at least 3, ideally 5–8.** More specific = better targeting.

**Include the inverse** — what the user says when they DON'T know which agent to use:
- "what do I do if the GPU shader crashes"
- "help me figure out the routing"

---

## Example Quality

Examples must be:
1. **Realistic** — something an actual developer on this project would type
2. **Specific** — reference actual files, systems, or outputs from this project
3. **Show the agent's value** — why is this agent better than just asking directly?

**Bad example:**
```
user: "Help me with the database"
assistant: sql-skill helps with database queries.
```

**Good example:**
```
user: "How many active sessions are in the inference_recall table?"
assistant: sql-skill reads trainer/inference_recall.db, runs SELECT COUNT(*) FROM sessions WHERE status='active', returns result with column schema context.
```

---

## Color Guide

| Color | Use for |
|---|---|
| `blue` | Data, storage, database agents |
| `green` | Build, test, CI/CD agents |
| `purple` | Orchestration, coordination, meta agents |
| `orange` | UI, frontend, shader, visual agents |
| `red` | Security, validation, audit agents |
| `yellow` | Documentation, spec, glossary agents |

---

## Model Guide

| Model | Use when |
|---|---|
| `sonnet` | Default — most tasks (code, analysis, generation) |
| `opus` | Deep reasoning, architecture decisions, complex refactors |
| `haiku` | Fast lookups, simple queries, high-frequency tasks |

---

## Full Example — Good Agent

```markdown
---
name: kuhul-shader
description: "Use this agent when working with HLSL compute shaders in the KUHUL GPU runtime. Handles writing, editing, and debugging kuhul.hlsl, kuhul.css.hlsl, kuhul.html.hlsl, and domain-specific shader files. Trigger on: 'edit the shader', 'add a glyph opcode', 'write a compute kernel', 'fix the fold dispatch', 'kuhul.hlsl', 'GPU shader', 'HLSL', 'domain shader', 'fold kernel', 'add to glyph table', 'KPI1 bytecode'\n\n<example>\nuser: \"Add a new glyph opcode for SVG gradient operations\"\nassistant: kuhul-shader reads gpu/kuhul.hlsl, identifies the glyph constant block (lines 20-35), adds GLYPH_SVG_GRADIENT = 0x30, updates the COMPUTE_FOLD dispatch branch to route it to expert nodes 200-299, verifies it matches the expected KPI1 sub-category field.\n</example>\n\n<example>\nuser: \"The CSS shader isn't dispatching spacing ops to STORAGE_FOLD\"\nassistant: kuhul-shader reads gpu/kuhul.css.hlsl, traces the spacing_op branch (Sek fold target), checks the FDISPATCH_STORAGE write, cross-references grammar/scx-atomic-css.ebnf for the expected scale_value encoding.\n</example>"
model: sonnet
color: orange
---
```

---

## Anti-Patterns to Avoid

1. **Generic role** — "assists with development tasks" — fires everywhere, useful nowhere
2. **No examples** — Claude Code can't infer the agent's actual value
3. **Examples using placeholder paths** — "reads path/to/file" — useless without real paths
4. **Only one trigger phrase** — misses too many real invocations
5. **Overlapping triggers with another agent** — causes ambiguous routing; make them distinct
6. **model: opus for everything** — opus is slow; only use for genuinely deep reasoning tasks
