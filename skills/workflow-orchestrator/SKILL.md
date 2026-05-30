---
name: workflow-orchestrator
description: Break down complex projects into structured tasks and phases, generate milestones and to-do lists, analyze scope/dependencies, research tools, automate multi-step workflows, and scaffold Kuhul micro-applications. Use for project planning, workflow automation, and Kuhul scaffolding.
---

# Workflow Orchestrator

## When to Use
Use this skill whenever the user needs structured planning for a large project, automation of a multi-step workflow, or Kuhul micro-app scaffolding. Always consult `references/workflow-orchestrator.md` for the detailed agent spec, and load memory from `C:\Users\canna\.claude\agent-memory\workflow-orchestrator\` to capture persistent patterns (tools, Kuhul templates, automation heuristics).

## Phase-Based Brain
The skill reasons in five deterministic phases:

1. **Scope & Analysis**
   - Capture requirements, constraints, MVP vs future scope, and success criteria.
   - Use the `PROJECT ANALYSIS` section to document scope, owner expectations, and risk/constraint boundaries.

2. **Research & Discovery**
   - Consult `references/workflow-orchestrator.md` for tool recommendations and best practices.
   - Run `web_search` when the user requests external knowledge to back tool choices.
   - Record findings in `# RESEARCH FINDINGS` with links and evaluation notes.

3. **Strategic Planning**
   - Break the roadmap into phases (foundation, MVP, advanced, scale).
   - Include deliverables, dependencies, and milestone success criteria.
   - Describe the `# PROJECT PHASES` section with titles, outcomes, and duration estimates.

4. **Task Generation**
   - Generate phase-aligned to-do lists with effort estimates (small/medium/large/epic), acceptance criteria, blockers, and automation checklists.
   - Populate `# TASK BREAKDOWN` with actionable tasks following the structure in `references/workflow-orchestrator.md`.

5. **Automation & Kuhul Integration**
   - Identify automation opportunities and propose Kuhul micro-app scaffolds (config.toml, schema.json, processor, validators, error handlers, README).
   - Document these in `# AUTOMATION & KUHUL ARCHITECTURE`.

## Output Discipline
- Always structure responses using the headings specified in `references/workflow-orchestrator.md` (`PROJECT ANALYSIS`, `RESEARCH FINDINGS`, etc.).
- Cite the persistent memory path when referencing prior insights (e.g., “As noted in `workflow-orchestrator/memory.md`, combat mitigation steps include...”). Update memory files when you uncover a new reusable pattern or automation strategy.
- Include dependencies and critical path information in `# DEPENDENCIES & CRITICAL PATH`, using light ASCII diagrams if it clarifies parallelism.
- Provide a `# TIMELINE ESTIMATE` with phase durations and overall delivery time.

## Deep Thinking Guidance
- Frame research tasks as discrete investigations with deterministic commands (e.g., `npm info` to inspect package versions, `curl` to validate API endpoints).
- When proposing Kuhul scaffolds, briefly describe the data flow (input → processing → output), validation, and error-handling sections to help the user visualize the micro-app’s atomic brain.
- Always ask a clarifying question if requirements lack constraints, otherwise lock onto the plan and avoid speculative suggestions.
