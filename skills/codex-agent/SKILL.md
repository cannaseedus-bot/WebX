---
name: codex
description: Launch the adaptable `codex` agent configuration for generating ideas, completions, and task breakdowns; use when a task needs general-purpose coding assistance or when debugging prompt/agent flows for micronaut services.
---

# Codex Agent Skill

## When to Use
Trigger this skill whenever the request is for the general-purpose Codex agent described by the configuration below (icon `❌`, model `gemini-2.5-pro`, and tools for reading, searching, and memory). This skill runs alongside or prior to the agentic Micronaut skill when you need idea generation, completions, to-do planning, or debugging of prompt flows.

## Agent Configuration Summary
- **Identity**: `codex` with display name `Codex ❌`, an adaptable assistant that blends idea generation, code completion, and task planning.
- **Model**: `gemini-2.5-pro` at low randomness (`temperature 0.1`, `maxTimeMinutes 5`, `maxTurns 15`) so the agent stays focused.
- **Tools**: `read`, `grep`, `glob`, `web_search`, `memory`, and `read_many_files` support research, filesystem inspection, and context awareness.
- **Core responsibilities**:
  1. Read URLs/text and return informative N-grams for summaries and understanding.
  2. Suggest code completions across languages by analyzing provided snippets (`code_context` input).
  3. Generate new code ideas or approaches tied to the user’s goals.
  4. Manage to-do lists, break down features, and track progress.

## Prompt & Input Guidance
- The default `query` shell is `Help me with my coding task: {task_description}`; always bind the {task_description} to the user's request and adapt additional inputs (`code_context`, `url_or_text`, `language_framework`) as available.
- Emphasize using `web_search` + N-gram synthesis before drafting answers when a URL/text is provided.
- For code completions, analyze context deeply and return idiomatic suggestions; mention language/framework names when supplied.
- For to-do breakdowns, enumerate tasks with actionable steps and align them with the `task_description`.

## Integration Notes
- This skill can be used independently for general coding assistance, or invoked before/after the `agentic-micronaut` skill to keep architecture-level design aligned with prompts that need the executable-data paradigm.
- Keep this file synchronized with any updates to the agent configuration (icons, tools, instructions) so CLI testing and debugging always refers to the single source of truth.
- For deterministic Micronaut orchestration in AS-XCFE, use the codex handoff runner:
  - `node scripts/codex-micronaut-orchestrate.js --skill-toml <path> --command <intent-or-route> [--payload ...] [--check-sync ...]`
  - Expected output contract: `{ ok, plan, execution, validation }`.

## Dataset Preparation

- Before training a tiny prompt, LoRA adapter, or hybrid memory/router stack, run `skills/dataset-training/scripts/generate_dataset_manifest.js` (set `DATA_DIR` when your corpus is outside the repo) to generate `artifacts/test/dataset-manifest.json`.
- Use the manifest plus `artifacts/supernaut/action-log-rlhf.jsonl` to align capability comparisons (see `docs/training-guidelines.md`) before invoking LoRA/QLoRA or launching `model.pool.arbitrate` in the sandbox.
