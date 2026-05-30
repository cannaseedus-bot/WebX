---
name: micronaut-agent-factory
description: Turn micronauts into Codex-callable agents and skills with command-first wrappers, command shims, and reusable invocation patterns. Use when the user wants to wrap a micronaut as an agent, skill, command, or tool-facing workflow.
---

# Micronaut Agent Factory

Use this skill when a micronaut needs to become something Codex can call directly.

## Goal

Convert a micronaut into one or more of these forms:

- an agent wrapper for delegated execution
- a skill wrapper for repeatable procedures
- a command shim for shell-first use
- a dataset example for tool-call learning

## Workflow

1. Identify the micronaut name, entrypoint, and command surface.
2. Map each public action to one of: `agent`, `skill`, `command`, `tool`.
3. Keep the wrapper thin; preserve the micronaut's existing behavior.
4. Expose command-style invocation strings that Codex can reuse directly.
5. Add glyph-token examples when the workflow should be learned by the dataset builder.

## Conversion Rules

- Prefer one wrapper per micronaut responsibility boundary.
- Keep names short and command-oriented.
- Preserve existing CLI args, environment variables, and file paths.
- When an action is interactive, surface it as an agent.
- When an action is repeatable and procedural, surface it as a skill.
- When an action is shell-native, surface it as a command shim.

## Output Expectations

Produce:

- a skill folder with a concise `SKILL.md`
- optional command helpers or scripts
- optional dataset rows showing tool, agent, or micronaut glyph patterns

## Dataset Guidance

When preparing training data, include explicit structured rows such as:

- `{"tokens":[1,120,...]}`
- `{"tool":"Read","arg":"config.toml","text":"read the config"}`
- `{"tool":"Agent","arg":"spawn a micronaut worker"}`
- `{"tool":"Skill","arg":"call the skill wrapper"}`

Use exact token IDs for structured glyph sequences when available. Do not hash those fields if the record already supplies `tokens`.
