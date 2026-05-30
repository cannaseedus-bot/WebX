---
name: bot-builder
description: Build and wire bot pipelines around C:\\public_html\\models\\gpt2_medium_dx11\\bot_helpers, including orchestrator pipelines, helper method selection, and the local HTTP run endpoint. Use when the user wants to create, extend, or invoke a bot-builder workflow from the DX11 bot helper bundle.
---

# Bot Builder

Use this skill when working with `C:\public_html\models\gpt2_medium_dx11\bot_helpers`.

## What It Covers

- Map bot goals to helper categories and methods
- Build pipelines for the local orchestrator
- Invoke the HTTP run endpoint
- Extend helper implementations without breaking the pipeline shape

## Core Files

- `bot_helpers/core.cjs` for helper categories and composition patterns
- `bot_helpers/bot_impls.cjs` for actual helper methods
- `bot_helpers/orchestrator.cjs` for pipeline composition and execution
- `bot_helpers/server.cjs` for the local HTTP endpoint

## Workflow

1. Decide the bot category first: `code`, `data`, `comm`, `automation`, or `learning`.
2. Pick the smallest helper method that fits the goal.
3. Build a pipeline when multiple helpers must run in sequence.
4. Keep the payload shape simple: `pipeline`, `create`, `bots`, `input` or `payload`.
5. Prefer extending `bot_impls.cjs` over adding new orchestration branches.

## Pipeline Patterns

- `pipeline`: one bot feeds the next
- `parallel`: independent bots fan out and then aggregate
- `conditional`: choose a helper based on input state
- `ensemble`: vote across multiple bots

## HTTP Usage

POST to `http://127.0.0.1:5780/run` with JSON like:

```json
{
  "pipeline": "example",
  "create": true,
  "bots": [
    { "name": "lint", "category": "code", "method": "lint", "params": { "language": "javascript" } }
  ],
  "input": { "code": "const value = 1;" }
}
```

## Helper Script

- `scripts/run_bot_pipeline.cjs` posts a sample pipeline to the local server
- Set `BOT_HELPERS_PORT` if the server is not on `5780`

## Extension Rules

- Keep helper methods deterministic when possible
- Return small JSON objects that are easy to chain
- Preserve the existing method names in `bot_impls.cjs`
- Do not change the orchestrator contract unless the whole pipeline shape changes
