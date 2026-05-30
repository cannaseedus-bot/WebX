---
name: mx2lm-local-agent
description: Use the local MX2LM model server, command registry, and Micronaut-style routes to run agent, skill, and command actions. Use when working with the DX11 local model, command panels, Micronaut bridge routes, or skill wrappers that should call the local server instead of remote APIs.
---

# MX2LM Local Agent

Use this skill when the task should run through the local DX11 model stack.

## Core workflow

1. Read the command manifest from `C:\public_html\models\gpt2_medium_dx11\micronaut_commands.json`.
2. Prefer the designed UI at `C:\public_html\models\gpt2_medium_dx11\MX2LM\brain\index.html`.
3. Route model prompts through `C:\public_html\models\gpt2_medium_dx11\server.py`.
4. Use local agent or skill wrappers for repeatable actions before inventing new ones.

## What to use

- `server.py` for local chat, command routing, and model-backed responses.
- `micronaut_commands.json` for launch paths and callable entries.
- `micronaut_commands.index.json` for grouped helpers, bots, and specs.
- `MX2LM/brain/index.html` for the designed command-and-skill UI.

## Rules

- Keep wrappers thin and command-oriented.
- Preserve local file paths and manifest-backed routes.
- Treat `powershell_src`, helper scripts, and bot specs as source entries, not launch commands.
- Use the local model first; fall back only when the local server is unavailable.

## Agent actions

Use an agent when the task is interactive or requires stepwise reasoning over local files.
Use a skill when the task is repeatable, like launching commands, querying the manifest, or invoking the local model with a fixed prompt frame.

