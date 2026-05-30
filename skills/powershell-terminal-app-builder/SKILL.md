---
name: powershell-terminal-app-builder
description: Build and launch Micronaut-style PowerShell terminal and app-builder flows using the locked KUHUL/MX2 entrypoints without rewriting them.
---

# PowerShell Terminal / App Builder

Use this skill when the task needs the PowerShell-based terminal, chat app builder, or MX2 integration flow exposed by the locked Micronaut bundle.

## Read-only base

- Treat `C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18` as the base example tree.
- Do not edit the base files directly.
- Use the PowerShell entrypoints as invocation surfaces only.

## Primary entrypoints

- `C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18\micronaut-ui-chat-app.ps1`
- `C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18\mx2_integration.ps1`
- `C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18\kuhul.svg`
- IDE/code-editor mode is supported through a separate launch target that opens the local editor/terminal surface.

## What this skill does

- Launches PowerShell terminal/app-builder flows.
- Bridges the KUHUL SVG identity asset into the UI workflow.
- Keeps app-builder behavior command-oriented and local.
- Preserves the base bundle as a locked system specification.

## Usage pattern

1. Choose the entrypoint.
2. Pass through the user’s command arguments unchanged.
3. Keep the workflow local and deterministic.
4. If a new app should be built, do it in a separate target folder.

## Boundary rule

- The base bundle is a reference implementation, not an editable project.
- Any derived app-builder work must write elsewhere.
