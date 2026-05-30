---
name: micronaut-builder-from-mx2
description: Build new Micronaut bundles from the locked MX-2 example as a read-only base, without editing the source bundle itself.
---

# Micronaut Builder from MX-2

Use this skill when you need to create a new Micronaut bundle that follows the MX-2 layout, registry shape, and evolution-engine conventions, but must not modify the locked MX-2 source tree.

## Hard rule

- Treat `C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18\MX-2` as read-only.
- Do not edit, rewrite, or relocate files inside that folder.
- Build only into a new target folder selected by the user.

## Workflow

1. Read the MX-2 bundle as the base example.
2. Derive a new target name and output root.
3. Copy the structural patterns: `folds.toml`, `micronaut.registry.xjson`, `model.runtime.json`, `bots.py`, `evolution/`, and `agents/`.
4. Rewrite references so the new bundle points at its own files only.
5. Keep the builder deterministic and path-based.

## What to generate

- a new bundle folder with the same directory shape
- a thin builder script or shim
- a manifest that records the copied source paths
- optional dataset rows for tool-call learning

## Notes

- Preserve MX-2 semantics as an example, not as an editable dependency.
- If a file is locked for system specs, reference it, mirror it, or wrap it, but never mutate it.
- Prefer clear, command-oriented names for the new bundle.
