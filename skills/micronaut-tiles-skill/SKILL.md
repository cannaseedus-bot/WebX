---
name: micronaut-tiles-skill
description: Query Micronaut tile retrieval and constrained local LLM inference for coding-focused assistance. Use when looking up tiles, asking tile-backed questions, or running the constrained tile API server.
---

# Usage
- Run the API server (already running at http://localhost:5003 by default).
- From the skill folder run: `node run.js --query "your question" [--tile tile_001] [--mode coding|default]`

# Endpoints
- POST /api/infer { query, tile?, mode? } — returns { ok, tile, score, reply, source }

# Notes
- The skill prioritizes tiles whose tile.meta.json lists the 'code' lane when mode=coding.
- The LLM is constrained with a system prompt to avoid hallucination; it will reply "I don't know." when context lacks an answer.
- Suitable as a lightweight agent for code lookups, examples, and minimal code completions.
