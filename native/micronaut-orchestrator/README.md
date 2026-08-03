# Micronaut Orchestrator TUI

This Rust crate is the native terminal UI option for the MX2LM Micronaut control plane.

## Current scope

- Talks to the MX2LM Flask inference/control server at `http://127.0.0.1:8000`
- Reads `/health`, `/micronaut/status`, and `/chat`
- Provides a lightweight TUI for chat/control, status, and result summary

## Build

```powershell
scripts\launch-micronaut-orchestrator.ps1 -Build
scripts\launch-micronaut-orchestrator.ps1
```

## Environment

- `MX2LM_API_BASE` overrides the API base URL

This is intentionally a lean, buildable first pass that matches the current MX2LM Micronaut API instead of assuming every speculative service already exists.
