---
name: micronaut-skill
description: Deterministic Micronaut-style actions for project init, endpoint config, security checks, DI injection, scheduling, and event emission.
---

# Micronaut Skill

Use this skill when you need Micronaut-flavored workflows executed via deterministic XCFE actions:
- `startProject` — initialize project metadata from `config/application.json`.
- `configureEndpoint` — update gateway path/method.
- `secureEndpoint` — check role authorization.
- `injectBean` — simulate DI bean availability/scope.
- `scheduleTask` — set cron/task pair.
- `eventBridge` — emit events to registered listeners.
- `toolDiResolve` / `toolDiInject` — deterministic DI planning and injection handles.
- `toolStateGet` / `toolStateSet` — deterministic state snapshots and writes.
- `toolScheduleTask` / `toolEventEmit` — deterministic scheduling and event emissions.
- `toolRuntimeStep` / `toolRuntimeRun` — deterministic cognition tick controls.
- `toolClusterMigrate` / `toolClusterReplicate` — deterministic cluster migration and replication envelopes.
- Runtime tools are wired to a mini-model handler via `model.dispatch` (manifest-backed summary) for hybrid handler mode.

Routes are defined in `micronaut_manifest.json`; actions live in `actions/MicronautActions.json`. Data root is `config/` for this skill.
