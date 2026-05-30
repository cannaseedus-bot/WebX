---
name: sk-coordinator
description: Coordinate learning plans, multi-call orchestration, scheduling, and Micronaut DI injection via semantic planner/scheduler/DI actions.
---

# Semantic Kernel Coordinator

Purpose: high-level intent router that:
- builds learning plans (`SemanticPlannerActions.generate_learning_plan`)
- assembles multi-step orchestration plans (`SemanticPlannerActions.build_multi_step_plan`)
- schedules learning (`LearningSchedulerActions.schedule_learning`)
- injects Micronaut components (`MicronautDIActions.inject_component`)

Routes live in `sk-coordinator_manifest.json`; skill config is `sk-coordinator.toml`; actions are under `actions/`.

Use when you need structured goal planning, multi-call choreography, or DI wiring for Micronaut contexts. Keep calls deterministic and reference the intentions in `examples/intentions/intentions.yaml`.
