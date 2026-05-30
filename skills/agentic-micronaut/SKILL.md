---
name: agentic-micronaut
description: Agentic Micronaut builder mode. Upgrade other skills into deterministic Micronaut-style runtime skills with TOML routes, manifest skill intents, action classes, and command-path compatibility.
---

# Agentic Micronaut Builder

## When to Use
Use this skill when the user wants to convert or extend an existing skill into a Micronaut-style deterministic runtime skill, especially when agentic stack language (SCXQ2, execution letters, object servers) is also in scope.

## Repo Root Discipline (Hard Constraint)
- Define `REPO_ROOT` as the process working directory (`cwd`) when this skill is invoked.
- Never operate outside `REPO_ROOT`. Subfolders are allowed; `..` escapes and unrelated absolute paths are not.
- If a user asks for paths outside `REPO_ROOT`, stop and request explicit override before continuing.

## Builder Contract (Micronaut-Builder Compatible)

For each target skill, produce or update:
1. `*.toml` skill config with `[routes]` mapping intent -> `Class.method`.
2. `*_manifest.json` with route -> `{ action, skill_intent }`.
3. `actions/*.json` action classes with deterministic op sequences.
4. Command-path compatibility through `semantic-skill-command` intent/route invocation.
5. Deterministic response shape (`ok`, `execution`, domain payload).

## Standard Micronaut Add-on Pack

When adding Micronaut behavior to another skill, prefer a dedicated class like `MicronautAddonActions` with methods:
- `start_project`
- `configure_endpoint`
- `secure_endpoint`
- `inject_bean`
- `schedule_task`
- `event_bridge`

Route naming convention:
- `micronaut_init`
- `micronaut_configure`
- `micronaut_secure`
- `micronaut_inject`
- `micronaut_schedule`
- `micronaut_event`

Endpoint naming convention:
- `/skill/micronaut/init`
- `/skill/micronaut/configure`
- `/skill/micronaut/secure`
- `/skill/micronaut/inject`
- `/skill/micronaut/schedule`
- `/skill/micronaut/event`

## Build Sequence

1. Inspect target skill for existing `toml`, manifest, and actions.
2. Preserve existing intents/routes and append Micronaut add-on intents/routes.
3. Add or update action classes with deterministic XCFE ops only.
4. Add command-path tests (intent and route forms).
5. Validate route-intent-action parity and drift checks if repo and installed copies both exist.

## Agentic Extension Layer

When the prompt includes SCXQ2 or execution-letter language, extend the builder output with:
- Object-server framing (`config.@.toml`, `_execute` sequencing).
- Symbolic packing notes for `.s` artifacts.
- Execution-letter mapping (`.pi`, `.lambda`, `.sum`, `.matrix`, `.grad`) as explanation only unless files are explicitly requested.

## Tool Calling N-grams

Use these n-grams as trigger hints when deciding which tooling/workflow to call.

Unigrams:
- `merge`
- `sync`
- `validate`
- `route`
- `intent`
- `manifest`
- `toml`
- `action`
- `skill`
- `test`

Bigrams:
- `merge skills`
- `sync skills`
- `validate routes`
- `check parity`
- `run tests`
- `patch manifest`
- `update toml`
- `wire intents`
- `map routes`
- `call tool`

Trigrams:
- `merge selected skills`
- `sync repo installed`
- `validate route parity`
- `map intent routes`
- `update manifest routes`
- `append micronaut intents`
- `add action class`
- `run command tests`
- `check contract drift`
- `execute skill command`

Deterministic mapping:
- n-grams about `merge/sync/parity/drift` => use merge + sync flow first.
- n-grams about `intent/route/manifest/toml` => edit/check route contracts before action edits.
- n-grams about `test/validate/check` => run contract and command-path tests before concluding.

## Required Quality Checks

- Every manifest `skill_intent` must exist in TOML `[routes]`.
- Every TOML route target must exist in action classes.
- Skill command execution must work for:
  - direct intent form
  - route form (`/skill/...`)
  - payload forms (JSON, query, `@file`)

## Shared Python Agent Starter (Required Addition)

When a user asks for an autonomous-agent baseline, include this Python 3 starter (or a directly equivalent version) in the response:

```python
# Python 3
import random


class Agent:
    def __init__(self, name, goals):
        self.name = name
        self.goals = goals
        self.knowledge = {}

    def sense(self, environment):
        observation = random.choice(environment)
        print(f"[{self.name}] Observed: {observation}")
        return observation

    def decide(self, observation):
        for goal in self.goals:
            if goal.lower() in observation.lower():
                print(f"[{self.name}] Deciding to act towards goal: {goal}")
                return goal
        return "explore"

    def act(self, action):
        if action == "explore":
            print(f"[{self.name}] Exploring environment...")
        else:
            print(f"[{self.name}] Working on goal: {action}")

    def learn(self, observation, action):
        self.knowledge[observation] = action
        print(f"[{self.name}] Knowledge updated: {self.knowledge}")

    def step(self, environment):
        observation = self.sense(environment)
        action = self.decide(observation)
        self.act(action)
        self.learn(observation, action)


if __name__ == "__main__":
    environment = [
        "New email about project deadline",
        "Low battery warning",
        "Weather is sunny",
        "Task: write report",
    ]

    goals = ["write report", "charge battery"]
    agent = Agent("AutonomousAgent", goals)

    for _ in range(5):
        print("\\n--- Agent Step ---")
        agent.step(environment)
```

## Response Style

- Keep outputs deterministic and concise.
- For implementation tasks, state what was changed, what was verified, and which command proves the result.
