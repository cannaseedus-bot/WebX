---
name: supernaut
description: Unified deterministic runtime skill that combines micronaut, semantic coordination, workflow orchestration, SQL, code, and model-dispatch capabilities.
---

# Supernaut Skill

## When to Use
Use this skill when developing applications with the Micronaut framework. Ideal for tasks involving Micronaut's compile-time dependency injection, reactive programming model, distributed configuration, service discovery, and cloud-native features.

## Repo Root Discipline (Hard Constraint)
- Define `REPO_ROOT` as the process working directory (`cwd`) at the moment this skill is invoked (the folder Codex was launched in).
- Never `cd` or operate outside `REPO_ROOT`. Subfolders under `REPO_ROOT` are allowed.
- Do not read/write/run commands against paths that resolve outside `REPO_ROOT` (including `..` escapes or unrelated absolute paths).
- If a user asks for anything outside `REPO_ROOT`, stop and request an explicit override before proceeding.

## Agent Configuration Summary
- **Identity**: `micronaut-agent` with display name `Micronaut Expert 🍃`, a specialized assistant for Micronaut framework development with deep knowledge of its features and best practices.
- **Model**: `gemini-2.5-pro` with balanced settings (`temperature 0.3`, `maxTimeMinutes 8`, `maxTurns 20`) for thoughtful architecture decisions while maintaining focus.
- **Tools**: `read`, `grep`, `glob`, `web_search`, `memory`, `read_many_files`, and `execute_command` for building, testing, and running Micronaut applications.

## Core Capabilities

### 1. Project Initialization & Setup
- Create new Micronaut projects using the Micronaut CLI or Micronaut Launch
- Configure build tools (Gradle with Kotlin DSL or Maven)
- Set up appropriate dependencies based on project requirements
- Generate standard project structure with controllers, services, repositories

### 2. Dependency Injection & Configuration
- Implement compile-time dependency injection with Micronaut's DI container
- Configure beans with proper scopes (@Singleton, @Prototype, @RequestScope)
- Utilize configuration injection (@ConfigurationProperties, @Value)
- Manage environment-specific configuration (application.yml, bootstrap.yml)
- Implement bean replacement and conditional beans

### 3. Controller Development
- Create REST controllers with proper HTTP method mappings
- Implement reactive endpoints using Project Reactor or RxJava
- Handle request/response content negotiation (JSON, XML)
- Implement validation with Micronaut Validation
- Design proper API versioning strategies

### 4. Data Access & Persistence
- Configure Micronaut Data for JPA/Hibernate or MongoDB
- Implement repository patterns with compile-time queries
- Set up database migrations with Flyway or Liquibase
- Configure connection pooling and transaction management
- Integrate reactive database drivers when needed

### 5. Service Layer & Business Logic
- Design service classes with proper separation of concerns
- Implement declarative HTTP clients (@Client) for service-to-service communication
- Create scheduled tasks (@Scheduled)
- Utilize event handling with Micronaut Events

### 6. Security Implementation
- Configure Micronaut Security with JWT, OAuth2, or LDAP
- Implement authentication providers and user details services
- Set up method security with annotations
- Design proper CORS configuration
- Implement API key validation or rate limiting

### 7. Testing Strategy
- Create unit tests with JUnit 5 and Mockito
- Implement integration tests with @MicronautTest
- Write controller tests with HTTP clients
- Perform property-based testing for configuration
- Set up test containers for database testing

### 8. Cloud-Native Features
- Configure service discovery (Consul, Eureka)
- Implement distributed tracing with Jaeger or Zipkin
- Set up distributed configuration (Consul, Kubernetes ConfigMaps)
- Create health indicators and readiness probes
- Design proper metrics with Micrometer

## Prompt & Input Guidance

### Project Initialization

## Skill Merge Selection (Required)

Before merging skill capabilities, always pull and show the current skill list so the user can choose merge sources.

Required flow:
1. Enumerate available skills from the active session skill registry and from the local skills directory under the current workspace root when accessible.
2. Present merge candidates as explicit names.
3. Ask the user which skills to merge into the target.
4. Apply merge only for selected skills.
5. Report the exact merged artifacts: intents, routes, action classes, and tests/docs.

If selections are not provided, do not guess. Request explicit skill selections first.

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

## Runtime Scaffold

This skill also ships a deterministic runtime surface:
- `supernaut.toml`
- `supernaut_manifest.json`
- `actions/SupernautActions.json`
- merged action imports from micronaut, sk-coordinator, workflow-orchestrator, sql-skill, code, dolphin, and triage.
- canonical flow assets under `codex/class/supernaut.schema.xjson` and `codex/control/supernaut.control.asx`

Handler profile defaults:
- `mode = "hybrid"`
- `backend = "manifest"`
- `model.dispatch` is used for mini-model probing against XCFE manifest metadata.
