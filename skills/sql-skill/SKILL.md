---
name: sql-skill
description: Deterministic SQL skill for SQLite and SQL Server planning/execution flows. Use when you need SK-callable SQL actions like connect, list tables, inspect schema, and run parameterized queries across sqlite or sqlserver providers.
---

# SQL Skill

Use this skill for deterministic SQL operations with provider-scoped behavior:
- `connect` - resolve connection profile for `sqlite` or `sqlserver`.
- `listTables` - return deterministic table inventory by provider.
- `describeTable` - return stable schema metadata for a table.
- `query` - return structured query execution envelope (deterministic mock rows).
- `upsert` - return deterministic write plan/result envelope.

Routes are defined in `sql_manifest.json`; actions live in `actions/SqlActions.json`; provider profiles are in `config/sql_profiles.json`.
