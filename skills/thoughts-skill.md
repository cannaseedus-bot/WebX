# Thoughts Skill

Name: thoughts-skill

Description: Provides access to the repository-level @thoughts.json live memory. Agents and developers can read, update, and follow directives stored in @thoughts.json. Designed to make planning and on-the-fly instructions discoverable across sessions.

Usage:
- Read @thoughts.json from repo root to retrieve current plan and tasks
- Append new tasks as structured JSON in the "tasks" array
- When registered as a skill, agents should check "default_autonomous" before acting autonomously

Example call patterns:
- Agent: "Read @thoughts.json and generate a build plan"
- Agent: "Append task: {id: 'compile-wasm', description: 'build wasm modules', status: 'pending'}"

Notes:
- Keep entries concise. Large artifacts should reference files stored under tools/ or public/ and not embed blobs directly.
- This skill file is informational; implementors may wire a small helper that maps skill actions to file edits.
