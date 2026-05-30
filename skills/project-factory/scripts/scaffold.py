#!/usr/bin/env python3
"""
project-factory: scaffold.py
Generate skeleton files for agents, skills, and model contracts.
Skeletons contain correct schema structure — Codex fills project-specific content.

Usage:
  python scaffold.py agent <name> --purpose "<text>" [--project-root .]
  python scaffold.py skill <name> --purpose "<text>" [--project-root .]
  python scaffold.py model <name> --fold <FOLD> [--port <N>] [--project-root .]
  python scaffold.py audit [--project-root .]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent


# ── Skeleton generators ────────────────────────────────────────────────────────

def scaffold_agent(name: str, purpose: str, root: Path):
    """Generate .claude/agents/<name>.md and skills/<name>/SKILL.md"""

    # Claude Code agent
    claude_dir = root / ".claude" / "agents"
    claude_dir.mkdir(parents=True, exist_ok=True)
    claude_path = claude_dir / f"{name}.md"

    claude_content = dedent(f"""\
        ---
        name: {name}
        description: "Use this agent when... [FILL: specific trigger phrases from this project]\\n\\nTrigger on: '<phrase1>', '<phrase2>', '<phrase3>'\\n\\n<example>\\nuser: \\\"[FILL: realistic user message]\\\"\\nassistant: {name} [FILL: what it does].\\n</example>\\n\\n<example>\\nuser: \\\"[FILL: another realistic message]\\\"\\nassistant: {name} [FILL: response].\\n</example>"
        model: sonnet
        color: blue
        ---

        You are the **{name.replace('-', ' ').title()}** agent — [FILL: one-sentence role statement].

        ## Purpose

        {purpose}

        ## Your Domain

        [FILL: What files, patterns, and conventions in THIS project does this agent know? Reference actual paths.]

        ## Common Operations

        ### [FILL: Operation 1]
        [FILL: Step-by-step instructions with project-specific file paths]

        ### [FILL: Operation 2]
        [FILL: Step-by-step instructions]

        ## Conventions

        - [FILL: project-specific convention]
        - [FILL: project-specific convention]

        ## Key Files

        | File | What it does |
        |---|---|
        | `[FILL: actual/path/in/project]` | [FILL: description] |
        | `[FILL: actual/path/in/project]` | [FILL: description] |
    """)

    claude_path.write_text(claude_content, encoding="utf-8")
    print(f"Created: {claude_path}")

    # Codex skill
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    agents_dir = skill_dir / "agents"
    agents_dir.mkdir(exist_ok=True)

    skill_md_path = skill_dir / "SKILL.md"
    skill_md_content = dedent(f"""\
        ---
        name: {name}
        description: "[FILL: what this skill does and when to use it. Include trigger phrases.] Use when: '<phrase1>', '<phrase2>'."
        metadata:
          short-description: [FILL: 8-word max summary]
        ---

        # {name.replace('-', ' ').title()}

        ## Purpose

        {purpose}

        [FILL: Expand with project-specific context — what files this skill works with, what conventions it follows, what outputs it produces.]

        ## Operations

        ### [FILL: Operation Name]

        [FILL: Instructions specific to this project]

        ### [FILL: Operation Name]

        [FILL: Instructions specific to this project]

        ## Project Context

        [FILL: Project-specific knowledge: actual file paths, schemas, command patterns, naming conventions.]
    """)
    skill_md_path.write_text(skill_md_content, encoding="utf-8")
    print(f"Created: {skill_md_path}")

    openai_yaml_path = agents_dir / "openai.yaml"
    openai_yaml_content = dedent(f"""\
        interface:
          display_name: "[FILL: Human-readable name]"
          short_description: "[FILL: One sentence, max 80 chars]"
          default_prompt: "[FILL: Example prompt a user would type]"
    """)
    openai_yaml_path.write_text(openai_yaml_content, encoding="utf-8")
    print(f"Created: {openai_yaml_path}")


def scaffold_skill(name: str, purpose: str, root: Path):
    """Generate skills/<name>/skill.matrix.toml + <Name>Actions.json + SKILL.md"""

    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    class_name = "".join(w.capitalize() for w in name.replace("-", "_").split("_")) + "Actions"

    # skill.matrix.toml
    toml_path = skill_dir / "skill.matrix.toml"
    toml_content = dedent(f"""\
        [skill]
        name        = "{name}"
        version     = "1.0.0"
        description = "{purpose} — [FILL: expand with trigger phrases and scope]"
        runtime     = "json.exe"
        category    = "domain"
        excluded_agents = []

        [actions.spec]
        entry       = "{class_name}.spec"
        runtime     = "json.exe"
        description = "Return the full {name} specification."
        resources   = {{ memory = "32MB", timeout = "3s", network = false, filesystem = "none" }}

        [actions.glossary]
        entry       = "{class_name}.glossary"
        runtime     = "json.exe"
        description = "Return index of all key files, schemas, and docs for {name}."
        resources   = {{ memory = "16MB", timeout = "2s", network = false, filesystem = "none" }}

        # [FILL: Add 3-8 domain-specific actions below]
        # [actions.my_action]
        # entry       = "{class_name}.my_action"
        # runtime     = "json.exe"
        # description = "Return [FILL]."
        # resources   = {{ memory = "16MB", timeout = "2s", network = false, filesystem = "none" }}
    """)
    toml_path.write_text(toml_content, encoding="utf-8")
    print(f"Created: {toml_path}")

    # *Actions.json
    actions_path = skill_dir / f"{class_name}.json"
    actions_data = {
        "class": class_name,
        "methods": {
            "spec": {
                "params": [],
                "body": [{"op": "json", "value": {
                    "ok": True,
                    "skill": name,
                    "version": "1.0.0",
                    "purpose": purpose,
                    "FILL": "Replace this entire value with real project data"
                }}]
            },
            "glossary": {
                "params": [],
                "body": [{"op": "json", "value": {
                    "ok": True,
                    "description": f"Key files and docs for {name}",
                    "files": [
                        {"path": "FILL/actual/path", "title": "FILL", "description": "FILL"}
                    ],
                    "quick_lookup": {
                        "FILL: common task": "FILL: which file to read"
                    }
                }}]
            }
        }
    }
    actions_path.write_text(json.dumps(actions_data, indent=2), encoding="utf-8")
    print(f"Created: {actions_path}")

    # SKILL.md
    skill_md_path = skill_dir / "SKILL.md"
    skill_md_content = dedent(f"""\
        ---
        name: {name}
        description: "[FILL: what this skill does and when Codex should use it. Include trigger phrases.]"
        metadata:
          short-description: [FILL: 8-word max]
        ---

        # {name.replace('-', ' ').title()}

        {purpose}

        [FILL: Expand with project-specific context, file patterns, and conventions.]

        ## Actions

        - `spec` — full specification
        - `glossary` — file index
        - [FILL: add domain-specific actions matching skill.matrix.toml]

        ## Key Files

        [FILL: List actual files in the project that this skill covers.]
    """)
    skill_md_path.write_text(skill_md_content, encoding="utf-8")
    print(f"Created: {skill_md_path}")


def scaffold_model(name: str, fold: str, port: int, root: Path):
    """Generate model/agents/<name>/model.json + model.runtime.json"""

    model_dir = root / "model" / "agents" / name
    model_dir.mkdir(parents=True, exist_ok=True)

    fold_to_hlsl = {
        "COMPUTE_FOLD": "CS_ComputeFold",
        "STORAGE_FOLD": "CS_StorageFold",
        "META_FOLD":    "CS_MetaFold",
        "UI_FOLD":      "CS_UIFold",
    }
    hlsl_entry = fold_to_hlsl.get(fold, "CS_ComputeFold")

    fold_to_z = {
        "COMPUTE_FOLD": 0.5,
        "STORAGE_FOLD": 0.3,
        "META_FOLD":    0.7,
        "UI_FOLD":      0.9,
    }
    z_layer = fold_to_z.get(fold, 0.5)

    # model.json
    model_json_path = model_dir / "model.json"
    model_json = {
        "schema": "xcfe-model-1",
        "id": name,
        "displayName": name.replace("-", " ").title(),
        "coordFrame": "grid",
        "zLayer": z_layer,
        "fold": fold,
        "micronaut": f"{name.upper()[:4]}-1",
        "description": "[FILL: What this model does, its role in the project architecture]",
        "scxGraph": {
            "nodes": [
                {
                    "id": f"{name}-node-0",
                    "type": "[FILL: expert|router|attention|decoder]",
                    "zCoord": z_layer,
                    "role": "[FILL: describe this node's role]"
                }
            ],
            "arcs": [
                {
                    "id": f"arc_{name.upper()[:4]}_MAIN",
                    "from": "[FILL: source node or fold]",
                    "to": "[FILL: target node or fold]",
                    "entropy": 0.25,
                    "description": "[FILL: what flows along this arc]"
                }
            ]
        },
        "architecture": {
            "type": "[FILL: dense_attention|moe_sparse|hybrid]",
            "moe": False,
            "hiddenDim": 2048,
            "numHeads": 16,
            "numLayers": 4,
            "contextWindow": 2048,
            "vocabSize": 32768,
            "dtype": "fp16",
            "attention": "full_causal"
        },
        "training": {
            "phases": ["[FILL: phase1_supervised]"],
            "datasets": ["[FILL: path/to/training/data.jsonl]"]
        },
        "clearanceLayer": {
            "fold": fold,
            "description": "[FILL: what this model is cleared to read and write]"
        }
    }
    model_json_path.write_text(json.dumps(model_json, indent=2), encoding="utf-8")
    print(f"Created: {model_json_path}")

    # model.runtime.json
    runtime_json_path = model_dir / "model.runtime.json"
    runtime_json = {
        "schema": "xcfe-runtime-1",
        "id": name,
        "port": port,
        "lane": "[FILL: PROMPT|COMPUTE|STORAGE|META]",
        "fold": fold,
        "computeProfile": {
            "dtype": "fp16",
            "contextWindow": 2048,
            "temperatureBase": 1.0,
            "topP": 0.9,
            "maxNewTokens": 1024,
            "hlslEntry": hlsl_entry,
            "textureRegisters": "t0-t7",
            "cm1Gate": "0x0001"
        },
        "foldInputs": {
            "t0": "[FILL: what arrives from upstream fold]",
            "t1": "[FILL: what this model reads from storage]"
        },
        "foldOutputs": {
            "u0": "[FILL: primary output]",
            "u1": "[FILL: secondary output]"
        },
        "verifierRules": [
            "[FILL: V0 — describe input validation rule]",
            "[FILL: V1 — describe output constraint]"
        ]
    }
    runtime_json_path.write_text(json.dumps(runtime_json, indent=2), encoding="utf-8")
    print(f"Created: {runtime_json_path}")


def audit_project(root: Path):
    """Scan project and report infrastructure status."""

    checks = {
        "CLAUDE.md":                ("Project instructions for Claude Code",    "claude_md"),
        "AGENTS.md":                ("Agent roster",                             "agents_md"),
        ".claude/agents":           ("Project-local Claude Code agents",         "claude_agents"),
        "skills":                   ("Skill packages directory",                 "skills"),
        "skills/project-spec":      ("Project spec skill",                       "project_spec"),
        "model/agents":             ("Model contract directory",                 "model_agents"),
    }

    print("\nAUDIT REPORT")
    print("─" * 60)

    missing = []
    present = []
    partial = []

    for path_str, (description, key) in checks.items():
        p = root / path_str
        if p.exists():
            if p.is_dir():
                children = list(p.iterdir())
                if children:
                    present.append((path_str, description, len(children)))
                else:
                    partial.append((path_str, description, "exists but empty"))
            else:
                size = p.stat().st_size
                present.append((path_str, description, f"{size} bytes"))
        else:
            missing.append((path_str, description))

    for path_str, description, info in present:
        print(f"  ✓  {path_str:<35} {description} ({info})")
    for path_str, description, issue in partial:
        print(f"  ~  {path_str:<35} {description} — {issue}")
    for path_str, description in missing:
        print(f"  ✗  {path_str:<35} {description} — MISSING")

    print("─" * 60)
    if missing:
        print(f"Missing: {len(missing)} items — run: scaffold.py init to generate")
    else:
        print("All baseline infrastructure present.")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Project Factory Scaffolding CLI")
    sub = parser.add_subparsers(dest="op", required=True)

    p_agent = sub.add_parser("agent", help="Scaffold a new agent")
    p_agent.add_argument("name")
    p_agent.add_argument("--purpose", default="[FILL: describe what this agent does]")
    p_agent.add_argument("--project-root", default=".")

    p_skill = sub.add_parser("skill", help="Scaffold a new skill")
    p_skill.add_argument("name")
    p_skill.add_argument("--purpose", default="[FILL: describe what this skill covers]")
    p_skill.add_argument("--project-root", default=".")

    p_model = sub.add_parser("model", help="Scaffold a model contract")
    p_model.add_argument("name")
    p_model.add_argument("--fold", default="COMPUTE_FOLD",
                         choices=["COMPUTE_FOLD", "STORAGE_FOLD", "META_FOLD", "UI_FOLD"])
    p_model.add_argument("--port", type=int, default=3200)
    p_model.add_argument("--project-root", default=".")

    p_audit = sub.add_parser("audit", help="Report missing infrastructure")
    p_audit.add_argument("--project-root", default=".")

    args = parser.parse_args()
    root = Path(args.project_root).resolve()

    if args.op == "agent":
        scaffold_agent(args.name, args.purpose, root)
    elif args.op == "skill":
        scaffold_skill(args.name, args.purpose, root)
    elif args.op == "model":
        scaffold_model(args.name, args.fold, args.port, root)
    elif args.op == "audit":
        audit_project(root)


if __name__ == "__main__":
    main()
