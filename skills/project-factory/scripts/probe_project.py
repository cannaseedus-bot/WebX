#!/usr/bin/env python3
"""
project-factory: probe_project.py
Probes the current project and emits project.probe.json.
Run from the project root directory.
"""

import json
import subprocess
import sys
from pathlib import Path

STACK_SIGNALS = [
    ("package.json",        "nodejs"),
    ("pyproject.toml",      "python"),
    ("setup.py",            "python"),
    ("requirements.txt",    "python"),
    ("CMakeLists.txt",      "cpp"),
    ("Cargo.toml",          "rust"),
    ("go.mod",              "go"),
    ("build.gradle",        "java"),
    ("pom.xml",             "java"),
    ("mix.exs",             "elixir"),
    ("composer.json",       "php"),
]

AGENT_INFRA_SIGNALS = [
    "CLAUDE.md",
    "AGENTS.md",
    ".claude/agents",
    "skills",
    "model/agents",
]


def detect_stacks(root: Path) -> list[dict]:
    stacks = []
    for signal_file, stack_name in STACK_SIGNALS:
        p = root / signal_file
        if p.exists():
            info = {"stack": stack_name, "signal": signal_file}
            # Try to extract version info
            if signal_file == "package.json":
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    info["name"] = data.get("name", "")
                    info["version"] = data.get("version", "")
                    info["main_deps"] = list(data.get("dependencies", {}).keys())[:10]
                    info["scripts"] = list(data.get("scripts", {}).keys())
                except Exception:
                    pass
            elif signal_file in ("pyproject.toml",):
                try:
                    text = p.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        if line.startswith("name"):
                            info["name"] = line.split("=")[-1].strip().strip('"\'')
                        if line.startswith("version"):
                            info["version"] = line.split("=")[-1].strip().strip('"\'')
                except Exception:
                    pass
            elif signal_file == "Cargo.toml":
                try:
                    text = p.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        if line.startswith("name"):
                            info["name"] = line.split("=")[-1].strip().strip('"\'')
                        if line.startswith("version"):
                            info["version"] = line.split("=")[-1].strip().strip('"\'')
                except Exception:
                    pass
            stacks.append(info)
    return stacks


def read_readme(root: Path) -> str:
    for name in ("README.md", "README", "README.txt", "readme.md"):
        p = root / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            # First 100 lines
            lines = text.splitlines()[:100]
            return "\n".join(lines)
    return ""


def read_claude_md(root: Path) -> str:
    p = root / "CLAUDE.md"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")[:3000]
    return ""


def git_log(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            cwd=str(root),
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip().splitlines()
    except Exception:
        return []


def top_dirs(root: Path) -> list[dict]:
    dirs = []
    for item in sorted(root.iterdir()):
        if item.is_dir() and not item.name.startswith(".") and item.name not in ("node_modules", "__pycache__", "build", "dist", ".git"):
            # Count files
            try:
                file_count = sum(1 for _ in item.rglob("*") if _.is_file())
            except Exception:
                file_count = 0
            dirs.append({"name": item.name, "files": file_count})
    return dirs[:20]


def audit_agent_infra(root: Path) -> dict:
    result = {}
    for signal in AGENT_INFRA_SIGNALS:
        p = root / signal
        if p.exists():
            if p.is_dir():
                children = [c.name for c in p.iterdir() if not c.name.startswith(".")]
                result[signal] = {"exists": True, "contents": children[:20]}
            else:
                result[signal] = {"exists": True}
        else:
            result[signal] = {"exists": False}
    return result


def scan_key_files(root: Path) -> list[str]:
    """Find files likely to be architecturally significant."""
    key_patterns = [
        "*.toml", "*.json", "*.md", "*.hlsl", "*.ebnf",
        "src/*.js", "src/*.ts", "src/*.py",
        "core/*.py", "core/*.cpp", "core/*.h",
    ]
    found = []
    for pattern in key_patterns:
        for p in root.glob(pattern):
            if p.is_file() and p.name not in (".gitignore", "package-lock.json"):
                found.append(str(p.relative_to(root)))
    return sorted(set(found))[:40]


def infer_purpose(readme: str, claude_md: str, git_log_lines: list[str], stacks: list[dict]) -> str:
    """Extract best-guess purpose from available signals."""
    # Use first non-empty heading from README
    for line in readme.splitlines():
        line = line.strip()
        if line.startswith("# ") and len(line) > 3:
            return line.lstrip("# ").strip()
        if line and not line.startswith("#") and len(line) > 20:
            return line[:200]
    if git_log_lines:
        return f"Active development — recent: {git_log_lines[0]}"
    if stacks:
        return f"{'+'.join(s['stack'] for s in stacks)} project"
    return "Unknown — read README.md for context"


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    root = root.resolve()

    print(f"Probing: {root}", file=sys.stderr)

    stacks     = detect_stacks(root)
    readme     = read_readme(root)
    claude_md  = read_claude_md(root)
    log        = git_log(root)
    dirs       = top_dirs(root)
    infra      = audit_agent_infra(root)
    key_files  = scan_key_files(root)
    purpose    = infer_purpose(readme, claude_md, log, stacks)

    probe = {
        "schema":       "project-probe-v1",
        "root":         str(root),
        "project_name": root.name,
        "purpose":      purpose,
        "stacks":       stacks,
        "top_dirs":     dirs,
        "key_files":    key_files,
        "git_log":      log,
        "readme_head":  readme[:1500],
        "claude_md":    claude_md[:2000] if claude_md else None,
        "agent_infra":  infra,
        "missing_infra": [k for k, v in infra.items() if not v["exists"]],
    }

    out_path = root / "project.probe.json"
    out_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(json.dumps(probe, indent=2))
    print(f"\nWrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
