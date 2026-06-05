import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message):
    raise SystemExit(f"validation failed: {message}")


def check_path(relative, *, directory=False):
    candidate = (ROOT / relative).resolve()
    if directory:
        if not candidate.is_dir():
            fail(f"missing directory: {relative}")
    elif not candidate.exists():
        fail(f"missing path: {relative}")
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        fail(f"path escapes repo root: {relative}")


def main():
    required = [
        "manifest.json",
        "README.md",
        "registry/agents-net.registry.json",
    ]
    for relative in required:
        check_path(relative)

    manifest = load_json(ROOT / "manifest.json")
    registry = load_json(ROOT / "registry/agents-net.registry.json")

    if manifest.get("kind") != "native_dotnet_agents_release":
        fail("manifest kind must be native_dotnet_agents_release")

    for relative in registry.get("directories", {}).values():
        check_path(relative, directory=True)

    for group_name in ("agent_projects", "connector_projects", "experimental_projects", "docs"):
        for relative in registry.get(group_name, {}).values():
            check_path(relative)

    check_path(registry["scx_control_flow_release"], directory=True)
    print("validation ok: Agents.NET.V_1")


if __name__ == "__main__":
    main()
