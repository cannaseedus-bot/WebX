#!/usr/bin/env python3
"""
KUHUL 3D compiler (bootstrap).

Compiles KUHUL 3D source into a normalized IR JSON for backend dispatch.
This parser is intentionally lightweight and line-oriented for now.

Usage:
    python kuhul3d_compile.py --input scene.kuhul3d --output scene.ir.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


COMMAND_RE      = re.compile(r"^\[(?P<body>.+?)\]\s*$")
DX12_BLOCK_START_RE = re.compile(r"^\s*dx12\s*\{\s*$")
DX12_BLOCK_END_RE   = re.compile(r"^\s*\}\s*$")
DECL_RE = re.compile(
    r"^\s*(Tensor|Cluster|Model|Buffer|Shader)\s+(?P<name>[A-Za-z_]\w*)\s*=.*;\s*$"
)


def normalize_line(line: str) -> str:
    return line.split("//", 1)[0].strip()


def parse_command(line: str) -> Dict[str, Any]:
    m = COMMAND_RE.match(line)
    if not m:
        return {}
    body  = m.group("body").strip()
    parts = body.split()
    if not parts:
        return {}
    return {
        "op":   "command",
        "name": parts[0],
        "args": parts[1:],
        "raw":  line,
    }


def parse_declaration(line: str) -> Dict[str, Any]:
    m = DECL_RE.match(line)
    if not m:
        return {}
    return {
        "op":   "declare",
        "kind": m.group(1),
        "name": m.group("name"),
        "raw":  line,
    }


def compile_source(text: str, source_name: str) -> Dict[str, Any]:
    ops: List[Dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw  = lines[i]
        line = normalize_line(raw)
        i   += 1
        if not line:
            continue

        # dx12 { ... } block
        if DX12_BLOCK_START_RE.match(line):
            block_lines: List[str] = []
            depth = 1
            while i < len(lines):
                raw2  = lines[i]
                line2 = normalize_line(raw2)
                i    += 1
                if not line2:
                    continue
                depth += line2.count("{")
                depth -= line2.count("}")
                if depth <= 0:
                    break
                block_lines.append(line2)
            ops.append({"op": "dx12_block", "statements": block_lines})
            continue

        # [command args...]
        cmd = parse_command(line)
        if cmd:
            ops.append(cmd)
            continue

        # Tensor/Cluster/... name = ...;
        decl = parse_declaration(line)
        if decl:
            ops.append(decl)
            continue

        ops.append({"op": "unknown", "raw": line})

    return {
        "ir_version": "0.1.0",
        "source":     source_name,
        "dialect":    "kuhul3d-v3-bootstrap",
        "ops":        ops,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="K'UHUL 3D bootstrap compiler")
    ap.add_argument("--input",  required=True, help="Path to KUHUL 3D source file")
    ap.add_argument("--output", required=True, help="Path to output IR JSON")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    src = in_path.read_text(encoding="utf-8")
    ir  = compile_source(src, in_path.as_posix())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
    print(f"[OK] KUHUL3D IR written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
