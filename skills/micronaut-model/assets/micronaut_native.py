"""
micronaut_native.py
Pure-Python bridge mirroring C++ headers:
  - micronaut_todo_creator.hpp  (TodoItem, TodoCreator, TodoJsonSerializer)
  - deterministic_v6.h          (DeterministicV6, TraceLogger)

Imported by DC-1 (dolphin-coder) and GR-1 (github-codereview) bots.py.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# TodoItem  —  mirrors the C++ struct in micronaut_todo_creator.hpp
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    id: str = ""
    title: str = ""
    description: str = ""
    category: str = ""        # "bug"|"feature"|"refactor"|"test"|"doc"|"perf"|"security"
    priority: int = 3         # 1-5  (5 = CRITICAL)
    service: str = ""
    line_number: int = 0
    confidence: float = 0.0
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TodoCreator  —  mirrors the C++ class in micronaut_todo_creator.hpp
# ---------------------------------------------------------------------------

class TodoCreator:
    _TODO_RE   = re.compile(r"#?\s*TODO[:\s](.*)", re.IGNORECASE)
    _FIXME_RE  = re.compile(r"#?\s*FIXME[:\s](.*)", re.IGNORECASE)
    _HACK_RE   = re.compile(r"#?\s*(?:HACK|XXX)[:\s](.*)", re.IGNORECASE)
    _RAISE_RE  = re.compile(r"raise\s+Exception\b(.*)")
    _BARE_RAISE_RE = re.compile(r"^\s*raise\s*$", re.MULTILINE)
    _ALLOC_RE  = re.compile(r"\b(?:open|socket|connect|acquire)\s*\(")
    _CTX_RE    = re.compile(r"\bwith\b")

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self._counter = 0

    # --- public -----------------------------------------------------------

    def extract_todos(self, content: str) -> list[TodoItem]:
        items: list[TodoItem] = []
        items.extend(self.extract_todo_comments(content))
        items.extend(self.extract_fixme_comments(content))
        items.extend(self.extract_hack_comments(content))
        items.extend(self.extract_issue_patterns(content))
        items.extend(self.extract_from_configuration(content))

        # Assign IDs, sort by priority desc
        for item in items:
            item.id = self.generate_todo_id(item, item.line_number)
        items.sort(key=lambda t: t.priority, reverse=True)
        return items

    def extract_todo_comments(self, content: str) -> list[TodoItem]:
        return self._scan_lines(content, self._TODO_RE, default_category="feature")

    def extract_fixme_comments(self, content: str) -> list[TodoItem]:
        items = self._scan_lines(content, self._FIXME_RE, default_category="bug")
        for item in items:
            if item.priority < 4:
                item.priority = 4
        return items

    def extract_hack_comments(self, content: str) -> list[TodoItem]:
        items = self._scan_lines(content, self._HACK_RE, default_category="refactor")
        for item in items:
            if item.priority > 2:
                item.priority = 2
        return items

    def extract_issue_patterns(self, content: str) -> list[TodoItem]:
        """
        Mirrors C++ logic that detects bare throw / new-without-delete.
        Python equivalent: bare `raise Exception(...)`, bare `raise`, and
        resource allocations not wrapped in a `with` block.
        """
        items: list[TodoItem] = []
        lines = content.splitlines()

        for lineno, line in enumerate(lines, start=1):
            m = self._RAISE_RE.search(line)
            if m:
                item = TodoItem(
                    title=f"Replace bare Exception raise: {m.group(1).strip()[:60]}",
                    description=line.strip(),
                    category="bug",
                    priority=3,
                    service=self.service_name,
                    line_number=lineno,
                    confidence=0.70,
                )
                items.append(item)
                continue

            if re.match(r"^\s*raise\s*$", line):
                item = TodoItem(
                    title="Bare re-raise detected; verify exception context",
                    description=line.strip(),
                    category="bug",
                    priority=2,
                    service=self.service_name,
                    line_number=lineno,
                    confidence=0.60,
                )
                items.append(item)
                continue

            # Resource allocation without context manager on the same line
            if self._ALLOC_RE.search(line) and not self._CTX_RE.search(line):
                item = TodoItem(
                    title=f"Potential resource leak at line {lineno}",
                    description=line.strip(),
                    category="bug",
                    priority=3,
                    service=self.service_name,
                    line_number=lineno,
                    confidence=0.55,
                )
                items.append(item)

        return items

    def extract_from_configuration(self, content: str) -> list[TodoItem]:
        """Finds TODO / FIXME / WIP markers anywhere in config text."""
        CONFIG_RE = re.compile(r"(?:TODO|FIXME|WIP)[:\s]+(.*)", re.IGNORECASE)
        items: list[TodoItem] = []
        for lineno, line in enumerate(content.splitlines(), start=1):
            m = CONFIG_RE.search(line)
            if m:
                comment = m.group(1).strip()
                priority = self.assess_priority("TODO", comment)
                item = TodoItem(
                    title=comment[:80],
                    description=line.strip(),
                    category="feature",
                    priority=priority,
                    service=self.service_name,
                    line_number=lineno,
                    confidence=0.65,
                )
                items.append(item)
        return items

    # --- priority logic ---------------------------------------------------

    def assess_priority(self, marker: str, comment: str) -> int:
        upper = comment.upper()
        if any(k in upper for k in ("CRITICAL", "URGENT", "BLOCKER")):
            return 5
        if any(k in upper for k in ("ASAP", "HIGH")):
            return 4
        if any(k in upper for k in ("LOW", "LATER", "OPTIONAL")):
            return 1
        base = 3
        m = marker.upper()
        if m == "FIXME":
            return max(base, 4)
        if m in ("HACK", "XXX"):
            return min(base, 2)
        return base

    # --- ID generation ----------------------------------------------------

    def generate_todo_id(self, item: TodoItem, line: int) -> str:
        conf_hex = int(item.confidence * 1000)
        svc = item.service or self.service_name
        return f"{svc}:{line:05d}:{conf_hex:x}"

    # --- private helpers --------------------------------------------------

    def _scan_lines(
        self,
        content: str,
        pattern: re.Pattern,
        default_category: str,
    ) -> list[TodoItem]:
        items: list[TodoItem] = []
        for lineno, line in enumerate(content.splitlines(), start=1):
            m = pattern.search(line)
            if not m:
                continue
            comment = m.group(1).strip()
            marker = pattern.pattern.split("[")[0].lstrip("#?\\s*").strip()
            priority = self.assess_priority(marker, comment)
            item = TodoItem(
                title=comment[:80],
                description=line.strip(),
                category=default_category,
                priority=priority,
                service=self.service_name,
                line_number=lineno,
                confidence=0.90,
            )
            items.append(item)
        return items


# ---------------------------------------------------------------------------
# TodoJsonSerializer  —  mirrors the C++ class (static methods only)
# ---------------------------------------------------------------------------

class TodoJsonSerializer:

    @staticmethod
    def to_json(todo: TodoItem) -> str:
        return json.dumps(
            {
                "id": todo.id,
                "title": todo.title,
                "description": todo.description,
                "category": todo.category,
                "priority": todo.priority,
                "service": todo.service,
                "line_number": todo.line_number,
                "confidence": todo.confidence,
                "status": todo.status,
                "depends_on": todo.depends_on,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def to_json_array(todos: list[TodoItem]) -> str:
        return json.dumps(
            {
                "@version": "v6.0",
                "@format": "micronaut-todos",
                "total": len(todos),
                "todos": [json.loads(TodoJsonSerializer.to_json(t)) for t in todos],
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def to_html_dashboard(todos: list[TodoItem]) -> str:
        _PRIORITY_LABELS = {1: "LOW", 2: "MINIMAL", 3: "MEDIUM", 4: "HIGH", 5: "CRITICAL"}
        rows = ""
        for t in todos:
            label = _PRIORITY_LABELS.get(t.priority, "MEDIUM")
            rows += (
                f'  <tr class="p{t.priority}">'
                f"<td>{t.id}</td>"
                f"<td>{t.priority} — {label}</td>"
                f"<td>{t.category}</td>"
                f"<td>{t.title}</td>"
                f"<td>{t.service}</td>"
                f"<td>{t.line_number}</td>"
                f"<td>{t.status}</td>"
                f"</tr>\n"
            )

        return (
            "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
            "<title>Micronaut TODO Dashboard</title><style>"
            "body{background:#1e1e2e;color:#cdd6f4;font-family:monospace;}"
            "table{width:100%;border-collapse:collapse;}"
            "th,td{padding:6px 10px;border:1px solid #313244;text-align:left;}"
            "th{background:#313244;}"
            ".p5{background:#3d1f2a;color:#f38ba8;}"
            ".p4{background:#3d2e1f;color:#fab387;}"
            ".p3{background:#2e2e1f;color:#f9e2af;}"
            ".p2{background:#1f2e1f;color:#a6e3a1;}"
            ".p1{background:#1f1f2e;color:#89b4fa;}"
            "</style></head><body>"
            "<h1>Micronaut TODO Dashboard</h1>"
            "<table><thead><tr>"
            "<th>ID</th><th>Priority</th><th>Category</th>"
            "<th>Title</th><th>Service</th><th>Line</th><th>Status</th>"
            f"</tr></thead><tbody>\n{rows}</tbody></table>"
            "</body></html>"
        )

    @staticmethod
    def to_markdown(todos: list[TodoItem]) -> str:
        header = "| ID | Pri | Category | Title | Service | Line | Status |\n"
        sep    = "|---|---|---|---|---|---|---|\n"
        rows = "".join(
            f"| {t.id} | {t.priority} | {t.category} | {t.title} "
            f"| {t.service} | {t.line_number} | {t.status} |\n"
            for t in todos
        )
        return header + sep + rows


# ---------------------------------------------------------------------------
# DeterministicV6  —  mirrors deterministic_v6.h
# ---------------------------------------------------------------------------

class DeterministicV6:

    @staticmethod
    def sha256(input: str) -> bytes:
        return hashlib.sha256(input.encode("utf-8")).digest()

    @staticmethod
    def sha256_hex(input: str) -> str:
        return hashlib.sha256(input.encode("utf-8")).hexdigest()

    @staticmethod
    def json_canonical(json_str: str) -> str:
        obj = json.loads(json_str)
        return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)

    @staticmethod
    def parse_canonical(json_str: str) -> dict:
        return json.loads(json_str)

    @staticmethod
    def iso8601_now() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def create_tool_result(
        tool_name: str,
        input_hash: str,
        output_hash: str,
        result,
        latency_ms: float,
        status: str = "success",
    ) -> str:
        if isinstance(result, str):
            try:
                result_payload = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                result_payload = result
        else:
            result_payload = result

        return json.dumps(
            {
                "@version": "v6.0",
                "@format": "micronaut-tool-result",
                "timestamp": DeterministicV6.iso8601_now(),
                "tool": tool_name,
                "status": status,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "latency_ms": latency_ms,
                "result": result_payload,
            },
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# TraceLogger  —  mirrors deterministic_v6.h TraceLogger
# ---------------------------------------------------------------------------

class TraceLogger:
    _FLUSH_THRESHOLD = 10

    def __init__(self, trace_file_path: str) -> None:
        self._path = trace_file_path
        self._pending: list[str] = []

    def log_tool_execution(
        self,
        tool_name: str,
        input,
        output,
        latency_ms: float,
        status: str = "success",
    ) -> None:
        input_str  = input  if isinstance(input,  str) else json.dumps(input)
        output_str = output if isinstance(output, str) else json.dumps(output)
        input_hash  = DeterministicV6.sha256_hex(input_str)
        output_hash = DeterministicV6.sha256_hex(output_str)
        entry = DeterministicV6.create_tool_result(
            tool_name, input_hash, output_hash, output, latency_ms, status
        )
        self._pending.append(entry)
        if len(self._pending) >= self._FLUSH_THRESHOLD:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        with open(self._path, "a", encoding="utf-8") as fh:
            for entry in self._pending:
                fh.write(entry.replace("\n", " ") + "\n")
        self._pending.clear()

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Optional ctypes binding to native shared library
# ---------------------------------------------------------------------------

def _try_load_native() -> ctypes.CDLL | None:
    _here = Path(__file__).parent
    build_dir = _here.parent / "build"
    candidates = [
        build_dir / "code-micronaut-native.dll",
        build_dir / "libcode-micronaut-native.so",
    ]
    for path in candidates:
        if path.exists():
            try:
                return ctypes.CDLL(str(path))
            except OSError:
                pass
    return None


_native_lib: ctypes.CDLL | None = _try_load_native()
NATIVE_AVAILABLE: bool = _native_lib is not None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== micronaut_native.py self-test ===\n")

    # --- TodoCreator / TodoItem ---
    sample = """\
# TODO: refactor the auth flow CRITICAL
# FIXME: null pointer possible here
# HACK: workaround for upstream bug
raise Exception("unexpected state")
open("data.bin")
"""
    creator = TodoCreator("test-service")
    todos = creator.extract_todos(sample)
    print(f"[TodoCreator] extracted {len(todos)} items")
    for t in todos:
        print(f"  pri={t.priority} cat={t.category:10s} id={t.id}  title={t.title[:50]}")

    # --- TodoJsonSerializer ---
    arr_json = TodoJsonSerializer.to_json_array(todos)
    parsed = json.loads(arr_json)
    assert parsed["@version"] == "v6.0"
    assert parsed["total"] == len(todos)
    print(f"\n[TodoJsonSerializer] JSON array OK  (total={parsed['total']})")

    md = TodoJsonSerializer.to_markdown(todos)
    assert "| ID |" in md
    print("[TodoJsonSerializer] Markdown OK")

    html = TodoJsonSerializer.to_html_dashboard(todos)
    assert "dark" not in html or "#1e1e2e" in html
    print("[TodoJsonSerializer] HTML dashboard OK")

    # --- DeterministicV6 ---
    h = DeterministicV6.sha256_hex("hello")
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    print(f"\n[DeterministicV6] sha256_hex OK  ({h[:16]}…)")

    ts = DeterministicV6.iso8601_now()
    assert "T" in ts and ts.endswith("Z")
    print(f"[DeterministicV6] iso8601_now OK  ({ts})")

    canonical = DeterministicV6.json_canonical('{"b":2,"a":1}')
    assert '"a": 1' in canonical and canonical.index('"a"') < canonical.index('"b"')
    print("[DeterministicV6] json_canonical OK")

    tool_result = DeterministicV6.create_tool_result(
        "test-tool", "aabbcc", "ddeeff", '{"ok":true}', 42.0
    )
    tr = json.loads(tool_result)
    assert tr["@format"] == "micronaut-tool-result"
    assert tr["result"] == {"ok": True}
    print("[DeterministicV6] create_tool_result OK")

    # --- TraceLogger ---
    log_path = str(Path(__file__).parent / "_selftest_trace.jsonl")
    logger = TraceLogger(log_path)
    for i in range(3):
        logger.log_tool_execution("dummy-tool", f"input-{i}", f"output-{i}", 10.0 * i)
    logger.flush()
    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    print(f"[TraceLogger] flush OK  ({len(lines)} entries written to {log_path})")
    Path(log_path).unlink(missing_ok=True)

    # --- Native binding ---
    print(f"\n[Native] NATIVE_AVAILABLE = {NATIVE_AVAILABLE}")

    print("\n=== All self-tests passed ===")

