#!/usr/bin/env python3
"""
micronaut-model: scaffold_micronaut.py
Scaffold a new micronaut xjson + bots.py + registry entries.

Usage:
  scaffold_micronaut.py new <ID> <Name> --backend <type> [options] [--project-root .]
  scaffold_micronaut.py register <path/to/ID.xjson>  [--project-root .]
  scaffold_micronaut.py list [--project-root .]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from textwrap import dedent

# ── Fold registry ──────────────────────────────────────────────────────────────

FOLD_MAP = {
    "CONTROL":  "⟁CONTROL_FOLD⟁",
    "DATA":     "⟁DATA_FOLD⟁",
    "TIME":     "⟁TIME_FOLD⟁",
    "STATE":    "⟁STATE_FOLD⟁",
    "STORAGE":  "⟁STORAGE_FOLD⟁",
    "COMPUTE":  "⟁COMPUTE_FOLD⟁",
    "PATTERN":  "⟁PATTERN_FOLD⟁",
    "UI":       "⟁UI_FOLD⟁",
    "META":     "⟁META_FOLD⟁",
}

FOLD_TO_LANE = {
    "CONTROL": "EDGE",  "DATA": "DICT",  "TIME": "LANE",
    "STATE":   "FIELD", "STORAGE": "FIELD", "COMPUTE": "BATCH",
    "PATTERN": "DICT",  "UI": "LANE",    "META": "DICT",
}

FOLD_TO_HLSL = {
    "CONTROL": "CS_ControlFold", "DATA": "CS_DataFold",   "TIME": "CS_TimeFold",
    "STATE":   "CS_StateFold",   "STORAGE": "CS_StorageFold", "COMPUTE": "CS_ComputeFold",
    "PATTERN": "CS_PatternFold", "UI": "CS_UIFold",        "META": "CS_MetaFold",
}


# ── xjson builder ─────────────────────────────────────────────────────────────

def build_xjson(id: str, name: str, fold: str, backend: str, backend_opts: dict,
                persona: str = None, domain: str = None, experts: list = None) -> dict:
    fold_key   = fold.upper()
    fold_sym   = FOLD_MAP.get(fold_key, f"⟁{fold_key}_FOLD⟁")
    domain     = domain or name.lower().replace(" ", "_")
    persona    = persona or f"Expert {name.lower()} assistant"
    experts    = experts or [{"id": "exp-0", "name": "General", "domain": "general"}]

    xjson = {
        "@meta": {
            "id": id,
            "name": name,
            "description": f"Micronaut micro-model — {name} ({fold_key} fold)",
            "version": "1.0",
            "format": "atomic_block_fold",
            "spec": "docs/specs/horizontal-fold-linear-system.md"
        },

        "@lanes": {
            "agent":   {"@dict": 0, "@role": "stateful cognitive tensor space",  "@bus": "cognition"},
            "skills":  {"@dict": 1, "@role": "event routing triggers",           "@bus": "intent"},
            "experts": {"@dict": 2, "@role": "compute endpoints (MoE)",          "@bus": "execution"},
            "runtime": {"@dict": 3, "@role": "hardware abstraction",             "@bus": "hardware"},
            "router":  {"@dict": 4, "@role": "control plane",                    "@bus": "control", "@kernel": "implicit"}
        },

        "@phases": {
            "Pop":   {"@op": "load",    "@desc": "load state vector from BSON snapshot"},
            "Wo":    {"@op": "resolve", "@desc": "resolve lanes → build x blocks from @lanes"},
            "Sek":   {"@op": "execute", "@desc": f"x_next = A·x, top_k dispatch to {name} experts"},
            "Ch'en": {"@op": "update",  "@desc": "A ← A + ΔA, persist learned_biases to BSON"}
        },

        "@variables": {
            "routing_bias": {
                "@mutability": "MUTABLE",
                "@update": "reinforce(success) - decay(failure)",
                "@range": [0, 1],
                "@persist": "@agent.main.learned_biases"
            },
            "learned": {
                "@mutability": "MUTABLE",
                "@update": "A + ΔA per turn",
                "@range": [-1, 1],
                "@persist": f"models/{id}_state.bson"
            },
            "memory": {
                "@mutability": "MUTABLE",
                "@update": "last_topic, context_window",
                "@persist": "@agent.main.memory"
            },
            "model": {
                "@mutability": "IMMUTABLE",
                "@source": "@model.core",
                "@note": "vocab_size, num_layers, num_experts — schema-locked"
            },
            "experts": {
                "@mutability": "IMMUTABLE",
                "@source": "@experts",
                "@note": "expert structure schema-locked; routing_bias MUTABLE within"
            }
        },

        "@edges": _build_edges(experts),

        "@agent.main": {
            "@lane": 0, "@dict": 0, "@kind": "atomic_block", "@phase": "Pop",
            "id": "agent-main",
            "name": name,
            "persona": persona,
            "system_prompt": f"You are {name}, {persona}.",
            "temperature": 0.7,
            "max_tokens": 2048,
            "interaction_count": 0,
            "learned_biases": {},
            "memory": {}
        },

        "@model.core": _build_model_core(id, backend, backend_opts),

        "@runtime.gpu": {
            "@lane": 3, "@dict": 0, "@kind": "atomic_block", "@phase": "Pop",
            "enabled": True,
            "backend": "d3d12",
            "@condition_exports": ["gpu_enabled", "memory_available"]
        },

        "@moe.router": {
            "@lane": 4, "@dict": 0, "@kind": "atomic_block", "@kernel": "implicit",
            "top_k_experts": min(2, len(experts)),
            "allow_self_modify": True,
            "persist_learning": True,
            "state_file": f"models/{id}_state.bson",
            "@control": {
                "mode": "route",
                "variable": "routing_bias",
                "budget_ms": 50,
                "max_latency_ms": 200,
                "ops": ["read", "write", "query", "sek", "infer"]
            },
            "@phase": "Sek"
        },

        "@skills": _build_skills(domain, experts),
        "@experts": _build_experts(experts)
    }

    return xjson


def _build_model_core(id: str, backend: str, opts: dict) -> dict:
    base = {"@xcfe": "IMMUTABLE", "backend": backend}

    if backend == "local_gguf":
        return {**base,
            "path":         opts.get("weights", f"models/{id}.gguf"),
            "hash":         "sha256:pending",
            "vocab_size":   opts.get("vocab_size", 32000),
            "num_layers":   opts.get("num_layers", 32),
            "num_experts":  1,
            "hidden_size":  opts.get("hidden_size", 4096),
            "context_length": opts.get("context_length", 4096),
            "quantization": opts.get("quantization", "Q4_K_M")
        }

    elif backend == "local_scxqdds":
        return {**base,
            "shard_dir":   opts.get("shard_dir", f"models/shards/{id}/"),
            "shard_count": opts.get("shard_count", 8),
            "shard_format":"SCXQDDS-v1",
            "vocab_size":  opts.get("vocab_size", 32000),
            "num_layers":  opts.get("num_layers", 8),
            "hidden_size": opts.get("hidden_size", 1024),
            "dtype":       "INT8"
        }

    elif backend == "fetch_oss":
        hf_repo = opts.get("hf_repo", "")
        hf_file = opts.get("hf_file", "model.gguf")
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", hf_repo.split("/")[-1] if hf_repo else id)
        return {**base,
            "hf_repo":     hf_repo,
            "hf_file":     hf_file,
            "local_cache": f"models/oss-cache/{safe_name}.gguf",
            "vocab_size":  opts.get("vocab_size", 32064),
            "num_layers":  opts.get("num_layers", 32),
            "hidden_size": opts.get("hidden_size", 3072),
            "quantization":"fp16",
            "license":     opts.get("license", "check-hf-repo")
        }

    elif backend == "api_openai":
        return {**base,
            "model":    opts.get("model", "gpt-4o"),
            "api_base": opts.get("api_base", "https://api.openai.com/v1"),
            "env_key":  "OPENAI_API_KEY",
            "stream":   True,
            "max_tokens": opts.get("max_tokens", 4096),
            "temperature": 0.7
        }

    elif backend == "api_anthropic":
        return {**base,
            "model":    opts.get("model", "claude-sonnet-4-6"),
            "env_key":  "ANTHROPIC_API_KEY",
            "stream":   True,
            "max_tokens": opts.get("max_tokens", 4096)
        }

    elif backend == "api_ollama":
        return {**base,
            "model":    opts.get("model", "phi3:mini"),
            "api_base": opts.get("api_base", "http://localhost:11434/v1"),
            "stream":   True
        }

    elif backend == "api_custom":
        return {**base,
            "model":    opts.get("model", ""),
            "api_base": opts.get("api_base", ""),
            "env_key":  opts.get("env_key", "API_KEY"),
            "stream":   True
        }

    return base


def _build_edges(experts: list) -> list:
    edges = []
    for i, exp in enumerate(experts):
        eid = exp.get("id", f"exp-{i}")
        domain = exp.get("domain", "general")
        edges.append({
            "@id": f"{domain}→{eid}",
            "@from": f"@skills.skill-{domain}",
            "@to": f"@experts.{eid}",
            "@control": "route",
            "@condition": f"match(intent={domain})",
            "@weight": f"@variable.routing_bias[{eid}]",
            "@phase": "Sek"
        })

    edges.append({
        "@id": "agent→router",
        "@from": "@agent.main",
        "@to": "@moe.router",
        "@control": "control",
        "@condition": "always",
        "@weight": 1.0,
        "@phase": "Sek",
        "@xcfe": "IMMUTABLE"
    })
    edges.append({
        "@id": "router→experts",
        "@from": "@moe.router",
        "@to": "@experts.*",
        "@control": "dispatch",
        "@condition": "top_k(@variable.routing_bias, k=2) && @runtime.gpu.enabled",
        "@weight": "@variable.routing_bias",
        "@phase": "Sek"
    })
    edges.append({
        "@id": "experts→learned",
        "@from": "@experts.*",
        "@to": "@agent.main.learned_biases",
        "@control": "feedback",
        "@condition": "always",
        "@weight": "@variable.signal",
        "@update": "reinforce(success) - decay(failure)",
        "@phase": "Ch'en",
        "@note": "CRITICAL: closed-loop learning — without this edge, adaptation is not possible"
    })
    return edges


def _build_skills(domain: str, experts: list) -> dict:
    skills = {}
    for i, exp in enumerate(experts):
        edomain = exp.get("domain", "general")
        key = f"skill-{edomain}"
        skills[key] = {
            "@lane": 1, "@dict": i, "@kind": "atomic_block", "@phase": "Wo",
            "enabled": True,
            "name": exp.get("name", edomain.replace("_", " ").title()),
            "trigger": f"user asks about {edomain.replace('_', ' ')}",
            "action": f"handle_{edomain}",
            "priority": 0.8,
            "@control": {
                "mode": "infer",
                "variable": "skill_active",
                "route": key,
                "throttle": {"max_calls_per_min": 120, "burst": 10, "cooldown_ms": 250},
                "metrics": {"track_latency_ms": True, "track_tokens": True}
            }
        }
    return skills


def _build_experts(experts: list) -> dict:
    result = {}
    for i, exp in enumerate(experts):
        eid = exp.get("id", f"exp-{i}")
        result[eid] = {
            "@lane": 2, "@dict": i, "@kind": "atomic_block", "@phase": "Sek",
            "name": exp.get("name", f"Expert {i}"),
            "domain": exp.get("domain", "general"),
            "weights_ref": f"expert_{i}",
            "trainable": True,
            "layer_start": 0,
            "layer_end": 8,
            "routing_bias": 0.0,
            "@control": {
                "mode": "route",
                "variable": "routing_bias",
                "route": eid,
                "throttle": {"max_calls_per_min": 240, "burst": 20, "cooldown_ms": 100},
                "metrics": {"track_latency_ms": True, "track_tokens": True, "track_hits": True}
            }
        }
    return result


# ── bots.py builder ───────────────────────────────────────────────────────────

BOTS_TEMPLATE_DIR = Path(__file__).parent.parent / "assets"


def build_bots_py(id: str, name: str, fold: str, backend: str, backend_opts: dict,
                  tools: list, system_prompt: str) -> str:
    fold_sym = FOLD_MAP.get(fold.upper(), f"⟁{fold.upper()}_FOLD⟁")

    # Backend-specific import + loader block
    if backend == "local_gguf":
        backend_imports = "from llama_cpp import Llama"
        backend_init = f"""\
        model_path = xjson.get("@model.core", {{}}).get("path", "models/{id}.gguf")
        self.llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        self.backend = "local_gguf"
        logger.info(f"Loaded GGUF: {{model_path}}")"""
        backend_infer = """\
        response = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=True)
        for chunk in response:
            token = chunk["choices"][0]["text"]
            yield token"""

    elif backend == "local_scxqdds":
        backend_imports = "import subprocess, struct"
        backend_init = f"""\
        shard_dir = xjson.get("@model.core", {{}}).get("shard_dir", "models/shards/{id}/")
        self.shard_dir = shard_dir
        self.backend = "local_scxqdds"
        logger.info(f"SCXQDDS shards: {{shard_dir}}")"""
        backend_infer = """\
        # Route through scxqdds_vector_runner for expert shards
        result = subprocess.run(
            ["artifacts/native-asx/scxqdds_vector_runner.exe", self.shard_dir, prompt[:512]],
            capture_output=True, text=True, timeout=30
        )
        yield result.stdout or "[no output]" """

    elif backend == "fetch_oss":
        backend_imports = "from huggingface_hub import hf_hub_download\nfrom llama_cpp import Llama"
        backend_init = f"""\
        core = xjson.get("@model.core", {{}})
        cache_path = core.get("local_cache", "models/oss-cache/{id}.gguf")
        if not Path(cache_path).exists():
            logger.info(f"Fetching OSS weights: {{core.get('hf_repo')}} / {{core.get('hf_file')}}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=core.get("hf_repo", ""),
                filename=core.get("hf_file", "model.gguf"),
                local_dir=str(Path(cache_path).parent)
            )
        self.llm = Llama(model_path=cache_path, n_ctx=4096, verbose=False)
        self.backend = "fetch_oss"
        logger.info(f"Loaded OSS model: {{cache_path}}")"""
        backend_infer = """\
        response = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, stream=True)
        for chunk in response:
            token = chunk["choices"][0]["text"]
            yield token"""

    elif backend == "api_anthropic":
        backend_imports = "import anthropic"
        backend_init = """\
        self.client = anthropic.Anthropic()
        self.model = xjson.get("@model.core", {}).get("model", "claude-sonnet-4-6")
        self.backend = "api_anthropic"
        logger.info(f"Anthropic backend: {self.model}")"""
        backend_infer = """\
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text"""

    elif backend in ("api_openai", "api_ollama", "api_custom"):
        backend_imports = "from openai import OpenAI"
        init_extra = ""
        if backend == "api_ollama":
            init_extra = '\n        api_base = xjson.get("@model.core", {}).get("api_base", "http://localhost:11434/v1")'
        elif backend == "api_custom":
            init_extra = '\n        api_base = xjson.get("@model.core", {}).get("api_base", "")'
        else:
            init_extra = '\n        api_base = None'

        backend_init = f"""\
        core = xjson.get("@model.core", {{}}){init_extra}
        self.client = OpenAI(
            api_key=os.environ.get(core.get("env_key", "OPENAI_API_KEY"), ""),
            base_url=api_base
        )
        self.model = core.get("model", "gpt-4o")
        self.backend = "{backend}"
        logger.info(f"OpenAI-compat backend: {{self.model}}")"""
        backend_infer = """\
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta"""
    else:
        backend_imports = ""
        backend_init = 'self.backend = "unknown"'
        backend_infer = 'yield "[backend not implemented]"'

    tools_repr = repr(tools)
    escaped_prompt = system_prompt.replace('"', '\\"')

    return dedent(f'''\
        """
        {name} ({id}) — micronaut bot
        Fold: {fold_sym}
        Backend: {backend}
        Tools: {tools_repr}
        Generated by micronaut-model factory.

        Native deps (co-located micronaut_native.py bridge):
          micronaut_todo_creator.hpp → TodoCreator, TodoItem, TodoJsonSerializer
          deterministic_v6.h         → DeterministicV6, TraceLogger
        """

        from __future__ import annotations
        import json
        import logging
        import os
        import sys as _sys
        import time as _time
        from pathlib import Path
        from typing import Generator, Any
        {backend_imports}

        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger = logging.getLogger("{id}")

        # ── Native bridge ─────────────────────────────────────────────────────────
        _sys.path.insert(0, str(Path(__file__).parent))
        try:
            from micronaut_native import (  # type: ignore
                TodoCreator, TodoItem, TodoJsonSerializer,
                DeterministicV6, TraceLogger, NATIVE_AVAILABLE,
            )
            _NATIVE_BRIDGE = True
            _trace_logger  = TraceLogger(str(Path(__file__).parent / "{id}_trace.jsonl"))
            _todo_creator  = TodoCreator("{id}")
        except ImportError:
            _NATIVE_BRIDGE = False
            _trace_logger  = None
            _todo_creator  = None

        MICRONAUT_ID   = "{id}"
        MICRONAUT_NAME = "{name}"
        MICRONAUT_FOLD = "{fold_sym}"
        TOOLS          = {tools_repr}
        XJSON_PATH     = Path(__file__).parent / "{id}.xjson"


        class {name.replace(" ", "")}Bot:
            """Model-specific bot for {name}."""

            def __init__(self) -> None:
                self.system_prompt = "{escaped_prompt}"
                xjson = self._load_xjson()
                {backend_init}

            def _load_xjson(self) -> dict:
                if XJSON_PATH.exists():
                    return json.loads(XJSON_PATH.read_text(encoding="utf-8"))
                logger.warning(f"xjson not found: {{XJSON_PATH}}")
                return {{}}

            def stream(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7) -> Generator[str, None, None]:
                """Stream inference tokens for the given prompt."""
                {backend_infer}

            def infer(self, prompt: str, max_tokens: int = 2048,
                      temperature: float = 0.7) -> str:
                """Run inference and return full response string."""
                return "".join(self.stream(prompt, max_tokens, temperature))

            def dispatch(self, tool: str, args: dict) -> Any:
                """Dispatch a named tool call (from model_api_registry).
                
                Wraps result in DeterministicV6 V6 envelope when native bridge active.
                """
                if tool not in TOOLS:
                    raise ValueError(f"Unknown tool: {{tool}} — available: {{TOOLS}}")

                t0 = _time.perf_counter()
                try:
                    # Route tool calls to fold-appropriate handlers
                    if tool in ("emit_token", "stream_tokens"):
                        prompt = args.get("prompt", "")
                        result = list(self.stream(prompt))
                        output = {{"result": result}}
                    elif tool == "score_logits":
                        output = {{"scores": [], "note": "logit scoring requires local backend"}}
                    else:
                        prompt = f"[TOOL: {{tool}}]\\n{{json.dumps(args)}}"
                        output = {{"result": self.infer(prompt)}}

                    latency = (_time.perf_counter() - t0) * 1000
                    if _NATIVE_BRIDGE and _trace_logger:
                        _trace_logger.log_tool_execution(
                            tool, str(args), json.dumps(output), latency
                        )
                    return output
                except Exception as exc:
                    latency = (_time.perf_counter() - t0) * 1000
                    if _NATIVE_BRIDGE and _trace_logger:
                        _trace_logger.log_tool_execution(
                            tool, str(args), str(exc), latency, status="error"
                        )
                    raise

            def status(self) -> dict:
                return {{
                    "id":      MICRONAUT_ID,
                    "name":    MICRONAUT_NAME,
                    "fold":    MICRONAUT_FOLD,
                    "backend": self.backend,
                    "tools":   TOOLS,
                    "ready":   True
                }}


        # ── Fold dispatch shim ─────────────────────────────────────────────────────
        # Register in micronaut/src/orchestrator_bot.py:
        #   _MICRONAUT_TO_FOLD["{id}"] = "{fold_sym}"

        _bot: {name.replace(" ", "")}Bot | None = None


        def get_bot() -> {name.replace(" ", "")}Bot:
            global _bot
            if _bot is None:
                _bot = {name.replace(" ", "")}Bot()
            return _bot


        if __name__ == "__main__":
            import sys
            bot = get_bot()
            if len(sys.argv) > 1:
                prompt = " ".join(sys.argv[1:])
                for token in bot.stream(prompt):
                    print(token, end="", flush=True)
                print()
            else:
                print(json.dumps(bot.status(), indent=2))
    ''')


# ── Registry updaters ──────────────────────────────────────────────────────────

def update_api_registry(registry_path: Path, id: str, name: str, fold: str,
                        skill: str, tools: list, binds_to: str):
    if not registry_path.exists():
        print(f"  ! model_api_registry.json not found at {registry_path} — skipping")
        return

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    fold_sym = FOLD_MAP.get(fold.upper(), f"⟁{fold.upper()}_FOLD⟁")
    lane     = FOLD_TO_LANE.get(fold.upper(), "BATCH")

    if id in data.get("models", {}):
        print(f"  ~ {id} already in model_api_registry.json")
        return

    data.setdefault("order", []).append(id)
    data.setdefault("models", {})[id] = {
        "id":              id,
        "name":            name,
        "skill":           skill,
        "role":            "token_signal_generator",
        "fold":            fold_sym,
        "lane":            lane,
        "api_namespace":   f"/kuhul/api/models/{id}",
        "skill_namespace": f"/kuhul/api/skills/{skill}",
        "dispatch_path":   f"/kuhul/api/models/{id}/dispatch",
        "status_path":     f"/kuhul/api/models/{id}/status",
        "tool_path":       f"/kuhul/api/models/{id}/tools/{{tool}}",
        "aliases":         [f"/kuhul/api/skills/{skill}"],
        "tools":           tools,
        "brain_profile":   f"micronaut/brains/micronaut-profiles.json#{id}",
        "brain_intent":    f"micronaut/brains/meta-intent-map.json#{skill}",
        "registry_entry":  f"micronaut/micronaut.registry.xjson#{id}",
        "binds_to":        [binds_to]
    }

    registry_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  + Registered {id} in model_api_registry.json")


def update_micronaut_registry(registry_path: Path, id: str, name: str,
                               domain: str, role: str, responsibilities: list,
                               forbidden: list, binds_to: str):
    if not registry_path.exists():
        print(f"  ! micronaut.registry.xjson not found at {registry_path} — skipping")
        return

    data = json.loads(registry_path.read_text(encoding="utf-8"))

    if id in data.get("micronauts", {}):
        print(f"  ~ {id} already in micronaut.registry.xjson")
        return

    data.setdefault("micronauts", {})[id] = {
        "id":               id,
        "name":             name,
        "domain":           domain,
        "role":             role,
        "responsibilities": responsibilities or [f"process_{domain}"],
        "forbidden":        forbidden or ["alter_law", "branch_execution"],
        "binds_to":         binds_to
    }

    registry_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  + Registered {id} in micronaut.registry.xjson")


# ── CLI ────────────────────────────────────────────────────────────────────────

def cmd_new(args):
    root         = Path(args.project_root).resolve()
    id           = args.id
    name         = args.name
    fold         = args.fold.upper()
    backend      = args.backend
    domain       = args.domain or id.lower()
    skill        = args.skill or domain
    persona      = args.persona or f"Expert {name.lower()} assistant"
    tools        = (args.tools or "emit_token,stream_tokens").split(",")
    binds_to     = args.binds_to or "KUHUL_π"
    responsibilities = (args.responsibilities or f"infer_{domain}").split(",")
    forbidden    = (args.forbidden or "alter_law,branch_execution").split(",")

    # Expert list
    experts_raw  = args.experts or domain
    experts = []
    for i, e in enumerate(experts_raw.split(",")):
        e = e.strip()
        experts.append({"id": f"exp-{i}", "name": e.replace("_"," ").title(), "domain": e})

    # Backend options
    backend_opts = {}
    if args.weights:     backend_opts["weights"]     = args.weights
    if args.shard_dir:   backend_opts["shard_dir"]   = args.shard_dir
    if args.hf_repo:     backend_opts["hf_repo"]     = args.hf_repo
    if args.hf_file:     backend_opts["hf_file"]     = args.hf_file
    if args.model:       backend_opts["model"]       = args.model
    if args.api_base:    backend_opts["api_base"]    = args.api_base
    if args.max_tokens:  backend_opts["max_tokens"]  = args.max_tokens

    # Output directory — honour --out-dir if provided, else default to model/agents/<ID>
    if getattr(args, 'out_dir', None):
        out_dir = Path(args.out_dir).resolve() / id
    else:
        out_dir = root / "model" / "agents" / id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build xjson
    xjson = build_xjson(id, name, fold, backend, backend_opts, persona, domain, experts)
    xjson_path = out_dir / f"{id}.xjson"
    xjson_path.write_text(json.dumps(xjson, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created: {xjson_path}")

    # Build bots.py
    system_prompt = xjson["@agent.main"]["system_prompt"]
    bots_code = build_bots_py(id, name, fold, backend, backend_opts, tools, system_prompt)
    bots_path = out_dir / "bots.py"
    bots_path.write_text(bots_code, encoding="utf-8")
    print(f"Created: {bots_path}")

    # Copy native bridge alongside bots.py
    native_src = Path(__file__).parent.parent / "assets" / "micronaut_native.py"
    if native_src.exists():
        import shutil
        native_dst = out_dir / "micronaut_native.py"
        shutil.copy2(native_src, native_dst)
        print(f"Copied:  {native_dst}")

    # Registry updates
    api_reg = root / "micronaut" / "micronaut" / "model_api_registry.json"
    xjson_reg = root / "micronaut" / "micronaut" / "micronaut.registry.xjson"
    update_api_registry(api_reg, id, name, fold, skill, tools, binds_to)
    update_micronaut_registry(xjson_reg, id, name, domain, f"{domain}_processor",
                               responsibilities, forbidden, binds_to)

    # Remind about orchestrator_bot.py patch
    fold_sym = FOLD_MAP.get(fold.upper(), f"⟁{fold.upper()}_FOLD⟁")
    print(f"""
Done! One manual step:
  Add to micronaut/src/orchestrator_bot.py _MICRONAUT_TO_FOLD dict:
    "{id}": "{fold_sym}",
""")


def cmd_register(args):
    xjson_path = Path(args.xjson_path).resolve()
    root       = Path(args.project_root).resolve()

    if not xjson_path.exists():
        print(f"ERROR: {xjson_path} not found")
        sys.exit(1)

    data = json.loads(xjson_path.read_text(encoding="utf-8"))
    meta = data.get("@meta", {})
    id   = meta.get("id", xjson_path.stem)
    name = meta.get("name", id)

    # Infer fold from router if present
    router  = data.get("@moe.router", {})
    fold    = "COMPUTE"

    api_reg  = root / "micronaut" / "micronaut" / "model_api_registry.json"
    xjson_reg = root / "micronaut" / "micronaut" / "micronaut.registry.xjson"
    update_api_registry(api_reg, id, name, fold, id.lower(), ["emit_token"], "KUHUL_π")
    update_micronaut_registry(xjson_reg, id, name, id.lower(), "token_signal_generator",
                               [f"infer_{id.lower()}"], ["alter_law"], "KUHUL_π")
    print("Done.")


def cmd_list(args):
    root = Path(args.project_root).resolve()
    api_reg = root / "micronaut" / "micronaut" / "model_api_registry.json"
    if not api_reg.exists():
        print("model_api_registry.json not found")
        return
    data = json.loads(api_reg.read_text(encoding="utf-8"))
    print(f"\nRegistered micronauts ({len(data.get('models', {}))}):")
    for mid in data.get("order", []):
        m = data["models"].get(mid, {})
        print(f"  {mid:12s}  {m.get('fold',''):<25s}  {m.get('role','')}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Micronaut Model Scaffold CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # new
    p = sub.add_parser("new", help="Scaffold a new micronaut")
    p.add_argument("id",          help="Micronaut ID, e.g. SM-2")
    p.add_argument("name",        help="Human name, e.g. SummarizationMicronaut")
    p.add_argument("--fold",      default="COMPUTE",
                   choices=["CONTROL","DATA","TIME","STATE","STORAGE","COMPUTE","PATTERN","UI","META"])
    p.add_argument("--backend",   required=True,
                   choices=["local_gguf","local_scxqdds","fetch_oss",
                            "api_openai","api_anthropic","api_ollama","api_custom"])
    p.add_argument("--domain",    help="Domain name (default: id.lower())")
    p.add_argument("--skill",     help="Primary skill name (default: domain)")
    p.add_argument("--persona",   help="Agent persona description")
    p.add_argument("--experts",   help="Comma-separated expert domain names")
    p.add_argument("--tools",     help="Comma-separated tool names (default: emit_token,stream_tokens)")
    p.add_argument("--binds-to",  help="Parent system (default: KUHUL_π)")
    p.add_argument("--responsibilities", help="Comma-separated responsibility verbs")
    p.add_argument("--forbidden", help="Comma-separated forbidden operations")
    # Backend-specific
    p.add_argument("--weights",   help="local_gguf: path to .gguf file")
    p.add_argument("--shard-dir", help="local_scxqdds: path to shard directory")
    p.add_argument("--hf-repo",   help="fetch_oss: HuggingFace repo id (org/model)")
    p.add_argument("--hf-file",   help="fetch_oss: filename in repo (default: model.gguf)")
    p.add_argument("--model",     help="api_*: model identifier")
    p.add_argument("--api-base",  help="api_custom/ollama: base URL")
    p.add_argument("--max-tokens",type=int, help="Max tokens for API backends")
    p.add_argument("--project-root", default=".")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory for xjson+bots.py (default: <project-root>/model/agents/<ID>). "
                        "Pass '.' to scaffold directly into the current terminal directory.")

    # register
    r = sub.add_parser("register", help="Register existing xjson into registries")
    r.add_argument("xjson_path")
    r.add_argument("--project-root", default=".")

    # list
    l = sub.add_parser("list", help="List registered micronauts")
    l.add_argument("--project-root", default=".")

    args = parser.parse_args()
    if args.cmd == "new":       cmd_new(args)
    elif args.cmd == "register": cmd_register(args)
    elif args.cmd == "list":    cmd_list(args)


if __name__ == "__main__":
    main()
