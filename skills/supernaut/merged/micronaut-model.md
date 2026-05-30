---
name: micronaut-model
description: "Factory for micronaut micro-models in the xjson model system. Use when defining a new model, choosing a weight backend, generating a model-specific bots.py, or registering in model_api_registry.json and micronaut.registry.xjson. Handles all three backend modes: local weights (GGUF/SCXQDDS), API stream (OpenAI/Anthropic/Ollama/custom), and GPT OSS fetch from HuggingFace. Trigger on: 'create a micronaut', 'new micro-model', 'add bot model', 'xjson model', 'scaffold bots.py', 'gpt oss weights', 'huggingface weights', 'api stream model', 'register micronaut', 'add to model registry', 'build fold model', 'new expert model', 'model with local weights', 'attach api backend', 'model bot', 'phi-3 model', 'ollama micronaut'."
metadata:
  short-description: Define and scaffold xjson micro-models with any backend
---

# Micronaut Model Factory

Scaffolds the full micronaut lifecycle: xjson definition → weight backend → bots.py → registry.

## Quick Start

```bash
# Scaffold a new micronaut with local GGUF weights
python scripts/scaffold_micronaut.py new <ID> <Name> --backend local_gguf --weights models/my.gguf --fold COMPUTE

# Scaffold with OpenAI API stream backend
python scripts/scaffold_micronaut.py new <ID> <Name> --backend api_openai --model gpt-4o --fold COMPUTE

# Scaffold with HuggingFace OSS weights (downloads on first run)
python scripts/scaffold_micronaut.py new <ID> <Name> --backend fetch_oss --hf-repo microsoft/Phi-3-mini-4k-instruct --fold COMPUTE

# Scaffold with SCXQDDS local shards
python scripts/scaffold_micronaut.py new <ID> <Name> --backend local_scxqdds --shard-dir models/shards/mymodel/ --fold COMPUTE

# Register an existing xjson into both registries
python scripts/scaffold_micronaut.py register model/agents/<ID>/<ID>.xjson

# Fetch GPT OSS weights from HuggingFace
python scripts/fetch_oss_weights.py --repo <hf-repo-id> --file <filename> --out models/oss-cache/
```

## What Gets Generated

For each new micronaut (`<ID>`):

```
model/agents/<ID>/
├── <ID>.xjson          — model definition (lanes, phases, experts, edges)
└── bots.py             — model-specific inference bot
```

Plus entries written into:
- `micronaut/micronaut/model_api_registry.json` — pool registration (dispatch_path, tools, api_namespace)
- `micronaut/micronaut/micronaut.registry.xjson` — law registration (role, responsibilities, forbidden)

## Backend Options

| Backend | When to use | Key field |
|---|---|---|
| `local_gguf` | Local .gguf/.mgguf file ready | `path: models/x.gguf` |
| `local_scxqdds` | SCXQDDS tensor shards | `shard_dir: models/shards/x/` |
| `fetch_oss` | Pull from HuggingFace on first run | `hf_repo: org/model-name` |
| `api_openai` | OpenAI API stream | `model: gpt-4o` |
| `api_anthropic` | Anthropic API stream | `model: claude-sonnet-4-6` |
| `api_ollama` | Local Ollama server | `model: phi3:mini` |
| `api_custom` | Any OpenAI-compat endpoint | `api_base: http://...` |

Full field schemas for each backend: see `references/backends.md`

## xjson Format

Full format reference with all `@`-key semantics: see `references/xjson-format.md`

## bots.py Pattern

The `assets/bots_template.py` is the base. Each generated bots.py customizes:
- Backend loader class (GGUF/SCXQDDS/API)
- Model-specific tools list (matches `model_api_registry.json tools`)
- System prompt from `@agent.main.system_prompt`
- Fold identifier (matches `_MICRONAUT_TO_FOLD` mapping)
- Streaming handler (for API backends) or token sampler (for local)

## Fold Mapping

```python
_MICRONAUT_TO_FOLD = {
    "CM-1": "⟁CONTROL_FOLD⟁",   "PM-1": "⟁DATA_FOLD⟁",
    "TM-1": "⟁TIME_FOLD⟁",      "HM-1": "⟁STATE_FOLD⟁",
    "SM-1": "⟁STORAGE_FOLD⟁",   "MM-1": "⟁COMPUTE_FOLD⟁",
    "XM-1": "⟁PATTERN_FOLD⟁",   "VM-1": "⟁UI_FOLD⟁",
    "VM-2": "⟁META_FOLD⟁",
    # Add new micronauts here
}
```

New models append their `"<ID>": "⟁<FOLD>_FOLD⟁"` entry to this dict in `micronaut/src/orchestrator_bot.py`.

## Popular OSS Weight Sources

See `references/backends.md#oss-weight-catalog` for tested HuggingFace repos and license summary.
