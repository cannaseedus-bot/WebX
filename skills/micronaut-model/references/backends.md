# Micronaut Backend Reference

Full field schemas, runtime requirements, and OSS weight catalog for all 7 micronaut backends.

---

## Backend: `local_gguf`

**Use when**: You have a local `.gguf` or `.mgguf` weight file ready.

**Runtime dependency**: `llama-cpp-python` (pip install llama-cpp-python)

**`@model.core` fields**:
```json
{
  "@xcfe": "IMMUTABLE",
  "backend": "local_gguf",
  "path": "models/my-model.gguf",
  "hash": "sha256:<sha256hex>",
  "vocab_size": 32000,
  "num_layers": 32,
  "num_experts": 1,
  "hidden_size": 4096,
  "context_length": 4096,
  "quantization": "Q4_K_M"
}
```

**`@backend` block**: Not required (local models don't need an HTTP backend block).

**bots.py init pattern**:
```python
from llama_cpp import Llama
self.model = Llama(model_path=model_core["path"], n_ctx=model_core["context_length"], n_gpu_layers=-1)
```

**Notes**:
- `n_gpu_layers=-1` uses all GPU layers (requires llama-cpp-python[cuda] or [metal])
- `hash` can be `sha256:pending` until you've verified the file
- Use `fetch_oss_weights.py` to download and auto-compute hash

---

## Backend: `local_scxqdds`

**Use when**: You have SCXQDDS tensor shard files from the native pipeline.

**Runtime dependency**: `scxqdds_vector_runner.exe` (artifacts/native-asx/)

**`@model.core` fields**:
```json
{
  "@xcfe": "IMMUTABLE",
  "backend": "local_scxqdds",
  "shard_dir": "models/shards/mymodel/",
  "shard_count": 8,
  "shard_format": "SCXQDDS-v1",
  "vocab_size": 32000,
  "num_layers": 8,
  "hidden_size": 1024,
  "dtype": "INT8"
}
```

**bots.py init pattern**:
```python
import subprocess, json
# scxqdds_vector_runner reads shards and returns token stream via stdout
self.runner = "artifacts/native-asx/scxqdds_vector_runner.exe"
self.shard_dir = model_core["shard_dir"]
```

**Notes**:
- SCXQDDS is the native INT8 tensor container format for the AS-XCFE pipeline
- Shards are produced by `scripts/gguf_stream_to_scxqdds.py`
- Use `npm run scxqdds:selftest` to verify shard integrity

---

## Backend: `fetch_oss`

**Use when**: You want to pull weights from HuggingFace on first run (no local files yet).

**Runtime dependency**: `huggingface_hub` + `llama-cpp-python`

**`@model.core` fields**:
```json
{
  "@xcfe": "IMMUTABLE",
  "backend": "fetch_oss",
  "hf_repo": "microsoft/Phi-3-mini-4k-instruct",
  "hf_file": "Phi-3-mini-4k-instruct-q4.gguf",
  "local_cache": "models/oss-cache/Phi-3-mini-4k-instruct-q4.gguf",
  "vocab_size": 32064,
  "num_layers": 32,
  "hidden_size": 3072,
  "quantization": "Q4_K_M",
  "license": "MIT"
}
```

**bots.py init pattern**:
```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
cache_path = Path(model_core["local_cache"])
if not cache_path.exists():
    hf_hub_download(repo_id=model_core["hf_repo"], filename=model_core["hf_file"],
                    local_dir=cache_path.parent)
self.model = Llama(model_path=str(cache_path), n_ctx=4096, n_gpu_layers=-1)
```

**Notes**:
- First run downloads the file (~2-8GB) — expect delay
- `license` field is REQUIRED — MIT/Apache 2.0 preferred; GPL requires isolation
- Use `fetch_oss_weights.py --list-catalog` to see tested repos and their licenses

---

## Backend: `api_openai`

**Use when**: Routing to OpenAI's API (gpt-4o, gpt-4.1, gpt-4-turbo, etc.)

**Runtime dependency**: `openai` (pip install openai)

**`@model.core` fields**:
```json
{
  "@xcfe": "IMMUTABLE",
  "backend": "api_openai",
  "model": "gpt-4o",
  "api_base": "https://api.openai.com/v1",
  "env_key": "OPENAI_API_KEY",
  "stream": true,
  "max_tokens": 4096,
  "temperature": 0.7
}
```

**`@backend` block** (required):
```json
{
  "type": "api_openai",
  "endpoint": "https://api.openai.com/v1/chat/completions",
  "stream": true
}
```

**bots.py init pattern**:
```python
import os
from openai import OpenAI
self.client = OpenAI(api_key=os.getenv(model_core["env_key"]))
```

---

## Backend: `api_anthropic`

**Use when**: Routing to Anthropic's API (claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5, etc.)

**Runtime dependency**: `anthropic` (pip install anthropic)

**`@model.core` fields**:
```json
{
  "@xcfe": "IMMUTABLE",
  "backend": "api_anthropic",
  "model": "claude-sonnet-4-6",
  "env_key": "ANTHROPIC_API_KEY",
  "stream": true,
  "max_tokens": 4096
}
```

**bots.py init pattern**:
```python
import os
import anthropic
self.client = anthropic.Anthropic(api_key=os.getenv(model_core["env_key"]))
```

**Notes**:
- Use `claude-sonnet-4-6` as default — best balance of speed and quality
- `claude-opus-4-6` for deep reasoning micronauts (PATTERN/META folds)
- `claude-haiku-4-5-20251001` for high-frequency lightweight tasks (DATA/STORAGE folds)

---

## Backend: `api_ollama`

**Use when**: Running a local Ollama server (`ollama serve`).

**Runtime dependency**: `ollama` running locally on port 11434

**`@model.core` fields**:
```json
{
  "@xcfe": "IMMUTABLE",
  "backend": "api_ollama",
  "model": "phi3:mini",
  "api_base": "http://localhost:11434/v1",
  "stream": true
}
```

**bots.py init pattern**:
```python
from openai import OpenAI
self.client = OpenAI(base_url=model_core["api_base"], api_key="ollama")
```

**Notes**:
- Ollama exposes an OpenAI-compatible API — use OpenAI client with custom base_url
- `api_key="ollama"` is a dummy key (Ollama doesn't validate it)
- Common models: `phi3:mini`, `llama3.1:8b`, `qwen2.5:7b`, `deepseek-r1:7b`, `mistral:7b`

---

## Backend: `api_custom`

**Use when**: Any OpenAI-compatible endpoint (LM Studio, text-generation-webui, vLLM, llama.cpp server, etc.)

**Runtime dependency**: `openai`

**`@model.core` fields**:
```json
{
  "@xcfe": "IMMUTABLE",
  "backend": "api_custom",
  "model": "local-model",
  "api_base": "http://localhost:8080/v1",
  "env_key": "LOCAL_API_KEY",
  "stream": true
}
```

**bots.py init pattern**:
```python
import os
from openai import OpenAI
self.client = OpenAI(base_url=model_core["api_base"], api_key=os.getenv(model_core["env_key"], "none"))
```

---

## OSS Weight Catalog

Tested HuggingFace repos. Run `python fetch_oss_weights.py --list-catalog` for the live list.

| Repo | Size | License | Recommended Use |
|---|---|---|---|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | **MIT** | Code, math, fast COMPUTE tasks |
| `microsoft/Phi-3-medium-4k-instruct` | 14B | **MIT** | Planning, reasoning |
| `microsoft/phi-2` | 2.7B | **MIT** | Lightweight classification |
| `Qwen/Qwen2.5-7B-Instruct-GGUF` | 7B | **Apache 2.0** | Multilingual, code, 32K context |
| `Qwen/Qwen2.5-3B-Instruct-GGUF` | 3B | **Apache 2.0** | Edge/low-memory micronauts |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | **Apache 2.0** | Structured instruction following |
| `lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF` | 7B | **MIT** | Reasoning/CoT, PATTERN fold |
| `NousResearch/Hermes-3-Llama-3.1-8B-GGUF` | 8B | ⚠ Llama License | Tool use, agentic tasks |
| `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | 8B | ⚠ Llama License | General purpose, 128K context |
| `google/gemma-2-9b-it-GGUF` | 9B | ⚠ Gemma Terms | Math/reasoning |

**License policy**:
- ✅ MIT, Apache 2.0 — use freely in any micronaut
- ⚠ Llama Community License — permitted for most uses; check at https://llama.meta.com/llama-downloads
- ⚠ Gemma Terms of Use — permitted with attribution; check terms
- ❌ GPL — requires source disclosure; must run in isolated process if used

---

## Backend Selection Guide

```
Is latency critical (< 100ms first token)?
├── YES → local_gguf or local_scxqdds
│          (GPU-accelerated local inference)
│
└── NO → Is offline operation required?
         ├── YES → local_gguf / local_scxqdds / fetch_oss
         └── NO → API stream
                  ├── Need best quality?          → api_anthropic (opus)
                  ├── Need good quality + speed?  → api_anthropic (sonnet) or api_openai (gpt-4o)
                  ├── High frequency, low cost?   → api_anthropic (haiku) or api_openai (gpt-4o-mini)
                  ├── Already running Ollama?     → api_ollama
                  └── Custom server?              → api_custom
```
