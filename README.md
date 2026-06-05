# KUHUL WebX-3D  ·  v3.5.0

> Canonical open-source release of the KUHUL ML Micronaut + Supernaut geometric compute system.

---

## Quick Start

```bat
START.bat
```

Opens the landing page at `http://127.0.0.1:7430` and the trainer bridge at `http://127.0.0.1:7431`.

Requires **Node.js** (any modern version).

---

## What's Inside

### App Shell (`index.html`)

- **Top bar** — KUHUL logo (far left), nav links, version badge, server status dot, user widget (right)
- **Left sidebar** — collapsible sections: Demos · JS Examples · Source · Docs · Registry
- **Content area** — iframe for live demos, markdown renderer for docs, JSON viewer for registry files
- **SVG-3D splash** — live K'uhul 7-brain canvas animation on the home page (uses `src/svg3d.js` projection math)
- **Hash routing** — `#app`, `#trainer`, `#demo:<name>`, `#doc:<name>`

### 3D Runtime (`src/index.html`)

Open via **`⟁ Launch 3D App`** or `http://127.0.0.1:7430/app`

- 3D perspective canvas — 7 brain organs, BRAIN_EDGES, field particles, orbit drag, scroll-to-zoom
- **3-tab left panel**: K3D compiler · Train · Fields
- **Train tab** — 7 training modes with mode-aware config fields and live SSE loss canvas
- **Right panel** — 7 brain mini-renderers (one per brain organ)
- **8-phase K'ayab' runtime** — pip ring, auto-advance, phase label

### Training Modes (`native/trainer-server.cjs`)

| Mode | Engine | Micronaut |
|---|---|---|
| Causal GPU | `gpt2_trainer.exe` D3D11 | Coder pretrain |
| Shard Chain | `.exe` + `--shard` DDS | Large datasets |
| Tool-Call CPU | `finetune_toolcall_pt.py` | TC-1/TC-2, agent |
| Curriculum CPU | PyTorch easy→hard | Math micronaut |
| Glyph Pretrain | PyTorch KXML prefix | Kuhul/KXML |
| Fiber Chain | Multi-chunk orchestrated | Any large domain |
| **Hybrid iGPU→CPU** | `hybrid_math_micronaut.py` | **Math · Coder** |

The **Hybrid** mode runs `[Sek]` D3D11 iGPU pretrain → `[Ch'en]` CPU finetune with geodesic+ARC attention from `E:\models\GPT2\geodesic_cache\`.

### Domain JSONL Generator (`E:\models\GPT2\build_domain_jsonl.py`)

```bat
python E:\models\GPT2\build_domain_jsonl.py --domain all --n 200 --out domain_train.jsonl
```

Generates KXML-prefixed training records across 5 domains:
`powershell` · `coder` · `math` · `agent` · `kuhul`

PowerShell domain includes full error→retry cycles (bash fallback, python fallback, corrected cmdlet).

---

## File Map

```
v3.5.0-WebX/
├── START.bat                   ← launch everything
├── index.html                  ← landing page shell (SVG-3D splash + sidebar + nav)
├── server.cjs                  ← static file server port 7430 (COOP/COEP for SAB)
├── server.manifest.json        ← server config, routes, env
├── cache.manifest.json         ← static asset inventory + cache policy
│
├── src/
│   ├── index.html              ← 3D runtime app (open via /app)
│   ├── index.js                ← 172 exports — full library surface
│   ├── d12webx.js              ← SharedArrayBuffer GPU substrate
│   ├── mind-binder.js          ← 8-phase K++ unified runtime
│   ├── svg3d.js                ← SVG-3D tensor serialization
│   ├── geometric-operators.js  ← 7 geometric ops (⊗ ⊕ ⊖ ⊛ ⊜ ⊝ ⊞)
│   ├── field-system.js         ← π_field_v1 physics (4 field types)
│   └── k3d/compiler.js         ← K3D source → IR JSON
│
├── native/
│   ├── trainer-server.cjs      ← HTTP+SSE trainer bridge port 7431
│   ├── bin/
│   │   ├── gpt2_trainer.exe    ← D3D11 iGPU trainer (HD 4600 confirmed)
│   │   ├── cso/                ← 17 compiled DXBC shaders
│   │   └── tokens_*.bin        ← token datasets
│   ├── shaders/                ← 29 HLSL/WGSL source shaders
│   └── klsl/                   ← KLSL compiler (C++ source + examples)
│
├── tools/trainers/
│   ├── hybrid_math_micronaut.py    ← [Pop][Sek][Ch'en][Xul] pipeline
│   ├── finetune_toolcall_pt.py     ← PyTorch CPU finetune (geodesic attn)
│   ├── geodesic_attention_bridge.py← sphere projection + ARC weights
│   └── hybrid_train.py             ← generic hybrid orchestrator
│
├── examples/                   ← 11 standalone HTML demos
├── docs/                       ← 14 markdown docs
├── kuhul/                      ← K'UHUL++ compiler, runtime, stdlib
├── shaders/                    ← top-level HLSL shaders
└── registry/
    └── micronauts.registry.json← model manifest (TC-1/TC-2/MM-1/CM-1)
```

---

## Key APIs

```js
import { D12WebX, MindBinder, encodeToSVG, decodeFromSVG } from './src/index.js';

// GPU buffer
const gpu    = new D12WebX();
const buffer = gpu.createBuffer(65536);

// 8-phase geometric runtime
const binder = new MindBinder();
binder.beginPhase(0);
const T = binder.allocateTensor(1024 * 3);
binder.writeTensor(T, pointCloud);
binder.endPhase();
await binder.executeAllPhases();

// SVG-3D round-trip
const svg    = binder.serializeToSVG(T, { edges });
const { points, phases } = decodeFromSVG(svg);
```

---

## Screenshots

> _Add screenshots here — suggested captures:_
>
> 1. Landing page with SVG-3D splash animation + sidebar
> 2. 3D App — 7-brain orbit view with field particles
> 3. Train tab — Hybrid mode selected, loss canvas live
> 4. Docs — K'UHUL++ Language spec rendered in sidebar shell
> 5. MX2LM DirectWrite GPU demo in iframe

---

## Micronaut Models

See `registry/micronauts.registry.json` for the full model map.

| ID | Role | Best checkpoint |
|---|---|---|
| TC-1 | Base tool-caller (small) | `gpt2_small_ft_toolcall_fixed.safetensors` |
| TC-2 | Base tool-caller (medium) | `gpt2_medium_ft_toolcall.safetensors` |
| MM-1 | Math specialist | `math_micronaut/gpt2_medium_toolcall_s1500.safetensors` |
| CM-1 | Coder specialist | `coder_micronaut/dx11/model.safetensors` |
| GW-1 | Geodesic cache | `geodesic_cache/*.npy` (50257 sphere positions, k=32 kNN) |

---

## Release Lineage

See `CHANGELOG.md` for the full history from `v0.1.0-igpu-trainer` through `v3.5.0-WebX`.
