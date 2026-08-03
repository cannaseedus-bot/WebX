# K'UHUL Machine Learning Engine

## Architecture

The K'UHUL ML Engine is a physics-field-driven GPU training system built on
Win2D + D3D11 compute shaders. It was not designed top-down — it was
**revealed** through the convergence of Win2D's type system generator, the
π field physics system, and the KXML graph topology.

```
kxml_settings.xml           ← Win2D codegen spec (KXML node types)
      ↓ codegen.exe
C++ KXMLMicronaut classes   ← typed node wrappers (auto-generated)
      ↓
field_optimizer.hlsl/.cso   ← Win2D parallel physics optimizer
      ↓ D3D11 compute dispatch
gpt2_trainer.exe            ← GPU forward/backward (no CPU)
      ↓
πFieldCompositor.py         ← Python field orchestration
      ↓
hybrid_field_trainer.py     ← unified launcher
      ↓
Routing model               ← dispatches to specialist micronauts
    ├── coder_tool          ← GPT-2 fine-tuned on 7M+ coding Q&A
    └── math_tool           ← GPT-2 fine-tuned on mathematical reasoning
```

---

## Physics Fields → Optimizer Dynamics

The four Win2D physics fields from `brain-integration/field_system/` map
exactly to the four components of adaptive optimization:

| π Field Spec | Optimizer Role | HLSL Kernel |
|---|---|---|
| `attraction_well_spec.json` | Gravity toward loss minimum (⟁Grav⟁) | `attraction_well()` in `field_optimizer.hlsl` |
| `scroll_inertia_field_spec.json` | Adam momentum m1/m2 accumulation | `scroll_inertia()` |
| `wind_field_spec.json` | L2 weight decay (directional push) | `wind_field()` |
| `navigation_force_spec.json` | Arrival LR decay (cosine replacement) | `navigation_force_scale()` |

This mapping was **discovered, not designed**. The UI physics fields for
scroll momentum, wind forces, and navigation steering are structurally
identical to the optimizer dynamics of gradient descent.

---

## Internet Learning Pipeline

The ML engine is extended with an autonomous internet data harvesting layer for MM-CODER continuous training.

```
Boot
 └─ data-harvester.mjs starts (port 25120)
 └─ learning-engine.mjs starts (port 25121)

Every 5 minutes (K'ayab' learning_cycle):
  ⟁Pop⟁ harvest_knowledge
    → GitHub API (code repos, trending)
    → StackOverflow API (Python Q&A)
    → arXiv CS.LG (research papers)
    → HuggingFace API (model metadata)
    → Wikipedia, HN, NASA
  ⟁Sek⟁ rate_limited_fetch   (1 req/domain/sec)
  ⟁Wo⟁  data/harvested/batch_<ts>.jsonl

  ⟁Pop⟁ train_on_new_data
    → internet_harvester.py extracts (prompt, completion) pairs
    → writes E:\models\GPT2\med-GPT\training\harvest_<ts>.jsonl
    → learning-engine hot-swaps model
    → coordinator notified via POST /notify

  ⟁Xul⟁ — sleep 300s → next K'ayab' iteration
```

**Connection-aware:** harvesting pauses when offline; resumes automatically on reconnect.

**K'UHUL PS control:**
```powershell
. .\micronaut\kuhul\autonomous_learning.ps1
Start-AutonomousLearning          # infinite learning loop
Get-LearningStatus                # inspect harvester + engine
Invoke-HarvestKnowledge           # trigger one cycle manually
```

**Manifest:** `micronaut/internet-learning.xjson`

---

## Win2D Codegen → KXML Type System

Win2D's `tools/codegen/exe/` reads `Settings.xml` and emits typed C++
wrappers for D2D effects. The same pipeline reads `kxml_settings.xml` and
emits K'UHUL node types:

```xml
<!-- kxml_settings.xml — mirrors Win2D Settings.xml format -->
<Struct Name="ATTENTION_NODE" ProjectedNameOverride="AttentionNode">
  <Field Name="N_HEADS"  Type="uint32" Default="12"/>
  <Field Name="PHASE"    Type="PHASE"  Default="Sek"/>
  <Field Name="GRAVITY"  Type="GRAVITY" Default="Normal"/>
</Struct>
```

Any new KXML node type — `fibonacci_fold`, `dxsk_route`, `math_tool` —
is defined once in `kxml_settings.xml` and automatically generates:
- C++ `KXMLMicronaut` subclass
- JSON tool schema for `toolcall_data.py` TOOLS dict
- XCFE `@` namespace registration

---

## K'UHUL Gravity System

Each model layer has a `gravity_scale` (⟁Grav⟁ opcode 0x61):

| Layer | Scale | Behavior |
|---|---|---|
| token embedding | 0.5 | Half gravity — can float slightly |
| position embedding | 0.5 | Half gravity |
| attention | 1.0 | Normal — standard field dynamics |
| FFN / MLP | 1.0 | Normal |
| layer norm | 2.0 | Heavy — normalization must be precise |
| LM head | 2.0 | Heavy — output logits critical |
| debug/telemetry | 0.0 | Antigravity ⟁AntiGrav⟁ — bypasses update |

The field equation: **∇²Φ = ρ_gravity + ρ_antigravity**

Stable training requires `ratio = Σgravity / Σantigravity ≥ 10`.

---

## GPU Trainer (No CPU)

`native/gpu_trainer/gpt2_trainer.exe` runs entirely on D3D11 GPU:

```powershell
# Hybrid field training (GPU only)
python tools/trainers/hybrid_field_trainer.py `
  --model E:\models\GPT2\mini-GPT\model.safetensors `
  --data  E:\models\GPT2\coder_micronaut\tokens_coder_gpu.bin `
  --out_dir E:\models\GPT2\coder_micronaut\dx11 `
  --steps 5000 --lr 2e-5 --target_loss 0.5
```

- Forward + backward: D3D11 compute shaders (Intel HD 4600)
- Optimizer: `field_optimizer.cso` (Win2D parallel dispatch)
- CPU usage: ~8s total (launcher only)

---

## Specialist Micronauts

The routing model dispatches to:

| Tool | Model | Training Data |
|---|---|---|
| `coder_tool` | GPT-2 @ `E:\models\GPT2\coder_micronaut\` | 80k coder Q&A + 10k KodCode (verified) |
| `math_tool` | GPT-2 @ (pending) | Math xshard + reasoning data |
| `kuhul_agent` | Existing K'UHUL stack | XCFE @ tensor grammar |

---

## Files

```
native/win2d/shaders/
  field_optimizer.hlsl   — Win2D physics field optimizer (source)
  field_optimizer.cso    — Compiled cs_5_0 shader
  FibonacciCS.hlsl       — Batched Fibonacci GPU kernel
  StabilizeCS.hlsl       — Logit clamp + gradient clip

native/win2d/field_system/specs/
  attraction_well_spec.json    — Gravity field spec
  scroll_inertia_field_spec.json — Momentum field spec
  wind_field_spec.json         — LR direction field spec
  navigation_force_spec.json   — Arrival steering spec

native/kxml/
  kxml_settings.xml      — KXML node codegen spec (Win2D format)
  kxml_directxmath_runtime.h
  kxml_directxmath_bridge.h
  fibonacci_fold.h
  kuhul_functions.h
  kuhul_tool_runtime.h

tools/trainers/
  hybrid_field_trainer.py        — Unified GPU + field launcher
  layered_train.py               — Curriculum: instruct→toolcall
  field_composition.py           — πFieldCompositor (Python)
  field_composition_enhanced.py  — Extended field system
```
