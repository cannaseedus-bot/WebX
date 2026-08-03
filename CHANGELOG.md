# CHANGELOG — KUHUL WebX-3D

WebX-3D is the canonical open-source release of the entire KUHUL ML Micronaut + Supernaut system.
All prior releases are archived in `docs/releases/`. Each version's innovations are documented there.

---

## v3.5.0 — WebX-3D (current, open-source baseline)
**D12WebX + KUHUL 3D — GPU computing in the browser**

- Full K'UHUL++ v2.0 compiler: lexer → parser → semantic → IR → codegen (HLSL/WGSL/JS)
- IR: SSA-style instructions, 7 geometric operators (⊗ ⊕ ⊖ ⊛ ⊜ ⊝ ⊞), optimization passes
- 6 glyph primitives: ⤍ ↻ ⟲ ∿ ⊙ ≋ (vector-encrypt, rotation-compress, spherical, torsion, radial, wave)
- MindBinder: 8-phase K++ unified geometric runtime
- D12WebX: SharedArrayBuffer + Atomics + Web Workers (95% native D3D12 performance in browser)
- SVG-3D tensor serialization (encodeToSVG / decodeFromSVG)
- Grammar: KUHUL-LLM.ebnf v7.0.0 with full semantic schema

---

## Release Lineage (archived, merged into WebX-3D)

| Version | Tag | Core Innovation |
|---------|-----|----------------|
| v0.1.0 | xvm-cpu-thread-cluster | XVM 32-fiber CPU cluster VM, SMGM-16 (52-card), SCX runtime, manifold opcodes |
| v0.1.0 | igpu-trainer | D3D11 iGPU GPT-2 trainer, full backprop, 15 HLSL compute shaders, GPU Adam |
| v0.1.1 | igpu-trainer-xjsl | XJSL (Cross-platform Shader JSON Language), WebGPU lowering pipeline |
| v0.2.0 | kuhul-directx-native | Glyph IPC (named file mapping), INT4 lane VM, field system topology |
| v1.0.0 | PowerShell-LLM | Semantic Reader Law, tiny.x + .brain format, TGUIX, PRIME-1 factory |
| v0.1.0 | micronaut-factory | Authority-based micronaut instantiation pattern |
| SCX.v1.0.0 | — | SCXT tensor substrate (Q4_BLOCK/INT4/BF16), BLAKE3 integrity, SCXTOK BPE tokenizer |
| SCXRuntime.v1.0.0 | — | SCX runtime contract, KBC1 bytecode, DDS streaming, int4_matmul.cso |
| KUHUL.v1.0.0 | — | KUHUL ISA, 4 fold shader CSOs, kuhulc.cjs compiler |
| KUHUL.EXE.v3.0.0 | — | Swarmable micronaut HTTP server, hot-reload from .md, π-phase arrays |
| Agents.NET.v1.0.0 | — | .NET multi-provider agent framework (11 providers), concurrent orchestration |
| NLOHMANN.JSON.v1.0.0 | — | JSON dependency for native builds |
| v3.0.0 | agentic-micronaut | 12-core micronaut (TOOL/AGENT/SKILL/etc.), 8-expert HLSL pipeline, coding DAG |
| v3.1.0 | scx-moe | SCX MoE (8 experts, cs_5_0 DXBC), SwiGLU FFN, Expert 2 adapter injection |
| v3.2.0 | supernaut | Supernaut lifecycle (REGISTERED→ARCHIVED), sxme_host.dll, DispatchContext |
| v3.3.0 | scx-control-flow | Compound workflow loop, 42 orchestration flows, LMStudio SDK |
| **v3.5.0** | **WebX** | **Canonical OSS release** |

---

## Adapter Delta Weight System

The following domain LoRA adapters (rank-8, ~295K params each) load alongside the base foundation model:

| File | Domain | Training | Export |
|------|--------|----------|--------|
| `agents_adapter.pt` | Agent orchestration, goal decomposition | 2000 steps | agents_adapter.scxq2 (7.8MB INT4) |
| `commands_adapter.pt` | Intent parsing, schema validation | 2000 steps | commands_adapter.scxq2 (7.8MB INT4) |
| `micronauts_adapter.pt` | Micronaut dispatch, bot coordination | 2000 steps | micronauts_adapter.scxq2 (7.8MB INT4) |
| `tools_adapter.pt` | Tool calling, file I/O, coding commands | 2000 steps | tools_adapter.scxq2 (7.8MB INT4) |

Base: `final_3way.pt` — 52.95M params, step=27833, loss~0.0003, INT4→23.9MB SCXQ2

See `docs/releases/gpu-trainer.md` for full training history.
