# D12WebX + KUHUL 3D — Capabilities Reference

## What Is D12WebX?

**D12WebX** is a geometric execution substrate that brings near-native GPU computing to the browser.  
It does **not** wrap WebGPU — it accesses GPU memory directly through three browser primitives:

| Primitive | Role |
|-----------|------|
| `SharedArrayBuffer` | GPU-accessible memory arena |
| `Atomics` | Lock-free synchronisation (no mutex overhead) |
| `Web Workers` | Parallel command dispatch |

The result is **~5 % API overhead** versus WebGPU's ~20 %, closing the gap with native DirectX 12 to 95 %.

---

## What Is KUHUL 3D?

**KUHUL 3D** is the symbolic control and transformation layer that sits on top of D12WebX.  
It exposes GPU-accelerated 3D operations through a deterministic **glyph grammar** — each glyph maps to a geometric transformation in the execution manifold **M**.

KUHUL does **not** render; it computes. Visualisation is an optional projection step.

---

## The Architecture (Four Planes)

```
K'uhul (Control Grammar)
        ↓
Geometric IR — Tensor operations in M (⊗ ⊕ ⊖ ⊛ ⊜ ⊝ ⊞)
        ↓
SVG-3D  — Canonical tensor serialisation (storage, not execution)
        ↓
WASM / WebGPU / Native  — Execution back-ends
        ↓  (optional)
DOM / Canvas / WebGL    — Projection for visualisation
```

### Invariants

1. **SVG-3D is storage, not execution** — it encodes tensors geometrically; it does not run code.
2. **K'uhul is control, not compute** — it defines phase ordering and legality; it does not perform maths.
3. **M is the execution space** — all tensors are embedded in M; all operations happen inside M.
4. **Rendering is optional** — visualisation is a projection from M; the substrate runs headless.
5. **All operations are geometric** — there are no scalars; everything lives in a manifold.

---

## K'uhul Control Grammar

K'uhul is a deterministic grammar for sequencing tensor operations through **phase cycles**.

| Token | Meaning |
|-------|---------|
| `[Pop]` | Enter scope / begin fold |
| `[Xul]` | Exit scope / end fold |
| `[Wo]` | Allocate tensor in manifold M |
| `[Yax]` | Read tensor from M |
| `[Ch'en]` | Write tensor to M |
| `[Sek]` | Apply geometric operator |
| `[K'ayab']` | Begin phase iteration |
| `[Kumk'u]` | End phase iteration |
| `[Muwan]` | Invoke folded (sub-)process |

### Example — Dense Layer as Geometry

```kuhul
[Pop dense_layer]
  [Yax X]   → [Sek load_from_manifold]
  [Wo W]    → [Sek encode_as_transform]
  [Wo b]    → [Sek encode_as_translation]

  [Yax X]   → [Sek ⊗ W]  → [Ch'en XW]     # matrix multiply
  [Yax XW]  → [Sek ⊕ b]  → [Ch'en XWb]    # translate
  [Yax XWb] → [Sek ⊝ relu] → [Ch'en out]  # ReLU = constraint clamp
[Xul]
```

---

## Geometric Operators (in M)

| Symbol | Name | Meaning |
|--------|------|---------|
| `⊗` | Geometric product | Matrix / tensor multiplication |
| `⊕` | Manifold composition | Additive overlay in shared space |
| `⊖` | Difference | Subtraction in M |
| `⊛` | Constraint projection | Project tensor onto valid subspace |
| `⊜` | Constraint validation | Assert tensor satisfies invariants |
| `⊝` | Clamp / threshold | Restrict tensor to bounded region |
| `⊞` | Fold (accumulate) | Reduce tensors across a dimension |

---

## SVG-3D Tensor Encoding

SVG-3D is the on-disk / in-memory **storage format** for tensors. Each SVG element encodes one or more tensors geometrically:

| SVG Element | Tensor Role |
|-------------|-------------|
| `<circle cx cy [cz] r data-*>` | Point cloud — position = coordinates in M, `r` = norm/density |
| `<path d data-*>` | Adjacency / dataflow topology — curvature ∝ computational weight |
| `<g transform class>` | Composite tensor / fold boundary — `transform` = manifold mapping |
| `<torus major minor>` | Memory hierarchy — major-radius = bandwidth, minor-radius = latency |

**Compression insight:** Because tensors share a schema (glyph definitions, manifold bounds, tensor types), each individual tensor only needs to store its **deltas** from the schema.  
This yields **~90 % compression naturally** — not clever encoding, just inherent geometric structure.

---

## KUHUL 3D Glyph Operations

| Glyph | Name | Operation | Target Time |
|-------|------|-----------|-------------|
| `(⤍)` | Vector Encrypt | Affine transform on vector field | 0.2 ms |
| `(↻)` | Rotational Compression | Geometry compression via rotation matrices | 0.8 ms |
| `(⟲)` | Spherical Loop | Spherical coordinate transformation | 1.2 ms |
| `(∿)` | Torsion Field | Torsion-based mesh deformation | 1.5 ms |
| `(⊙)` | Radial Projection | 3D→radial-basis projection | 0.5 ms |
| `(≋)` | Wave Modulation | Wave functions applied to mesh surfaces | 0.3 ms |

### JavaScript API

```js
import { KUHULEngine } from './src/kuhul.js';

const engine = new KUHULEngine();
await engine.init();

// Affine transform (0.2 ms target)
await engine.executeGlyph('⤍', mesh.vertices, transformMatrix);

// Geometry compression (0.8 ms target)
await engine.executeGlyph('↻', mesh, { angle: 45 });

// Torsion deformation (1.5 ms target)
await engine.executeGlyph('∿', mesh, { torsion: 0.3, axis: [0,1,0] });
```

---

## Phase Cycles

A **phase cycle** is the deterministic unit of execution. It is independent of the browser's animation frame — `requestAnimationFrame` may be used to *project* results, but execution happens inside M.

```
Phase 0    → Load tensors from M
Phase π/4  → Apply geometric operators
Phase π/2  → Validate constraints (⊜)
Phase 3π/4 → Commit results to M
Phase π    → Fold complete → advance to next cycle
```

---

## Memory Model

```
SharedArrayBuffer (GPU-accessible arena)
├── Tensor region  [offset 0 …]
├── Command ring   [offset N …]
└── Sync word      [offset N+M …] ← Atomics operate here
```

Allocation is **bump-pointer** (O(1)); synchronisation is a **single Atomics.store/load** — no mutex, no GC pressure.

---

## Performance Targets

| Operation | D12WebX | WebGPU | Native D3D12 |
|-----------|---------|--------|--------------|
| Buffer creation | < 1 ms | 1–2 ms | < 0.1 ms |
| Synchronisation | < 0.1 ms | 0.8 ms | < 0.01 ms |
| Compute dispatch | < 1 ms | 2–3 ms | < 0.5 ms |
| API memory overhead | ~5 % | ~20 % | 0 % |
| Throughput (tensors/ms) | ~950 | ~800 | ~1000 |

---

## Browser Requirements

| Feature | Minimum version |
|---------|----------------|
| `SharedArrayBuffer` | Chrome 91 / Firefox 79 / Safari 15.2 (requires COOP + COEP headers) |
| `Atomics` | Chrome 68 / Firefox 78 / Safari 15 |
| `Web Workers` | All modern browsers |
| `WebGPU` (optional compute back-end) | Chrome 113 / Edge 113 |

---

## Quick Start

```bash
git clone https://github.com/cannaseedus-bot/WebX.git
cd WebX
npm install
npm run build
# open examples/geometric-execution-demo.html in your browser
```

See also:
- [`docs/benchmarks.md`](benchmarks.md) — detailed performance comparisons  
- [`docs/use-cases.md`](use-cases.md) — real-world application patterns  
- [`examples/geometric-execution-demo.html`](../examples/geometric-execution-demo.html) — live manifold visualisation  
- [`examples/neural-network-demo.html`](../examples/neural-network-demo.html) — GPU neural network layer  
- [`examples/mesh-transform-demo.html`](../examples/mesh-transform-demo.html) — KUHUL 3D mesh transforms
