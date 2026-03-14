# D12WebX + KUHUL 3D — Real-World Use Cases

## Overview

D12WebX + KUHUL 3D targets any workload that needs **GPU-speed computation inside a browser tab** without shipping a native binary. The geometric execution model (K'uhul → Geometric IR → SVG-3D → Execution back-end) applies uniformly across domains.

---

## 1. Game Development

### Real-Time Mesh Deformation

Stream character meshes into the manifold M, apply KUHUL torsion-field glyphs per frame, and commit results back — all without round-tripping through the browser's WebGPU process.

```js
// Each animation frame
const mesh = manifold.getTensor('character_mesh');
await kuhul.executeGlyph('∿', mesh, {
  torsion: Math.sin(t) * 0.4,
  axis: [0, 1, 0]
});
manifold.commit(mesh);
renderer.draw(manifold.project(mesh)); // optional projection
```

Target: 60 fps on a 50 000-vertex mesh at < 2 ms deformation cost.

### GPU Particle Systems

Encode particle positions as a point cloud in M, step the simulation with a `⊗` geometric-product update, and project to screen.

```js
// Particle step
const particles = manifold.getTensor('particles');
await manifold.applyOperator('⊗', particles, velocityTransform);
await manifold.applyOperator('⊕', particles, gravityDelta);
```

### Physics Simulation

Rigid-body constraints are naturally expressed as `⊜` (constraint validation) and `⊛` (constraint projection) operators — the same primitives used for neural network weight clipping.

---

## 2. Machine Learning

### Neural Network Inference

Each layer is a geometric transformation in M:

| Layer type | Geometric operation |
|------------|---------------------|
| Dense / Linear | `⊗` (geometric product) + `⊕` (bias translation) |
| ReLU | `⊝` (clamp to non-negative subspace) |
| BatchNorm | `⊛` (project onto unit-norm subspace) |
| Attention | `⊗` Q⊗K followed by `⊝` softmax clamping |

```kuhul
[Pop inference_pass]
  [K'ayab' layer_loop]
    [Yax input]   → [Sek ⊗ W]   → [Ch'en h]
    [Yax h]       → [Sek ⊕ b]   → [Ch'en h]
    [Yax h]       → [Sek ⊝ relu] → [Ch'en h]
  [Kumk'u layer_loop]
[Xul]
```

No special ML runtime needed — the geometric substrate handles everything.

### On-Device Fine-Tuning

Because SVG-3D compresses weight matrices 10×, a BERT-base model (110 MB) fits in ~10 MB of `SharedArrayBuffer`. Fine-tuning gradients are computed as geometric differences (`⊖`) in M.

### Streaming Inference

The phase-cycle model naturally supports streaming: each cycle processes one token (or one frame), commits results, and advances. Latency is bounded by the per-cycle time, not by batch size.

---

## 3. Scientific Computing

### Fluid Dynamics

Velocity and pressure fields are point clouds in M. Each simulation step applies:

1. `⊗` — advection (velocity × transform)
2. `⊝` — boundary clamping
3. `⊛` — divergence-free projection (pressure solve)
4. `⊕` — external forces

All steps run inside the manifold; the browser never touches raw float arrays.

### Volumetric Rendering

Voxel grids are encoded as `<torus>` elements in SVG-3D (major-radius = grid extent, minor-radius = voxel size). Rendering is a projection step that fires a WASM ray-marcher against the manifold.

### Point Cloud Processing

LiDAR scans or photogrammetry outputs (millions of points) load as a single SVG-3D `<circle>` group. KUHUL's `⊙` radial-projection glyph downsamples to target density in < 1 ms per 10 000 points.

---

## 4. Data Visualisation

### Large Dataset Rendering

A 1 M-row DataFrame encodes as a point cloud in M. Filtering, aggregation, and projection all happen in the manifold — the DOM only receives the final N visible points.

```js
const data = manifold.allocateTensor('dataset', rows);
await manifold.applyOperator('⊛', data, filterConstraint);   // filter
await manifold.applyOperator('⊞', data, aggregateFold);      // aggregate
const visible = manifold.project(data);                        // render
renderEngine.draw(visible);
```

### Interactive 3D Exploration

Camera transforms are `⊗` operations in M — the same primitive used for ML layers. Pan/zoom/rotate is free once the geometric substrate is running.

### Real-Time Streaming Data

Each new data batch is a `⊕` (manifold composition) into the existing tensor. No re-allocation, no re-upload: the bump-pointer allocator appends in O(1).

---

## 5. Creative / Generative Art

### Procedural Geometry

KUHUL glyphs compose naturally:

```
(⤍) → affine warp the base mesh
(⟲) → apply spherical symmetry
(∿) → add torsion for organic feel
(≋) → wave-modulate the surface
```

Each composition is deterministic and phase-cycle-reproducible — the same K'uhul program always produces the same geometry.

### Generative Music Visualisation

Audio FFT data encodes as a 1D point cloud; KUHUL maps frequency bins to 3D positions via `⊙` radial projection, producing real-time reactive geometry at < 0.5 ms per frame.

---

## 6. Development Tooling

### In-Browser Profilers

Because every tensor is in M and every operation is geometric, a debugger can inspect M at any phase boundary — no special instrumentation needed.

### Differential Testing

Two versions of a model / shader / simulation produce tensors in M. Their difference is a single `⊖` operation, rendering the divergence as a visual heat-map.

### Headless CI

The substrate runs without a display. CI pipelines can execute K'uhul programs, capture SVG-3D snapshots of M, and diff them against golden files — GPU compute in CI, no GPU required.

---

## Getting Started with a Use Case

1. Pick a use case above.
2. Open the relevant demo:
   - [`geometric-execution-demo.html`](../examples/geometric-execution-demo.html) — manifold fundamentals
   - [`neural-network-demo.html`](../examples/neural-network-demo.html) — ML inference
   - [`mesh-transform-demo.html`](../examples/mesh-transform-demo.html) — 3D mesh transforms
3. Read [`capabilities.md`](capabilities.md) for the full API.
4. Check [`benchmarks.md`](benchmarks.md) for realistic performance expectations.
