# Release Audit: KUHUL.v1.0.0

**Path:** `releases/KUHUL.v1.0.0/`
**Audited:** 2026-05-28

---

## Artifacts

| File | Purpose |
|------|---------|
| kuhulc.cjs | K'UHUL compiler — Node.js CJS, 4-phase, 989 lines |
| kuhul_fold_compute.cso | Fold compute shader (D3D11, cs_5_0) |
| kuhul_fold_meta.cso | Fold metadata shader (D3D11, cs_5_0) |
| kuhul_fold_storage.cso | Fold storage shader (D3D11, cs_5_0) |
| glyph_compute.cso | Glyph stream shader (D3D12, cs_6_0) |

---

## K'UHUL Language

**6-phase compiler:** Lexer → Parser → Semantic → KSON Generator → (shader codegen → D3D11 upload — Phase 5-6 not in this release)

### Glyph Names (valid)

| Glyph | Role |
|-------|------|
| Pop | Function definition |
| Wo | Tensor declaration (`shape=`, `dtype=`) |
| Yax | Read/load |
| Ch'en / Chen | Write/store |
| Sek | Execute operation |
| Xul | Sync/end marker |
| K'ayab' | Loop |
| Kumk'u | (additional glyph) |
| Muwan | (additional glyph) |

### Operators

| Symbol | Type | Operation |
|--------|------|-----------|
| ⊗ | OP_TENSOR_DOT | tensor_dot |
| ∫ | OP_INTEGRAL | integrate |
| ∇ | OP_GRADIENT | gradient |
| ⊙ | OP_ATTEND | attend |
| → | OP_PIPE | pipe |

### KSON Output Format

```json
{
  "$schema": "https://kuhul.dev/kson/v1",
  "version": "1.0.0",
  "manifest": { "name", "type": "compute_kernel", "target": "directx_12" },
  "tensors": [{ "id", "glyph": "Wo", "role": "input|scratch", "shape", "dtype" }],
  "kernels": [{ "id", "glyph": "Pop", "entry": "<name>_CS", "thread_group": [16,16,1], "operations": [...] }],
  "schedule": { "dispatch": { "kernel", "grid": [64,64,1], "phase_gate": "π/2" } }
}
```

Operation phases (π-based): load → π/4 → π/2 → 3π/4 → π → store

---

## KUHUL ISA Stance (from docs/KUHUL_RUNTIME_AND_ISA.md)

KUHUL is treated as an ISA-like system when:
1. Language surface maps to finite μ-ops
2. Execution is domain-constrained (fold, cache, compute, glyph)
3. I/O is hash-locked + replay-traceable

Runtime: `kuhul-native` v1.0.0, D3D11, d3d-cso, shader_version=1

### Shader Artifact Roles

| Role | CSO | Notes |
|------|-----|-------|
| fold_storage | kuhul_fold_storage.cso | Persistent trunk activations |
| fold_compute | kuhul_fold_compute.cso | Fold compute ops |
| fold_meta | kuhul_fold_meta.cso | Router decision persistence |
| glyph_compute | glyph_compute.cso | cs_6_0 — D3D12/DXC path only |

Note: glyph_compute.cso uniquely requires cs_6_0 (DXC); the other three are cs_5_0 (FXC/D3D11). Intel HD 4600 may not support cs_6_0.

---

## Semantic Analyzer

- Validates glyph names against known set
- Validates `[Wo name shape=N,M dtype=float32]` — shape must be positive int tuple
- Validates `[Yax name]` / `[Ch'en name]` — symbol must be previously declared
- Tracks symbol table per-compilation unit

---

## Merge Target

`src/kuhul/kuhulc.js` — ES module port of kuhulc.cjs (removed `fs`/`path` CLI wrapper; pure in-memory `compileKUHUL(source) → KSON`).
