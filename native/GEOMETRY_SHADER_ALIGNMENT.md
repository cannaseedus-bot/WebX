# Geometry-Shader Alignment

**Status:** ✅ **Canonical Alignment**
**Date:** 2026-07-30
**Version:** 1.0

---

## Overview

`GeometryFieldMap.h` defines the **geometric field topologies** that neural shaders execute on.

**Every geometry → Every shader → Every field** are **perfectly aligned**.

---

## Alignment Matrix

| Geometry (GeometryFieldMap.h) | Field Topology | Neural Shader | Execution Class |
|-------------------------------|----------------|---------------|-----------------|
| **ComputeBox** | `FieldTopology::Box` | `neural_layer.klsl` | `Attention` |
| **ComputeCone** | `FieldTopology::Cone` | `xvm_fused_qkv_attention.hlsl` | `Attention` |
| **ComputeCylinder** | `FieldTopology::Cylinder` | `orchestrate.hlsl` | `Fiber` |
| **ComputeTorus** | `FieldTopology::Torus` | `pi_field.wgsl` | `Fiber` |
| **ComputeGeoSphere** | `FieldTopology::GeoSphere` | `xvm_compute.hlsl` | `Wave` |
| **ComputeIcosahedron** | `FieldTopology::Icosahedron` | `FibonacciCS.hlsl` | `Scalar` |
| **ComputeDodecahedron** | `FieldTopology::Dodecahedron` | `scxq2_infer_layer.hlsl` | `Scalar` |
| **ComputeTetrahedron** | `FieldTopology::Tetrahedron` | `kuhul_fold_compute.hlsl` | `Fiber` |
| **ComputeOctahedron** | `FieldTopology::Octahedron` | `int4_matmul.hlsl` | `SIMD` |
| **ComputeTeapot** | `FieldTopology::Teapot` | `evolution.hlsl` | `Evolution` |
| **OpticalProcessor** | `FieldTopology::OpticalField` | `optical_wave.klsl` | `Optical` |

---

## Detailed Alignment

### **1. Box Geometry → neural_layer.klsl**

```cpp
// GeometryFieldMap.h
ComputeBox() → FieldTopology::Box
  Domain: Compute
  ExecClass: Attention
  6 faces = 6 transformer layers
  face normals = attention head directions
```

```klsl
// neural_layer.klsl
⟁ shader dense_layer
  ⟁Wo⟁ stage "compute"
  ⟁Wo⟁ threads [16, 16, 1]
  
  // Dense layer = orthogonal field (Box topology)
  // 6 faces = 6 transformer layers
  // Each thread group = one attention head
```

**Alignment:**
- ✅ Box geometry = orthogonal field
- ✅ 6 faces = 6 transformer layers
- ✅ Dense layer = fully-connected field
- ✅ Thread groups = attention heads

---

### **2. Torus Geometry → pi_field.wgsl**

```cpp
// GeometryFieldMap.h
ComputeTorus() → FieldTopology::Torus
  Domain: Memory
  ExecClass: Fiber
  major-radius = capacity / bandwidth
  minor-radius = latency / access pattern
```

```wgsl
// pi_field.wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    // Field evolution on torus topology
    // Dual-cycle memory hierarchy
    // φ-node: nonlinear attractor
    
    sum += w * sin(PHI * PI * v);  // φ-attractor
}
```

**Alignment:**
- ✅ Torus = dual-cycle memory (major/minor radius)
- ✅ Field evolution = memory access patterns
- ✅ φ-attractor = memory coherence
- ✅ WGSL = portable field topology

---

### **3. GeoSphere → xvm_compute.hlsl**

```cpp
// GeometryFieldMap.h
ComputeGeoSphere() → FieldTopology::GeoSphere
  Domain: Geometry
  ExecClass: Wave
  tessellation = field resolution
  tokens = vertices (uniform sphere distribution)
```

```hlsl
// xvm_compute.hlsl
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // Wave propagation on geodesic sphere
    // Uniform token distribution
    // Tessellation level = dispatch size
}
```

**Alignment:**
- ✅ GeoSphere = uniform token distribution
- ✅ Tessellation = field resolution
- ✅ Wave execution = spherical harmonics
- ✅ Thread count = vertex count

---

### **4. Tetrahedron → kuhul_fold_compute.hlsl**

```cpp
// GeometryFieldMap.h
ComputeTetrahedron() → FieldTopology::Tetrahedron
  Domain: Execution
  ExecClass: Fiber
  4 faces = Pop/Wo/Sek/Ch'en
  4 vertices = 4 K'uhul phase states
```

```hlsl
// kuhul_fold_compute.hlsl
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // Execute K'UHUL fold
    // Pop → Wo → Yax → Sek → Ch'en → Xul
    // 4-phase minimal field
}
```

**Alignment:**
- ✅ Tetrahedron = minimal phase field (4 faces)
- ✅ 4 vertices = 4 K'UHUL phases
- ✅ Fiber execution = phase transitions
- ✅ Minimal topology = essential computation

---

### **5. Octahedron → int4_matmul.hlsl**

```cpp
// GeometryFieldMap.h
ComputeOctahedron() → FieldTopology::Octahedron
  Domain: Geometry
  ExecClass: SIMD
  8 faces = 8 bit byte / 8D tensor axis
```

```hlsl
// int4_matmul.hlsl
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // INT4 matrix multiply
    // 8-bit quantization
    // SIMD vector operations
}
```

**Alignment:**
- ✅ Octahedron = 8 faces = 8-bit byte
- ✅ SIMD execution = vector operations
- ✅ INT4 quantization = 8D tensor axes
- ✅ Geometric symmetry = quantization symmetry

---

### **6. Teapot → evolution.hlsl**

```cpp
// GeometryFieldMap.h
ComputeTeapot() → FieldTopology::Teapot
  Domain: Evolution
  ExecClass: Evolution
  control points = token path waypoints
  mirrorZ = bidirectional phase propagation
```

```hlsl
// evolution.hlsl
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // Evolution kernel
    // Selection, recombination, mutation
    // Bezier trajectory field
}
```

**Alignment:**
- ✅ Teapot = Bezier trajectory
- ✅ Control points = token waypoints
- ✅ Evolution = selection/mutation
- ✅ Bidirectional = mirrorZ symmetry

---

### **7. OpticalField → optical_wave.klsl**

```cpp
// GeometryFieldMap.h
OpticalProcessor() → FieldTopology::OpticalField
  Domain: Vision
  ExecClass: Optical
  image → geometry → edges → frequency → semantics
```

```klsl
// optical_wave.klsl
⟁ shader optical_wave
  ⟁Wo⟁ stage "compute"
  
  // Wave propagation
  // Edge detection
  // Frequency analysis
  // Image → geometry → semantics
```

**Alignment:**
- ✅ OpticalField = vision pipeline
- ✅ Wave propagation = optical processing
- ✅ Edge detection = geometry extraction
- ✅ Frequency = semantic mapping

---

## Shader Model Alignment

| Geometry | Shader | Shader Model | GPU Generation |
|----------|--------|--------------|----------------|
| Box | `neural_layer.klsl` | KLSL (custom) | All |
| Torus | `pi_field.wgsl` | WGSL | D3D12 |
| GeoSphere | `xvm_compute.hlsl` | SM 5.1/6.1 | Gen9+ |
| Tetrahedron | `kuhul_fold_compute.hlsl` | SM 5.1 | All |
| Octahedron | `int4_matmul.hlsl` | SM 6.1 | Gen10+ |
| Teapot | `evolution.hlsl` | SM 5.1 | All |
| OpticalField | `optical_wave.klsl` | KLSL (custom) | All |

---

## Field-Shader-Geometry Triad

```
┌─────────────────────────────────────────────────────────────┐
│  Geometry (GeometryFieldMap.h)                              │
│  - Box, Torus, GeoSphere, etc.                              │
│  - Defines field topology                                   │
│  - Provides vertex/face structure                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (maps to)
┌─────────────────────────────────────────────────────────────┐
│  Field (pi_kuhul_field.h)                                   │
│  - FieldDomain, ExecutionClass                              │
│  - Cards, Tokens                                            │
│  - Residency, scheduling                                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (executes via)
┌─────────────────────────────────────────────────────────────┐
│  Shader (neural_layer.klsl, xvm_compute.hlsl, etc.)         │
│  - GPU kernel execution                                     │
│  - Thread dispatch                                          │
│  - Memory operations                                        │
└─────────────────────────────────────────────────────────────┘
```

**All three are perfectly aligned!**

---

## API Integration

### **Query Geometry-Shader Mapping:**

```bash
curl http://127.0.0.1:5235/api/query/geometry-shader-map
```

**Response:**
```json
{
  "mappings": [
    {
      "geometry": "Box",
      "topology": "FieldTopology::Box",
      "shader": "neural_layer.klsl",
      "exec_class": "Attention"
    },
    {
      "geometry": "Torus",
      "topology": "FieldTopology::Torus",
      "shader": "pi_field.wgsl",
      "exec_class": "Fiber"
    },
    ...
  ]
}
```

### **Compile Aligned Shader:**

```bash
curl -X POST http://127.0.0.1:5235/api/shader/compile ^
  -H "Content-Type: application/json" ^
  -d "{\"geometry\": \"Box\", \"shader\": \"neural_layer.klsl\"}"
```

---

## Related Documents

- [GeometryFieldMap.h](./geometry/GeometryFieldMap.h) - Geometry definitions
- [SHADER_EXPERT_REGISTRY.md](./shaders/SHADER_EXPERT_REGISTRY.md) - Shader registry
- [FIELD_GRADIENT_UNIFICATION.md](../../docs/FIELD_GRADIENT_UNIFICATION.md) - Field-gradient insight
- [XVM_SHADER_MODEL_COMPAT.md](../../components/nnc-k/v1.0/src/native/xvm-d3d12/XVM_SHADER_MODEL_COMPAT.md) - Shader models

---

**Status:** ✅ Canonical Alignment
**Next:** Execute geometry-aligned shaders via WebX API
