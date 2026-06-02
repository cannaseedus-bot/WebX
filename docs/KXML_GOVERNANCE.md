# KXML Governance — What KXML Governs

KXML is not just a graph topology format. It is the **governance layer** for
all K'UHUL compute. Every node that computes, routes, or transforms must be
declared in KXML before it executes.

## KXML Governs

### 1. Standard Compute Nodes
Phase-gated execution: Pop → Wo → Sek → Ch'en → Xul.
Every attention, FFN, normalization, and activation node.

### 2. Numeric Micronauts
Mathematical specialist agents that govern numerical computation domains:

| Micronaut | Domain | Phase | Opcode |
|---|---|---|---|
| `fibonacci_fold` | Compression via golden ratio windowing | Sek | 0x2D (Act) |
| `zeckendorf_encode` | Non-consecutive Fibonacci sum encoding | Sek | 0x22 (Tok) |
| `pi_field` | Transcendental constants + geodesic curvature | Wo | 0x21 (Wey) |
| `mayan_fold` | Base-20 vigesimal + Long Count operations | Sek | 0x23 (Log) |
| `matmul_kernel` | Tiled matrix multiply (GEMM) | Sek | 0x30 (TPROD) |
| `geodesic_router` | Hyperbolic distance routing | Wo | 0x65 (Path) |
| `linalg_solver` | LU decomp, eigenvalue, OLS | Sek | 0x2A (Attn) |
| `dxsk_route` | Geodesic MoE expert routing via mesh | Sek | 0x65 (Path) |

Each Numeric Micronaut is declared as a KXML node:

```xml
<node id="fib_compress" phase="Sek" domain="compute" device="gpu">
  <mathml>
    <apply><times/><ci>F</ci><ci>n</ci></apply>
  </mathml>
  <bind from="tensor_in" to="tensor_folded" transform="fibonacci_fold"/>
  <soft_landing lipschitz="1.618"/>
</node>
```

### 3. DXSK Expert Nodes
DirectX Semantic Kernel fiber→expert routing nodes:
- `dxsk_run` — Execute KXML program on 1024-fiber kernel
- `dxsk_train` — CP-1 bridge JSON IPC training
- `dxsk_evolve` — Breed/mutate/prune genome (commands/skills/tools/plans)
- `dxsk_route` — Geodesic MoE token routing

### 4. Physics Field Nodes
Win2D field optimizer nodes that govern training dynamics:
- `attraction_well` — Gravity toward loss minimum (⟁Grav⟁)
- `scroll_inertia` — Adam momentum accumulation
- `wind_field` — L2 weight decay directional push
- `navigation_force` — Arrival LR decay steering

### 5. Specialist Micronaut Nodes
Dispatch nodes that route to fine-tuned specialist models:
- `coder_tool` — GPT-2 fine-tuned on 7M+ coding Q&A
- `math_tool` — GPT-2 fine-tuned on mathematical reasoning
- `kuhul_agent` — K'UHUL agent stack
- `kxj_store` — KXJ store interface

## KXML Node Lifecycle

Every node — whether a Numeric Micronaut, a DXSK expert, or a standard
attention block — follows the same phase lifecycle:

```
Pop  (load inputs, validate preconditions)
 ↓
Wo   (declare intent, bind parameters)
 ↓
Sek  (execute forward pass)
 ↓
Ch'en (compute backward pass / gradient)
 ↓
Xul  (emit outputs, verify soft landing)
```

Soft landing guarantee: `||∇f|| ≤ L × ||x||` (Lipschitz bound).
The `<soft_landing lipschitz="N"/>` attribute is verified at Xul.

## The kxml_settings.xml Type System

Win2D codegen reads `native/kxml/kxml_settings.xml` and auto-generates:
- C++ `KXMLMicronaut` subclass for each `<Struct>` node type
- JSON tool schema for toolcall routing model
- XCFE `@` namespace registration

Defining a new Numeric Micronaut is a single XML addition:
```xml
<Struct Name="FFT_NODE" ShouldProject="true" ProjectedNameOverride="FftNode">
  <Field Name="N_POINTS" Type="uint32" Default="512"/>
  <Field Name="PHASE"    Type="PHASE"  Default="Sek"/>
  <Field Name="GRAVITY"  Type="GRAVITY" Default="Normal"/>
</Struct>
```
→ codegen emits `KXMLNode_Fft` C++ class + tool schema + XCFE export automatically.
