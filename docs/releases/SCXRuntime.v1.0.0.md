# Release Audit: SCXRuntime.v1.0.0

**Path:** `releases/SCXRuntime.v1.0.0/`
**Audited:** 2026-05-28

---

## Artifacts

Binary-only runtime with C++ header contracts. No C++ source included.

| Artifact | Purpose |
|----------|---------|
| scx_runtime.exe | Inference + train_step entry binary |
| dxc.exe / dxcompiler.dll / dxil.dll | DXC shader compiler (D3D12/DXIL path) |
| dxv.exe | DXIL validator |
| include/*.h | C++ API contracts |
| shaders/*.hlsl + *.cso | GPU compute kernels (FXC + DXC compiled) |
| model/ | scx-moe-16l model artifacts |
| cache/sco-cache-index.json | SHA256 manifest (48 files, uppercase hex) |

---

## Runtime Contract (include/runtime.h)

```
scx_runtime.exe infer <prompt>        → string summary
scx_runtime.exe train_step <id> <p>  → JSON {loss}
run_train_from_shard(path, ctx=8, lr=1e-3) → JSON summary
```

`run_train_from_shard`: slides a context window of length `context_len` over packed INT8 token sequence; next-token prediction per window; compatible with SCXQDDS shards from `shard-artifacts.js`.

D3D12 runtime: `init_gpu_if_available()` → `run_d3d12(KBC1_Program)` → `run_cpu(KBC1_Program)` fallback.

---

## SCX IR (include/scx_ir.h)

Sits above KBC1 — abstract op-graph before KBC1 lowering.

| Op | Value | Description |
|----|-------|-------------|
| NOP | 0 | no-op |
| ROUTE | 1 | expert routing |
| FETCH | 2 | tensor slice load |
| DISPATCH | 3 | dispatch to expert |
| TENSOR_MATMUL | 4 | matrix multiply |
| TENSOR_ADD | 5 | element-wise add |
| MOE_ROUTE | 6 | MoE route |

SCXOperand: { u32, f32, str } — polymorphic arg.

---

## KBC1 (include/kbc1.h)

Adds OP_LINEAR (6) and OP_GELU (7) to the op set we already had from v0.2.0-kuhul-directx-native.

```
KBC1_Inst: uint16 op, uint16 argc, uint32 args[4]
```

Note: C header uses sequential ints (1..11), our JS KBC1_OP uses grouped hex (0x0001, 0x0010...). OP_LINEAR=0x0014, OP_GELU=0x0015 added to JS.

---

## ScxGraph (include/scxgraph.h)

```
ScxNode: { id, role, q, device, pos[2] }
  roles: vector, matrix, micronaut-di, intent, dispatch, executor
  q: int4, fp16, fp32
  device: cpu, gpu

ScxEdge: { id, from, to, type, metric, entropy, phase }
  types: brain-edge, geodesic-entropy-arc
```

---

## GTA1 Loader (include/gta1_loader.h)

GTA1 binary contains: header + topology XML block (kind=2) + FIELDS tensor metadata. Optional zstd compression.

GtaTensor: { id: uint32, shape[], dtype: uint16, q_scheme: uint16, scale: float }
GtaEdge types: brain-edge, geodesic-entropy-arc (same as ScxGraph)

---

## ManifestInfo (include/manifest_loader.h)

```
ManifestTensor: { id, shape[], dtype, q_scheme, source, data: bytes }
ManifestInfo: { ok, coord_frame, nodes, edges, tensors[], kbc1_bytes, scx_graph }
```

Model manifest kind: `scx.model.manifest.v1`
Artifacts: { kbc1, dds, config, experts, vocab }

---

## Model Config (model/config.xjson)

scx-moe-16l: hidden=2048, layers=16, experts=32, top_k=2, quant=int4
Runtime: gpu=d3d12, fallback=cpu, threads=auto, paging=true, kv_delta=int4

---

## KV Cache Delta (include/runtime_state.h)

INT4 delta encoding for KV cache streaming:
- `kv_delta_encode(v, prev)` → nibble [0..15] (4-bit two's complement)
- `kv_delta_decode(prev, nib)` → float
- Scale: 1 nibble step = 1/8 float

---

## DDS Stream (include/dds_stream.h)

Streaming byte-range reader over a `.dds` weight file. Provides:
- `open(path)`, `read(offset, dst, size)`
- `make_tiles(tileBytes)` → DDSPage[] for paged loading

Not portable to browser (filesystem dependent).

---

## Shaders Added

- `shaders/moe_route.hlsl` — top-K expert routing (cs_5_0)
- `shaders/kuhul_fold_compute.hlsl` — fold compute (KUHUL ISA)
- `shaders/kuhul_fold_meta.hlsl` — fold metadata persistence
- `shaders/kuhul_fold_storage.hlsl` — persistent trunk activations
- `shaders/matmul_int4.hlsl` — INT4 GEMM (note: duplicate of int4_matmul.hlsl with different name)

---

## sco-cache-index.json Format

- `@context`: "sco-cache-index://v1"
- `release`: { name, version, runtime_version }
- `files[]`: { path, sha256 (UPPERCASE), bytes }

Note: SHA256 values here are UPPERCASE; SCX.v1.0.0 uses lowercase. Both are valid — compare case-insensitively.

---

## Merge Targets

| Innovation | Target |
|-----------|--------|
| SCX IR | `src/scx/ir.js` |
| ScxGraph + GTA1 | `src/scx/graph.js` |
| ManifestInfo + model manifest schema | `src/scx/manifest.js` |
| KV cache delta INT4 | `src/scx/kv-delta.js` |
| KBC1 OP_LINEAR + OP_GELU | `src/kbc1/kbc1-program.js` |
| Shader files | `shaders/` |
| DDS stream | Not ported (filesystem, browser incompatible) |
| run_train_from_shard | Not ported (Windows exe; interface documented) |
