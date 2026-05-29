# Release Audit: Agents.NET.v1.0.0

**Path:** `releases/Agents.NET.v1.0.0/`
**Audited:** 2026-05-28

---

## Artifacts

| File | Purpose |
|------|---------|
| manifest.json | Release manifest — source_root points to `../../native/dotnet` |
| registry/agents-net.registry.json | Registry of all .NET agent/connector/experimental projects |
| dotnet-bridge/dotnet_http_bridge.js | Node.js HTTP bridge (uses `http` module) |
| dotnet-bridge/DotNetBridge.java | Java bridge via subprocess (`dotnet worker.dll payload`) |
| dotnet-bridge/Dispatcher.java | Java @op router |
| dotnet-workers/Workers/SharedMemoryStateReader.cs | Windows MMF reader (`Local\KuhulGeometricState`) |
| dotnet-workers/Workers/SyncWorker.cs | Tick-based worker (polls shared state, acts on tick change) |
| dotnet-workers/Workers/SecurityPlugin.cs | Semantic Kernel plugin for identity verification |
| dotnet-workers/HttpWorker/Program.cs | ASP.NET minimal API: POST /run → OpRouter.Dispatch |

---

## Core Innovations

### SharedStateHeader (Windows MMF)

Binary layout — `LayoutKind.Sequential, Pack=1`, 64 bytes total:

| Offset | Type | Field |
|--------|------|-------|
| 0 | uint32 | Version |
| 4 | uint32 | ActiveFold |
| 8 | uint32 | TickCount |
| 12 | float32 | Entropy |
| 16 | float32 | Attention |
| 20 | float32 | Pressure |
| 24 | float32[10] | Reserve |

Named mapping: `Local\KuhulGeometricState` (Windows named memory-mapped file).
High-precision poll: 1ms tick interval, 16ms display interval.

### SyncWorker Tick Model

Worker executes only when `TickCount` changes — not on wall-clock timer. This ensures:
- Zero spurious task executions (deterministic trigger)
- Natural back-pressure (fast producers → ticks accumulate)
- Semantic Kernel calls only happen when the C++ geometric kernel signals

### @op Dispatch Protocol

POST /run with JSON body containing `@op` field:

```json
{ "@op": "DOTNET_MATH_ADD", "a": 1.5, "b": 2.5 }
```

Supported ops:
- `DOTNET_RUN` — generic run
- `DOTNET_MATH_ADD` — scalar addition (demo)
- `DOTNET_SIMD_DOT` — SIMD dot product
- `DOTNET_TENSOR_MATMUL` — matrix multiply
- `DOTNET_GPU_INFO` — GPU capability query

Java bridge spawns `dotnet worker.dll payload` as subprocess and parses stdout JSON.
JS bridge uses `http.request` to POST to the HttpWorker.

### SecurityPlugin (Semantic Kernel)

Three `[KernelFunction]` methods:
- `VerifyIdentityAsync(identity, threshold=0.9)` — manifold coherence check against wallet/user ID
- `IssueSecuroLink(identity)` — generates a SecuroLink vault URL
- `RevokeIdentity(identity)` — revokes all vault links for identity

Coherence is a synthetic metric in the stub (real system: reads from SHM tensor region).

### Multi-Provider Architecture

From registry — agent projects:
- Abstractions, Core, Runtime.Abstractions, Runtime.Core, Runtime.InProcess
- YAML agent definitions (`Agents.Yaml`)
- CopilotStudio, Magentic, OpenAI, Orchestration

Connector projects: HuggingFace, Ollama, OpenAI, Google
Experimental: Orchestration.Flow, Process.Abstractions/Core/LocalRuntime/Dapr

---

## WebX-3D Port

Three pure ES modules:

**`src/agents-net/shared-state.js`**
- `SHARED_STATE_BYTES=64`, `SHARED_STATE_MMF_NAME`, `SHARED_STATE_OFFSETS`
- `readSharedState(buffer)` → typed state object
- `writeSharedState(state)` → Uint8Array
- `createSharedState(overrides)` → default state

**`src/agents-net/op-dispatcher.js`**
- `DOTNET_OPS` array, `DOTNET_OP_SCHEMAS` per-op required fields
- `validateOp(op)` → {ok, error}
- `dispatchOp(op, url)` — fetch-based (replaces Node `http.request`; works in browser + Node 18+)

**`src/agents-net/sync-worker.js`**
- `SyncWorker(stateSupplier, taskHandler)` — tick-based execution
- `start(pollMs)` / `stop()` — uses `setInterval`/`clearInterval`
- `SECURITY_PLUGIN_OPS` — plugin function names from SecurityPlugin.cs

---

## Notes

- The Windows MMF (`Local\KuhulGeometricState`) is native-only. The JS port accepts a caller-supplied `stateSupplier` function instead, enabling in-memory simulation or WebSocket state feed.
- The Dapr process runtime (`Process.Runtime.Dapr`) enables distributed micronaut orchestration outside the single-machine model.
- `scx_control_flow_release` in the registry links back to `v3.3.0-scx-control-flow` — these releases share the same SCX control flow specs.
