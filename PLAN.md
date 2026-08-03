# WebX Runtime Plan

## Goal

Make `kuhul_engine.exe` the native execution host for a field-centric,
multi-model runtime:

```text
Prompt
  -> intent and complexity routing
  -> optional research/data sources
  -> FieldGraph
  -> model nodes and S7 Micronauts
  -> bimodal token + SVG3D attention
  -> native providers
  -> SCX semantic cache
  -> CSO/S7 compiled artifacts
  -> inference, reward, replay
```

SCX means Symbolic Cache Execution. CSO is the compiled binary artifact
format, and `target.cso` represents the S7 Micronaut. Native DLLs remain
providers, not Micronauts.

## Runtime Law

K'UHUL is a fold-oriented runtime, not a conventional language or model
framework. `runtime.kuhul` is the authoritative runtime contract:

```text
runtime(x) = Xul(Chen(Sek(Yax(Wo(Pop(x))))))
```

The six control folds are immutable scheduler algebra:

```text
Pop -> Wo -> Yax -> Sek -> Chen -> Xul
```

The folds are evaluators, not graph nodes:

```text
FieldGraph.kuhul
  -> Pop: discover reachable nodes
  -> Wo: materialize working set
  -> Yax: admit by pressure and capability
  -> Sek: topological dispatch
  -> Chen: update evidence and runtime metrics
  -> Xul: persist the next graph state
```

Graph topology is immutable during ordinary evaluation. Explicit graph
evolution may create a new validated graph version, preserving provenance
between the prior and next DAG.

The runtime cycle is a graph-state transformation:

```text
FieldGraph_t
  -> Pop: reachable and dependency-ready nodes
  -> Wo: materialized working set
  -> Yax: pressure/capability admission
  -> Sek: local topological sort and provider dispatch
  -> Chen: pressure, confidence, provenance, and evidence updates
  -> Xul: persistence and survivor collapse
  -> FieldGraph_(t+1)
```

Sek must not topologically sort the entire graph on every cycle. Yax first
selects an admitted working set, and Sek validates the induced dependency
subgraph before dispatch. A node whose required predecessor is outside the
admitted set remains pending rather than executing against incomplete state.
This preserves deterministic dependency order while allowing pressure-driven
partial execution.

The graph state is the SCX cache: node inputs, outputs, provider decisions,
pressure, confidence, residency, provenance, history, and completion state
are resumable runtime data. Xul must persist the resulting state before
discarding a completed or superseded working set.

SCX has a two-level cache boundary:

```text
SCX symbolic cache
  -> FieldGraph DAG state, node results, provenance, pressure, and history

shader_cache / D3DSCache.dll
  -> compiled shader artifacts, CSO/DXBC reuse, and driver cache entries
```

The registered `shader_cache` provider is therefore useful to SCX, but it is
not the SCX graph store. A node's SCX artifact record should reference the
shader-cache entry by a deterministic key containing the node identity,
provider, shader/compiler version, resource contract, and device signature.
Graph state remains resumable even when a shader-cache entry is evicted; the
runtime must then recompile or select another admitted provider.

`native/runtime/SCXcache.manifest.json` is the dedicated symbolic cache
manifest. It records the persistent DAG contract, fold cycle, working-set
policy, required node state, Xul persistence boundary, and deterministic
references to compiled shader-cache artifacts. It is intentionally separate
from the root `cache.manifest.json`, which describes browser/static assets.

### Token RAG

Token RAG is a fine-grained retrieval subgraph synchronized with generation:

```text
MM-1 token generator
  -> TokenBuffer
  -> vector retrieval node
  -> retrieved evidence/conditioning state
  -> next token step
```

Each token update may raise pressure on the retrieval node, but retrieval is
bounded by an admission threshold, provider budget, and micro-batch window.
The runtime must not turn every token into an unconditional external call.
Yax admits retrieval when the TokenBuffer change is semantically relevant;
Sek dispatches the admitted provider; Chen updates token/node confidence,
pressure, provenance, and evidence; Xul persists or prunes the resulting
state.

Token RAG uses the same SCX DAG and does not create a second KV-cache
authority. TokenBuffer state, retrieval keys, evidence references, provider
results, and generation positions are symbolic graph state; hardware-specific
KV/logit representations remain provider-owned.

```text
confidence < threshold
  -> decay retrieval pressure
  -> bounded prune or reroute

confidence >= threshold
  -> retain evidence in the active working set
  -> allow the next dependent token step
```

Every retrieval result must retain provenance and a generation position.
Stale, contradictory, or low-confidence evidence cannot silently remain
active across cycles.

The native `TokenRAGEngine` contract in
`native/runtime/token_rag.{h,cpp}` implements this bounded admission model:
it pressure-sorts pending token queries, admits only a capped micro-batch,
records generation-positioned evidence, and prunes evidence below the
retention-confidence threshold. Provider-specific vector search remains
outside this policy service.

`FieldGraph.kuhul` is the persistent declarative execution DAG. Runtime
capability folds evaluate that DAG inside the control cycle and must not
silently redefine its topology. Compute, Meta, and Storage are the first
explicitly registered capability folds; Parser, Compiler, Tensor, Network,
Tool, Memory, and Provider folds can be added by the same contract pattern.
Micronauts own capabilities, while native providers and CSOs are replaceable
implementations selected by the host.

Nodes are the atomic deterministic transformations within that state:

```text
Runtime -> FieldGraph DAG -> Fold evaluation -> Node -> Capability -> Provider
```

The DAG owns dependencies, data flow, control flow, and provenance edges.
Folds operate on the same graph: Pop discovers reachable nodes, Wo
materializes the working set, Yax admits high-pressure nodes, Sek executes
the admitted nodes in topological order, Chen updates pressure/confidence/
history, and Xul persists the resulting graph state. Node pressure,
confidence, residency, inputs, outputs, provider selection, and execution
history belong to the node. Micronauts provide implementations for node
capabilities rather than becoming the dispatch unit.

Every Micronaut carries a bounded runtime state contract for the nodes it
owns or exposes. The contract must include:

```text
Micronaut
  -> Node identity and node-type declarations
  -> pressure       (scheduling demand)
  -> confidence     (evidence and validation quality)
  -> residency      (HOT, WARM, or STREAMING)
  -> provider       (admitted implementation binding)
  -> history        (replayable execution and state transitions)
```

These values are not model-generated authority. Forge may propose initial
node metadata and provider preferences, but FieldGraph, TaskEngine, and the
phase runtime validate and update the live values. Pressure controls
scheduling, confidence controls trust and follow-up work, residency controls
placement, provider identifies the implementation, and history preserves
provenance, replay, failures, and transitions across `Pop -> Xul`.

The Personality Micronaut (`PT-0001`) owns node-selection policy. It may
prioritize nodes, configure node parameters, and express provider preferences,
but it cannot execute nodes, alter node operations, or change fold order.

### Intentional and operational graph layers

The runtime separates two graph views without separating their underlying
nodes:

```text
Intentional layer
  plans, tasks, milestones, owners, deadlines, budgets
  -> plan-edge lens over the persistent execution DAG

Operational layer
  plots, ICP routing, workflows, APIs, data movement, schematics
  -> operational-edge lens over the persistent execution DAG
```

A shared FieldGraph node may therefore carry both plan edges and operational
edges. The plan graph answers **what must happen and in what order**; the
operational graph answers **how the running system routes and transforms
data**. Neither projection is allowed to overwrite the other.

Para-Graph documents are the human-readable interchange form for this
intersection. A paragraph/block is the canonical payload and may declare
typed edges in both lenses:

```text
node Auth_Service {
  plan.depends_on -> Database_Setup
  plan.blocks -> Frontend_Login_UI
  workflow.validates_against => Enterprise_Security_Policy
  workflow.routes_to_failure => Rate_Limiter_Block
}
```

Lens selection is a view operation over the same provenance-bearing nodes:
`PROJECT_PLAN` exposes the intentional DAG, while `SCHEMATIC_WORKFLOW`
exposes the operational FieldGraph. TaskEngine consumes the intentional lens; BOSS, providers, RAG, and runtime
folds consume the operational lens. Both lenses address the same persistent
nodes and edges.

### Schematic Pressure

Schematic Pressure is the operational force accumulated when workflows,
payloads, providers, and graph routes compete for finite capacity. It is
distinct from Plan Pressure:

```text
Plan Pressure       = deadlines, budgets, milestones, staffing
Schematic Pressure  = load, contention, bottlenecks, data friction,
                      provider limits, residency and context capacity
```

Schematic Pressure belongs to operational nodes and their typed edges. It
increases when multiple routes converge on a node, a provider is saturated,
working-set or context size exceeds capacity, writes contend, or a required
fallback route is absent. It decreases when the graph branches, filters,
compresses, queues, caches, changes residency, or selects an alternate
provider.

Each operational node should expose pressure inputs and thresholds:

```text
node Payment_Gateway {
  pressure.capacity = 5000
  pressure.inputs = [Mobile_Checkout, Wholesale_Invoicing, Auto_Renewals]
  pressure.policy = queue | branch | cache | reroute | reject
}
```

The runtime records Schematic Pressure separately from confidence. Pressure
answers **can this node safely accept more work?**; confidence answers **can
the result be trusted and considered complete?**. FieldGraph owns the live
pressure state, BOSS applies routing and residency relief, and TaskEngine
reports admission failures without treating them as successful execution.
Plan completion must never clear unresolved Schematic Pressure.

### Code Generation Graph

Code generation is a three-layer projection over the same task and
provenance model:

```text
user specification
  -> Plan Graph       (tasks, tickets, dependencies, acceptance criteria)
  -> Schematic Graph  (APIs, types, state, data flow, pressure)
  -> AST Graph        (functions, declarations, imports, generated artifacts)
```

The Plan Graph is the intentional edge lens consumed by TaskEngine. The
Schematic Graph is the operational edge lens consumed by BOSS, providers, and
pressure analysis. Both are views of the same persistent execution DAG.
The AST Graph is the concrete compiler-facing projection; it must be
validated by the language compiler and tests before its node is marked
complete.

For each code-generation node, the contract should preserve:

```text
plan.requires      -> intentional dependencies
schematic.input    -> typed inputs and upstream edges
schematic.output   -> typed outputs and downstream edges
schematic.throws   -> failure and fallback edges
ast.artifact       -> source range, language, target, and compiler result
```

Generation is bounded and incremental. RAG retrieves only adjacent
provenance-bearing nodes and contracts, reducing context pressure. The
validation loop is explicit:

```text
generate AST/source
  -> compile
  -> test and inspect
      -> accepted: lock artifact and release dependents
      -> failed: record history, increase pressure, bounded repair retry
```

The model may propose code nodes and task metadata, but it cannot mark an
artifact trusted, bypass compiler/test validation, or execute arbitrary
commands. TaskEngine owns admission, the compiler owns syntax/type validity,
and FieldGraph/BOSS owns operational routing.

The native `CodeGenGraph` contract in
`native/runtime/codegen_graph.{h,cpp}` stores the three-layer metadata on one
node, reuses `DAGScheduler` for plan ordering, records compiler/test results,
and bounds repair attempts. It is a graph contract only; execution remains
behind TaskEngine and the existing provider/helper boundaries.

`native/runtime/APIWriter.cpp`/`api_writer.h` provides the bounded API artifact
writer for `api_writer` CodeGenGraph nodes. MicrosoftSDK may describe API
generation in a TaskList, but APIWriter only validates endpoint/schema
metadata and emits OpenAPI; TaskEngine remains responsible for admission and
the runtime does not execute generated handlers implicitly.

### Grammar Graph

`native/runtime/grammar_graph.{h,cpp}` provides the grammar-layer graph
contract. It supports EBNF production edges, GBNF/FSM transitions, and PEG
ordered-choice priorities without making any of them a second parser:

```text
EBNF -> production and repetition graph
GBNF -> token/state transition graph
PEG  -> prioritized deterministic transition graph
```

The existing EBNF files remain the canonical KUHUL grammar authorities.
GrammarGraph validates node/edge identity and accepting states, while PEG
outgoing edges are returned in priority order. A future Para-Graph parser
can compile EBNF into this graph, use PEG ordering for deterministic parsing,
and emit GBNF constraints for model-generated TaskLists, CodeGenNodes, and
`.pg` blocks.

## Hybrid Execution Rule

GEMM remains the authoritative path for attention projections, QKV products,
and other dense linear algebra. GPU-resident Shader Experts and SVG3D
Micronauts handle geometry, topology, routing, pressure, and 3D semantic
operations in parallel. Bimodal attention and typed adapters join the two
without replacing model weights or forcing geometric work through GEMM.

`Micronaut.bson` is allowed as a semantic/document envelope for compact
nodes, folds, AST references, and graph metadata. It is not a replacement
for CSO bytecode, SafeTensors weights, or streamed xshard storage.

## Unified XJSON/K'UHUL tensor baseline

The JSON Runtime and K'UHUL runtimes share one logical tensor contract.
XJSON owns declaration, orchestration, and XCFE routing; K'UHUL owns fold
execution; Micronauts resolve capabilities; providers perform numerical
execution. Moving a tensor between providers changes residency and backend
metadata, not tensor identity.

The current JSON Runtime baseline is implemented in
`bin/json-runtime/src/tensor_runtime.*`:

- bounded CPU `f32` tensor allocation;
- `MATMUL`/`GEMM`, `RELU`, `SOFTMAX`, and CPU copy operations;
- process-local shared tensor registry through `TENSOR_REGISTER`,
  `TENSOR_GET`, `TENSOR_EXISTS`, `TENSOR_REMOVE`, and `TENSOR_LIST`;
- Khanary `dml_gemm_bt_f32` loading from `bin/ggml/dml_gemm.dll`;
- DirectML GEMM with explicit `cpu-fallback` behavior and backend provenance.

The Khanary GEMM ABI is:

```cpp
int dml_gemm_bt_f32(
    const float* A, const float* B_transposed, float* C,
    unsigned M, unsigned N, unsigned K);
```

The JSON Runtime transposes XJSON row-major `[K,N]` weights to Khanary's
`[N,K]` input convention. The current tensor registry is process-local and
JSON-backed; native `UnifiedTensor` objects, CSO persistence, residency
synchronization, and additional DirectML operators remain planned work.

## Bimodal attention status

Bimodal attention is compatible with the unified contract and remains a
bounded adapter between token/model tensors and SVG3D semantic tensors:

```text
token embeddings -> token attention ─┐
                                     ├─> projector/cross-attention -> output
SVG3D nodes      -> spatial attention┘
```

The branches must preserve separate tensor identities and model weights.
Fusion is performed through explicit projectors, cross-attention, or
per-layer conditioning; geometry tensors must not be silently merged into
token weights. The current runtime has the required GEMM and softmax
primitives but does not yet implement QK-transpose, masking, KV-cache
fusion, or a bimodal attention kernel. Token-only execution must remain
unchanged when the adapter is disabled.

## What Exists

### Native host and build

- `native/kuhul_engine.cpp` is the canonical executable entry point.
- `CMakeLists.txt` builds the unified x64 `kuhul_engine.exe`.
- `build_release.bat` configures, builds, and copies runtime dependencies.
- `native/webx_compute.h` provides the simplified runtime, FieldGraph,
  pressure/confidence tracking, ProviderManager, Micronaut Forge, artifact
  contracts, and S7/CSO artifact labeling.
- `native/FieldGraph.kuhul` is the canonical semantic contract.
- `native/llama_runtime.*` provides an optional native llama.cpp model node.
- `native/http_api_server.*` provides a localhost native API.

### Providers

Provider discovery currently covers:

- DirectML and DirectML.Debug.
- D3D11, D3D12, D3D10 WARP, D3D compiler, and shader cache.
- Intel OpenGL/D3D user-mode drivers and Intel graphics JIT.
- Intel OpenCL GPU/ICD components.
- Intel CPU OpenCL device, runtime, executor, TBB, and Clang components.
- Clang native compilation.

`KUHUL_DRIVER_ROOT` makes the driver directory configurable.

### Existing inference and streaming pieces

- `native/gpu_trainer/xshard.h` describes the XSQ2 shard header used by
  existing streamable layer data.
- `native/gpu_trainer/xshard_attention.*` provides a working native D3D11
  Q/K/V attention proof path, currently limited to FP32 tile assumptions.
- `native/gpu_trainer/scx_stream_engine.*` contains the DX12 SCX streaming
  engine, tile queues, decode workers, layer readiness, and batch inference
  entry points.
- `native/gpu_trainer/dx12_inference_pipeline.*` contains the intended
  multi-layer QKV, KV-cache, FFN, logits, and asynchronous generation path.
- `native/gpu_trainer/speculative_decoder.*` contains draft/main model
  speculative decoding design.
- `native/gpu_trainer/asx_ram_controller.h` defines bounded CPU/GPU tile
  residency and priority behavior.
- `native/gpu_trainer/xshard_validate.cpp` and `xshard_tile_test.cpp` contain
  validation and attention reference implementations.
- `shaders/fabric_kernels.hlsl` contains multi-head attention, MoE routing,
  decompression, and expert merge shader work.

### SVG3D, graph, and semantic layers

- `src/xcfe/svg3d-compute.js` implements SVG3D compute nodes, topology,
  propagation, phase scheduling, and SVG3D output.
- `src/xcfe/fold-algebra.js`, `pressure-mapper.js`, `node-runtime.js`,
  `semantic-atlas.js`, and `tensor-algebra.js` provide semantic graph
  concepts.
- `src/pggtf/` provides fold tensors, phase tensors, geodesic tensors, and
  tensor-to-fold operations.
- `src/kxml/` provides graph parsing, dispatch, operations, and shard
  registry support.
- `src/mx2lm/` provides model views, tensor manifests, and brain topology.
- `src/gpu/webgpu-runtime.js` provides browser WebGPU/WebGL/CPU capability
  selection and fallback behavior.
- `src/scx/`, `src/scxq7/`, and adapter modules provide JavaScript SCX,
  manifest, quantization, and adapter concepts.

## What Is Missing

### P0: Make native model loading authoritative

1. Add `native/inference/model_manifest.*` for model metadata:
   layer count, hidden size, heads, head dimension, FFN size, vocabulary,
   tokenizer, dtype, tensor names, shard locations, and backend policy.
2. Add `native/inference/xshard_model_loader.*` to discover layer shards,
   validate headers, map tensor names to Q/K/V/O/FFN/embedding weights, and
   stream tiles into the native pipeline.
3. Add `--model`, `--validate-xshard`, `--stream-xshard`, and
   `--infer <model> <prompt>` commands to `kuhul_engine.exe`.
4. Replace the current demo-only xshard attention path with a real
   layer-by-layer generation path using the existing stream engine.
5. Add tokenizer loading and model-node query contracts.

### P0: Re-enable the real inference path

The following files are currently excluded or not fully integrated in the
unified target:

- `scx_stream_engine.cpp`
- `scx_transformer_block.cpp`
- `scx_infer_layer.cpp`
- `dx12_inference_pipeline.cpp`
- `speculative_decoder.cpp`
- `xshard_validate.cpp`
- `xshard_tile_test.cpp`
- `asx_ram_test.cpp`

Re-enable them incrementally after resolving their missing shader binaries,
headers, duplicate entry points, and visibility/ownership issues. Keep
standalone demos and tests out of the production executable.

### P1: Add optional llama.cpp GPU backends

`native/llama_runtime.*` currently loads the CPU llama.cpp provider and
forces `n_gpu_layers = 0`. Add a backend selection contract without changing
the native xshard path or making GGUF inference authoritative:

1. **DirectML native option** — build llama.cpp/ggml with its DirectML backend,
   load the matching backend DLLs beside `llama.dll`, detect the DirectML
   device through the existing provider registry, and expose an explicit
   `cpu|directml|auto` policy. The llama API binding must set GPU-layer and
   backend parameters only when the loaded runtime exports the required
   ABI; otherwise it must reject the request and fall back deterministically.
2. **WebGL2 browser option** — do not treat WebGL2 as a drop-in llama.cpp
   backend. WebGL2 belongs behind the WebX/browser adapter and should expose
   bounded tensor kernels through the localhost/native bridge or a WASM
   projection. It is a UI/browser execution tier, not a replacement for the
   native xshard executor.
3. **Hardware policy** — on Intel HD 4600-class hardware, keep CPU-first
   behavior, cap context/batch/layer residency, and allow DirectML only after
   device/feature validation. WARP is a correctness fallback, not a
   performance claim.
4. **Verification** — compare CPU and DirectML logits on a small deterministic
   GGUF fixture, record backend/device/layer residency in the response, and
   keep native xshard hot-swap tests independent from llama.cpp fallback tests.

### P1: Add SafeTensors interoperability

Add a dependency-free native reader:

- `native/tensors/safetensors_reader.*`
- Parse the 8-byte header length and JSON tensor index.
- Validate offsets, dtype, shape, and file bounds.
- Expose tensors through the same model manifest used by xshards.
- Support read-only loading first; add export only after ingestion is proven.

SafeTensors and SVG3D are separate representations of the same tensor
contract. SafeTensors remains conventional model storage; SVG3D remains the
semantic/structural sidecar.

### P1: Implement bimodal attention

Create a typed bridge between:

- token/model tensors: embeddings, QKV, KV cache, logits;
- SVG3D tensors: nodes, edges, topology, pressure, folds, provenance, and
  persistent semantic state.

Target files:

- `native/inference/bimodal_attention.*`
- `native/inference/svg3d_tensor_bridge.*`
- `shaders/bimodal_attention.hlsl`
- `native/FieldGraph.kuhul`

The bridge must use explicit projectors, cross-attention, or per-layer
conditioning. It must not merge model weights or overwrite token tensors.

### P1: Make models queryable graph nodes

Define `ModelNode` metadata and lifecycle:

```text
load -> validate -> bind provider -> expose query contract
     -> stream weights -> infer -> emit provenance/reward -> unload
```

Each node keeps its own tokenizer, weights, execution state, and namespace.
GPT-OSS, LFM, GPT-small, and future models can then be queried as isolated
capabilities through one FieldGraph.

### P1: Complete SVG3D semantic persistence

Add native or bridge-level support for:

- SVG3D tensor serialization and loading.
- Field/card/token/fold/node provenance.
- Pressure and residency metadata.
- Reward and replay records.
- Conversion between SVG3D sidecars and SCX cache entries.

The existing JavaScript implementation is a reference, not yet the native
inference contract.

### P2: Research and distillation pipeline

Keep online research optional and policy-gated. For research-intent prompts:

```text
source documents
 -> normalized passages and n-grams
 -> provenance-bearing FieldGraph
 -> S7 Micronauts
 -> shader experts
 -> SVG3D semantic weights
 -> reward evaluation
 -> replay record
 -> SCX cache
```

The existing Python harvesting/training scripts can remain data-preparation
tools. They must not become the inference owner; native execution remains in
`kuhul_engine.exe`.

### P2: CSO/S7 artifact production

Replace placeholder Forge payloads with real artifacts:

- compile selected HLSL/KLSL through D3DCompiler or DXC;
- validate shader model and resource layout;
- emit an S7 envelope containing valid CSO bytecode plus compact semantic
  metadata: AST references, graph/node identifiers, fold bindings, tensor
  names, and provider/resource layout;
- hash and register the artifact in SCX;
- reload and execute the cached CSO.

The S7 envelope must keep executable CSO bytes independently addressable.
Large token/weight data remains in streamed buffers or referenced xshard and
SafeTensors storage. Only minimal text and graph metadata belongs in the S7
semantic pocket.

### P1: Microsoft Semantic Kernel Micronaut

`native/semantic-kernel/MicrosoftSDK.ps1` is the management surface for the
vendored C# Semantic Kernel solution. It exposes project discovery, capability
manifest generation, the bound SDK orchestration persona, build, test, and
format actions through the explicit `SCX::command::microsoft-sdk` binding.

The Python SDK requirement remains intentionally minimal at this stage:
`pip install semantic-kernel`. Python agent/plugin examples are integration
references for the portal contract; they do not add LangGraph, LlamaIndex, or
other orchestration dependencies, and they do not become the native inference
owner.

The C# agent path is independent of that Python package. `MicrosoftSDK.ps1`
uses `dotnet build SK-dotnet.slnx` to compile the vendored Semantic Kernel
assemblies, including `src/Agents/Core/Agents.Core.csproj`, which references
the agent abstractions and `SemanticKernel.Core`. `dotnet.exe` therefore
provides the C# build/runtime toolchain and Semantic Kernel agent assemblies.
`dnx.cmd` is only the .NET tool runner for executing a NuGet tool from source;
it is not the Semantic Kernel agent runtime and is not required by the bridge.
The installed Mono workload manifests are optional platform workloads, not
dependencies of the normal C# agent path.

Semantic Kernel already provides cloud model connector surfaces, including
Azure OpenAI and OpenAI. These connectors are activated per generated user app,
not globally: the app manifest declares a provider and environment-variable
references, the host validates the user-owned `.env` configuration, and only
then constructs the app's `ChatCompletionAgent` service. Secret values never
enter prompts, TaskLists, manifests, generated source, logs, or SCX state.

```json
{
  "model_provider": "azure_openai",
  "deployment_env": "AZURE_OPENAI_DEPLOYMENT",
  "endpoint_env": "AZURE_OPENAI_ENDPOINT",
  "api_key_env": "AZURE_OPENAI_API_KEY"
}
```

The existing generated app `manifest.json` is the correct place for this
binding; no separate cloud-connection file is required. `KuhulAppCreator`
now accepts an optional, secret-free `model_provider` block when the user
requests cloud inference. `KuhulAppHost` still must resolve the declared
environment-variable references privately for that specific app and activate
the corresponding connector. Apps that omit the block remain
local/provider-neutral.

The creator allowlists `azure_openai` and `openai` and validates the four
environment-reference names without reading their values. Connector
activation remains deferred to the host/SDK boundary, so app creation cannot
leak credentials or claim that a cloud service is reachable.

The `microsoft-sdk-planner` persona is a deterministic, user-authorized
orchestration policy. It may translate intent into Semantic Kernel capabilities,
TaskList dependencies, provider recommendations, and validation points, but it
cannot execute tools, invent results, bypass TaskEngine admission, or write and
run generated handlers. `MicrosoftSDK.ps1 -Command persona` exposes the same
system policy used by `invoke` and `tasklist`, keeping the bridge contract
inspectable and consistent.

The bridge also accepts a caller-supplied agent name, instructions, explicit
plugin manifest, and structured response schema. These remain declarative
`ChatCompletionAgent` configuration; plugin registration/function choice and
schema validation belong to the bridge host or TaskEngine, not the model.

MicrosoftSDK is also the stack portal boundary: `stack-manifest` inventories
the root, server, cache, RPC/API/model (when present), SCX cache, and registry
manifests without treating absent optional manifests as failures. `tasklist`
continues to generate a validated declarative TaskList for
`kuhul_engine.exe task-boss`; passing `-DispatchToBoss` performs that dispatch
explicitly rather than executing generated code implicitly.

`registry/sdk-system-micronauts.registry.json` supplies model-facing reminders
for the C# Semantic Kernel agent, optional Python Semantic Kernel agent,
`dotnet.exe` toolchain, stack-manifest portal, and TaskEngine-to-BOSS bridge.
These are policy/capability contracts, not additional model weights. The
MicrosoftSDK bridge injects their reminders into `invoke` and `tasklist`
system context so models retain the language, ownership, and execution
boundaries.

`ELIZA-1` is the deterministic pattern-router Micronaut. It is an optional
zero-token preprocessor for Para-Graph paragraphs: decomposition rules produce
capture groups, and validated reassembly routes those captures to FieldGraph
nodes, pressure updates, Token RAG keys, and provenance. It is not a therapist,
LLM, provider executor, or unrestricted graph mutator. The existing
`src/xjsl/jsonl-compiler.js` capture-map behavior is reusable prior art; a
native RegexVM implementation should preserve the same bounded capture and
successor-graph rules.

`ADAM12-1` defines the artifact-ingestion boundary for semantic Micronauts.
`native/runtime/adam12.schema.json` specifies an integrity-addressed,
non-executable JSON payload containing semantic tensor dimensions/weights,
bounded ELIZA pressure rules, and platform hardware mappings. The K'UHUL
engine must validate the SHA-256 digest and schema before projecting an ADAM12
artifact into a versioned FieldGraph successor. Hardware mappings are
capability preferences, not proof that a provider is installed; normal Yax
admission still resolves DirectML, OpenCL, CPU SIMD, or another available
provider.

`REGEX-1` is the specialized zero-token ADAM12 provider profile behind the
ELIZA graph. Its bounded `regex_pipeline` extracts captures from raw
paragraphs or task text and emits only allowlisted structural intents:
`DrawEdge`, `InjectPressure`, `RouteCapture`, or `EmitDiagnostic`. PCRE2 JIT
is an optional provider preference with `STD_Regex_C11` as the fallback; no
PCRE2 dependency is assumed until the native provider is installed and probed.
Regex matches may propose graph-state changes, but validation, provenance,
topology versioning, and fold admission remain host/runtime responsibilities.

`ROSLYN-1` is the compilation and diagnostics Micronaut for generated C#.
Its ADAM12 payload may declare `csc.exe`, `vbc.exe`, `VBCSCompiler.exe`, and
`dotnet-format.exe` toolchain paths plus shared-server execution settings.
Paths are capabilities discovered from the installed .NET SDK, not guaranteed
fixed locations; the host must resolve and probe them before admission.
Roslyn diagnostics become structured provenance and bounded repair input for
the Code Generation Graph. Successful compilation may advance a node after
validation and formatting, while failures raise upstream pressure and create
a bounded successor repair state. Generated assemblies are never executed
implicitly by the compiler Micronaut.

This Micronaut owns orchestration capabilities such as connectors, agents,
plugins, and vector-data adapters. It is model-agnostic and must not become the
native inference owner; model nodes and `kuhul_engine.exe` retain ownership of
tokenization, weights, GEMM attention, KV cache, and generation.

### P1: Native fold algebra runtime

The native runtime sandbox makes the fold algebra executable and explicit:

```text
runtime(x) =
  Xul(Chen(Sek(Yax(Wo(Pop(x))))))
```

`native/runtime/runtime.cpp` owns only the `RuntimeContext` and invokes the
fixed composition. `Pop.cpp`, `Wo.cpp`, `Yax.cpp`, `Sek.cpp`, `Chen.cpp`, and
`Xul.cpp` are state operators with one responsibility each. Pressure,
working-set selection, Micronaut reuse/forging, backend selection, execution,
reflection, and SCX persistence remain phase-local transformations of the
shared context.

Micronauts are semantic operators and providers are interchangeable
implementations selected by the operators. Replacing D3DCompiler with DirectML,
OpenCL, or CPU SIMD therefore changes the backend implementation, not the fold
algebra or FieldGraph meaning.

The declarative contracts live beside the current native host:

```text
native/runtime/runtime.kuhul
  -> Pop.kuhul -> Wo.kuhul -> Yax.kuhul
  -> Sek.kuhul -> Chen.kuhul -> Xul.kuhul
```

The `.kuhul` layer defines runtime composition, fold responsibilities,
admission algebra, capability requests, and SCX persistence. The C++ files in
the same directory are the present execution host for those contracts; they
must not become the semantic source of truth. Future runtime specializations
can import the same fold contracts and replace only the composition or policy
data.

The first-class runtime algebras are now declared in `runtime.kuhul`:

| Algebra | Purpose |
|---|---|
| `SymbolicExecution` | parse, rewrite, prove, evaluate, and persist symbolic state |
| `NeuralTraining` | ingest batches, execute tensor work, reward, and checkpoint |
| `Inference` | tokenize, stage, generate, validate, and cache model output |
| `ShaderCompilation` | load, select compiler capability, compile, validate, and cache CSO |
| `MicronautForging` | turn intent into admitted, validated runtime operators |
| `ProviderResolution` | discover, rank, bind, verify, and release providers |
| `SCXWorkingSet` | page, admit, schedule, reflect, and persist bounded state |

These are payload specializations of the same runtime composition, not
separate execution laws. ML remains one domain algebra among many.

The first provider fold contracts are now explicit under
`native/runtime/folds/`:

| Runtime fold | Control phase | Micronaut | Provider artifact |
|---|---|---|---|
| `ComputeFold` | `Sek` | `MM-1` | `kuhul_fold_compute.cso` |
| `MetaFold` | `Chen` | `VM-2` | `kuhul_fold_meta.cso` |
| `StorageFold` | `Xul` | `SM-1` | `kuhul_fold_storage.cso` |

`native/runtime/manifest.kuhul` maps these capability folds to their control
phases. The manifest is declarative; the native dispatcher remains responsible
for validating artifacts, resolving providers, and dispatching only admitted
fold instances.

## Proposed Runtime Structure

```text
native/
  kuhul_engine.cpp                 # CLI and native host
  webx_compute.h                   # runtime facade and Forge
  FieldGraph.kuhul                 # semantic contract
  inference/
    model_manifest.h/.cpp          # model node contract
    model_registry.h/.cpp          # isolated model nodes
    xshard_model_loader.h/.cpp     # XSQ2 layer/shard ingestion
    safetensors_reader.h/.cpp      # SafeTensors index/data reader
    bimodal_attention.h/.cpp       # token + SVG3D attention
    svg3d_tensor_bridge.h/.cpp     # semantic tensor bridge
    inference_session.h/.cpp       # prompt -> tokens -> generation
    provenance_log.h/.cpp          # source, reward, replay records
  scx/
    scx_cache.h/.cpp               # Symbolic Cache Execution
    cso_registry.h/.cpp            # CSO/S7 artifact index
    residency_manager.h/.cpp       # bounded CPU/GPU state
  gpu_trainer/
    scx_stream_engine.*            # streaming GPU layer engine
    dx12_inference_pipeline.*      # multi-layer execution
    speculative_decoder.*          # draft/main decoding
    asx_ram_controller.h           # tile residency policy
  providers/
    provider_manager.*             # future extraction from webx_compute.h
  semantic-kernel/
  MicrosoftSDK.ps1               # explicit orchestration Micronaut command surface
  dotnet/                         # vendored Microsoft Semantic Kernel source
  runtime/
  phase_runtime.h                # RuntimeContext and fold operator contract
  runtime.cpp                    # minimal runtime() composition
  Pop.cpp Wo.cpp Yax.cpp         # admission, organization, pressure evaluation
  Sek.cpp Chen.cpp Xul.cpp       # execution, reflection, collapse
  runtime.kuhul                 # declarative runtime algebra
  Pop.kuhul Wo.kuhul Yax.kuhul   # declarative admission pipeline
  Sek.kuhul Chen.kuhul Xul.kuhul
  tensors/
    tensor_contract.h/.cpp         # shared shape/layout/dtype contract

shaders/
  fabric_kernels.hlsl              # multi-head/MoE reference
  bimodal_attention.hlsl           # token/SVG3D cross-attention
  scx_decode.hlsl                  # SCX tile decode
  cso/                              # compiled CSO artifacts

src/
  xcfe/                             # semantic/fold/SVG3D reference runtime
  pggtf/                            # tensor/fold/geodesic reference layer
  kxml/                             # graph and shard serialization
  mx2lm/                            # model/tensor manifests
  gpu/                              # browser WebGPU/WebGL/CPU fallback
  scx/                              # portable SCX contracts and command surface
    command-parser.js               # subject symbols + action verbs -> typed commands
    capability-resolver.js          # allowlisted capability/opcode/phase resolution

tools/trainers/
  data preparation, distillation,
  reward/replay generation only
```

## Dependency-Ordered Build Sequence

1. **Freeze contracts**: finalize model manifest, tensor contract, model-node
   query API, SVG3D sidecar schema, SCX cache schema, and CSO/S7 metadata.
2. **Load one model**: validate an existing GPT-OSS xshard directory and run
   a CPU/readback inference smoke test.
3. **Stream one layer**: connect validated tiles to the existing DX12
   pipeline and prove layer readiness/residency.
4. **Run complete generation**: embeddings -> QKV -> attention -> FFN ->
   logits -> tokenizer output.
5. **Add SafeTensors**: load the same model manifest from HF SafeTensors.
6. **Add model nodes**: load GPT-OSS, LFM, or GPT-small as isolated nodes.
7. **Add SVG3D sidecars**: load/persist semantic tensors independently of
   model weights.
8. **Add bimodal attention**: connect sidecars through bounded adapters.
9. **Add Forge output**: compile real S7 CSOs and register them in SCX.
10. **Add research distillation**: source provenance, n-grams, reward,
    replay, and cache reuse.
11. **Re-enable speculative decoding and optimization** after correctness
    tests pass.
12. **Expose browser/WebGL integration** through a thin API over the native
    host; WebGL is a projection/fallback, not the native model executor.
13. **Lower SCX commands** through `src/scx/command-parser.js` and
    `capability-resolver.js`; SCX remains the human-facing scripting layer,
    while K'UHUL validates phase legality and native providers execute the
    resolved capability.

## Verification Gates

- `kuhul_engine.exe --providers` reports the selected provider stack.
- `--validate-xshard` rejects malformed headers, offsets, dtypes, and shapes.
- A tiny known model produces deterministic logits/tokens on CPU.
- The same model produces matching readback results through the GPU path
  within the declared precision tolerance.
- Layer streaming never exceeds the configured CPU/GPU residency budget.
- SafeTensors and xshard loaders produce identical manifest tensor shapes.
- SVG3D sidecars round-trip without changing model weights.
- Bimodal attention can be disabled and the token-only path remains unchanged.
- Each model node remains isolated and provenance identifies the node and
  source used for every result.
- A generated `target.cso` is valid CSO, identifies S7, hashes correctly,
  registers in SCX, and can be reloaded for execution.
- Research/replay is opt-in, source-attributed, and never silently replaces
  the base model.

## Current execution baseline

The authoritative build inventory is the generated CMake project under
`build-llama`, especially `build-llama/kuhul_engine.vcxproj`. The active
runtime probe is `build-llama/bin/Release/kuhul_engine.exe`; its `--help`,
active command set, and `--providers` output are the source of truth for
runtime capability and driver discovery. The separate `build` directory must
not be used as the current validation target.

The working trainer baseline is the Khanary llama.cpp/GGML Release bundle:

```text
C:\Users\canna\khanary-llama-build\ggml\build\bin\Release
```

It provides the operational `ggml-xcfe.dll` and DirectML path, alongside
`ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `DirectML.dll`, `dml_gemm.dll`,
`xcfe_probe.exe`, and `xcfe_matmul_test.exe`. In llama.cpp mode, llama.cpp
and GGML own token inference; `kuhul_engine.exe` and `MicrosoftSDK.ps1`
remain the orchestration, provider, shard, and Task List boundary. The
MicrosoftSDK bridge may normalize local llama.cpp requests using the same
model-agnostic contract used for cloud models, but it must not become the
inference owner.

The full system and shard-to-Task List workflow are treated as operational.
Any remaining D3D11 smoke failure is therefore scoped to that validation path,
not evidence that the GPU trainer or shard runtime is stale.

The first Atomic Shell control-plane slice is now implemented in
`native/runtime/atomic_shell_manifest.*`. The active engine command
`atomic-shell PATH` validates required identity, block type, version, local
schema route, and `execution_gated=true` before reporting a manifest ready.
`atomic-shell PATH --render` adds a presentation-only terminal FRAME preview;
it does not execute Java, model inference, or provider actions.
FRAME manifests may now declare a runtime `blocks` array; the validator
checks each composed block and the renderer displays that array. The sample
`native/runtime/atomic.frame.manifest.json` composes the standard shell, so
block additions and updates are data changes and do not require a binary
rebuild.
`AtomicDOM.cmd`, `AtomicChat.cmd`, `AtomicGame.cmd`, and `AtomicPage.cmd` are
manifest-driven launchers. They resolve the active native engine and render
validated terminal FRAMEs without a browser, WebView, CSS, or SCSS runtime.
The chat, game, and page manifests are independently bound compositions; the
game path now admits an OpenGL FRAME and renders a bounded OBJ asset through
the native adapter.
The validator also accepts terminal-native widget blocks: `BUTTON`, `IMAGE`,
`VIDEO`, `TEXT`, `INPUT`, `CARD`, `PANEL`, and `GAME`. Their dimensions, borders,
colors, shapes, data, and routes remain manifest data; CSS is not part of the
terminal contract.
`GAME` is the OpenGL scene block for a 3D world. The existing
`native/runtime/atomic.game.manifest.json` defines the scene, asset, feed, and
game handoff contract. `opengl-game-smoke MANIFEST [FRAMES]` now resolves the
manifest `asset_uri` and renders the selected OBJ scene through the OpenGL
FRAME; `--interactive` keeps it open. Interactive mode now provides a native
camera rig: arrow keys orbit the scene and `W`/`S` zoom within bounded limits.
The OBJ scene also uses native directional lighting and computed face normals;
glTF/GLB and STL loading, materials, and simulation are subsequent stages.
The sample `native/runtime/atomic.manifest.json` uses
`atomics://local.dns.route`. New runtime sources must be added to the
generated CMake project before building `build-llama`.

### Atomic Shell integration boundary

The Atomic Shell C++ example is an architectural illustration, not a law,
schema, or drop-in implementation. Broad generated examples must not be
treated as validated runtime code: they may contain syntax errors, incorrect
bytecode offsets, unsafe parsing, hardcoded manifests, or APIs that do not
exist in the active build.

The implementation boundary is intentionally split:

```text
atomic.manifest.json
  -> manifest parser and validator
  -> Atomic Shell presentation state
  -> FRAME and block renderer

K'UHUL/XJSON state
  -> kuhul_engine.exe control-plane command
  -> TaskEngine/provider admission
  -> normalized presentation update
```

`kuhul_engine.exe` may validate manifests, expose shell state, and emit
explicit presentation events, but it does not become the DOM renderer or
replace the working Khanary llama.cpp/GGML trainer. `MicrosoftSDK.ps1` may
provide shell commands for selecting and updating manifests. Rendering remains
owned by the controlled Atomic Shell DOM, while execution remains owned by
TaskEngine and native providers.

Atomic manifests use local schema routes. `$schema` values must resolve
through the local Atomic route registry, for example
`atomics://local.dns.route`. Deterministic hash routes and cache-object routes
are preferred when they provide faster lookup or immutable identity. Hot-path
manifests must not require network access or external `http://`/`https://`
schema URLs.

The Atomic Shell FEED supports two native rendering tiers:

```text
XML/SVG vector feed
  -> bounded parser
  -> terminal green-screen/vector output

XML/SVG plus native scene extensions or glTF/GLB assets
  -> OpenGL FRAME backend
  -> 3D meshes, transforms, materials, and animation
```

OpenGL is an optional native FRAME renderer, not a browser or WebView
dependency. Three.js assets may be reused through compatible formats such as
glTF/GLB, OBJ, or STL; Three.js itself is not required by the terminal
runtime. The terminal renderer remains the fallback for 2D/vector content.
`atomic-shell` accepts `"backend": "opengl"` and preserves that selection;
actual provider capability remains an explicit `--providers` check rather
than loading graphics DLLs during manifest validation.
OpenGL FRAME manifests may select `asset_format` values `gltf`, `glb`, `obj`,
or `stl`; the native loader rejects 3D asset declarations for terminal
backends. `native/runtime/opengl_frame_adapter.*` now provides a native WGL
FRAME smoke adapter linked against the installed OpenGL system libraries;
`opengl-frame-smoke [FRAMES]` creates a context and presents bounded frames.
The adapter also provides a bounded OBJ path through
`opengl-obj-smoke PATH [FRAMES]`, parsing vertices and triangulated faces and
rendering them in the native FRAME. `opengl-obj-smoke PATH --interactive`
keeps that FRAME open until the window is closed; numeric frame counts remain
non-interactive smoke runs. glTF/GLB and STL decoding, textures, and full scene
updates remain subsequent stages.

### Manifest-bound Micronaut presentation

An Atomic Page is a lightweight Micronaut presentation surface. The manifest
defines identity, routes, style tokens, blocks, capabilities, and optional
persona metadata; AtomicDOM renders the current state in the selected backend.
The current terminal surfaces are:

| Surface | Manifest | Role |
| :--- | :--- | :--- |
| Station Control | `native/runtime/atomic.page.manifest.json` | informational page and route index |
| Station Guide | `native/runtime/atomic.chat.manifest.json` | NPC persona and terminal I/O plane |
| Station Game | `native/runtime/atomic.game.manifest.json` | OpenGL scene and asset contract |

The live feedback loop is:

```text
user input
  -> manifest persona, context, and capability contract
  -> native or cloud/API provider
  -> structured reply/question/action
  -> reply log and runtime state
  -> AtomicDOM rerender
```

NPC behavior is manifest-defined rather than hard-coded in the renderer. The
chat manifest supplies the character identity, monotone system prompt, rules,
context window, provider route, and reply schema. Provider execution remains
behind `task://chat.submit`; results return through
`state://chat.reply_log`. This allows a model to ask questions, adapt
assistance, push semantic state, or produce a character script while the
presentation plane remains deterministic and execution-gated.

The first cloud adapter target is the OpenAI-compatible
`/v1/chat/completions` protocol. The terminal does not upload
`atomic.chat.manifest.json`; the native route compiler extracts the required
persona and context fields into a normalized request:

```json
{
  "model": "${KUHUL_CHAT_MODEL}",
  "messages": [
    {"role": "system", "content": "<compiled NPC prompt and rules>"},
    {"role": "user", "content": "<terminal input>"}
  ],
  "metadata": {
    "request_route": "task://chat.submit",
    "response_route": "state://chat.reply_log",
    "reply_format": "atomic.npc.reply.v1"
  }
}
```

`KUHUL_CHAT_API_URL` supplies the provider base URL and
`KUHUL_CHAT_MODEL` selects the model. The adapter owns HTTP authentication,
bounded response parsing, and error admission; AtomicDOM receives only the
normalized NPC reply and never handles provider credentials or raw manifests.

Style is expressed through backend-neutral manifest tokens such as theme,
accent, surface, text, density, and radius. Terminal and OpenGL consume those
tokens directly. A future browser adapter may export them as CSS variables;
SCSS is optional build tooling for that adapter and is not a native runtime
dependency.

The approved WebGL examples under `E:\Downloads` are behavioral references
for the native GAME roadmap:

```text
webgl_camera.html                 -> perspective/orthographic camera state
webgl_lights_spotlight.html       -> lights, shadows, and material state
webgl_geometry_terrain_raycast.html -> terrain FEED and ray picking
webgl_geometry_minecraft*.html    -> instanced voxel/world geometry
webgl_batch_lod_bvh.html          -> batching, LOD, frustum culling, and BVH
```

These examples do not add a browser or Three.js dependency to the terminal
runtime. Their scene behavior is to be represented by Atomic `GAME`, `FEED`,
camera, lighting, and asset manifests and implemented by the native OpenGL
backend.

FEED follows this authoritative pipeline:

```text
manifest FEED
  -> XCFE Tree-sitter WASM parser
  -> syntax tree and validated contract
  -> native kuhul_engine Atomic DOM
  -> widget composition and rendering
```

The native `kuhul_engine.exe` remains authoritative for manifest resolution,
capabilities, widget composition, and rendering. XCFE WASM provides
deterministic, sandboxed, incremental parsing only; it does not own CSS,
WebView, DOM composition, provider execution, or model execution. Do not add
a competing ad-hoc FEED parser to the native shell path. FEED validation now
also resolves the local `tree-sitter-xcfe/tree-sitter-xcfe.wasm` artifact, or
an explicit `KUHUL_XCFE_WASM` override, before admitting a FEED composition.

Before any Atomic Shell code is admitted to the active `build-llama` target,
the implementation must pass:

- C++ compilation and type checking through the generated CMake project;
- manifest schema, block-type, version, route, and permission validation;
- bounded K'UHUL bytecode parsing with explicit length checks;
- command-path tests through `build-llama/bin/Release/kuhul_engine.exe`;
- provider and Task List regression checks;
- a presentation-only smoke test proving that rendering cannot bypass
  execution admission.

## Immediate Next Task

Completed: `native/inference/model_manifest.*`,
`native/inference/xshard_model_loader.*`, and the native
`--validate-xshard` command now provide the first dependency-free ingestion
gate. The command accepts one shard or recursively scans `.xshard`/`.xsq2`
directories, validates headers, dtypes, tile sizes, payload bounds, and
reports layer/tensor totals.

Next, connect the validated manifest to `scx_stream_engine`. The dependency-free
`native/inference/scx_manifest_bridge.*` now converts validated xshard records
into deterministic SCX layer/tile readiness plans exposed by
`--stream-xshard`. The DX12 engine is now part of the native target, uses
movable atomic state and loads its offline CSO files from
`native/shaders/bin/cso`. `streamXShardTilesToEngine` now maps validated
records, tile geometry, payload bytes, and dtypes directly to
`ScxStreamEngine::submitChunk`; the CLI can read a bounded tile prefix without
loading the full model. `scx-dx12-smoke PATH [MAX_TILES]` now creates a
hardware-or-WARP D3D12 compute device, initializes one 2,880-wide layer,
streams a bounded prefix, and verifies layer readiness. Full multi-layer
execution remains separate work. The upload worker now converts XSQ2 FP16,
FP32, and INT8 payloads into float storage while retaining the GPU INT4
decode path. The smoke command accepts an optional tensor type filter; a
targeted GPT-OSS Q tile (type 0, FP16) now reaches `ready=true` without
reading the preceding expert shards.

The bounded expert path now also has a native INT8 route:
`XShardBlockReader::readRawBlock` reads one-byte INT8 blocks without float
expansion, `sxme_expert_block_int8.cso` decodes signed bytes through a raw
`ByteAddressBuffer`, and `D3D11ExpertBlockInt8Gemm` binds the block with a
raw SRV and explicit scale/zero-point parameters. The
`xshard-block-gemm-int8` command exposes a smoke path. The new
`xshard-quantize-int8 INPUT OUTPUT` command creates a symmetric per-tensor
INT8 shard and stores its scale in the reserved header metadata bytes; GEMM
reads that scale automatically, while explicit scale/zero arguments remain
available for externally produced shards.

The packed INT4 workflow is now available through
`xshard-quantize-int4 INPUT OUTPUT` and `xshard-block-gemm-int4`. It stores
two signed nibbles per byte, records a symmetric scale in the reserved header
metadata, reads only the requested packed block, and executes through the
SM5 `ByteAddressBuffer` adapter. A 32-expert gate shard measured about
132.8 MB after conversion, and the packed block GEMM smoke completed.

The fold providers are now emitted as SM5 artifacts:
`kuhul_fold_compute.cso`, `kuhul_fold_meta.cso`, and
`kuhul_fold_storage.cso`. META now uses initialized verification state,
correct SHA-256 padding for concatenated hashes, and does not gate compute
execution. STORAGE now tracks dirty/sealed snapshot state, retrieves into a
separate buffer, and claims replay slots atomically.

The fold-runtime contract is now formalized in
`native/runtime/folds/ComputeFold.kuhul`,
`native/runtime/folds/MetaFold.kuhul`, and
`native/runtime/folds/StorageFold.kuhul`. The runtime manifest registers
these folds independently from the six control contracts and assigns them to
Sek, Chen, and Xul respectively. The next implementation step is a native
fold registry/dispatcher that loads this manifest, validates CSO entry points
and resource contracts, and dispatches only folds admitted by Yax.

The first node contracts are registered under `native/runtime/nodes/` for
Parse, AST, Compute, Tensor, Storage, Verification, and Memory.
`SemanticNode` now carries node kind, capability, residency, graph I/O,
confidence, provider, and execution-history fields. The next node-layer task
is contract loading and capability-to-provider resolution for admitted nodes.

`native/runtime/Personality.kuhul` and `registry/personality.micronaut.json`
now define `PT-0001` as the policy Micronaut for node selection, pressure
prioritization, provider preference, and style. Its constraints preserve
deterministic node execution and the immutable fold cycle.

`native/runtime/node_registry.*` now loads the registered `.node` contracts
and resolves each node capability against the detected provider registry.
`node-manifest PATH` exposes this contract check without executing nodes;
Yax admission and Sek dispatch remain the only execution path.

PT-0001 is now applied during Wo, before Yax admission. It deterministically
derives a bounded node priority from confidence and residency, then Yax uses
that priority when clamping admission pressure. Personality remains policy
only: it cannot execute nodes, mutate node operations, or reorder folds. The
listed WWA binaries are host/runtime components and are not loaded as
providers by this policy.

`native/runtime/KuhulAppCreator.*` provides a bounded WWA host integration
surface. It validates the configured `WwaExt.dll`, `WwaApi.dll`, and
`WWAHost.exe` paths and can launch an app root through `WWAHost.exe`; it does
not load, replace, or modify the Windows system DLLs. The native host exposes
`wwa-status` for diagnostics and `wwa-create PARENT NAME` for non-overwriting
app scaffolding (`index.html`, `app.js`, `manifest.json`, and `app.kuhul`).
Built-in templates now include `basic`, `dashboard`, and `editor`; the CLI
accepts `wwa-create PARENT NAME TEMPLATE`, and the HTTP create request accepts
an optional `template` key. Design kits now include `default`, `midnight`, and
`high-contrast`; each generated app records its design kit and server
`host`/`port` in both `manifest.json` and `app.kuhul`.
The localhost API exposes the same bridge through `GET /v1/wwa/status` and
`POST /v1/wwa/apps` with `{ "parent": "...", "name": "...", "template": "dashboard" }`. WWAHost
remains a separate process; the native engine owns validation and scaffold
creation. The app-engine catalog now includes hologram, liquid, quantum,
prime-cockpit, atomic, blocks, and scx design kits; runtime-hologram,
liquid-runtime, atomic-compression, blocks-hud, node-graph, prime-cockpit,
three-file-runtime, scx-inspector, and atomic-blocks demos; and reusable
layout, dashboard, runtime, developer, controls, and game components.
`wwa-demos` and `wwa-components` expose the catalogs through the CLI, while
`/v1/wwa/templates`, `/v1/wwa/kits`, `/v1/wwa/demos`, and
`/v1/wwa/components` expose them through HTTP. Generated apps now include
`components.json`, `demo.html`, and a backward-compatible typed XJSON
bootstrap contract in `manifest.json` with `xjson`, `schema`, and `nodes`
entries for the app tape, route, design kit, components, and demo surface.
A dependency-free read-only
SafeTensors index reader is now present as
`native/inference/safetensors_reader.*` and is exposed through
`--validate-safetensors`; it validates tensor names, dtypes, shapes, offsets,
and file bounds without loading weights.

`native/runtime/TaskEngine.cpp` now plans declarative `TaskList.kuhul` files
against the native provider registry. Tasks are dependency-ordered and report
admitted, blocked, or cyclic states; provider DLL invocation still requires an
explicit executor ABI. The task list can therefore be generated by a small
local instruction model, validated by the host, and then routed to a provider
without making the model the scheduler. A valid two-task job using
`d3dcompiler` followed by `shader_cache` reaches `admitted` status in
dependency order.

The explicit execution boundary is now implemented by
`native/runtime/task_executor_abi.{h,cpp}` and the
`kuhul_task_helper.exe` process. `task-run PATH` executes only the allowlisted
`probe_provider`, `probe_opencl`, `stream_xshard`, and `verify_scx` actions
through a version-independent JSON-lines process boundary. Ordinary provider
probes retain a five-second timeout; bounded model and graphics actions use a
35-second timeout. Helper success is reported as `completed`; missing
libraries, rejected actions, crashes, and timeouts are reported as `failed`.
Task dependency failures propagate as `dependency_failed` instead of allowing
downstream work to run.

## Bounded Agentic RAG Execution Graph

The native runtime already has the correct host abstraction: **FieldGraph**.
Agentic RAG is a retrieval subgraph within the persistent execution DAG, not
a separate graph authority. Its retrieval, grading, rewrite, and synthesis
nodes use the same edges, pressure, provenance, and fold evaluation law.

RAG feedback is represented as bounded graph evolution rather than an
unbounded cycle in one DAG version. A failed or incomplete retrieval creates a
validated successor state with an explicit retry edge, counter, deadline, and
terminal fallback:

```text
query analyzer
  -> parallel vector/web retrieval
  -> merge, RRF rank, and semantic deduplication
  -> relevance grader
  -> synthesis
       or bounded query rewrite -> retrieval
```

No graph transition may loop indefinitely, and a failed provider must produce
a structured failure that can route to an alternate provider.
Retrieval results must remain provenance-bearing through merge and synthesis.

The first implementation target is a native provider graph using the existing
`Runtime -> Fold -> Node -> Capability -> Provider` hierarchy. Independent
retrieval nodes should execute through the Task Engine and executor ABI, then
fan in to a deterministic merger. The grader should initially use a bounded
structured scorer or small local model rather than a heavyweight generation
call on every retry. RRF and semantic deduplication are required before
context assembly so overlapping vector and web results do not consume the
entire model window.

The RAG implementation must register its fields, nodes, and typed edges
through the existing FieldGraph/DAG contracts in `native/webx_compute.h` and
`native/FieldGraph.kuhul`; it must not introduce a parallel graph authority.
Retrieval passages, grader scores, rewrite history, and synthesis provenance
remain field state and must be serializable/replayable with the existing
runtime records.

### System Micronauts and KuhulAppCreator apps

The `micronauts/` tree contains **system Micronauts**: reusable runtime
services with explicit capability boundaries and, where useful, a local
server endpoint. `micronauts/kuhul-micronaut/interop/run-interop.cjs` is one
such service surface for GPU tensor interoperability; its HTTP server is an
adapter detail, not the definition of the Micronaut itself.

`KuhulAppCreator.cpp` creates app contracts that bind selected system
Micronauts into an application. A generated app therefore consumes system
capabilities through manifest/app.kuhul bindings instead of copying or
redefining the service. The application may use the interop service, a
storage service, a UI service, or another provider according to its declared
capabilities. System Micronauts remain independently addressable and
replaceable.

### Native KuhulAppHost and hosting profiles

`native/http_api_server.cpp` is the current native HTTP transport and already
serves health, provider, inference, chat, and WWA routes. It is currently
localhost-bound and therefore functions as a stack/local host. The WWA
surface is supplied by `native/runtime/KuhulAppCreator.cpp`, which remains the
app factory rather than becoming the HTTP host.

`kuhul_engine.exe serve` should grow into the `KuhulAppHost` entrypoint by
adding host configuration, manifest loading, route policy, and lifecycle
around these existing classes. It must support distinct hosting profiles:

```text
user host
  public UI/API, TLS through the deployment proxy, authentication,
  rate limits, and restricted task/chat routes

stack host
  private native control plane, manifests, TaskEngine, BOSS, FieldGraph,
  providers, model runtime, SCX, diagnostics, and artifact operations
```

The user host may call the stack host through a restricted RPC boundary, but
must not expose provider, BOSS, SCX, or internal manifest control directly.
The planned command surface is `serve --profile user|stack` with explicit
host, port, root, and manifest options. A separate parallel app-host
implementation is not required: `HttpApiServer` remains transport/routing,
`KuhulAppCreator` remains app scaffolding, and `KuhulAppHost` is the missing
configuration and security-policy layer.

### Explicit portal verbs

Prompt text must not implicitly select an execution path. MicrosoftSDK emits
an allowlisted verb alongside its declarative output:

```text
chat.respond  -> ordinary conversational response
task.plan     -> validated TaskList planning
app.create    -> TaskEngine admission followed by app materialization
app.inspect   -> read-only app/runtime inspection
build.game    -> game artifact planning/materialization
build.website -> website artifact planning/materialization
build.program -> program artifact planning/materialization
build.micronaut -> Micronaut artifact planning/materialization
```

Natural language aliases are normalized by policy (`site` to `build.website`,
`software` to `build.program`, and so on) and carried with a normalized
`target_kind`. The runtime dispatches the verb, then applies authentication,
profile policy, TaskEngine validation, provider admission, and BOSS/FieldGraph
execution. A verb is a routing signal, not authorization to execute arbitrary
model output.

Website and app requests have a second classification layer so “build me a
website” can become a concrete artifact request rather than a generic code
task:

```text
dashboard     -> dashboard template and runtime metrics surface
landing_page  -> public informational/marketing surface
sign_in       -> authentication UI backed by an explicit auth provider
payment_form  -> payment UI backed by an explicit Stripe/provider contract
```

Payment generation is configuration-driven. Models may describe the provider
and required integration boundary, but may not invent, store, or emit Stripe
keys or other credentials. User-owned `.env` values are loaded only by the
host/runtime; prompts, TaskLists, manifests, generated source, logs, and SCX
provenance must carry secret names/references only. TaskEngine and the host
must validate the provider, secret references, and execution permissions
before materialization.

An app may also **modify a system Micronaut when it uses it**, but the
modification must be explicit and scoped. `KuhulAppCreator.cpp` should record
the base Micronaut, mutation/overlay contract, app owner, version, and
capability delta in the generated XJSON manifest and `app.kuhul`. The runtime
must resolve the app-scoped variant for that app while preserving the
unmodified system definition for other apps. A shared system Micronaut is
never silently mutated in place.

```text
system Micronaut
  -> app binding
  -> app-scoped overlay or evolved variant
  -> capability/provider resolution
  -> app execution
```

The native `kuhul_engine.exe` Micronauts are **out of bounds** for app-level
mutation. They remain host-owned runtime infrastructure and may only be
invoked through their published capability/provider contracts. App evolution
is limited to system Micronaut packages and app-scoped overlays exposed by
the app-engine boundary; it must not patch, replace, rebind, or mutate the
engine's internal Micronauts or executable state.

### Task-list hosted applications

This makes the app engine Docker-like in packaging and isolation, but more
semantic in execution. A complete Kuhul app is a contract-bearing package
(`manifest.json`, `app.kuhul`, assets, components, and app-scoped
Micronaut overlays) launched by a simple Task Engine task list. The task list
declares lifecycle, dependencies, required capabilities, providers, and
shutdown behavior; it is not an instruction to modify `kuhul_engine.exe`.

```text
app package
  -> XJSON/app.kuhul contract
  -> TaskList admission and dependency order
  -> app-scoped Micronaut overlays
  -> published system Micronaut capabilities
  -> isolated app execution
  -> structured result and teardown
```

Unlike a container that primarily isolates processes and files, the WebX app
engine also validates semantic contracts, phases, provider availability,
provenance, and capability boundaries. Multiple apps can share host-owned
system Micronauts while retaining separate app state, overlays, task history,
and execution records. The Task Engine executor ABI is the process boundary
for actions that require helper isolation.

The current implementation boundaries are:

| Runtime surface | Responsibility |
| :--- | :--- |
| `KuhulAppCreator.cpp` | Create the app package and its XJSON/app.kuhul contract |
| `TaskEngine.cpp` | Parse, validate, order, and admit the model-produced TaskList |
| `kuhul_engine.exe task-boss` | Route admitted tasks into BOSS and FieldGraph execution |
| `OpenCLTaskAdapter.cpp` | Probe risky OpenCL capability through an isolated probe |
| `task_executor_abi.cpp` | Invoke an admitted allowlisted action out of process |
| `task_helper.cpp` | Execute provider probes and delegate OpenCL probing to the isolated helper |
| `RAG.cpp` | Build and update retrieval fields, evidence, coverage, and provenance |
| `DAG.cpp` | Validate and topologically evaluate persistent DAG dependencies |

The model therefore only needs to emit the app/task contract. It does not
need to know whether the selected system Micronaut is backed by OpenCL,
D3D11.1, CPU, storage, or another provider; admission and execution resolve
that boundary after the TaskList is received.

### MicrosoftSDK conversational planners

The Phi-2 and specialized Phi-3 GGUF models are appropriate for the
`MicrosoftSDK.ps1` planning/chat role even when they are not strong code
generation models. Their useful contract is conversational understanding,
topic extraction, clarification, task decomposition, and TaskList drafting.
They should not be treated as authoritative coders or executors.

```text
Phi-2 / Phi-3 conversational model
  -> MicrosoftSDK.ps1 prompt and planning layer
  -> constrained TaskList / app intent
  -> native validation and provider resolution
  -> executor ABI or app host
```

The SDK should prefer these small local models for dialogue and planning,
while `TaskEngine.cpp` receives and executes the resulting code tasks through
validated TaskLists. The conversational model does not need to write the code
itself; it needs to describe the requested code job, files, dependencies,
capabilities, and acceptance conditions clearly enough for the Task Engine to
admit and route it. Code-producing work remains a task/provider concern, and
execution stays under the native runtime contracts.

```text
Phi-2 / Phi-3 conversation
  -> code-task description and TaskList
  -> TaskEngine.cpp
  -> dependency/provider admission
  -> BOSS / FieldGraph execution or helper executor
  -> structured result
```

`task-boss` is the explicit native bridge for the BOSS path. In the current
implementation, TaskEngine owns parsing, dependency ordering, provider
admission, and failure propagation; only admitted tasks become temporary
FieldGraph fields/cards and are executed through the simplified BOSS layers.
This proves the TaskEngine -> BOSS -> FieldGraph path, but it does not yet mean
that BOSS consumes the complete persistent graph system.

The remaining graph integration is explicit: BOSS must load the manifest
topology (`rpc.manifest.json`, `server.manifest.json`, `api.manifest.json`,
`model.manifest.json`, and related contracts when present), resolve it into the
persistent FieldGraph DAG, evaluate the admitted working set through the
`Pop -> Wo -> Yax -> Sek -> Chen -> Xul` fold law, and persist provenance,
pressure, confidence, and survivor state through SCX. Until that work is
complete, `native/runtime/SCXcache.manifest.json` is a declarative cache
boundary rather than a BOSS graph loader.

`task-run` remains the isolated helper-ABI path for actions that should not
enter BOSS. The model cannot call BOSS directly or bypass TaskEngine.

`RAG.cpp` and `DAG.cpp` are complementary rather than competing runtimes.
`RAG.cpp` owns retrieval semantics: query analysis, vector/web retrieval
requests, merge/deduplication, relevance feedback, rewrite state, and
provenance-bearing context. `DAG.cpp` validates and evaluates the persistent
execution DAG: dependency order, parallel fan-out/fan-in, and bounded
ready-task selection. Both operate on the FieldGraph DAG and must use the
Task Engine and executor ABI for actual provider work.

The RAG retry loop remains a bounded, versioned state transition over
FieldGraph DAG nodes. Each retry records its predecessor and retry metadata;
`DAG.cpp` must never turn the retry transition into an unbounded cycle.

The first native scaffolds are now present:
`native/runtime/RAG.cpp`/`rag.h` provide deterministic merge, lightweight
coverage/relevance grading, and a bounded rewrite counter;
`native/runtime/DAG.cpp`/`dag.h` provide deterministic topological scheduling
with duplicate, missing-dependency, and cycle errors. The
`native/runtime/SCXcache.manifest.json` defines the persisted DAG state and
artifact-cache boundary; provider calls remain outside these small policy
services.

`native/runtime/DirectML.cpp`/`directml.h` now provide an explicit DirectML
provider probe. The probe only confirms that `DirectML.dll` and the required
device-creation export are present; it does not claim that a model can run or
move work to the GPU. Actual DirectML execution remains gated on adapter
feature validation, bounded workload policy, and a compatible llama.cpp/ggml
backend. CPU remains authoritative on the target rig until those checks pass.

### ggml WebGL2 comparison and adapter direction

The local llama.cpp sources confirm that `ggml-webgpu` and `ggml-xcfe` are
currently the same Dawn/WebGPU backend: their CMake files have identical
SHA-256 content, both embed WGSL shader libraries, and both require Dawn when
not compiled through Emscripten. They are useful references for operation
coverage and shader organization, but they are not a usable backend for this
rig.

`ggml-webgl2` would therefore be a new browser backend, not a rename or
configuration switch. It should reuse the generic tensor/operator contracts
from `ggml.c`, allocation behavior from `ggml-alloc.c`, and CPU fallback
semantics from `ggml-cpu`, while replacing compute shaders with WebGL2
fragment/render-target or transform-feedback kernels. WebGL2 has no native
compute-shader model and cannot provide the D3D12 shared-resource path.

The planned boundary is:

```text
ggml core / allocator / CPU
  -> ggml-webgl2 browser adapter
  -> WebGL2 context and bounded kernels
  -> browser-side fallback only
```

Keep this separate from the native D3D11.1/D3D12 provider and do not make it
an authority for native llama.cpp execution. Start with small deterministic
ops and explicit tensor-size limits before considering quantized matmul or
attention kernels.

### Existing XCFE/DirectML build evidence

The separate `C:\Users\canna\khanary-llama-build\ggml\build\bin\Release`
bundle already provides a more relevant native path than Dawn:
`ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-xcfe.dll`,
`DirectML.dll`, and `dml_gemm.dll`, plus `xcfe_probe.exe` and
`xcfe_matmul_test.exe`. The probe reports `XCFE` and `CPU` backends with
`XCFE registered: YES`. The matmul test reports
`[ggml-xcfe] MUL_MAT path: DirectML (GPU)` and matches the CPU result with a
maximum absolute error of `5.960e-07`.

This establishes that the practical native GPU direction is the existing
`ggml-xcfe` DirectML backend, not WebGPU/Dawn. The next integration step is
to make the native provider resolver load and validate this bundle explicitly,
including `dml_gemm.dll`, before exposing an XCFE capability to a TaskList.

The matching standalone `llama-cli.exe` currently reports no devices from
`--list-devices`, even though Gemma and Dolphin GGUF inference works with
`-ngl 99` through the CPU build. Therefore standalone `xcfe_matmul_test.exe`
proves the DirectML GGML kernel path, but does not yet prove llama model
offload. A llama GPU claim requires rebuilding/linking llama against the
XCFE-enabled GGML backend and then repeating device discovery and generation.
The provider registry now exposes this bundle as `xcfe_directml` with
`TensorCompute` capability and `mul_mat`/`directml_gemm` outputs. Its presence
is still only admission evidence; an execution task must use the matching
GGML/llama ABI and report the actual operation result.

### GPU Micronaut backend constraint

For the target Intel HD 4600-class rig, WebGPU/Dawn is not an available
production backend. The Kuhul GPU Micronaut therefore treats native
D3D11.1/D3D11On12 interop as the authoritative GPU path, with CPU fallback
for unsupported operations. WebGL2 may serve browser clients as a separate
bounded compatibility/visualization adapter, but it does not provide native
D3D12 shared-resource execution. Dawn-based Node servers remain optional
experiments and must not be required by FieldGraph or Micronaut execution.

### Runtime confidence service

`native/runtime/Confidence.cpp` and `confidence.h` now provide the reusable
`ConfidenceEngine`. `Chen` owns the point in the phase cycle where confidence
is observed and updated, but the service owns the scoring mechanics. Its
inputs are normalized verification, replay, provider, and agreement evidence;
it does not inspect logits or make provider decisions. The service also accepts
semantic coverage, user feedback from topic discussion, and an explicit gap
penalty. This is important because a web-researched answer can still expose
missing definitions, unresolved follow-up questions, contradictions, or weak
explanations when users continue the discussion. Those conversational gaps
must lower confidence or trigger verification/retrieval, rather than being
mistaken for new evidence of correctness.

Confidence is therefore revisable across turns: discussion events attach to
the relevant FieldGraph node or fold, `Chen` observes them, and later `Yax`
can admit the node for clarification, additional retrieval, or replay.
Promotion, scheduling, and Yax admission may consume the resulting node
confidence without changing the fold order. Pressure remains a separate
scheduling signal.

### Discussion-driven Micronaut Forge

Prompts and replies are semantic context, not just transient chat strings.
`MicronautForge` should identify the related topic/domain represented by a
discussion and create or reuse a topic Micronaut with an explicit capability
and provenance contract. The resulting Micronaut is a bounded semantic
specialist; it does not become an unrestricted autonomous actor.

The discussion path is:

```text
prompt/reply
  -> topic and domain extraction
  -> FieldGraph topic fold/node
  -> Pressure / Evidence / Provenance
  -> Confidence update
  -> MicronautForge create-or-reuse
  -> Scheduler and ProviderResolver
```

`Pressure.cpp` determines whether the topic deserves work,
`Evidence.cpp` records what the discussion establishes or leaves unresolved,
`Provenance.cpp` tracks source and conversational lineage, and
`Confidence.cpp` evaluates coverage and trust. `Scheduler.cpp` and
`ProviderResolver.cpp` select bounded work and implementations.
`FieldGraph.cpp` owns the semantic relationships; `Chen` remains the phase
hook that observes results, while `MicronautForge.cpp` creates the topic
capability contract.

A user introducing A and B while omitting related C should create a
detectable coverage gap for C. The system may then create or reuse a C-topic
Micronaut and schedule clarification or retrieval, rather than silently
assuming the explanation is complete.

Forged Micronauts also carry field capability declarations. Forge resolves
field intent from the prompt and copies the resulting schema references into
the `MicronautNetwork`, each `MicronautConfig`, and the `ForgeResult`:

```text
prompt
  -> field intent extraction
  -> π_field_v1 schema selection
  -> Micronaut field_specs declaration
  -> native provider implementation
  -> FieldGraph/BOSS scheduling
```

The schema is the contract; it does not duplicate the implementation. For
example, an attraction or repulsion request declares
`native/win2d/field_system/specs/attraction_well_spec.json`, which resolves to
`PiKuhul::AttractionWell`. Navigation intent may additionally declare
`navigation_force_spec.json`. `Micronaut::getFieldSpecs()` exposes these
declarations to admission and inspection, while the native provider remains
responsible for execution and validation.

### Chat history and preference evidence

`ChatHistory.cpp` should be a separate runtime service for retaining
structured conversation events that affect semantic coverage, confidence,
provenance, and topic-Micronaut reuse. It may record prompt/reply lineage,
topic transitions, corrections, explicit user preferences, unresolved
questions, and accepted or rejected explanations.

This is **RLHF-compatible evidence**, but it is not itself an RLHF trainer.
Online ChatHistory events should update runtime confidence and FieldGraph state;
offline preference-pair or reward datasets may later be exported through an
explicit consent- and policy-gated training pipeline. The service must preserve
source attribution, avoid treating silence as approval, and distinguish user
correction from model-generated text.

```text
ChatHistory
  -> Evidence / Provenance
  -> Confidence and semantic coverage
  -> topic Micronaut reuse or creation
  -> Scheduler / ProviderResolver
```

The initial bounded service is implemented in
`native/runtime/ChatHistory.cpp` and exposes semantic coverage, user feedback,
and unresolved-topic gap metrics. `ConfidenceEngine::UpdateFromChat` consumes
those metrics so a user correction or an omitted related topic can revise a
node without treating the conversation as model-training data by default.

Planned graph states are `analyze`, `retrieve_vector`, `retrieve_web`,
`merge_context`, `grade_context`, `rewrite_query`, `synthesize`, `fallback`,
and `complete`. Each state needs a typed input/output contract, timeout,
provider identity, and replayable execution record. LangGraph/LlamaIndex are
not runtime dependencies; Python may be used for offline experiments, but the
authoritative graph and provider execution remain native and deterministic.

TaskEngine now also accepts dependency-free JSON task lists with explicit
`id`, `action`, `description`, `provider`, and `depends_on` keys. The compact
CLI form is:

```text
kuhul_engine.exe -task native/runtime/TaskList.json
```

The `.kuhul` and JSON forms share the same validation, dependency ordering,
provider admission, and execution path. The Intel 64-bit executor was probed

TaskEngine resolves repository-relative task paths while running from
`build/bin/Release`, so `native/runtime/TaskList.json` and
`native/runtime/TaskList.kuhul` remain valid input paths from the build output
directory. The executable must be rebuilt after adding the `-task` alias. The Intel 64-bit executor was probed
The refreshed `build-llama/bin/Release/kuhul_engine.exe` accepts both
`task-engine` and `-task`; the sample plans reach provider admission and
correctly report unavailable OpenCL CPU providers as blocked when their
executor DLLs are absent. The Intel 64-bit executor was probed successfully
and exposes `GetTaskExecutor -> ITaskExecutor*`; this is a private C++
interface, so TaskEngine reports it without guessing vtable methods.
`probe_provider` remains the generic library-presence action, while
`probe_opencl` is an explicit standard-OpenCL action.

`stream_xshard` and `verify_scx` are explicit helper actions. They require
`KUHUL_XSHARD_ROOT` and invoke bounded `stream-xshard` and
`scx-d3d11-smoke` commands in a child `kuhul_engine.exe`, preserving the
host/helper boundary. The D3D11 smoke path loads the precompiled `cs_5_0`
CSO and requests feature level 11_1. The first end-to-end run streamed four
bounded GPT-OSS xshard tiles successfully. The current `build-llama` runtime
must remain the validation target; its D3D11 smoke result is still isolated
from the known-good Khanary GGML/DirectML trainer baseline. Do not claim a
D3D11 WARP success unless the active `build-llama` executable reproduces it.

`native/runtime/OpenCLTaskAdapter.cpp` no longer loads or enumerates an
OpenCL DLL inside `kuhul_engine.exe`. It launches
`kuhul_opencl_helper.exe`, which searches the configured `KUHUL_DRIVER_ROOT`
for the standard Intel OpenCL names and performs platform/device discovery in
the child process. The helper supports both `--probe` for admission and
`--execute` for the `probe_opencl` TaskEngine action, returns bounded JSON, and
is terminated by the parent after five seconds. The separate
`kuhul_task_helper.exe` remains the TaskEngine ABI boundary and delegates
OpenCL requests to the OpenCL helper, so driver enumeration cannot stall the
native host.

The current driver directory contains the OpenCL CPU runtime, device,
backend, TBB, Clang, and task-executor DLLs under
`C:\DRIVERS\VDO\h2vdo66us14\Gfx\950fd7b1-7601-4c2a-ae29-d33825f748c9`.
Set `KUHUL_DRIVER_ROOT` to that directory before probing or running the
OpenCL task list. Provider discovery still treats the private
`GetTaskExecutor -> ITaskExecutor*` export as diagnostic only; no private
vtable invocation is permitted.

The next SCX stability pass added optional DX12 debug/GPU-assisted validation
(`KUHUL_D3D12_DEBUG=1`) and explicit upload/destination bounds plus command
reset guards. Upload-heap transition barriers were removed because upload
resources must remain in `GENERIC_READ`; this was not sufficient to resolve
the known multi-tile access violation. One FP16 Q tile remains stable, while
the two-tile smoke still exits with `0xC0000005` after the first decode,
including on WARP. Keep multi-tile execution blocked while isolating the
remaining command/resource or source-tile lifetime fault.

A fresh per-tile command allocator/list experiment was attempted, but it
caused the one-tile smoke to fail before decode and was reverted. The
single-list path remains the only known contract, while the current build
still needs a clean one-tile baseline before further SCX edits are safe.

An alternate D3D11 feature-level 11_1 / SM5 path was also tested through
`--infer-xshard`. The original generated softmax used
`numthreads(2880,1,1)`, which is invalid for SM5; it now uses a 256-thread
strided reduction and compiles successfully as `cs_5_0`. A full 2,880-wide
FP32 expert run is computationally impractical on this rig and was stopped,
so this path is a shader-compatibility fallback, not yet a replacement for
the bounded SCX smoke.

The D3D11 fallback now enforces a CPU-first work budget through
`KUHUL_D3D11_MAX_WORK` (default 500 million multiply/add units). GPT-OSS-scale
QKV/expert requests are rejected before GPU allocation or dispatch with an
explicit instruction to use the CPU model path; smaller SM5 validation cases
remain available.

Standalone D3D11 SM5 CSOs now exist for the GPT-OSS 2,880-wide expert shape:
`native/shaders/bin/d3d11/xshard_softmax_2880.cso` and
`native/shaders/bin/d3d11/xshard_vmul_2880.cso`. They are compiled with the
Windows SDK `fxc` target `cs_5_0` and loaded by `xshard_attention.cpp` for
matching dimensions; other shapes continue to use runtime `D3DCompile`.
The same expert geometry is present in both
`E:\models\GPT-OSS\all_layers\experts_staging` and `E:\models\GPT-OSS\hf`
(2,880 × 2,880, FP32, 32 tiles), so the generated CSOs cover either source.
The `hf` tree additionally contains the FP16 Q/K/V/O attention shards.

The expert files are not spatially tiled: each of their 32 records is a full
2,880 × 2,880 matrix. The next expert-specific task is therefore to preserve
the record/expert identity while repacking or block-reading each matrix into
bounded spatial tiles, then use a dedicated FFN/GEMM SM5 kernel. The current
attention CSOs are only a shader-loading compatibility probe and must not be
treated as the final MoE expert execution path.

`native/inference/xshard_block_reader.*` now provides bounded row/column
access to the full-matrix expert records without changing the XSQ2 container
or losing expert identity. `xshard-block PATH EXPERT ROW ROWS COL COLS`
exposes the reader through the native host; a 256 × 256 FP32 block was read
successfully from expert 0. This is the foundation for a dedicated streamed
FFN/GEMM provider and avoids uploading the full 33 MB expert record.
`xshard-block-plan PATH BLOCK_ROWS BLOCK_COLS` now emits the deterministic
expert/block schedule; a 256 × 256 plan produces 144 blocks per expert and
4,608 blocks across the 32-record expert shard.
The full-resident `sxme_expert.hlsl` path is not suitable for these blocks, so
`native/shaders/bin/d3d11/sxme_expert_block.cso` was added and compiled as
`cs_5_0`. It computes one streamed `X × W_block` phase with bounded input and
output buffers; gate/up/down accumulation and the D3D11 binding adapter remain
the next integration step. `D3D11ExpertBlockGemm` and
`xshard-block-gemm PATH EXPERT ROW ROWS` now bind the CSO, upload one bounded
weight block, dispatch on D3D11, and read back only the block output. Expert 0
rows 0–255 completed successfully with a streamed 256 × 2,880 block.
An INT4 companion, `native/shaders/bin/d3d11/sxme_expert_block_int4.cso`,
now uses `ByteAddressBuffer.Load` to unpack eight 4-bit values per `uint` and
apply scale/zero-point during GEMM. Its D3D11 raw SRV binding still needs a
dedicated adapter (`D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS`,
`DXGI_FORMAT_R32_TYPELESS`, and `D3D11_BUFFEREX_SRV_FLAG_RAW`).
