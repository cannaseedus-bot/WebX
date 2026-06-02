// D12WebX - Main entry point
// Exports all public APIs for GPU computing and KUHUL 3D operations.

export { default as D12WebX, GPU_FLAGS } from './d12webx.js';
export { default as KuhulD12WebX, GLYPHS, applyGlyph } from './kuhul.js';
export { default as CommandList } from './command-list.js';
export { default as GPUMemoryAllocator } from './gpu-allocator.js';

// K'UHUL++ v2.0 Compiler
export { tokenize, TokenType, KEYWORDS, GLYPHS as GLYPH_SYMBOLS, LexerError } from './compiler/lexer.js';
export { parse, NodeKind, ParseError } from './compiler/parser.js';
export { analyze, SemanticError } from './compiler/semantic.js';

// K'UHUL++ v2.0 Runtime
export { KuhulRuntime, RuntimeError, run } from './runtime/runtime.js';

// FLUX Runtime — deterministic state machine (Layer 2: the time machine)
// K'UHUL phase mapping: Pop=dequeue Wo=dispatch Sek=reduce Ch'en=commit+notify
// Math -> FLUX IR -> Runtime -> CPU/OS/Network
export { FluxRuntime, ActionQueue, StoreRegistry, EffectEngine,
         TimerEffect, PromiseEffect, TimeTraveler,
         createRuntime } from './runtime/flux-runtime.js';

// FLUX IR — intermediate representation (Layer 1: time made explicit)
// pure() store() action() reduce() query() effectCreator() view() module()
// fromDescriptor(): µMODEL descriptor -> FLUX IR program
// generateJS(): FLUX IR -> idiomatic JavaScript source
// COUNTER_PROGRAM: canonical usage example
export { pureFn, storeDef, actionDef, reduceDef, queryDef,
         effectCreatorDef, viewDef, moduleDef,
         FluxIRInterpreter, generateJS, fromDescriptor,
         COUNTER_PROGRAM } from './runtime/flux-ir.js';

// XVM CPU Cluster (v0.1.0-xvm-cpu-thread-cluster)
export { default as CPUCluster32, FLAG as XVM_FLAG, OP as XVM_OP, CLUSTER_SIZE_CONST, PHASE_COUNT_CONST, XVM_TRAINING_OPTIMUM } from './xvm/cpu-cluster.js';

// XVM 1000-fiber CUDA emulator (.gpu_trainer)
export { default as FiberPool1000, OP_EXT as XVM_OP_EXT, WARP_SIZE, DEFAULT_CONFIG as FIBER1000_CONFIG } from './xvm/fiber-1000.js';

// SMGM-16 model descriptor and runtime (v0.1.0-xvm-cpu-thread-cluster)
export { default as SMGM16Runtime, SMGM16_CONFIG, LOSS_LAMBDAS, SHARD_MAP, PARAM_COUNTS, tokenFeatures, cardSlotShape } from './smgm16/index.js';

// Atomic Brain Hypergraph v10.0 — 96-glyph opcode sugar for µMODELs
// Glyph groups: control_flow(0x20-0x31) tensor_core(0x32-0x41)
//               tensor_net(0x42-0x51)   phase_quantum(0x52-0x61)
//               geodesic_wave(0x62-0x71) layout(CJK IDC ⿰⿱⿲⿴⿵⿶⿷⿻◯)
// Pi token base: pi digits -> geodesic phase rotation -> matrix series S
// Deep Thinking Index: D = N*G*T*sum(kappa_i), 2048 for 8-brain chain
export { KuhulAtomicBrain, resolveGlyph, opcodesForFold,
         piGeodesicSequence, piMatrixSeries, deepThinkingIndex,
         MUPY_ATOMIC_BRAIN_KXML, MUPY_ATOMIC_BRAIN_SPEC,
         ATOMIC_BRAIN_IR,
         // Math functions (Layer 0) tied to glyph opcodes
         fibonacci, fibonacciFold, zeckendorf, PHI,     // 0x34 TENSOR_INT, 0x36 TRI_SUM
         mayanEncode, mayanDecode, longCount,            // 0x67 TRIPLE_GEO
         dot, matmul, outer, norm, normalize, softmax,   // 0x42 TENSOR_CORE, 0x44 TENSOR_PROD, 0x40 DOT
         cross3, geodesicDist,                           // 0x3F CROSS, 0x64 PARALLEL_GEO
         GLYPH_SPEC, PI_DIGITS } from './xcfe/atomic-brain.js';

// µPY — Python->KXML transpilation bridge (write Python-like tensors, compile to µBRAIN)
// MuPYTranspiler: Python op AST -> KXML nodes with Lipschitz bounds + backward edges
// MuPYRuntime:    execute KXML graphs, train with backward pass, export to µJSONL
// TWO_LAYER_MLP:  canonical 2-layer MLP spec (gemm->add->relu->gemm->add->softmax)
export { MuPYTranspiler, MuPYRuntime, MU_BUILTINS, TWO_LAYER_MLP } from './mupy/mu-py.js';

// µBRAIN — Atomic Cognitive Architecture (Think->Research->Reason->Plan->Execute)
// Four differentiable modules, each backed by a µJSONL grammar.
// backward(error) propagates through plan->reason->research->think.
export { MuBrain, MuThink, MuResearch, MuReason, MuPlan,
         MUPY_BRAIN_KXML, MUPY_BRAIN_SPEC } from './mupy/mu-brain.js';

// µJSONL — Atomic JSONL Grammar Units: differentiable linguistic patterns
// Each line = one provable µModel (Lipschitz bound + trainable weight + phase gate)
// Bridge: symbolic grammar rules <-> connectionist weights + gradient descent
// Forward = Sek, Backward = Ch'en, Compile = Wo, Load = Pop
export { MuJSONLCompiler, MuJSONLTrainer, MuJSONLOptimizer,
         MuJSONLGrammar, BASE_GRAMMAR,
         Loss, MU_MODEL_TYPES, PHASES } from './mupy/mu-jsonl.js';

// Base µMODEL registry — 11 canonical Drivers/Kernels shipped with every runtime
//   Numeric:  fibonacci_fold  pi_field  mayan_fold  linalg_solver  geodesic_router
//   Glyphs:   tensor_ops  geodesic_ops  logic_ops  phase_ops
//   Runtime:  flux_runtime  semantic_reader
export { BASE_MUMODELS,
         FIBONACCI_FOLD_SPEC, FIBONACCI_FOLD_KXML,
         PI_FIELD_SPEC,        PI_FIELD_KXML,
         MAYAN_FOLD_SPEC,      MAYAN_FOLD_KXML,
         LINALG_SOLVER_SPEC,   LINALG_SOLVER_KXML,
         GEODESIC_ROUTER_SPEC, GEODESIC_ROUTER_KXML,
         TENSOR_OPS_SPEC,      TENSOR_OPS_KXML,
         GEODESIC_OPS_SPEC,    GEODESIC_OPS_KXML,
         LOGIC_OPS_SPEC,       LOGIC_OPS_KXML,
         PHASE_OPS_SPEC,       PHASE_OPS_KXML,
         FLUX_RUNTIME_SPEC,    FLUX_RUNTIME_KXML,
         SEMANTIC_READER_SPEC, SEMANTIC_READER_KXML } from './mupy/base-models.js';

// muPY — Python evolutionary layer for µMODEL Drivers/Kernels
// µMODEL: Schema/TOML/YAML/XML/MD as behavioral spec — no hand-written weights needed
// µPY: Python trainer + evolver that turns specs into running specialist models
export { parseMupySpec, buildMupyDescriptor, registerMupy, getMupy, listMupy,
         MUPY_MATH_SPEC, MUPY_CODER_SPEC } from './mupy/index.js';

// KLSL compiler — K'UHUL Language Shading Language (.gpu_trainer)
export { default as klslCompile, KLSL_OP, GLYPHS as KLSL_GLYPHS } from './klsl/index.js';

// XCFE @ Namespace Runtime — complete executable tensor grammar (v3.5.0-WebX)
//
// 8-layer @ computational stack:
//   L8 @semantics  — meaning encoding      L4 @protocol — transport
//   L7 @ngram      — pattern recognition   L3 @opcodes  — instructions
//   L6 @notation   — representation        L2 @context  — environment
//   L5 @jsonl      — streaming data        L1 @folds/@micro_folds — atomic
//
// Execution hierarchy: @agent → @skill → @micronaut → @command → @tool
// Temporal fabric:     @flux → @tick → @thread → @batch → @round → @step → @mark → @map → @graph
// Math foundation:     @mathml → @formula → @linalg → @matmul → @loop → @biginteger → @fibonacci → @pi → @zero → @vigesimal
//
// Tensor theorem: XCFE = rank-R sparse tensor over @-space (39 basis vectors)
//   Block nesting ↔ ⊗   Parallel siblings ↔ ⊕   Dataflow ↔ Σᵢ (contraction)

// Node.js IPC + runtime primitives (@node.ipc @node.fs @node.http @node.worker etc.)
export { XCFENodeRuntime, xcfe, AT_OPCODE_MAP } from './xcfe/node-runtime.js';

// Execution hierarchy (@agent @skill @micronaut @command @tool @orchestrator)
export { XCFEAgentRuntime, Agent, Skill, Micronaut, Tool,
         AGENT_REGISTRY, SKILL_REGISTRY, MICRONAUT_REGISTRY,
         COMMAND_REGISTRY, TOOL_REGISTRY, AGENT_OPCODE_MAP } from './xcfe/agent-runtime.js';

// Fold algebra (@semantics @opcodes @context @folds @horizontal_folds @micro_folds)
export { SemanticLayer, OpcodeLayer, ContextLayer, Fold, HorizontalFold,
         MicroFold, FoldPipeline, registerFoldNamespaces,
         SEMANTIC_MODELS, MICRO_OPCODES } from './xcfe/fold-algebra.js';

// Communications stack (@protocol @jsonl @notation @ngram) + 8-layer @ map
export { ProtocolLayer, JsonlStream, NotationLayer, NgramAnalyzer,
         registerCommsNamespaces, AT_STACK } from './xcfe/comms-layer.js';

// Math foundation (@mathml @fibonacci @pi @matmul @linalg @loop @biginteger @formula @zero @vigesimal)
export { ZERO, Vigesimal, PI_CONSTANTS, PiCompute, Fibonacci, BigInteger,
         Loop, MatMul, LinAlg, Formula, MathML,
         registerMathNamespaces, MATH_OPCODE_MAP } from './xcfe/math-layer.js';

// Temporal fabric (@flux @tick @thread @batch @round @step @mark @map @graph)
export { Tick, Flux, ThreadPool, Batch, Round, Step, Mark, Mapper, Graph,
         registerTemporalNamespaces, TEMPORAL_OPCODE_MAP } from './xcfe/temporal-layer.js';

// XCFE Tensor Algebra — grammatical tensor theorem + Tucker decomposition
export { XCFETensor, blockToTensor, TuckerDecomposition, semanticSimilarity,
         estimateRank, AT_AXES, AT_BASIS_DIMENSION,
         SEMANTIC_CLUSTERS, XCFE_TENSOR_THEOREM, RANK_11_EXAMPLE } from './xcfe/tensor-algebra.js';

// XCFE Imperative Layer — @verbs @endpoints @pagination @generate @validate
//                          @parse @render @import @class @function @action @program
export { VERBS, VerbChain, EndpointRegistry, Paginator, Generator, Validator,
         Parser, Renderer, ImportManager, ClassDefinition, FunctionDef,
         ActionRunner, Program, registerImperativeNamespaces,
         IMPERATIVE_OPCODE_MAP } from './xcfe/imperative-layer.js';

// SK Schema Builder — auto-generate JSON Schema from Python/JS type annotations
// Port of Microsoft Semantic Kernel's KernelJsonSchemaBuilder.
// Every micronaut, agent, and DXSK tool can self-describe via buildFromClass()
// instead of hand-writing TOOLS dict entries.
// t.int() t.str() t.list() t.optional() t.enum() t.obj() — type DSL
// @kernelFunction / registerKernelFunction — method decorator
// buildMicronautManifest(registry) — CSV registry → full tool manifest
export { KernelJsonSchemaBuilder, kernelFunction, registerKernelFunction,
         micronautToTool, buildMicronautManifest, t } from './xcfe/sk-schema-builder.js';

// XCFE Gravity — K'UHUL physics engine (gravity/antigravity field + Poisson solver)
// ⟁Grav⟁ = constrained (phase-gated, Lipschitz-bounded, gradient-clipped)
// ⟁AntiGrav⟁ = float (debug nodes, telemetry, bypass phase machine)
// KuhulPhysicsSolver: auto-adjusts gravity_scale per node based on loss/grad observations
// Field equation: ∇²Φ = ρ_gravity + ρ_antigravity  (stable when ratio ≥ 10)
export { GravityField, PhysicsDispatcher, GravityAdvisor, KuhulPhysicsSolver,
         G, antigravity, heavyGravity, negativeGravity, parseGravityAttr,
         registerGravityNamespaces, GRAVITY_OPCODE_MAP } from './xcfe/gravity.js';

// XCFE KXML Bridge — graph topology → KU'HUL DAG + Micronaut compilation
// KXML <node> = KXMLMicronaut    KXML <edge> = FoldEntanglement
// Forward pass = Sek→Ch'en       Backward pass = Ch'en→Sek
// Soft landing = ||∇f|| ≤ L·||x|| (Lipschitz theorem verified)
export { KXMLBridge, KXMLMicronaut, FoldEntanglement, LipschitzAnalyzer,
         KXML_PHASES, PHASE_INDEX, KXML_OPCODE_TABLE, ALLOWED_TRANSITIONS,
         KXML_KUHUL_MAP, BRIDGE_THEOREMS, isValidTransition, canExecute,
         registerBridgeNamespaces } from './xcfe/kxml-bridge.js';

// XCFE Knowledge Layer — computational epistemology
// Aristotelian: @who @what @where @when
// Causal:       @cause @effect @event @mutate @reward @evolve
// I/O:          @cdata @in @out @save @post @read @write @search
export { WhoContext, WhatEntity, WhereContext, WhenContext, CausalModel,
         EffectMeasure, EventBus, MutationEngine, RewardFunction, EvolutionEngine,
         CData, IOBoundary, PersistenceLayer, Publisher, SearchEngine,
         registerKnowledgeNamespaces, KNOWLEDGE_OPCODE_MAP } from './xcfe/knowledge-layer.js';

// LoRA delta weight adapters (.gpu_trainer/trainer/)
export { default as LoRAAdapter, loadAdapterFromBuffer, createAdapterStub, applyLoRA, unpackINT4 } from './adapters/adapter-loader.js';
export { ADAPTER_REGISTRY, BASE_MODEL, ADAPTER_CONFIG, resolveAdapter, listAdapters } from './adapters/adapter-registry.js';

// Glyph IPC protocol + INT4 ISA (v0.2.0-kuhul-directx-native)
export { IPC_STATUS, IPC_HEADER, GLYPH_ENTRY, INT4_ISA, GLYPH_MODE, readIPCHeader, readGlyphEntry, readGlyphEntries, encodeINT4Program } from './ipc/glyph-ipc.js';

// KBC1 program builder (v0.2.0-kuhul-directx-native)
export { default as KBC1Program, KBC1Instruction, KBC1_OP } from './kbc1/kbc1-program.js';

// KUHUL 3D compiler — K3D source → IR JSON (v0.2.0-kuhul-directx-native)
export { default as compileK3D } from './k3d/compiler.js';

// SCXQ2 binary format parser (v0.1.0-igpu-trainer)
export { default as parseScxq2, parseScxq2OrThrow, extractInstructions } from './scx/scxq2-parser.js';

// D3D11 trainer architecture descriptor (v0.1.0-igpu-trainer)
export { XVM_D3D11_BUFFERS, TRAINER_BUFFERS, FORWARD_PIPELINE, BACKWARD_PIPELINE, D3D11_NAN_BUGS } from './d3d11/trainer-arch.js';

// XJSL — Cross-platform Shader JSON Language (v0.1.1-igpu-trainer-xjsl)
export { generateWGSL, generateHLSL, lowerXJSLDoc } from './xjsl/lowering.js';
export { XJSLWGPURuntime }                           from './xjsl/wgpu-runtime.js';
export { validateXJSLDoc, validateXJSLDocOrThrow }   from './xjsl/validate.js';
export { AUTOGRAD_RULES }                            from './xjsl/autograd.js';
export { GPT2_CONFIG, TrainerConfig, FUSED_OPS, XJSL_SCHEMA } from './xjsl/index.js';

// XJSL Fused Op schemas + Math Declarations + JSONL compiler (v3.5.0-WebX)
export { XJSL_OP_REGISTRY, FUSED_MLP_SCHEMA, FUSED_ATTENTION_SCHEMA, FUSED_NORM_SCHEMA, FUSED_OP_SCHEMAS, buildFusedNode } from './xjsl/fused-op-schemas.js';
export { MATH_DECL_SCHEMA, MATH_DECL_CONTRACT, KUHUL_PHASES as MATH_DECL_PHASES, MATH_DECL_DOMAINS, MATH_DECL_EXAMPLE, MATH_DECL_PIPELINE, validateMathDecl } from './xjsl/math-declarations.js';
export { compileJsonlToXjsl, compileMany, stripCompilerMeta, fillBindings } from './xjsl/jsonl-compiler.js';

// K'uhul Fold Engine + DOM (v3.5.0-WebX)
export { TEMPORAL_FOLDS, TEMPORAL_FOLD_ROLES, SEMANTIC_SUBGRAPH_TEMPLATES, createFoldNode, createFoldEdge, buildTemporalSpine, expandSemanticFold, compileFoldDAG, foldDagToXvm, KUHUL_GLYPHS as FOLD_ENGINE_GLYPHS, createKNode, buildKuhulDom, serializeKuhulDom } from './kuhul/fold-engine.js';

// MATRIX DAG Runtime — .mxb graph + UnifiedAPI + K'uhul/XCFE scheduling (v3.5.0-WebX)
export { LANE_TYPES, JOB_STATE, nodeToUnit, createMicronautJob, loadMxb, buildRuntime, runGraph, annotateKuhulFolds, getNodesByType, getNodeById, graphSummary } from './matrix/dag-runtime.js';

// ASX RAM — schemas, flux gate policy, replay verifier, CSS projection, FLASH RAM (v3.5.0-WebX)
export { ASX_RAM_SCHEMA, PI_MUTATION_SCHEMA, CSS_PROJECTION_SCHEMA, FLUX_GATE_POLICY, TICK_PHASES, CSS_BINDING_TABLE, createEmptyRam } from './asx/schemas.js';
export { setSha256, sha256Hex, stableStringify, deepClone, getByDotPath, setByDotPath, delByDotPath, assertGate, applyOp, computeInputHash, computeMutHash, computeStateHash, computeTickHash, replayVerify, compileProjection } from './asx/replay-verifier.js';
export { MEMORY_HIERARCHY, FLASH_MODES, FLASH_ALLOWED_KEYS, FLASH_FORBIDDEN_KEYS, createFlashSnapshot, verifyFlashSnapshot, rebuildRamFromFlash, createDeltaFlash, verifyFlashChain } from './asx/flash-ram.js';

// MX2LM DirectWrite GPU Execution — model viewer + 7-brain topology (v3.5.0-WebX)
export { π_vecNorm, π_softmax, π_entropy, π_ngramProb, π_pmi, π_angleFromVec, π_clamp, π_classToColor, isoCoords, buildOrbitalHaloData, buildStackGridData, buildTunnelStreamData, buildFractalTreeData, buildHudRingData, buildModelShells, buildMemoryConstellation, buildMicronautMesh } from './mx2lm/model-viewer.js';
export { BRAIN_ORGANS, BRAIN_ORGAN_COUNT, SHELL_TO_ORGAN, ORGAN_TO_SHELL, COMPUTE_UNIT_TO_ORGAN, POLYGON_KERNEL_MAP, getBrainOrgan, getOrganByShell, getMissingOrgans, getImplementedOrgans, brainCoherence } from './mx2lm/brain-topology.js';

// .brain binary format — tiny.x graph neural network substrate (v1.0.0-PowerShell-LLM)
export { readBrainHeader, writeBrainHeader, readBrainFeatures, readBrainTopology, readBrainRouting, buildBrainManifest, kExpertsFor, BRAIN_HEADER_BYTES, BRAIN_FEAT_DIM, BRAIN_FIELD } from './brain/brain-format.js';

// SMCA — Structural Manifold Cluster Architecture axioms (v1.0.0-PowerShell-LLM)
export { SMCA_LAYERS, SMCA_AUTHORITY_GRADIENT, SMCA_CLUSTER_ROLES, SMCA_COLLAPSE_CLASSES, IDB_PROPERTIES, KXC_FORBID_LIST, SMCA_VERSION } from './smca/index.js';

// KXC IR format — kernel descriptor builder/validator (v1.0.0-PowerShell-LLM)
export { createKernelIR, validateKernelIR, computeStackCid, KXC_CAPABILITY_FLAGS, KXC_STACK_ID, KXC_IR_VERSION } from './kxc/ir-format.js';

// CM-1 control gate dictionary (v1.0.0-PowerShell-LLM)
export { CM1_BYTES, CM1_BYTE_NAMES, CM1_ID, CM1_LANE, encodeCM1Frame, decodeCM1Frame, recordSeparate } from './cm1/dict.js';

// Micronaut Code Review Engine (micronaut-coder — multi-language, security + perf rules)
export { CodeReviewEngine, detectLanguage, LANGUAGE_PATTERNS, SECURITY_RULES, PERFORMANCE_RULES } from './micronaut/coder-engine.js';

// Micronaut Factory Core (micronaut-factory — .micronaut dir scanner, authority-based registry)
export { MicronauntFactory, createMicronaut, defaultPersonality, getFactory } from './micronaut/factory-core.js';

// Micronaut Core spec v3.0.0 (cognitive state, memory system, required methods)
export { MicronauntCore, validateMicronaunt, MICRONAUT_CORE_VERSION, COGNITIVE_DEFAULTS, REQUIRED_METHODS } from './micronaut/micronaut-core.js';

// Micronaut factory — authority-based instantiation contract (v0.1.0-micronaut-factory)
export { FACTORY_POLICY, FACTORY_VERSION, MICRONAUT_STATUS, createRegistryEntry, MicronanutRegistry, createMutationRecord, DEFAULT_REGISTRY } from './micronaut/factory.js';

// SCXT tensor format + SCXTOK tokenizer + U1 unary alphabet (SCX.v1.0.0)
export { readScxtHeader, writeScxtHeader, hashScxtTensor, verifyScxtBuffer, calcDataSize, calcStrides, SCXT_MAGIC, SCXT_VERSION, SCXT_HEADER_SIZE, SCXT_HASH_SIZE, SCX_DTYPE, SCX_DTYPE_BYTES, SCXT_STRIDE_MODE, Q4_BLOCK_ELEMENTS, Q4_SCALE_BYTES } from './scx/tensor-format.js';
export { SCXTokenizer, readScxtokHeader, SCXTOK_MAGIC, SCX_BYTE_OFFSET, SCX_BYTE_COUNT, SCX_SPECIAL_OFFSET, SCXTOK_VOCAB_TYPE, SCXTOK_SPECIAL } from './scx/tokenizer.js';
export { U1_GLYPHS, U1_BY_CODEPOINT, U1_FOLD_LATTICE, U1_VERSION, U1_ALPHABET_SHA256, expandU1Capsule, u1HashInput, parseGlyphName } from './scx/u1-alphabet.js';

// SCX IR, graph, manifest, KV-delta (SCXRuntime.v1.0.0)
export { SCXOp, SCX_OP_NAMES, createOperand, createInstruction, createProgram, programToJSON } from './scx/ir.js';
export { SCX_NODE_ROLES, SCX_EDGE_TYPES, SCX_QUANT, SCX_DEVICE, createScxNode, createScxEdge, createScxGraph, createGta1Tensor, createGta1Node, createGta1Edge } from './scx/graph.js';
export { MANIFEST_KIND, SCX_MOE_16L_CONFIG, createManifestTensor, createManifestInfo, parseModelManifest, parseScoIndex, validateScoIndex } from './scx/manifest.js';
export { kvDeltaEncode, kvDeltaDecode, packNibs, unpackNibs, KVCacheDelta } from './scx/kv-delta.js';

// K'UHUL compiler + KSON format (KUHUL.v1.0.0)
export { compileKUHUL, KuhulLexer, KuhulParser, KuhulSemanticAnalyzer, KSONGenerator, KuhulSyntaxError, KuhulSemanticError, KUHUL_GLYPHS, KUHUL_DTYPES, KUHUL_OP_MAP, KSON_PHASES, KUHUL_SHADER_ROLES, KUHUL_RUNTIME_CONTRACT } from './kuhul/kuhulc.js';

// Swarm runtime — PhaseArray routing + SwarmManager (KUHUL.EXE.v3.0.0)
export { PhaseArray, createSkill, createAgent, createSwarm, computeSwarmCoherence, SwarmManager, PHASE_ARRAY_SIZE, SWARM_STRATEGIES, SWARM_API_ROUTES } from './micronaut/swarm.js';

// Hive Micronaut Atlas — 85-agent canonical registry (v3.5.0-WebX)
export { HIVE_MICRONAUTS, UNIFIED_SERVICES, HIVE_SKILL_PACKS, HIVE_AGENT_COUNT, HIVE_FOLDS, getHiveMicronaut, getHiveService, getHiveMicronautsByFold, getHiveMicronautsByExpert, buildHiveAtlas } from './micronaut/hive-registry.js';

// Agentic Micronaut skill — fleet agents, XCFE ops, super alias routes (v3.5.0-WebX)
export { AGENTIC_MICRONAUT_ACTIONS, FLEET_AGENTS, MICRONAUT_XCFE_OPS, SUPER_ALIAS_ROUTES, getFleetAgent, getXcfeOp } from './micronaut/agentic-skill.js';

// SCX Control Delta + LMStudio binding types (v3.3.0-scx-control-flow)
export { createDelta, validateDelta, stableJson, djb2Hex, SURFACE_FOLDS, SURFACE_EFFECTS, COMPOUND_LOOP, CONTROL_TRANSITIONS } from './control/delta.js';
export { lmStudioDeltaParams, createToolBinding, createAgentBinding, createSkillBinding, createCommandBinding, createChatActBinding, LMSTUDIO_SDK_VERSION, LMSTUDIO_SURFACE_BY_KIND } from './control/lmstudio.js';

// Supernaut lifecycle + dispatch — monotonic 7-state, Mayan fold (v3.2.0-supernaut)
export { LifecycleState, LIFECYCLE_ORDER, lifecycleIndex, EXPERT_MAP, ADDON_EXPERT, createExpertRoute, DispatchContext, runDispatch, archiveContext } from './supernaut/lifecycle.js';
export { SkillRunResult, RoundContext, mayanFold, foldReport, parseSigilActions, SIGIL_ACTIONS, SIGIL_TRANSITIONS, STOCK_SKILLS } from './supernaut/dispatch.js';

// Supernaut skill route manifest — 130+ routes across all action namespaces (v3.5.0-WebX)
export { SUPERNAUT_MANIFEST, SUPERNAUT_ROUTES, SUPERNAUT_ROUTE_COUNT, SUPERNAUT_MERGED_SKILLS, SUPERNAUT_SKILL_MATRIX, resolveRoute, getRoutesByActionNs, getRoutesByIntent } from './supernaut/routes.js';

// SCX-MoE top-K router + SwiGLU expert forward (v3.1.0-scx-moe)
export { routeTopK, routeToken, MOE_NUM_EXPERTS, MOE_NUM_LAYERS, MOE_HIDDEN_SIZE, MOE_INTERMEDIATE, MOE_NUM_HEADS, MOE_HEAD_DIM, MOE_MAX_SEQ, MOE_VOCAB_SIZE, MOE_TOP_K, ROUTER_PARAMS_SCHEMA } from './moe/router.js';
export { expertForward, expertForwardBatch, expertReduce, AMPLIFY_EXPERT_ID, EXPERT_PARAMS_SCHEMA } from './moe/expert.js';

// Micronaut expert registry + SCXQ2 ISA + coding DAG (v3.0.0-agentic-micronaut)
export { SCXQ2_OPCODES, SCXQ2_DOMAINS, SCXQ2_LANES, SCXQ2_FLAGS, SCXQ2_INSTR_BYTES, MICRONAUT_EXPERTS, EXPERT_COUNT, ADAPTATION_SHADERS, FOLD_BARRIER_LAWS, RECOGNITION_RULE, getExpertById, getExpertByName } from './micronaut/expert-registry.js';
export { CORE_MICRONAUTS, CORE_MICRONAUT_COUNT, getMicronaught, CODING_DAG_NODES, DAG_ENTRY_NODE, DAG_TERMINAL_NODES, getDagNode, CODING_RECORD_TYPES, CODING_ROLES, createCodingRecord, CORE_FOLDS, FOLD_TYPES } from './micronaut/coding-dag.js';

// Mayan Math — base-20 glyph system, Long Count calendar, AtomicMayan ops
export { MAYAN_GLYPHS, MAYAN_GLYPH_BY_VALUE, LONG_COUNT_POSITIONS, MAYAN_JD_CORRELATION, bigIntToDigits, digitsToBase, glyphToValue, valueToGlyph, mayanGlyphToBigInt, bigIntToMayanGlyph, mayanAdd, mayanSub, mayanMul, mayanToJD, jdToMayan, AtomicMayan, ATOMIC_MATH_GRAMMAR } from './mayan/mayan-math.js';

// Linear Algebra — Float64Array vectors/matrices/tensors, attention, eigenvalue
export { vectorAdd, vectorSub, vectorScale, dotProduct, vectorNorm, vectorNormalize, matMul, matTranspose, matVecMul, matIdentity, matScale, matInverse, powerIteration, softmaxInPlace, batchMatMul, tensorSoftmax, scaledDotAttention, tensorContract, matFromArray, matToArray } from './linalg/linalg.js';

// Mayan-Linear Algebra Hybrid — calendar regression, linear transforms, OLS
export { mayanToVector, vectorToMayanDigits, mayanDigitsToScalar, scalarToMayanDigits, mayanLinearTransform, mayanLinearSolve, mayanLinearRegression, predictMayanDays, buildCalendarRegression, IDENTITY_TRANSFORM, mayanAffineShift, calendarRoundAlignment } from './linalg/mayan-linalg.js';

// Mayan Tensor — CP/Tucker decomposition (ALS, HOSVD+HOOI), MayanTensor class
export { MayanTensor, cpDecompose, reconstructCP, tuckerDecompose, tuckerReconstruct, tensorUnfold, khatriRao, tensorSize, strides } from './mayan/mayan-tensor.js';

// Mayan Crypto — GL(20,n) ring ops, GL20N key/encrypt/sign/ZKP, mayanHash
export { mod, gcd, modInverse, matMulMod, matDetMod, matInverseMod, isInvertibleMod, GL20N, mayanHash, createMayanGL } from './mayan/mayan-crypto.js';

// Intel HD4600 Gen7.5 optimization constants + Morton Z-curve tiling
export { HD4600, MORTON_CONFIG, MORTON_TABLE, mortonEncode, mortonDecode, buildMortonTable, hd4600DispatchSize, slmFloat32Capacity, HD4600_CBUFFER } from './gpu/hd4600.js';

// VRAM Compression — delta (Mayan digit streams, ~2.1x) + DCT spectral (~2.8x)
export { deltaCompress, deltaDecompress, dctCompress, dctDecompress, ZIGZAG, estimateDeltaRatio, estimateDctRatio } from './gpu/vram-compression.js';

// Power Scheduler — P0-P3/RC6 states, thermal rules, adaptive dispatch config
export { POWER_STATES, POWER_STATE_ORDER, MAYAN_PRECISION_BY_STATE, THERMAL_THRESHOLDS, selectPowerState, batchSizeForState, thermalAction, shouldOffloadToCPU, dispatchConfig } from './gpu/power-scheduler.js';

// WebGPU Fallback Manager — 4-tier fallback, LRU VRAM eviction, OOM handling
export { BACKEND, MIN_WEBGPU_BUFFER_BYTES, SAB_TOTAL_BYTES, detectWebGPU, detectWasmSimd, detectSAB, WebGPUFallbackManager } from './gpu/webgpu-runtime.js';

// Mayan Atomic Ops v0.1 — unified base-20 atomic semantics (SAB + CPU backends)
export { AtomicMayanOpsSAB, AtomicMayanOpsCPU, createAtomicMayanOps, initMayanBuffer, readMayanBuffer } from './gpu/mayan-atomic-ops.js';

// AdaptiveHardwareRuntime v0.1 — GPU/WebGPU/WASM/CPU unified executor
export { AdaptiveHardwareRuntime, createExecContext, selectBackend, getAdaptiveRuntime } from './gpu/adaptive-hardware-runtime.js';

// KXML v7.2 — bidirectional geometric graph runtime
// Nodes = computation units, Edges = geodesic transport (forward+backward channels),
// Phase gates = soft landing guarantees, MathML = Lipschitz-bounded semantics
export { parseKXML }                        from './kxml/kxml-parser.js';
export { PhaseGatedDispatcher, PHASE_ORDER as KXML_PHASE_ORDER } from './kxml/kxml-dispatcher.js';
export { KXMLGraph }                        from './kxml/kxml-graph.js';
export {
  OPS as KXML_OPS, dispatchOp as kxmlDispatchOp,
  geodesicDist, parallelTransport, ricciFlowStep,
  geometricAttention, foldCompress, crossEntropy,
} from './kxml/kxml-ops.js';
export {
  ShardRegistry, PHASE_RESIDENCY, KXML_FOLD_TO_SCXQ2,
  SCXQ2_OPCODES as KXML_SCXQ2_OPCODES, SCXQ2_DOMAIN, SCXQ2_FOLD_ID,
} from './kxml/kxml-shard-registry.js';

// Agents.NET shared-state, @op dispatch, SyncWorker (Agents.NET.v1.0.0)
export { readSharedState, writeSharedState, createSharedState, SHARED_STATE_VERSION, SHARED_STATE_BYTES, SHARED_STATE_MMF_NAME, SHARED_STATE_OFFSETS } from './agents-net/shared-state.js';
export { validateOp, dispatchOp, DOTNET_OPS, DOTNET_OP_SCHEMAS, DOTNET_WORKER_URL_DEFAULT } from './agents-net/op-dispatcher.js';
export { SyncWorker, SECURITY_PLUGIN_OPS } from './agents-net/sync-worker.js';

// PGGTF — Phase-Gated Geodesic Tensor Field v0.1
// System: Omega = (P, G, F, M, Pi)  evolution: Omega_{t+1} = M(P, G, F, Pi)
// Inference law: predict P(F_{t+1}|F_t,G,P) — next field state, not next token
export { PGGTF }                                           from './pggtf/pggtf.js';
export { PhaseTensor, PHASE_INDEX as PGGTF_PHASE_INDEX, PHASE_NAMES, N_PHASES } from './pggtf/phase-tensor.js';
export { GeodesicTensor }                                  from './pggtf/geodesic-tensor.js';
export { FoldTensor, SCXQ2_LANES as PGGTF_LANES }          from './pggtf/fold-tensor.js';
export { PiPhaseTensor }                                   from './pggtf/pi-tensor.js';
export {
  memoryMicronaut, routingMicronaut,
  compressionMicronaut, inferenceMicronaut,
  applyOperator as applyMicronautOp,
} from './pggtf/micronaut-ops.js';

// MX2LM Object — SafeTensors -> Tensor Registry -> Fold Objects -> MX2LM Cognitive Object
// Object = (Glyphs, Folds, Graph, Memory, State, TensorRefs) where TensorRefs -> SafeTensors
export { TensorManifest, TensorRegistry, TensorGraph,
         FoldObject, FoldStore, MX2LMObject } from './mx2lm/tensor-manifest.js';
