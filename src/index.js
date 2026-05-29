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

// XVM CPU Cluster (v0.1.0-xvm-cpu-thread-cluster)
export { default as CPUCluster32, FLAG as XVM_FLAG, OP as XVM_OP, CLUSTER_SIZE_CONST, PHASE_COUNT_CONST, XVM_TRAINING_OPTIMUM } from './xvm/cpu-cluster.js';

// XVM 1000-fiber CUDA emulator (.gpu_trainer)
export { default as FiberPool1000, OP_EXT as XVM_OP_EXT, WARP_SIZE, DEFAULT_CONFIG as FIBER1000_CONFIG } from './xvm/fiber-1000.js';

// SMGM-16 model descriptor and runtime (v0.1.0-xvm-cpu-thread-cluster)
export { default as SMGM16Runtime, SMGM16_CONFIG, LOSS_LAMBDAS, SHARD_MAP, PARAM_COUNTS, tokenFeatures, cardSlotShape } from './smgm16/index.js';

// KLSL compiler — K'UHUL Language Shading Language (.gpu_trainer)
export { default as klslCompile, KLSL_OP, GLYPHS as KLSL_GLYPHS } from './klsl/index.js';

// XCFE JSON Program Format (.gpu_trainer)
export { default as XCFERuntime, XCFEProgram, parseMicronauts } from './xcfe/index.js';

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

// .brain binary format — tiny.x graph neural network substrate (v1.0.0-PowerShell-LLM)
export { readBrainHeader, writeBrainHeader, readBrainFeatures, readBrainTopology, readBrainRouting, buildBrainManifest, kExpertsFor, BRAIN_HEADER_BYTES, BRAIN_FEAT_DIM, BRAIN_FIELD } from './brain/brain-format.js';

// SMCA — Structural Manifold Cluster Architecture axioms (v1.0.0-PowerShell-LLM)
export { SMCA_LAYERS, SMCA_AUTHORITY_GRADIENT, SMCA_CLUSTER_ROLES, SMCA_COLLAPSE_CLASSES, IDB_PROPERTIES, KXC_FORBID_LIST, SMCA_VERSION } from './smca/index.js';

// KXC IR format — kernel descriptor builder/validator (v1.0.0-PowerShell-LLM)
export { createKernelIR, validateKernelIR, computeStackCid, KXC_CAPABILITY_FLAGS, KXC_STACK_ID, KXC_IR_VERSION } from './kxc/ir-format.js';

// CM-1 control gate dictionary (v1.0.0-PowerShell-LLM)
export { CM1_BYTES, CM1_BYTE_NAMES, CM1_ID, CM1_LANE, encodeCM1Frame, decodeCM1Frame, recordSeparate } from './cm1/dict.js';

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

// SCX-MoE top-K router + SwiGLU expert forward (v3.1.0-scx-moe)
export { routeTopK, routeToken, MOE_NUM_EXPERTS, MOE_NUM_LAYERS, MOE_HIDDEN_SIZE, MOE_INTERMEDIATE, MOE_NUM_HEADS, MOE_HEAD_DIM, MOE_MAX_SEQ, MOE_VOCAB_SIZE, MOE_TOP_K, ROUTER_PARAMS_SCHEMA } from './moe/router.js';
export { expertForward, expertForwardBatch, expertReduce, AMPLIFY_EXPERT_ID, EXPERT_PARAMS_SCHEMA } from './moe/expert.js';

// Micronaut expert registry + SCXQ2 ISA + coding DAG (v3.0.0-agentic-micronaut)
export { SCXQ2_OPCODES, SCXQ2_DOMAINS, SCXQ2_LANES, SCXQ2_FLAGS, SCXQ2_INSTR_BYTES, MICRONAUT_EXPERTS, EXPERT_COUNT, ADAPTATION_SHADERS, FOLD_BARRIER_LAWS, RECOGNITION_RULE, getExpertById, getExpertByName } from './micronaut/expert-registry.js';
export { CORE_MICRONAUTS, CORE_MICRONAUT_COUNT, getMicronaught, CODING_DAG_NODES, DAG_ENTRY_NODE, DAG_TERMINAL_NODES, getDagNode, CODING_RECORD_TYPES, CODING_ROLES, createCodingRecord, CORE_FOLDS, FOLD_TYPES } from './micronaut/coding-dag.js';

// Agents.NET shared-state, @op dispatch, SyncWorker (Agents.NET.v1.0.0)
export { readSharedState, writeSharedState, createSharedState, SHARED_STATE_VERSION, SHARED_STATE_BYTES, SHARED_STATE_MMF_NAME, SHARED_STATE_OFFSETS } from './agents-net/shared-state.js';
export { validateOp, dispatchOp, DOTNET_OPS, DOTNET_OP_SCHEMAS, DOTNET_WORKER_URL_DEFAULT } from './agents-net/op-dispatcher.js';
export { SyncWorker, SECURITY_PLUGIN_OPS } from './agents-net/sync-worker.js';
