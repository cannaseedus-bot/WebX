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
export { default as CPUCluster32, FLAG as XVM_FLAG, OP as XVM_OP, CLUSTER_SIZE_CONST, PHASE_COUNT_CONST } from './xvm/cpu-cluster.js';

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
