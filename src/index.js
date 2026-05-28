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

// SMGM-16 model descriptor and runtime (v0.1.0-xvm-cpu-thread-cluster)
export { default as SMGM16Runtime, SMGM16_CONFIG, LOSS_LAMBDAS, SHARD_MAP, PARAM_COUNTS, tokenFeatures, cardSlotShape } from './smgm16/index.js';
