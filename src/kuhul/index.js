// src/kuhul/index.js — K'UHUL language layer exports
//
// Hierarchy:
//   ngram-opcodes.js  — 147 ngrams: 67 opcodes + 80 syntax sugar
//   parser.js         — tokenize → expand sugar → parse tree
//   builder.js        — parse → ML detect → codegen → artifact

export { CORE_OPCODES, ML_OPCODES, DISTRIBUTED_OPCODES, XCFE_OPCODES,
         SYNTAX_SUGAR, ALL_NGRAMS, STATS } from './ngram-opcodes.js';
export { KuhulParser }  from './parser.js';
export { KuhulBuilder } from './builder.js';
export { FLAGS, DTYPE, ACTIVATION, OPTIMIZER, LOSS_TYPE, NORM_TYPE, ATTENTION_TYPE,
         CORE_BYTECODE, ML_BYTECODE, DIST_BYTECODE, XCFE_BYTECODE,
         TENSOR_HEADER, validateBytecode, BytecodeAssembler } from './bytecode.js';
export { MAYAN, LinearEntropyFold, GeodesicEntropyFold, PiPhase,
         MayanOrchestrator, coherenceMatrix } from './mayan-orchestrator.js';
export { FOLD_SHADERS, FoldOrchestrator } from './fold-shaders.js';
