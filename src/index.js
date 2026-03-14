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
