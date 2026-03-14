/**
 * @fileoverview Main entry point for the @webx/kuhul package.
 *
 * Re-exports all public classes and constants from the compiler, IR,
 * runtime, stdlib, and LLM sub-modules.
 *
 * @module kuhul
 */

// ------------------------------------------------------------------ //
// Compiler
// ------------------------------------------------------------------ //

export { KuhulCompiler, CompileError } from './compiler/kuhul-compiler.js';
export { KuhulLexer, TokenType, KEYWORDS, GLYPHS, LexerError } from './compiler/lexer.js';
export { KuhulParser, NodeKind, ParseError } from './compiler/parser.js';
export { SemanticAnalyzer, SemanticError } from './compiler/semantic-analyzer.js';
export { IRGenerator } from './compiler/ir-generator.js';
export { BaseCodegen } from './compiler/codegen.js';
export { JavaScriptCodegen } from './compiler/codegen/js-codegen.js';
export { WasmCodegen } from './compiler/codegen/wasm-codegen.js';
export { WebGPUCodegen } from './compiler/codegen/webgpu-codegen.js';

// ------------------------------------------------------------------ //
// IR
// ------------------------------------------------------------------ //

export { IRBuilder } from './ir/ir-builder.js';
export { IROptimizer } from './ir/ir-optimizer.js';
export { IRVerifier } from './ir/ir-verifier.js';
export { IRPrinter } from './ir/ir-printer.js';
export { TensorType, Phase, Instruction, GeometricIR, ExecutableProgram } from './ir/ir-types.js';

// ------------------------------------------------------------------ //
// Runtime
// ------------------------------------------------------------------ //

export { KuhulVM } from './runtime/kuhul-vm.js';
export { PhaseManager } from './runtime/phase-manager.js';
export { MemoryManager } from './runtime/memory-manager.js';
export { ExecutionEngine } from './runtime/execution-engine.js';
export { Scheduler } from './runtime/scheduler.js';

// ------------------------------------------------------------------ //
// Stdlib
// ------------------------------------------------------------------ //

export {
  GLYPHS as STDLIB_GLYPHS,
  GLYPH_TENSOR_PRODUCT,
  GLYPH_ADDITION,
  GLYPH_SUBTRACTION,
  GLYPH_CONVOLUTION,
  GLYPH_EQUALITY,
  GLYPH_NEGATION,
  GLYPH_DIRECT_SUM,
  executeGlyph,
  tensorProduct,
  addition,
  subtraction,
  convolution,
  equality,
  negation,
  directSum,
} from './stdlib/glyphs.js';

export { Tensor } from './stdlib/tensor.js';
export * as KuhulMath from './stdlib/math.js';
export { KuhulIO } from './stdlib/io.js';

// ------------------------------------------------------------------ //
// LLM
// ------------------------------------------------------------------ //

export { SyntaxCompleter } from './llm/syntax-completer.js';
export { GlyphValidator } from './llm/glyph-validator.js';
export { LLMOptimizer } from './llm/optimizer.js';
export { DocGenerator } from './llm/doc-generator.js';
export { DiagnosticEngine } from './llm/diagnostic-engine.js';

// ------------------------------------------------------------------ //
// Grammar
// ------------------------------------------------------------------ //

export { parseEBNF } from './grammar/ebnf-parser.js';
export { validateGrammar } from './grammar/grammar-validator.js';

// ------------------------------------------------------------------ //
// Tools
// ------------------------------------------------------------------ //

export { KuhulREPL } from './tools/repl.js';
export { KuhulDebugger } from './tools/debugger.js';
export { KuhulProfiler } from './tools/profiler.js';
export { KuhulLinter } from './tools/linter.js';
export { KuhulFormatter } from './tools/formatter.js';
