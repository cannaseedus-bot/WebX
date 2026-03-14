/**
 * @fileoverview Main KUHUL compiler – orchestrates the full pipeline.
 *
 * Pipeline stages:
 *  1. Lexer       → token stream
 *  2. Parser      → AST
 *  3. Semantic    → validated / annotated AST
 *  4. IRGenerator → GeometricIR
 *  5. IROptimizer → optimised GeometricIR
 *  6. IRVerifier  → validation report
 *  7. Codegen     → target source string
 *
 * @example
 * import { KuhulCompiler } from './kuhul-compiler.js';
 * const compiler = new KuhulCompiler();
 * const result   = await compiler.compile(source, 'js');
 * console.log(result.code);
 *
 * @module kuhul/compiler/kuhul-compiler
 */

import { KuhulLexer }        from './lexer.js';
import { KuhulParser }       from './parser.js';
import { SemanticAnalyzer }  from './semantic-analyzer.js';
import { IRGenerator }       from './ir-generator.js';
import { IROptimizer }       from '../ir/ir-optimizer.js';
import { IRVerifier }        from '../ir/ir-verifier.js';
import { JavaScriptCodegen } from './codegen/js-codegen.js';
import { WasmCodegen }       from './codegen/wasm-codegen.js';
import { WebGPUCodegen }     from './codegen/webgpu-codegen.js';
import { ExecutableProgram } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// Target → codegen map
// ------------------------------------------------------------------ //

const CODEGENS = {
  js:     () => new JavaScriptCodegen(),
  wasm:   () => new WasmCodegen(),
  webgpu: () => new WebGPUCodegen(),
};

// ------------------------------------------------------------------ //
// CompileError
// ------------------------------------------------------------------ //

/** Thrown when any compiler stage fails. */
export class CompileError extends Error {
  /**
   * @param {string}   stage  - Pipeline stage name
   * @param {string[]} errors - Detailed error messages
   */
  constructor(stage, errors) {
    super(`[${stage}] ${errors[0] ?? 'unknown error'}`);
    this.name   = 'CompileError';
    this.stage  = stage;
    this.errors = errors;
  }
}

// ------------------------------------------------------------------ //
// KuhulCompiler
// ------------------------------------------------------------------ //

/** Full KUHUL compilation pipeline. */
export class KuhulCompiler {
  /**
   * Compile KUHUL source to a target language.
   *
   * @param {string} source          - KUHUL source text
   * @param {'js'|'wasm'|'webgpu'} [target='js'] - Target language
   * @returns {Promise<ExecutableProgram>}
   */
  async compile(source, target = 'js') {
    // 1. Lex
    const lexer  = new KuhulLexer();
    const tokens = lexer.lex(source);

    // 2. Parse
    const parser = new KuhulParser();
    const ast    = parser.parse(tokens);

    // 3. Semantic analysis
    const analyzer = new SemanticAnalyzer();
    const { errors: semErrors } = analyzer.analyze(ast);
    if (semErrors.length > 0) {
      throw new CompileError('semantic', semErrors.map(e => e.toString()));
    }

    // 4. IR generation
    const irGen = new IRGenerator();
    const ir    = irGen.generate(ast);

    // 5. IR optimisation
    const optimizer = new IROptimizer();
    const optIR     = optimizer.optimize(ir);

    // 6. IR verification
    const verifier = new IRVerifier();
    const { valid, errors: verErrors } = verifier.verify(optIR);
    if (!valid) {
      throw new CompileError('ir-verifier', verErrors);
    }

    // 7. Code generation
    const codegenFactory = CODEGENS[target];
    if (!codegenFactory) {
      throw new CompileError('codegen', [`Unknown target "${target}". Valid targets: ${Object.keys(CODEGENS).join(', ')}`]);
    }
    const codegen = codegenFactory();
    const code    = codegen.generate(optIR);

    return new ExecutableProgram(optIR, target, code);
  }
}
