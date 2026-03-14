/**
 * @fileoverview KUHUL semantic analyser.
 *
 * Validates an AST produced by KuhulParser:
 *  - Identifiers are declared (via [Wo]) before they are read/written.
 *  - Glyph symbols used in [Sek] are from the recognised KUHUL set.
 *  - Tensor element types are valid.
 *  - Folds and phase cycles introduce nested scopes.
 *
 * Usage:
 * ```js
 * const { SemanticAnalyzer } = await import('./semantic-analyzer.js');
 * const analyzer = new SemanticAnalyzer();
 * const { errors, warnings } = analyzer.analyze(ast);
 * ```
 *
 * @module kuhul/compiler/semantic-analyzer
 */

import { NodeKind } from './parser.js';
import { GLYPHS }   from './lexer.js';

// ------------------------------------------------------------------ //
// SemanticError
// ------------------------------------------------------------------ //

/** Represents a semantic error in a KUHUL program. */
export class SemanticError {
  /**
   * @param {string} message
   * @param {object|null} astNode - The AST node where the error occurred
   */
  constructor(message, astNode = null) {
    this.message = message;
    this.node    = astNode;
  }

  toString() {
    const loc = this.node ? ` (line ${this.node.line ?? '?'})` : '';
    return `SemanticError${loc} — ${this.message}`;
  }
}

// ------------------------------------------------------------------ //
// Symbol scope
// ------------------------------------------------------------------ //

class Scope {
  /** @param {Scope|null} parent */
  constructor(parent = null) {
    this.parent  = parent;
    /** @type {Map<string, { kind: string, node: object }>} */
    this.symbols = new Map();
  }

  /**
   * Declare a symbol in this scope.
   * @param {string} name
   * @param {{ kind: string, node: object }} info
   */
  define(name, info) { this.symbols.set(name, info); }

  /**
   * Look up a symbol, walking parent scopes.
   * @param {string} name
   * @returns {{ kind: string, node: object }|null}
   */
  lookup(name) {
    if (this.symbols.has(name)) return this.symbols.get(name);
    return this.parent ? this.parent.lookup(name) : null;
  }

  /** Create a child scope. */
  child() { return new Scope(this); }
}

// ------------------------------------------------------------------ //
// Valid element types
// ------------------------------------------------------------------ //

const VALID_ELEMENT_TYPES = new Set([
  'float32', 'float64', 'int32', 'int64', 'uint8', 'bool', 'complex64',
]);

// ------------------------------------------------------------------ //
// SemanticAnalyzer
// ------------------------------------------------------------------ //

/** Performs semantic analysis on a KUHUL AST. */
export class SemanticAnalyzer {
  /**
   * Analyse the given AST and return errors and warnings.
   *
   * @param {{ kind: 'Program', body: object[] }} ast
   * @returns {{ errors: SemanticError[], warnings: string[] }}
   */
  analyze(ast) {
    this._errors   = [];
    this._warnings = [];

    const globalScope = new Scope();
    this._visitProgram(ast, globalScope);

    return { errors: this._errors, warnings: this._warnings };
  }

  // ---------------------------------------------------------------- //
  // Visitor helpers
  // ---------------------------------------------------------------- //

  /** @param {string} message @param {object|null} node */
  _error(message, node = null) {
    this._errors.push(new SemanticError(message, node));
  }

  /** @param {string} message */
  _warn(message) {
    this._warnings.push(message);
  }

  // ---------------------------------------------------------------- //
  // Program & body
  // ---------------------------------------------------------------- //

  _visitProgram(ast, scope) {
    if (ast.kind !== NodeKind.Program) {
      this._error('Root node must be a Program');
      return;
    }
    for (const stmt of ast.body) {
      this._visitStatement(stmt, scope);
    }
  }

  /**
   * Dispatch to the appropriate visit method.
   * @param {object} stmt
   * @param {Scope} scope
   */
  _visitStatement(stmt, scope) {
    switch (stmt.kind) {
      case NodeKind.Fold:        return this._visitFold(stmt, scope);
      case NodeKind.Allocation:  return this._visitAllocation(stmt, scope);
      case NodeKind.Read:        return this._visitRead(stmt, scope);
      case NodeKind.Write:       return this._visitWrite(stmt, scope);
      case NodeKind.Operation:   return this._visitOperation(stmt, scope);
      case NodeKind.PhaseCycle:  return this._visitPhaseCycle(stmt, scope);
      case NodeKind.Invocation:  return this._visitInvocation(stmt, scope);
      default:
        this._warn(`Unknown statement kind "${stmt.kind}" – skipping`);
    }
  }

  // ---------------------------------------------------------------- //
  // Fold
  // ---------------------------------------------------------------- //

  _visitFold(node, scope) {
    const inner = scope.child();
    for (const stmt of node.body) {
      this._visitStatement(stmt, inner);
    }
  }

  // ---------------------------------------------------------------- //
  // Allocation  [Wo X tensor<float32, [10]>]
  // ---------------------------------------------------------------- //

  _visitAllocation(node, scope) {
    const name = node.identifier.name;

    if (scope.lookup(name)) {
      this._warn(`"${name}" is already declared; shadowing previous declaration`);
    }

    // Validate element type
    const { elementType, shape } = node.tensorType;
    if (!VALID_ELEMENT_TYPES.has(elementType)) {
      this._error(`Unknown tensor element type "${elementType}"`, node);
    }

    // Shape dims must be positive integers or '?'
    for (const dim of shape) {
      if (dim !== '?' && (!Number.isInteger(dim) || dim <= 0)) {
        this._error(`Tensor dimension must be a positive integer or "?" but got "${dim}"`, node);
      }
    }

    scope.define(name, { kind: 'tensor', elementType, shape, node });
  }

  // ---------------------------------------------------------------- //
  // Read  [Yax X]
  // ---------------------------------------------------------------- //

  _visitRead(node, scope) {
    const name = node.identifier.name;
    if (!scope.lookup(name)) {
      this._error(`Identifier "${name}" is used before declaration`, node);
    }
  }

  // ---------------------------------------------------------------- //
  // Write  [Ch'en X value]
  // ---------------------------------------------------------------- //

  _visitWrite(node, scope) {
    const name = node.identifier.name;
    if (!scope.lookup(name)) {
      // Implicitly declare the variable so subsequent reads succeed.
      // We also warn the user to prefer an explicit [Wo] declaration.
      this._warn(`Writing to undeclared identifier "${name}" – consider using [Wo] first`);
      scope.define(name, { kind: 'implicit', node });
    }
    this._visitValue(node.value, scope);
  }

  // ---------------------------------------------------------------- //
  // Operation  [Sek ⊗ X W]
  // ---------------------------------------------------------------- //

  _visitOperation(node, scope) {
    if (!GLYPHS.has(node.glyph)) {
      this._error(`Unknown glyph operator "${node.glyph}"`, node);
    }
    for (const operand of node.operands) {
      this._visitValue(operand, scope);
    }
  }

  // ---------------------------------------------------------------- //
  // Phase cycle  [K'ayab'] ... [Kumk'u]
  // ---------------------------------------------------------------- //

  _visitPhaseCycle(node, scope) {
    const inner = scope.child();
    for (const stmt of node.body) {
      this._visitStatement(stmt, inner);
    }
  }

  // ---------------------------------------------------------------- //
  // Invocation  [Muwan fn arg*]
  // ---------------------------------------------------------------- //

  _visitInvocation(node, scope) {
    // The invoked identifier doesn't need to be a tensor; it's a function ref
    for (const arg of node.args) {
      this._visitValue(arg, scope);
    }
  }

  // ---------------------------------------------------------------- //
  // Value
  // ---------------------------------------------------------------- //

  /**
   * Validate a value node.
   * @param {object} node
   * @param {Scope} scope
   */
  _visitValue(node, scope) {
    if (!node) return;
    if (node.kind === NodeKind.Identifier) {
      const name = node.name;
      if (!scope.lookup(name)) {
        this._error(`Identifier "${name}" is used before declaration`, node);
      }
    }
    // NumberLit and StringLit are always valid
  }
}
