/**
 * @fileoverview IR generator – lowers a KUHUL AST to Geometric IR.
 *
 * Walks the annotated (semantic-checked) AST and emits Instruction objects
 * via IRBuilder.
 *
 * Opcode reference:
 *  ALLOC  name elementType shape[]
 *  READ   name
 *  WRITE  name valueRef
 *  OP     glyph ...operandRefs
 *  CALL   name ...args
 *  PHASE_START
 *  PHASE_END
 *  FOLD_START
 *  FOLD_END
 *
 * @module kuhul/compiler/ir-generator
 */

import { NodeKind } from './parser.js';
import { IRBuilder } from '../ir/ir-builder.js';
import { TensorType } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// IRGenerator
// ------------------------------------------------------------------ //

/** Generates Geometric IR from a KUHUL AST. */
export class IRGenerator {
  /**
   * Generate a GeometricIR from the given AST.
   *
   * @param {{ kind: 'Program', body: object[] }} ast
   * @returns {import('../ir/ir-types.js').GeometricIR}
   */
  generate(ast) {
    this._builder = new IRBuilder().createProgram();
    this._visitProgram(ast);
    return this._builder.build();
  }

  // ---------------------------------------------------------------- //
  // Helpers
  // ---------------------------------------------------------------- //

  /**
   * @param {string} opcode
   * @param {*[]}    operands
   * @param {object} [meta]
   */
  _emit(opcode, operands = [], meta = {}) {
    this._builder.addInstruction(opcode, operands, meta);
  }

  // ---------------------------------------------------------------- //
  // Visitors
  // ---------------------------------------------------------------- //

  _visitProgram(ast) {
    for (const stmt of ast.body) this._visitStatement(stmt);
  }

  _visitStatement(stmt) {
    switch (stmt.kind) {
      case NodeKind.Fold:        return this._visitFold(stmt);
      case NodeKind.Allocation:  return this._visitAllocation(stmt);
      case NodeKind.Read:        return this._visitRead(stmt);
      case NodeKind.Write:       return this._visitWrite(stmt);
      case NodeKind.Operation:   return this._visitOperation(stmt);
      case NodeKind.PhaseCycle:  return this._visitPhaseCycle(stmt);
      case NodeKind.Invocation:  return this._visitInvocation(stmt);
      default:
        throw new Error(`IRGenerator: unknown node kind "${stmt.kind}"`);
    }
  }

  _visitFold(node) {
    this._emit('FOLD_START', [], { line: node.line });
    for (const stmt of node.body) this._visitStatement(stmt);
    this._emit('FOLD_END', []);
  }

  _visitAllocation(node) {
    const { name }               = node.identifier;
    const { elementType, shape } = node.tensorType;
    const tt = new TensorType(elementType, shape);
    this._builder.declareSymbol(name, tt);
    this._emit('ALLOC', [name, elementType, shape], { line: node.line });
  }

  _visitRead(node) {
    this._emit('READ', [node.identifier.name], { line: node.line });
  }

  _visitWrite(node) {
    const valueRef = this._valueRef(node.value);
    this._emit('WRITE', [node.identifier.name, valueRef], { line: node.line });
  }

  _visitOperation(node) {
    const operandRefs = node.operands.map(o => this._valueRef(o));
    this._emit('OP', [node.glyph, ...operandRefs], { line: node.line });
  }

  _visitPhaseCycle(node) {
    this._emit('PHASE_START', [], { line: node.line });
    for (const stmt of node.body) this._visitStatement(stmt);
    this._emit('PHASE_END', []);
  }

  _visitInvocation(node) {
    const argRefs = node.args.map(a => this._valueRef(a));
    this._emit('CALL', [node.identifier.name, ...argRefs], { line: node.line });
  }

  /**
   * Convert an AST value node to a simple scalar reference (string or number).
   * @param {object} node
   * @returns {string|number}
   */
  _valueRef(node) {
    switch (node.kind) {
      case NodeKind.Identifier: return node.name;
      case NodeKind.NumberLit:  return node.value;
      case NodeKind.StringLit:  return `"${node.value}"`;
      default:                  return String(node.kind);
    }
  }
}
