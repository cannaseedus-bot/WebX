/**
 * @fileoverview IR builder for the KUHUL Geometric IR.
 *
 * Provides a fluent API for constructing a GeometricIR object.
 *
 * @example
 * const builder = new IRBuilder();
 * builder
 *   .addInstruction('ALLOC', ['X', 'float32', [10]])
 *   .addInstruction('READ',  ['X'])
 *   .addInstruction('WRITE', ['Y']);
 * const ir = builder.build();
 *
 * @module kuhul/ir/ir-builder
 */

import { Instruction, GeometricIR, TensorType } from './ir-types.js';

// ------------------------------------------------------------------ //
// IRBuilder
// ------------------------------------------------------------------ //

/** Fluent builder for GeometricIR objects. */
export class IRBuilder {
  constructor() {
    this._instructions = [];
    this._symbolTable  = new Map();
    this._metadata     = {};
  }

  /**
   * Start building a new program (resets state).
   * @returns {IRBuilder}
   */
  createProgram() {
    this._instructions = [];
    this._symbolTable  = new Map();
    this._metadata     = {};
    return this;
  }

  /**
   * Add an instruction to the program.
   *
   * @param {string} opcode    - Instruction opcode
   * @param {*[]}    operands  - Operand list
   * @param {object} [metadata] - Optional metadata
   * @returns {IRBuilder}
   */
  addInstruction(opcode, operands = [], metadata = {}) {
    this._instructions.push(new Instruction(opcode, operands, metadata));
    return this;
  }

  /**
   * Register a tensor symbol in the symbol table.
   *
   * @param {string}     name
   * @param {TensorType} type
   * @returns {IRBuilder}
   */
  declareSymbol(name, type) {
    this._symbolTable.set(name, type);
    return this;
  }

  /**
   * Attach arbitrary metadata to the program (e.g. source file, version).
   *
   * @param {object} meta
   * @returns {IRBuilder}
   */
  setMetadata(meta) {
    Object.assign(this._metadata, meta);
    return this;
  }

  /**
   * Finalise and return the GeometricIR object.
   * @returns {GeometricIR}
   */
  build() {
    return new GeometricIR(
      [...this._instructions],
      new Map(this._symbolTable),
      { ...this._metadata },
    );
  }
}
