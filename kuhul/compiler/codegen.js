/**
 * @fileoverview Base code-generator class.
 *
 * Concrete code generators (JS, Wasm, WebGPU) extend this class and
 * override the `generate` method and optionally the per-opcode handlers.
 *
 * @module kuhul/compiler/codegen
 */

import { GeometricIR } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// BaseCodegen
// ------------------------------------------------------------------ //

/**
 * Base class for KUHUL code generators.
 *
 * Subclasses must implement `generate(ir)` which accepts a GeometricIR
 * and returns target source code as a string.
 */
export class BaseCodegen {
  /**
   * Generate target code from a GeometricIR program.
   *
   * @param {GeometricIR} ir
   * @returns {string}
   */
  generate(ir) {
    throw new Error(`${this.constructor.name}.generate() is not implemented`);
  }

  // ---------------------------------------------------------------- //
  // Shared utilities for subclasses
  // ---------------------------------------------------------------- //

  /**
   * Render an operand (string identifier or numeric/string literal) as
   * a valid target-language expression string.
   *
   * @param {*} operand
   * @returns {string}
   */
  _renderOperand(operand) {
    if (typeof operand === 'number') return String(operand);
    if (typeof operand === 'string') {
      if (operand.startsWith('"')) return operand; // already a string literal
      return operand; // identifier
    }
    return JSON.stringify(operand);
  }

  /**
   * Indent each line of a multi-line string.
   *
   * @param {string} code
   * @param {number} [spaces=2]
   * @returns {string}
   */
  _indent(code, spaces = 2) {
    const pad = ' '.repeat(spaces);
    return code.split('\n').map(l => pad + l).join('\n');
  }
}
