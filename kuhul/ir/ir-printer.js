/**
 * @fileoverview IR printer – converts a GeometricIR to a human-readable string.
 *
 * @module kuhul/ir/ir-printer
 */

import { GeometricIR } from './ir-types.js';

// ------------------------------------------------------------------ //
// IRPrinter
// ------------------------------------------------------------------ //

/** Converts a GeometricIR object into a readable textual representation. */
export class IRPrinter {
  /**
   * Print the given IR as a string.
   *
   * @param {GeometricIR} ir
   * @returns {string}
   */
  print(ir) {
    if (!(ir instanceof GeometricIR)) {
      return '; <invalid IR>\n';
    }

    const lines = [];

    // Header
    lines.push('; === KUHUL Geometric IR ===');
    if (ir.metadata && Object.keys(ir.metadata).length > 0) {
      for (const [k, v] of Object.entries(ir.metadata)) {
        lines.push(`; meta: ${k} = ${JSON.stringify(v)}`);
      }
    }
    lines.push('');

    // Symbol table
    if (ir.symbolTable.size > 0) {
      lines.push('; --- Symbol Table ---');
      for (const [name, type] of ir.symbolTable) {
        lines.push(`; ${name}: ${type}`);
      }
      lines.push('');
    }

    // Instructions
    lines.push('; --- Instructions ---');
    let indent = 0;

    for (const instr of ir.instructions) {
      const { opcode, operands } = instr;

      // Decrease indent before closing delimiters
      if (opcode === 'FOLD_END' || opcode === 'PHASE_END') {
        indent = Math.max(0, indent - 2);
      }

      const pad  = ' '.repeat(indent);
      const ops  = operands.map(o => JSON.stringify(o)).join(', ');
      const meta = instr.metadata && Object.keys(instr.metadata).length > 0
        ? `  ; ${JSON.stringify(instr.metadata)}`
        : '';

      lines.push(`${pad}${opcode}(${ops})${meta}`);

      // Increase indent after opening delimiters
      if (opcode === 'FOLD_START' || opcode === 'PHASE_START') {
        indent += 2;
      }
    }

    lines.push('');
    lines.push('; === END ===');
    return lines.join('\n');
  }
}
