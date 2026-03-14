/**
 * @fileoverview IR optimizer for the KUHUL Geometric IR.
 *
 * Applies a series of optimisation passes over a GeometricIR object:
 *  1. Dead-store elimination – removes WRITE instructions whose target is
 *     never subsequently READ.
 *  2. Constant folding – replaces numeric literal operands in OP instructions
 *     with their computed value when all operands are numbers.
 *  3. Redundant READ elimination – removes consecutive READ instructions for
 *     the same identifier.
 *
 * @module kuhul/ir/ir-optimizer
 */

import { GeometricIR, Instruction } from './ir-types.js';

// ------------------------------------------------------------------ //
// IROptimizer
// ------------------------------------------------------------------ //

/** Optimises a GeometricIR program. */
export class IROptimizer {
  /**
   * Run all optimisation passes and return a new (optimised) GeometricIR.
   *
   * @param {GeometricIR} ir
   * @returns {GeometricIR}
   */
  optimize(ir) {
    let instrs = [...ir.instructions];
    instrs = this._eliminateRedundantReads(instrs);
    instrs = this._eliminateDeadStores(instrs);
    instrs = this._foldConstants(instrs);
    return new GeometricIR(instrs, new Map(ir.symbolTable), { ...ir.metadata, optimized: true });
  }

  // ---------------------------------------------------------------- //
  // Pass 1 – redundant READ elimination
  // ---------------------------------------------------------------- //

  /**
   * Remove a READ instruction when it is immediately preceded by another
   * READ for the same name without any intervening mutation.
   *
   * @param {Instruction[]} instrs
   * @returns {Instruction[]}
   */
  _eliminateRedundantReads(instrs) {
    const result = [];
    let lastRead = null;

    for (const instr of instrs) {
      if (instr.opcode === 'READ') {
        const name = instr.operands[0];
        if (lastRead === name) continue; // redundant
        lastRead = name;
      } else if (['WRITE', 'ALLOC', 'OP', 'CALL'].includes(instr.opcode)) {
        lastRead = null;
      }
      result.push(instr);
    }
    return result;
  }

  // ---------------------------------------------------------------- //
  // Pass 2 – dead-store elimination
  // ---------------------------------------------------------------- //

  /**
   * Remove WRITE instructions whose target identifier is never READ
   * after the write.
   *
   * @param {Instruction[]} instrs
   * @returns {Instruction[]}
   */
  _eliminateDeadStores(instrs) {
    // First pass: collect identifiers that are READ at any point after each write
    const readSet = new Set();

    // Gather all identifiers that are ever read
    for (const instr of instrs) {
      if (instr.opcode === 'READ') readSet.add(instr.operands[0]);
      if (instr.opcode === 'OP') {
        for (const op of instr.operands.slice(1)) {
          if (typeof op === 'string') readSet.add(op);
        }
      }
      if (instr.opcode === 'CALL') {
        for (const op of instr.operands.slice(1)) {
          if (typeof op === 'string') readSet.add(op);
        }
      }
      if (instr.opcode === 'WRITE') {
        const val = instr.operands[1];
        if (typeof val === 'string') readSet.add(val);
      }
    }

    return instrs.filter(instr => {
      if (instr.opcode !== 'WRITE') return true;
      const target = instr.operands[0];
      return readSet.has(target);
    });
  }

  // ---------------------------------------------------------------- //
  // Pass 3 – constant folding
  // ---------------------------------------------------------------- //

  /**
   * Fold OP instructions where all operands are numeric literals into a
   * CONST instruction (representing a pre-computed scalar result).
   *
   * Supported glyphs: ⊕ (add), ⊖ (subtract), ⊗ (multiply), ⊝ (negate).
   *
   * @param {Instruction[]} instrs
   * @returns {Instruction[]}
   */
  _foldConstants(instrs) {
    return instrs.map(instr => {
      if (instr.opcode !== 'OP') return instr;
      const [glyph, ...operands] = instr.operands;
      if (!operands.every(o => typeof o === 'number')) return instr;

      let result;
      switch (glyph) {
        case '⊕': result = operands.reduce((a, b) => a + b, 0); break;
        case '⊖': result = operands[0] - operands.slice(1).reduce((a, b) => a + b, 0); break;
        case '⊗': result = operands.reduce((a, b) => a * b, 1); break;
        case '⊝': result = -operands[0]; break;
        default:   return instr;
      }

      return new Instruction('CONST', [result], { ...instr.metadata, folded: true });
    });
  }
}
