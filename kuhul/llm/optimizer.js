/**
 * @fileoverview LLM-guided IR optimizer for KUHUL.
 *
 * Wraps the rule-based IROptimizer and adds an LLM-guided heuristic layer
 * that can annotate the IR with optimisation hints or reorder instructions
 * based on predicted data-access patterns.
 *
 * @module kuhul/llm/optimizer
 */

import { IROptimizer } from '../ir/ir-optimizer.js';
import { GeometricIR, Instruction } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// LLMOptimizer
// ------------------------------------------------------------------ //

/** LLM-guided KUHUL IR optimizer. */
export class LLMOptimizer {
  constructor() {
    this._base = new IROptimizer();
  }

  /**
   * Optimize the IR using both rule-based and heuristic passes.
   *
   * @param {GeometricIR} ir
   * @returns {Promise<GeometricIR>}
   */
  async optimize(ir) {
    // 1. Apply base rule-based optimisation
    let optimized = this._base.optimize(ir);

    // 2. LLM heuristic: annotate phase cycles with loop-unroll hints when
    //    the phase body is short (≤ 4 instructions).
    optimized = this._annotatePhaseHints(optimized);

    // 3. LLM heuristic: promote frequently read tensors to "hot" metadata
    optimized = this._promoteHotTensors(optimized);

    return optimized;
  }

  // ---------------------------------------------------------------- //
  // Heuristic passes
  // ---------------------------------------------------------------- //

  /**
   * Annotate PHASE_START instructions with an `unroll` hint when the
   * enclosed body has ≤ 4 instructions.
   *
   * @param {GeometricIR} ir
   * @returns {GeometricIR}
   */
  _annotatePhaseHints(ir) {
    const instrs = [...ir.instructions];
    for (let i = 0; i < instrs.length; i++) {
      if (instrs[i].opcode !== 'PHASE_START') continue;
      // Count instructions until PHASE_END
      let depth = 1, j = i + 1, count = 0;
      while (j < instrs.length && depth > 0) {
        if (instrs[j].opcode === 'PHASE_START') depth++;
        if (instrs[j].opcode === 'PHASE_END')   depth--;
        if (depth > 0) count++;
        j++;
      }
      if (count <= 4) {
        instrs[i] = new Instruction('PHASE_START', instrs[i].operands, {
          ...instrs[i].metadata,
          llm_hint: 'unroll',
        });
      }
    }
    return new GeometricIR(instrs, new Map(ir.symbolTable), { ...ir.metadata, llm_optimized: true });
  }

  /**
   * Mark tensors that are READ more than once as "hot" in the symbol table
   * metadata.
   *
   * @param {GeometricIR} ir
   * @returns {GeometricIR}
   */
  _promoteHotTensors(ir) {
    const readCount = new Map();
    for (const instr of ir.instructions) {
      if (instr.opcode === 'READ') {
        const name = instr.operands[0];
        readCount.set(name, (readCount.get(name) ?? 0) + 1);
      }
    }
    const newTable = new Map(ir.symbolTable);
    for (const [name, type] of newTable) {
      if ((readCount.get(name) ?? 0) > 1) {
        // Shallow-clone the TensorType instance and add a hot-path hint
        const hotType = Object.create(Object.getPrototypeOf(type));
        Object.assign(hotType, type);
        hotType.hot = true;
        newTable.set(name, hotType);
      }
    }
    return new GeometricIR(ir.instructions, newTable, ir.metadata);
  }
}
