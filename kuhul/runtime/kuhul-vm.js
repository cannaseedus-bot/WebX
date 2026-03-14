/**
 * @fileoverview KUHUL Virtual Machine.
 *
 * Executes a GeometricIR program on a simple in-process stack machine.
 * Provides a bridge between the IR and the underlying ExecutionEngine /
 * PhaseManager / MemoryManager.
 *
 * @module kuhul/runtime/kuhul-vm
 */

import { GeometricIR } from '../ir/ir-types.js';
import { PhaseManager }  from './phase-manager.js';
import { MemoryManager } from './memory-manager.js';

// ------------------------------------------------------------------ //
// Glyph operation implementations (scalar fallbacks)
// ------------------------------------------------------------------ //

const GLYPH_FNS = {
  '⊗': (a, b) => a * b,
  '⊕': (a, b) => a + b,
  '⊖': (a, b) => a - b,
  '⊛': (a, b) => a * b, // simplified convolution scalar
  '⊜': (a, b) => (a === b ? 1 : 0),
  '⊝': (a)    => -a,
  '⊞': (a, b) => a + b, // scalar direct-sum
};

// ------------------------------------------------------------------ //
// KuhulVM
// ------------------------------------------------------------------ //

/** Executes a KUHUL GeometricIR program. */
export class KuhulVM {
  /**
   * Execute the given IR program.
   *
   * @param {GeometricIR} ir
   * @param {object} [context={}] - Initial variable bindings / environment
   * @returns {Promise<object>} Final memory state
   */
  async execute(ir, context = {}) {
    const mem   = new MemoryManager();
    const phase = new PhaseManager();
    const stack = []; // operand stack

    // Seed memory from context
    for (const [k, v] of Object.entries(context)) {
      const id = mem.allocate(1);
      mem.get(id).buffer[0] = typeof v === 'number' ? v : 0;
      mem._names = mem._names ?? new Map();
      mem._names.set(k, id);
    }

    /** Resolve an operand to a scalar value. */
    const resolve = (operand) => {
      if (typeof operand === 'number') return operand;
      if (typeof operand === 'string') {
        if (operand.startsWith('"')) return operand.slice(1, -1);
        const id  = mem._names?.get(operand);
        const blk = id != null ? mem.get(id) : null;
        return blk ? blk.buffer[0] : 0;
      }
      return 0;
    };

    for (const instr of ir.instructions) {
      switch (instr.opcode) {
        case 'ALLOC': {
          const [name, , shape] = instr.operands;
          const size = Array.isArray(shape) ? shape.reduce((a, d) => a * (d === '?' ? 1 : d), 1) : 1;
          const id   = mem.allocate(size);
          mem._names = mem._names ?? new Map();
          mem._names.set(name, id);
          break;
        }
        case 'READ': {
          const [name] = instr.operands;
          stack.push(resolve(name));
          break;
        }
        case 'WRITE': {
          const [target, valRef] = instr.operands;
          const val = resolve(valRef);
          const id  = mem._names?.get(target);
          if (id != null) {
            mem.get(id).buffer[0] = typeof val === 'number' ? val : 0;
          }
          break;
        }
        case 'OP': {
          const [glyph, ...operands] = instr.operands;
          const args = operands.map(resolve);
          const fn   = GLYPH_FNS[glyph];
          const result = fn ? fn(...args) : 0;
          stack.push(result);
          break;
        }
        case 'CALL':
          // External calls are no-ops in the VM
          break;
        case 'PHASE_START':
          phase.reset();
          break;
        case 'PHASE_END':
          phase.nextPhase();
          break;
        case 'FOLD_START':
        case 'FOLD_END':
        case 'CONST':
          break;
        default:
          // Unknown opcodes are silently skipped
          break;
      }
    }

    // Materialise the final memory state as a plain object
    const result = {};
    if (mem._names) {
      for (const [name, id] of mem._names) {
        const blk = mem.get(id);
        result[name] = blk ? Array.from(blk.buffer) : null;
      }
    }
    return result;
  }
}
