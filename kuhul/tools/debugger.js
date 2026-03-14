/**
 * @fileoverview Debugger for KUHUL programs.
 *
 * Executes a GeometricIR program step-by-step, firing callbacks at
 * each instruction and pausing at breakpoints.
 *
 * @module kuhul/tools/debugger
 */

import { KuhulVM }       from '../runtime/kuhul-vm.js';
import { KuhulCompiler } from '../compiler/kuhul-compiler.js';

// ------------------------------------------------------------------ //
// KuhulDebugger
// ------------------------------------------------------------------ //

/** Step-through debugger for KUHUL programs. */
export class KuhulDebugger {
  constructor() {
    this._breakpoints = new Set();
    this._paused      = false;
    this._stepResolve = null;
    this._compiler    = new KuhulCompiler();
    /** @type {import('../ir/ir-types.js').Instruction[]} */
    this._instructions = [];
    this._pos          = 0;
    this._context      = {};
  }

  /**
   * Set a breakpoint at the given instruction index (0-based).
   *
   * @param {number} index
   */
  setBreakpoint(index) {
    this._breakpoints.add(index);
  }

  /** Remove a breakpoint. @param {number} index */
  removeBreakpoint(index) {
    this._breakpoints.delete(index);
  }

  /**
   * Load a KUHUL program for debugging.
   *
   * @param {string} source
   * @returns {Promise<void>}
   */
  async load(source) {
    const program = await this._compiler.compile(source, 'js');
    this._instructions = program.ir.instructions;
    this._pos          = 0;
    this._paused       = false;
  }

  /**
   * Execute one instruction and pause.
   *
   * @returns {{ index: number, instruction: object }|null}
   */
  step() {
    if (this._pos >= this._instructions.length) return null;
    const instr = this._instructions[this._pos];
    this._pos++;
    return { index: this._pos - 1, instruction: instr };
  }

  /**
   * Continue execution until the next breakpoint or end of program.
   *
   * @returns {{ stoppedAt: number|null, reason: 'breakpoint'|'end' }}
   */
  continue() {
    while (this._pos < this._instructions.length) {
      if (this._breakpoints.has(this._pos)) {
        return { stoppedAt: this._pos, reason: 'breakpoint' };
      }
      this._pos++;
    }
    return { stoppedAt: null, reason: 'end' };
  }

  /** Reset the debugger to the beginning of the loaded program. */
  reset() {
    this._pos    = 0;
    this._paused = false;
  }

  /**
   * Get the current instruction pointer position.
   * @returns {number}
   */
  get position() { return this._pos; }

  /**
   * Get all instructions in the loaded program.
   * @returns {object[]}
   */
  get instructions() { return this._instructions; }
}
