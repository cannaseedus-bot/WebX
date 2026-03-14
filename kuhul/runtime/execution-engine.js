/**
 * @fileoverview Hardware execution engine for KUHUL programs.
 *
 * The ExecutionEngine bridges the VM-level GeometricIR execution and the
 * underlying hardware contexts (CPU, WebGPU GPU, WebAssembly).
 * It selects the best available backend and runs the compiled program.
 *
 * @module kuhul/runtime/execution-engine
 */

import { KuhulVM } from './kuhul-vm.js';

// ------------------------------------------------------------------ //
// ExecutionEngine
// ------------------------------------------------------------------ //

/** Hardware execution engine – selects the best available backend. */
export class ExecutionEngine {
  /**
   * @param {{ backend?: 'auto'|'cpu'|'gpu'|'wasm' }} [options]
   */
  constructor(options = {}) {
    this._backend = options.backend ?? 'auto';
    this._vm      = new KuhulVM();
  }

  /**
   * Run a compiled GeometricIR program.
   *
   * @param {import('../ir/ir-types.js').GeometricIR} program
   * @param {object} [context={}] - Initial variable bindings
   * @returns {Promise<object>} Execution result
   */
  async run(program, context = {}) {
    const backend = this._resolveBackend();

    switch (backend) {
      case 'gpu':
        return this._runGPU(program, context);
      case 'wasm':
        return this._runWasm(program, context);
      case 'cpu':
      default:
        return this._runCPU(program, context);
    }
  }

  // ---------------------------------------------------------------- //
  // Backend selection
  // ---------------------------------------------------------------- //

  _resolveBackend() {
    if (this._backend !== 'auto') return this._backend;
    // In a browser environment, prefer GPU if available
    if (typeof navigator !== 'undefined' && navigator.gpu) return 'gpu';
    return 'cpu';
  }

  // ---------------------------------------------------------------- //
  // CPU backend (uses KuhulVM)
  // ---------------------------------------------------------------- //

  async _runCPU(program, context) {
    return this._vm.execute(program, context);
  }

  // ---------------------------------------------------------------- //
  // WebGPU backend (stub – falls back to CPU when GPU unavailable)
  // ---------------------------------------------------------------- //

  async _runGPU(program, context) {
    // In environments without WebGPU, fall back to CPU
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      return this._runCPU(program, context);
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return this._runCPU(program, context);

      const device = await adapter.requestDevice();
      // A full WebGPU execution pipeline would dispatch compute shaders here.
      // For now we fall back to the CPU VM.
      device.destroy();
    } catch (_) {
      // Ignore GPU errors and fall back
    }
    return this._runCPU(program, context);
  }

  // ---------------------------------------------------------------- //
  // WebAssembly backend (stub – falls back to CPU)
  // ---------------------------------------------------------------- //

  async _runWasm(program, context) {
    // A full Wasm backend would instantiate a compiled .wasm module here.
    return this._runCPU(program, context);
  }
}
