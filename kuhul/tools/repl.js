/**
 * @fileoverview Interactive REPL for the KUHUL language.
 *
 * Provides a read-eval-print loop that compiles and executes KUHUL
 * source snippets interactively.  In a Node.js environment it uses the
 * built-in `readline` module; in other environments it exposes the same
 * async `eval` API for programmatic use.
 *
 * @module kuhul/tools/repl
 */

import { KuhulCompiler } from '../compiler/kuhul-compiler.js';
import { KuhulVM }       from '../runtime/kuhul-vm.js';

// ------------------------------------------------------------------ //
// KuhulREPL
// ------------------------------------------------------------------ //

/** Interactive REPL for the KUHUL language. */
export class KuhulREPL {
  /**
   * @param {{ prompt?: string, target?: string }} [options]
   */
  constructor(options = {}) {
    this._prompt   = options.prompt ?? 'kuhul> ';
    this._target   = options.target ?? 'js';
    this._compiler = new KuhulCompiler();
    this._vm       = new KuhulVM();
    this._running  = false;
    this._history  = [];
    this._rl       = null;
  }

  /**
   * Start the interactive REPL (Node.js only).
   * Resolves when the user exits with `.exit` or Ctrl-D.
   *
   * @returns {Promise<void>}
   */
  async start() {
    let readline;
    try {
      readline = await import('readline');
    } catch (_) {
      console.error('REPL requires a Node.js environment with readline support.');
      return;
    }

    this._running = true;
    this._rl = readline.createInterface({
      input:  process.stdin,
      output: process.stdout,
      prompt: this._prompt,
    });

    this._rl.prompt();
    this._rl.on('line', async (line) => {
      const src = line.trim();
      if (!src) { this._rl.prompt(); return; }
      if (src === '.exit' || src === '.quit') { this.stop(); return; }

      const result = await this.eval(src);
      console.log(result);
      this._rl.prompt();
    });

    await new Promise(resolve => this._rl.on('close', resolve));
  }

  /** Stop the REPL. */
  stop() {
    this._running = false;
    if (this._rl) { this._rl.close(); this._rl = null; }
  }

  /**
   * Evaluate a KUHUL source snippet.
   *
   * @param {string} source
   * @returns {Promise<string>} Human-readable result or error message
   */
  async eval(source) {
    this._history.push(source);
    try {
      const program = await this._compiler.compile(source, 'js');
      const result  = await this._vm.execute(program.ir);
      return JSON.stringify(result, null, 2);
    } catch (err) {
      return `Error: ${err.message}`;
    }
  }

  /** Return the command history. @returns {string[]} */
  getHistory() { return [...this._history]; }
}
