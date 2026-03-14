/**
 * @fileoverview Performance profiler for KUHUL programs.
 *
 * Records timing and instruction-count metrics across compilation and
 * execution phases.
 *
 * @module kuhul/tools/profiler
 */

// ------------------------------------------------------------------ //
// KuhulProfiler
// ------------------------------------------------------------------ //

/** Records performance metrics for KUHUL programs. */
export class KuhulProfiler {
  constructor() {
    this._records = [];
    this._active  = null;
  }

  /**
   * Start a profiling session with the given label.
   *
   * @param {string} [label='default']
   */
  start(label = 'default') {
    this._active = { label, startTime: Date.now(), startHR: this._hrNow() };
  }

  /**
   * Stop the current profiling session and record the result.
   *
   * @returns {{ label: string, durationMs: number }} The completed record
   */
  stop() {
    if (!this._active) throw new Error('Profiler: no active session (call start() first)');
    const endTime = Date.now();
    const endHR   = this._hrNow();
    const record  = {
      label:      this._active.label,
      durationMs: endTime - this._active.startTime,
      durationHR: endHR  - this._active.startHR,
    };
    this._records.push(record);
    this._active = null;
    return record;
  }

  /**
   * Generate a human-readable profiling report.
   *
   * @returns {string}
   */
  report() {
    if (this._records.length === 0) return 'No profiling data collected.';
    const lines = ['=== KUHUL Profiling Report ===', ''];
    for (const r of this._records) {
      lines.push(`  ${r.label.padEnd(30)} ${r.durationMs.toString().padStart(8)} ms  (${r.durationHR.toFixed(3)} ms HR)`);
    }
    lines.push('');
    const total = this._records.reduce((s, r) => s + r.durationMs, 0);
    lines.push(`  ${'TOTAL'.padEnd(30)} ${total.toString().padStart(8)} ms`);
    lines.push('');
    return lines.join('\n');
  }

  /**
   * Return all recorded profiles.
   * @returns {{ label: string, durationMs: number, durationHR: number }[]}
   */
  getRecords() { return [...this._records]; }

  /** Clear all records and reset. */
  reset() {
    this._records = [];
    this._active  = null;
  }

  // ---------------------------------------------------------------- //
  // High-resolution timer
  // ---------------------------------------------------------------- //

  _hrNow() {
    if (typeof performance !== 'undefined') return performance.now();
    // Node.js
    try {
      const [s, ns] = process.hrtime();
      return s * 1e3 + ns / 1e6;
    } catch (_) {
      return Date.now();
    }
  }
}
