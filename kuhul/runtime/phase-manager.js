/**
 * @fileoverview Phase-cycle manager for the KUHUL runtime.
 *
 * Phase advances from 0 to 2π in discrete steps.  The PhaseManager
 * tracks the current phase angle and provides step control.
 *
 * @module kuhul/runtime/phase-manager
 */

const TWO_PI = 2 * Math.PI;

// ------------------------------------------------------------------ //
// PhaseManager
// ------------------------------------------------------------------ //

/** Manages phase cycles in [0, 2π]. */
export class PhaseManager {
  /**
   * @param {number} [stepSize=Math.PI/8] - Phase increment per step
   */
  constructor(stepSize = Math.PI / 8) {
    this._angle    = 0;
    this._stepSize = stepSize;
    this._cycle    = 0;
  }

  /**
   * Advance the phase by one step.
   * Wraps around at 2π and increments the cycle counter.
   *
   * @returns {number} New phase angle in radians
   */
  nextPhase() {
    this._angle += this._stepSize;
    if (this._angle >= TWO_PI) {
      this._angle %= TWO_PI;
      this._cycle++;
    }
    return this._angle;
  }

  /**
   * Get the current phase angle.
   * @returns {number} Phase angle in radians (0 ≤ angle < 2π)
   */
  getCurrentPhase() {
    return this._angle;
  }

  /**
   * Get the number of complete cycles completed.
   * @returns {number}
   */
  getCycleCount() {
    return this._cycle;
  }

  /**
   * Reset phase angle and cycle counter to zero.
   */
  reset() {
    this._angle = 0;
    this._cycle = 0;
  }

  /**
   * Return a plain-object snapshot of the current phase state.
   * @returns {{ angle: number, cycle: number }}
   */
  snapshot() {
    return { angle: this._angle, cycle: this._cycle };
  }
}
