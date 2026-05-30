// pi-tensor.js — π Phase Tensor Π = A sin(ωt + φ)
//
// The clock signal of the fold field.
// Each node has its own oscillator: amplitude A, frequency ω, phase offset φ.
// Π[n, t] = A_n * sin(ω_n * t + φ_n)
//
// This drives:
//   - MoE routing: nodes with phase-aligned Π values route to same expert
//   - Kuramoto coupling: dφ/dt = ω + K Σ sin(φ_j - φ_i)
//   - Fold gating: Sek fires when Π > threshold

import { N_PHASES } from './phase-tensor.js';

export class PiPhaseTensor {
  /**
   * @param {number} n_nodes
   * @param {object} opts
   */
  constructor(n_nodes, opts = {}) {
    this.n_nodes = n_nodes;

    // Per-node oscillator parameters
    this.A    = new Float64Array(n_nodes).fill(opts.amplitude  ?? 1.0);
    this.omega = new Float64Array(n_nodes).fill(opts.frequency ?? 1.0);
    this.phi  = opts.phi ?? (() => {
      // Random initial phase offsets in [0, 2π)
      const a = new Float64Array(n_nodes);
      for (let i = 0; i < n_nodes; i++) a[i] = Math.random() * 2 * Math.PI;
      return a;
    })();

    // Kuramoto coupling constant
    this.K = opts.K ?? 0.1;

    // Current time
    this.t = 0;

    // Cached current values
    this._values = new Float64Array(n_nodes);
    this._updateValues();
  }

  // ── Core oscillation ────────────────────────────────────────────────────────

  /** Π[n] at current time */
  value(node) { return this._values[node]; }

  /** All values as Float64Array */
  get values() { return this._values; }

  _updateValues() {
    for (let n = 0; n < this.n_nodes; n++)
      this._values[n] = this.A[n] * Math.sin(this.omega[n] * this.t + this.phi[n]);
  }

  // ── Kuramoto synchronisation step ─────────────────────────────────────────
  // dφ_i/dt = ω_i + K/N Σ_j sin(φ_j - φ_i)
  // Called each tick to evolve phases toward synchrony.

  kuramoto_step(dt = 0.01, adjacency = null) {
    const N   = this.n_nodes;
    const dphi = new Float64Array(N);

    for (let i = 0; i < N; i++) {
      let coupling = 0;
      if (adjacency) {
        // Sparse coupling from geodesic neighbours
        for (const { j } of adjacency(i))
          coupling += Math.sin(this.phi[j] - this.phi[i]);
        coupling *= this.K / N;
      } else {
        // Mean-field all-to-all
        for (let j = 0; j < N; j++)
          coupling += Math.sin(this.phi[j] - this.phi[i]);
        coupling *= this.K / N;
      }
      dphi[i] = this.omega[i] + coupling;
    }
    for (let i = 0; i < N; i++) this.phi[i] += dphi[i] * dt;
    this.t += dt;
    this._updateValues();
  }

  // ── Phase coherence (Kuramoto order parameter r) ──────────────────────────
  // r = |1/N Σ exp(i φ_n)| ∈ [0,1]
  // r=1 → fully synchronised; r=0 → incoherent

  get coherence() {
    let re = 0, im = 0;
    for (let n = 0; n < this.n_nodes; n++) {
      re += Math.cos(this.phi[n]);
      im += Math.sin(this.phi[n]);
    }
    return Math.sqrt(re ** 2 + im ** 2) / this.n_nodes;
  }

  // ── MoE phase routing ─────────────────────────────────────────────────────
  // Geodesic routing: node routes to expert whose phase is closest.
  // expertPhases: Float64Array[n_experts] of expert phase values

  routeToExpert(node, expertPhases) {
    const phi_n = this.phi[node];
    let minDist = Infinity, best = 0;
    for (let e = 0; e < expertPhases.length; e++) {
      const dist = Math.abs(phi_n - expertPhases[e]) % (2 * Math.PI);
      const d = Math.min(dist, 2 * Math.PI - dist);
      if (d < minDist) { minDist = d; best = e; }
    }
    return { expert: best, phase_dist: minDist };
  }

  // ── Sek gating ────────────────────────────────────────────────────────────
  // Returns fold indices where Π > threshold (active for computation)

  sekActiveNodes(threshold = 0.0) {
    const active = [];
    for (let n = 0; n < this.n_nodes; n++)
      if (this._values[n] > threshold) active.push(n);
    return active;
  }

  // ── Serialisation ─────────────────────────────────────────────────────────

  toJSON() {
    return {
      "@tensor": "pi_phase",
      "@shape": [this.n_nodes],
      t: this.t, K: this.K,
      A: Array.from(this.A),
      omega: Array.from(this.omega),
      phi: Array.from(this.phi),
    };
  }

  static fromJSON(obj) {
    const n = obj["@shape"][0];
    const t = new PiPhaseTensor(n, {
      amplitude: 1, frequency: 1, K: obj.K,
      phi: new Float64Array(obj.phi),
    });
    t.A.set(obj.A); t.omega.set(obj.omega); t.t = obj.t;
    t._updateValues();
    return t;
  }
}
