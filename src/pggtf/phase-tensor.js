// phase-tensor.js — Phase Tensor P ∈ ℝ^(N×5)
//
// Execution-state tensor. One row per node, 5 columns for phases.
//   P[n, p] = activation of node n at phase p
//
// Canonical phase indices:
//   0=Pop  1=Wo  2=Sek  3=Ch'en  4=Xul
//
// Convention: P[n, :] is a probability distribution over phases.
//   sum(P[n, :]) ≈ 1.0 for active nodes.

export const PHASE_INDEX = Object.freeze({
  Pop: 0, Wo: 1, Sek: 2, "Ch'en": 3, Xul: 4,
});

export const PHASE_NAMES = Object.freeze(['Pop', 'Wo', 'Sek', "Ch'en", 'Xul']);
export const N_PHASES = 5;

export class PhaseTensor {
  /**
   * @param {number} n_nodes
   * @param {Float64Array|null} data — optional initial data (n_nodes × N_PHASES, row-major)
   */
  constructor(n_nodes, data = null) {
    this.n_nodes = n_nodes;
    this.data = data ?? new Float64Array(n_nodes * N_PHASES);
  }

  // ── Element access ─────────────────────────────────────────────────────────

  get(node, phase) {
    return this.data[node * N_PHASES + phase];
  }

  set(node, phase, value) {
    this.data[node * N_PHASES + phase] = value;
  }

  /** Row slice for node n: Float64Array[5] view */
  row(node) {
    return this.data.subarray(node * N_PHASES, (node + 1) * N_PHASES);
  }

  // ── Phase queries ──────────────────────────────────────────────────────────

  /** Phase index with highest activation for node n */
  dominantPhase(node) {
    const row = this.row(node);
    let max = -Infinity, idx = 0;
    for (let p = 0; p < N_PHASES; p++) { if (row[p] > max) { max = row[p]; idx = p; } }
    return idx;
  }

  dominantPhaseName(node) { return PHASE_NAMES[this.dominantPhase(node)]; }

  /** All nodes currently in Sek phase (activation > threshold) */
  nodesInPhase(phaseIdx, threshold = 0.5) {
    const result = [];
    for (let n = 0; n < this.n_nodes; n++)
      if (this.get(n, phaseIdx) > threshold) result.push(n);
    return result;
  }

  // ── Transition ─────────────────────────────────────────────────────────────

  /** Hard-set node n to phase p (one-hot) */
  setPhase(node, phaseIdx) {
    const base = node * N_PHASES;
    for (let p = 0; p < N_PHASES; p++) this.data[base + p] = p === phaseIdx ? 1.0 : 0.0;
  }

  /** Soft transition toward target phase by alpha */
  transitionPhase(node, targetPhase, alpha = 0.1) {
    const base = node * N_PHASES;
    for (let p = 0; p < N_PHASES; p++) {
      const target = p === targetPhase ? 1.0 : 0.0;
      this.data[base + p] += alpha * (target - this.data[base + p]);
    }
  }

  /** Advance every Sek node to Ch'en (called after compute phase completes) */
  advanceSekToChen() {
    for (let n = 0; n < this.n_nodes; n++)
      if (this.get(n, PHASE_INDEX.Sek) > 0.5) this.setPhase(n, PHASE_INDEX["Ch'en"]);
  }

  // ── Serialisation ──────────────────────────────────────────────────────────

  toJSON() {
    return { "@tensor": "phase", "@shape": [this.n_nodes, N_PHASES],
             data: Array.from(this.data) };
  }

  static fromJSON(obj) {
    const t = new PhaseTensor(obj["@shape"][0]);
    t.data.set(obj.data);
    return t;
  }
}
