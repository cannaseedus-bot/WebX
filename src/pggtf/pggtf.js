// pggtf.js — Phase-Gated Geodesic Tensor Field (PGGTF) v0.1
//
// System state: Ω = (P, G, F, M, Π)
// Evolution:    Ω_{t+1} = M(P, G, F, Π)
//
// Inference law: predict P(F_{t+1} | F_t, G, P) — next field state,
//   NOT next token. The model predicts the configuration of the semantic
//   field, which may then be decoded to tokens if needed.
//
// SafeTensors integration (Phase 1→2 progression):
//   PGGTF.fromSafeTensors(weightsDict) — load model weights into fold field.
//   The physical storage layer remains safetensors; PGGTF adds semantics.

import { PhaseTensor, PHASE_INDEX, N_PHASES }  from './phase-tensor.js';
import { GeodesicTensor }                      from './geodesic-tensor.js';
import { FoldTensor, SCXQ2_LANES }             from './fold-tensor.js';
import { PiPhaseTensor }                       from './pi-tensor.js';
import {
  memoryMicronaut, routingMicronaut,
  compressionMicronaut, inferenceMicronaut,
  applyOperator,
} from './micronaut-ops.js';

export class PGGTF {
  /**
   * @param {object} opts
   *   n_nodes    — number of nodes (rows of P and Π)
   *   n_folds    — F dim (default 512)
   *   n_lanes    — M dim (default 128)
   *   n_channels — C dim (default 256)
   *   k_nearest  — edges per node in G (default 8)
   *   K_kuramoto — coupling constant (default 0.1)
   */
  constructor(opts = {}) {
    const n  = opts.n_nodes    ?? 512;
    const nf = opts.n_folds    ?? 512;
    const nm = opts.n_lanes    ?? 128;
    const nc = opts.n_channels ?? 256;

    // Core tensors
    this.P  = new PhaseTensor(n);
    this.F  = new FoldTensor(nf, nm, nc);
    this.Pi = new PiPhaseTensor(n, { K: opts.K_kuramoto ?? 0.1 });

    // Geodesic tensor: initialise with random positions if none supplied
    if (opts.positions) {
      this.G = GeodesicTensor.fromPositions(
        opts.positions, n, opts.dim ?? 3, opts.k_nearest ?? 8, opts.curvature ?? 0);
    } else {
      // Random unit-cube positions
      const pos = new Float64Array(n * 3);
      for (let i = 0; i < pos.length; i++) pos[i] = Math.random();
      this.G = GeodesicTensor.fromPositions(pos, n, 3, opts.k_nearest ?? 8);
    }

    this._tick = 0;
    this._history = [];   // brief summary per tick for replay
  }

  // ── Phase initialisation ──────────────────────────────────────────────────

  /** Hard-set all nodes to Pop (start of execution) */
  initPhases(phase = PHASE_INDEX.Pop) {
    for (let n = 0; n < this.P.n_nodes; n++) this.P.setPhase(n, phase);
    return this;
  }

  /** Advance entire field to the next canonical phase */
  advanceAllPhases() {
    for (let n = 0; n < this.P.n_nodes; n++) {
      const cur  = this.P.dominantPhase(n);
      const next = (cur + 1) % N_PHASES;
      this.P.setPhase(n, next);
    }
    return this;
  }

  // ── SafeTensors integration ───────────────────────────────────────────────
  // Phase 1→2: project loaded weights into the fold field.
  // weightsDict: { "embedding.weight": Float32Array, ... }

  loadFromWeights(weightsDict, opts = {}) {
    let foldOffset = 0;
    for (const [name, tensor] of Object.entries(weightsDict)) {
      const flat = tensor instanceof Float64Array ? tensor
                 : new Float64Array(tensor.length).map((_, i) => tensor[i]);
      const C = Math.min(flat.length, this.F.n_channels);

      // Map tensor name to SCXQ2 lane
      let lane = SCXQ2_LANES.TENSOR;
      if (name.includes("embed"))   lane = SCXQ2_LANES.TENSOR;
      if (name.includes("attn"))    lane = SCXQ2_LANES.FIELD;
      if (name.includes("mlp"))     lane = SCXQ2_LANES.MEMORY;
      if (opts.laneMap?.[name] !== undefined) lane = opts.laneMap[name];

      this.F.projectFromFlat(foldOffset, lane, flat, C);
      foldOffset = (foldOffset + Math.floor(flat.length / C)) % this.F.n_folds;
    }
    return this;
  }

  // ── Single tick of field evolution ───────────────────────────────────────
  // Ω_{t+1} = M(P, G, F, Π)
  //
  // 1. Kuramoto step → update Π phase oscillators
  // 2. Identify Sek-active folds (Π > 0)
  // 3. Apply micronaut operators in order: memory → route → compress → infer
  // 4. Phase-gated fold update
  // 5. Advance phase tensor

  tick(dt = 0.01, opts = {}) {
    // 1. Evolve Kuramoto oscillators
    this.Pi.kuramoto_step(dt, (i) => this.G.neighbours(i));

    // 2. Sek-active folds (those with positive Π signal)
    const activeFolds = this.Pi.sekActiveNodes(opts.sekThreshold ?? 0.0);

    // Reflect Π into F's PHASE lane for micronauts to read
    for (const f of activeFolds) {
      const phaseLane = this.F.lane(f, SCXQ2_LANES.PHASE);
      phaseLane[PHASE_INDEX.Sek] = this.Pi.value(f % this.Pi.n_nodes);
    }

    if (activeFolds.length > 0) {
      // 3. Micronaut operators (phase-gated: only Sek folds update)
      const mem  = memoryMicronaut(this.F, activeFolds, opts.memory);
      const rte  = routingMicronaut(this.F, this.G, activeFolds, opts.routing);
      const cmp  = compressionMicronaut(this.F, activeFolds, opts.compression);
      const inf  = inferenceMicronaut(this.F, this.G, this.P, activeFolds, opts.inference);

      // 4. Phase-gated apply: ΔF only for Sek-active folds
      applyOperator(this.F, mem);
      applyOperator(this.F, cmp);
      applyOperator(this.F, inf);
      // Routing mutates G directly (no deltaF)

      this._history.push({
        tick: this._tick, active: activeFolds.length,
        coherence: this.Pi.coherence, edges_added: rte.edges_added,
      });
      if (this._history.length > 256) this._history.shift();
    }

    // 5. Advance phase tensor: Sek → Ch'en after computation
    this.P.advanceSekToChen();

    this._tick++;
    return { tick: this._tick, active: activeFolds.length, coherence: this.Pi.coherence };
  }

  // ── Run N ticks ───────────────────────────────────────────────────────────

  run(n_ticks = 10, dt = 0.01) {
    const results = [];
    for (let i = 0; i < n_ticks; i++) results.push(this.tick(dt));
    return results;
  }

  // ── Field state summary ───────────────────────────────────────────────────

  get state() {
    return {
      tick:      this._tick,
      coherence: this.Pi.coherence,
      active:    this.Pi.sekActiveNodes().length,
      p_shape:   [this.P.n_nodes, N_PHASES],
      g_nnz:     this.G.nnz,
      f_shape:   this.F.shape,
    };
  }

  get history() { return [...this._history]; }

  // ── Snapshot for serialisation ─────────────────────────────────────────────

  snapshot() {
    return {
      tick: this._tick,
      P:  this.P.toJSON(),
      G:  this.G.toJSON(),
      F:  this.F.toJSON(),
      Pi: this.Pi.toJSON(),
    };
  }

  // ── Factory: minimal field for smoke testing ───────────────────────────────

  static minimal(n = 32) {
    return new PGGTF({ n_nodes: n, n_folds: n, n_lanes: 6, n_channels: 16 });
  }
}
