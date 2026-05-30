// fold-tensor.js — Fold Tensor F ∈ ℝ^(F×M×C)
//
// Semantic state tensor. Three axes:
//   f = fold   (512 semantic folds)
//   m = memory lane (128 lanes — DICT/FIELD/PHASE/GEO/MEMORY/TENSOR + custom)
//   c = channel (256 feature channels per lane)
//
// The fold tensor IS the system's semantic memory.
// Micronauts read/write slices of it.
//
// SCXQ2 tensor lane mapping (from spec §8):
//   lane 0: DICT    — symbol lookup
//   lane 1: FIELD   — fold ownership
//   lane 2: PHASE   — phase state (mirrors PhaseTensor but denormalized)
//   lane 3: GEO     — geodesic edge cache
//   lane 4: MEMORY  — replay state
//   lane 5: TENSOR  — raw values
//   6+    : custom per-domain lanes

export const SCXQ2_LANES = Object.freeze({
  DICT: 0, FIELD: 1, PHASE: 2, GEO: 3, MEMORY: 4, TENSOR: 5,
});

export class FoldTensor {
  /**
   * @param {number} n_folds    — F
   * @param {number} n_lanes    — M
   * @param {number} n_channels — C
   */
  constructor(n_folds = 512, n_lanes = 128, n_channels = 256) {
    this.n_folds    = n_folds;
    this.n_lanes    = n_lanes;
    this.n_channels = n_channels;
    this.data       = new Float64Array(n_folds * n_lanes * n_channels);
  }

  // ── Index math ─────────────────────────────────────────────────────────────

  _idx(f, m, c) {
    return (f * this.n_lanes + m) * this.n_channels + c;
  }

  get(f, m, c)        { return this.data[this._idx(f, m, c)]; }
  set(f, m, c, value) { this.data[this._idx(f, m, c)] = value; }

  // ── Slice access ───────────────────────────────────────────────────────────

  /** Float64Array view of fold f, lane m — all C channels */
  lane(f, m) {
    const base = (f * this.n_lanes + m) * this.n_channels;
    return this.data.subarray(base, base + this.n_channels);
  }

  /** Copy a lane slice (returns new Float64Array) */
  readLane(f, m) {
    return Float64Array.from(this.lane(f, m));
  }

  writeLane(f, m, values) {
    const l = this.lane(f, m);
    const n = Math.min(l.length, values.length);
    for (let c = 0; c < n; c++) l[c] = values[c];
  }

  // ── Fold operations ────────────────────────────────────────────────────────

  /** Add delta to fold f, lane m */
  addDelta(f, m, delta) {
    const l = this.lane(f, m);
    const n = Math.min(l.length, delta.length);
    for (let c = 0; c < n; c++) l[c] += delta[c];
  }

  /** L2 norm of fold f, lane m */
  foldNorm(f, m) {
    const l = this.lane(f, m);
    let sq = 0;
    for (const v of l) sq += v * v;
    return Math.sqrt(sq);
  }

  /** Geodesic-weighted merge: F[f_out] += weight * F[f_src, lane] */
  geodesicMerge(f_out, lane_out, f_src, lane_src, weight) {
    const src = this.lane(f_src, lane_src);
    const dst = this.lane(f_out, lane_out);
    for (let c = 0; c < this.n_channels; c++) dst[c] += weight * src[c];
  }

  // ── Evolution (phase-gated ΔF) ────────────────────────────────────────────
  // Only Sek-active folds receive updates.
  // activeSet: array of fold indices currently in Sek phase.

  applyDelta(deltaF, activeFolds) {
    for (const f of activeFolds) {
      for (let m = 0; m < this.n_lanes; m++) {
        const base = (f * this.n_lanes + m) * this.n_channels;
        const dbase = (f * this.n_lanes + m) * this.n_channels;
        for (let c = 0; c < this.n_channels; c++)
          this.data[base + c] += deltaF.data[dbase + c];
      }
    }
  }

  /** Deep clone */
  clone() {
    const c = new FoldTensor(this.n_folds, this.n_lanes, this.n_channels);
    c.data.set(this.data);
    return c;
  }

  // ── SafeTensors projection ────────────────────────────────────────────────
  // Load a weight tensor into the TENSOR lane of a specific fold range.
  // This is Phase 1→2 of the SafeTensors progression.

  projectFromFlat(foldStart, lane, flatTensor, channels) {
    const c = Math.min(channels, this.n_channels);
    const rows = Math.floor(flatTensor.length / channels);
    const fEnd = Math.min(foldStart + rows, this.n_folds);
    for (let f = foldStart; f < fEnd; f++) {
      const row = f - foldStart;
      const l   = this.lane(f, lane);
      for (let ci = 0; ci < c; ci++) l[ci] = flatTensor[row * channels + ci];
    }
  }

  // ── Serialisation ──────────────────────────────────────────────────────────

  toJSON() {
    return {
      "@tensor": "fold",
      "@shape": [this.n_folds, this.n_lanes, this.n_channels],
      data: Array.from(this.data),
    };
  }

  static fromJSON(obj) {
    const [nf, nm, nc] = obj["@shape"];
    const t = new FoldTensor(nf, nm, nc);
    t.data.set(obj.data);
    return t;
  }

  get shape() { return [this.n_folds, this.n_lanes, this.n_channels]; }
  get byteSize() { return this.data.byteLength; }
}
