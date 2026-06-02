// entropic-weights.js — Entropic Geodesic Weights: fog of war on the semantic map
//
// An entropic weight = a geodesic weight that KNOWS its own uncertainty.
// The entropy tensor travels alongside the value tensor — same shape, same transport.
//
// What the entropy field adds to the map:
//   entropy=0.0  clear sky — model is certain here
//   entropy=0.5  haze      — moderate uncertainty
//   entropy=1.0  thick fog — never visited / highly uncertain
//
// How entropy changes:
//   prediction correct → entropy decreases (fog lifts)
//   prediction wrong   → entropy increases (fog thickens)
//   ARC replayed       → entropy along path decreases
//   curvature high     → entropy naturally higher (complex geometry = more uncertainty)
//
// K'UHUL physics connection:
//   entropy=0.0 ≡ gravity=HEAVY (tightly constrained, certain)
//   entropy=1.0 ≡ gravity=FLOAT (antigravity, near-zero constraint, exploring)
//   EntropyField.update() fires in Ch'en phase (after prediction, update uncertainty)
//
// Connection to KuhulPhysicsSolver:
//   mean entropy > 0.7  → physics Rule 1 fires (approaching uncertainty horizon)
//   entropy decreasing  → stable orbit → reserves charge
//   entropy spike       → escape velocity event

import { SphericalManifold, GeodesicWeight } from './geo-weights.js';

// ─── EntropyField ─────────────────────────────────────────────────────────────

export class EntropyField {
  constructor(size, initial = 0.5, opts = {}) {
    this.size      = size;
    this.values    = new Float32Array(size).fill(initial);
    this.decayRate = opts.decayRate ?? 0.99;    // entropy decays toward 0 with data
    this.diffusion = opts.diffusion ?? 0.01;    // entropy spreads to neighbours
    this._history  = [];
  }

  get(idx)        { return this.values[idx] ?? 0.5; }
  set(idx, v)     { this.values[idx] = Math.min(1, Math.max(0, v)); }
  mean()          { return this.values.reduce((s,v)=>s+v,0) / this.size; }
  max()           { return Math.max(...this.values); }
  min()           { return Math.min(...this.values); }

  /** Update entropy from prediction error (scalar or per-point array). */
  update(errors) {
    const errs = Array.isArray(errors) ? errors : new Array(this.size).fill(errors);
    for (let i = 0; i < this.size; i++) {
      const e = errs[i] ?? 0;
      // High error → more entropy; low error → entropy decays
      this.values[i] = Math.min(1, Math.max(0,
        this.values[i] * this.decayRate + e * (1 - this.decayRate)));
    }
    this._history.push(this.mean());
    if (this._history.length > 1000) this._history.shift();
  }

  /** Reduce entropy along a path (successful ARC replay clears fog). */
  clearAlongPath(indices, reduction = 0.05) {
    for (const idx of indices)
      this.values[idx] = Math.max(0, this.values[idx] * (1 - reduction));
  }

  /** Entropy coupled to curvature: high curvature → naturally higher entropy. */
  syncWithCurvature(curvatureValues, coupling = 0.5) {
    for (let i = 0; i < this.size; i++) {
      const κ_entropy = 1 / (1 + Math.exp(-coupling * curvatureValues[i]));
      this.values[i] = Math.min(1, Math.max(0, (this.values[i] + κ_entropy) / 2));
    }
  }

  /** Curiosity-driven exploration priorities (high entropy = explore first). */
  explorationPriority(dataOccurrence) {
    return this.values.map((e, i) => e * (dataOccurrence[i] ?? 1));
  }

  trend() {
    if (this._history.length < 2) return 0;
    const n = this._history.length;
    return this._history[n-1] - this._history[n-2];
  }

  summary() {
    return { mean: this.mean().toFixed(3), max: this.max().toFixed(3),
             min: this.min().toFixed(3), trend: this.trend().toFixed(4),
             foggy: this.values.filter(v=>v>0.7).length,
             clear:  this.values.filter(v=>v<0.2).length };
  }
}

// ─── EntropicGeodesicWeight ───────────────────────────────────────────────────

export class EntropicGeodesicWeight extends GeodesicWeight {
  constructor(manifold, basePoint, values, entropyInit = 0.3, opts = {}) {
    super(manifold, basePoint, values, opts);
    // Entropy tensor: same shape as values, values ∈ [0,1]
    this.entropy = new Float32Array(values.length).fill(entropyInit);
    this._temperature = Math.exp(entropyInit);
  }

  get temperature() { return Math.exp(this.entropy.reduce((s,v)=>s+v,0) / this.entropy.length); }

  /** Transport both value and entropy to targetPoint. */
  parallelTransportTo(targetPoint) {
    const transportedValues = super.parallelTransportTo(targetPoint);
    // Entropy also transports (simpler: diffuse slightly)
    const dist = this.manifold.geodesicDistance(this.basePoint, targetPoint);
    const diffusion = 0.01 * dist;
    const transportedEntropy = this.entropy.map(e =>
      Math.min(1, e + diffusion * (0.5 - e))
    );
    return { values: transportedValues, entropy: transportedEntropy };
  }

  /** Apply weight with temperature scaling from entropy. */
  applyAt(targetPoint, input) {
    const { values: W, entropy: E } = this.parallelTransportTo(targetPoint);
    const rows = this.shape[0], cols = this.shape[1];
    const temp = Math.exp(E.reduce((s,v)=>s+v,0) / E.length);  // mean entropy → temperature

    // Scaled matmul (high entropy → softer output)
    const output = new Array(Math.min(rows, 3)).fill(0);
    for (let r = 0; r < output.length; r++)
      for (let c = 0; c < cols; c++)
        output[r] += W[r * cols + c] * (input[c] ?? 0) / Math.sqrt(temp);

    return this.manifold.expMap(targetPoint, output);
  }

  /** Reduce entropy when prediction is correct (fog lifts). */
  clearFog(predictionQuality = 0.9, rate = 0.05) {
    for (let i = 0; i < this.entropy.length; i++)
      this.entropy[i] = Math.max(0, this.entropy[i] * (1 - rate * predictionQuality));
  }

  /** Increase entropy when surprised (fog thickens). */
  addFog(surprise = 0.5, rate = 0.1) {
    for (let i = 0; i < this.entropy.length; i++)
      this.entropy[i] = Math.min(1, this.entropy[i] + rate * surprise);
  }

  meanEntropy() { return this.entropy.reduce((s,v)=>s+v,0)/this.entropy.length; }

  snapshot() {
    return { ...super.snapshot(), meanEntropy: this.meanEntropy(), temperature: this.temperature };
  }
}

// ─── EntropicGeodesicAttention ────────────────────────────────────────────────
//
// Attention where score = geodesic_distance + entropy_penalty
// High entropy target → attend less (uncertain destination)
// High entropy source → attend more broadly (exploring from uncertain location)

export class EntropicGeodesicAttention {
  constructor(dim, curvature = 0.1, opts = {}) {
    this.dim           = dim;
    this.manifold      = new SphericalManifold(curvature);
    this.geoWeight     = opts.geoWeight  ?? 0.7;  // weight of geodesic distance
    this.entroWeight   = opts.entroWeight ?? 0.3; // weight of entropy penalty
    this.curiosity     = opts.curiosity  ?? 0.1;  // bonus for high-entropy targets
    this.baseTemp      = opts.baseTemp   ?? 1.0;
    this._entropyField = null;
  }

  setEntropyField(field) { this._entropyField = field; return this; }

  /** Compute attention weights for query position i over all key positions.
   *  Returns softmax-normalised array of length n. */
  attend(queryPos, keyPositions, queryEntropy = 0, keyEntropies = null) {
    const n = keyPositions.length;
    const scores = keyPositions.map((kp, j) => {
      const geoDist  = this.manifold.geodesicDistance(queryPos, kp);
      const kEnt     = keyEntropies ? (keyEntropies[j] ?? 0) : 0;
      const entroPen = this.entroWeight * kEnt;
      const curiBonus = this.curiosity * kEnt;         // reward exploring foggy areas
      const temp      = this.baseTemp * (1 + queryEntropy);  // uncertain source = wider attention
      return Math.exp(-(this.geoWeight * geoDist + entroPen - curiBonus) / temp);
    });
    const sum = scores.reduce((a,b)=>a+b,0) || 1;
    return scores.map(s => s/sum);
  }

  /** Full forward: tokenPositions on sphere, optional entropy per token. */
  forward(tokenPositions, entropies = null) {
    return tokenPositions.map((qi, i) => {
      const weights = this.attend(qi, tokenPositions,
        entropies?.[i] ?? 0, entropies);
      // Weighted combination with parallel transport
      const combined = [0,0,0];
      for (let j = 0; j < tokenPositions.length; j++) {
        const transported = this.manifold.parallelTransport(
          tokenPositions[j], tokenPositions[j], qi);
        combined.forEach((_, k) => { combined[k] += weights[j] * transported[k]; });
      }
      return this.manifold.expMap(qi, combined);
    });
  }

  /** Return which regions the attention is exploring vs exploiting. */
  diagnoseFog(tokenPositions, entropies) {
    const n = tokenPositions.length;
    const exploring = [], exploiting = [];
    for (let i = 0; i < n; i++) {
      const e = entropies?.[i] ?? 0;
      (e > 0.6 ? exploring : exploiting).push(i);
    }
    return { exploring: exploring.length, exploiting: exploiting.length,
             meanEntropy: (entropies?.reduce((s,v)=>s+v,0)??0)/n };
  }
}

// ─── EntropicOptimizer ────────────────────────────────────────────────────────
//
// SGD where learning rate scales with entropy:
//   high entropy → larger steps (explore aggressively)
//   low entropy  → smaller steps (fine-tune known territory)

export class EntropicOptimizer {
  constructor(weights, entropyField, opts = {}) {
    this._weights = weights;
    this._field   = entropyField;
    this._baseLr  = opts.lr ?? 0.001;
    this._step    = 0;
  }

  update(gradients) {
    this._step++;
    for (let i = 0; i < this._weights.length && i < gradients.length; i++) {
      const w   = this._weights[i];
      const g   = gradients[i];
      const ent = w instanceof EntropicGeodesicWeight ? w.meanEntropy() : 0.5;
      const lr  = this._baseLr * (1 + ent);   // high entropy = high LR
      w.values  = w.values.map((v, j) => v - lr * (g[j] ?? 0));
      // Update entropy based on gradient magnitude
      const gNorm = Math.sqrt(g.reduce((s,v)=>s+v*v,0));
      if (w instanceof EntropicGeodesicWeight) {
        const delta = Math.min(0.1, gNorm);
        w.entropy = w.entropy.map(e => Math.min(1, e + delta*(1-e)));
      }
    }
  }
}
