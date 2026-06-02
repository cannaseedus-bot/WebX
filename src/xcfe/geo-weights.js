// geo-weights.js — Geodesic Weights: tensors that live on a curved manifold
//
// A geodesic weight is NOT a flat matrix.
// It is a tensor that:
//   1. Lives AT a specific point on the manifold (base_point)
//   2. Has values in the TANGENT SPACE at that point
//   3. Must be PARALLEL TRANSPORTED when applied at a different point
//
// This is what makes the spherical map a usable coordinate system:
//   Without geodesic weights → just fancy coordinates, Euclidean math underneath
//   With geodesic weights    → the weights KNOW where they live and HOW to move
//
// The map provides:
//   position     = WHERE concepts live (sphere point)
//   geodesic     = HOW FAR apart they are (great-circle arc)
//   parallel transport = HOW meaning shifts as you travel (tangent rotation)
//   metric tensor = LOCAL geometry (curvature of space around each point)
//
// K'UHUL connection:
//   Pop    = GeodesicWeight.constructor() — weight born at base_point
//   Wo     = weight.applyAt(point) — transport to new location
//   Sek    = transported weight applied to input
//   Ch'en  = gradient flows back along geodesic (update tangent_values)
//   Xul    = base_point optionally moves along gradient
//
// Connection to spherical_map_compiler.hlsl:
//   The iGPU compiles the geodesicCache (distances + expmap)
//   GeodesicWeight.parallelTransportTo() reads from that cache
//   Zero-copy: same buffer, no serialisation

import { Manifold, GeoTensor } from './geo-ir.js';
import { dot, norm, cross3, normalize } from './atomic-brain.js';

// ─── SphericalManifold ────────────────────────────────────────────────────────

export class SphericalManifold {
  constructor(curvature = 0.1) {
    this.curvature = curvature;
    this.radius    = 1 / Math.sqrt(Math.abs(curvature));
    this.type      = 'spherical';
  }

  /** Project a flat vector onto the sphere surface. */
  project(v) {
    const n = norm(v) || 1;
    return v.map(x => x / n * this.radius);
  }

  /** Geodesic distance along great circle. */
  geodesicDistance(p, q) {
    const d = dot(p, q) / (this.radius * this.radius);
    return this.radius * Math.acos(Math.min(1, Math.max(-1, d)));
  }

  /** Exponential map: exp_p(v) = move from p in direction v by ||v||. */
  expMap(p, v) {
    const nv = norm(v);
    if (nv < 1e-9) return [...p];
    const c = Math.cos(nv / this.radius), s = Math.sin(nv / this.radius);
    return p.map((pi, i) => c * pi + s * v[i] / nv);
  }

  /** Logarithmic map: log_p(q) = direction + distance in tangent space at p. */
  logMap(p, q) {
    const dist = this.geodesicDistance(p, q);
    if (dist < 1e-9) return p.map(() => 0);
    // Direction: project q onto tangent space at p, then scale by dist
    const pDotQ = dot(p, q);
    const tang  = q.map((qi, i) => qi - (pDotQ / (this.radius * this.radius)) * p[i]);
    const nt    = norm(tang) || 1;
    return tang.map(t => t / nt * dist);
  }

  /** Parallel transport vector v from point `from` to `to`. */
  parallelTransport(v, from, to) {
    const axis  = cross3(from, to);
    const al    = norm(axis);
    if (al < 1e-9) return [...v];
    const aHat  = axis.map(x => x / al);
    const angle = this.geodesicDistance(from, to) / this.radius;
    return this._rotateVector(v, aHat, angle);
  }

  /** Metric tensor at point p (returns 3×3 flattened array, 9 elements). */
  metricAt(p) {
    const theta = Math.acos(Math.min(1, Math.max(-1, p[2] / this.radius)));
    const sinT  = Math.sin(theta);
    const R2    = this.radius * this.radius;
    // g = R² diag(1, sin²θ, sin²θ)
    return [R2,0,0, 0,R2*sinT*sinT,0, 0,0,R2*sinT*sinT];
  }

  randomPoint() {
    const theta = Math.acos(2*Math.random()-1);
    const phi   = 2*Math.PI*Math.random();
    return [
      this.radius * Math.sin(theta) * Math.cos(phi),
      this.radius * Math.sin(theta) * Math.sin(phi),
      this.radius * Math.cos(theta),
    ];
  }

  _rotateVector(v, axis, angle) {
    const c = Math.cos(angle), s = Math.sin(angle);
    const d = dot(v, axis);
    const cr = cross3(axis, v);
    return v.map((vi, i) => c*vi + s*cr[i] + (1-c)*d*axis[i]);
  }
}

// ─── GeodesicWeight ───────────────────────────────────────────────────────────

export class GeodesicWeight {
  /**
   * A weight tensor that lives at a specific point on a manifold.
   * @param {SphericalManifold} manifold
   * @param {number[]}          basePoint   coords on manifold
   * @param {number[]}          values      flat array (the actual weight values)
   * @param {object}            opts
   */
  constructor(manifold, basePoint, values, opts = {}) {
    this.manifold    = manifold;
    this.basePoint   = [...basePoint];
    this.values      = [...values];
    this.shape       = opts.shape ?? [Math.sqrt(values.length)|0, Math.sqrt(values.length)|0];
    this.learnBase   = opts.learnBase ?? false;
    this.id          = opts.id ?? `gw_${Date.now()}_${Math.random().toString(36).slice(2,6)}`;
    this._gradValues = null;
    this._gradBase   = null;
  }

  /**
   * Transport this weight from basePoint to targetPoint.
   * Returns a new flat array of the same shape, valid in targetPoint's tangent space.
   */
  parallelTransportTo(targetPoint) {
    const rows = this.shape[0], cols = this.shape[1];
    const transported = new Array(rows * cols);
    // Each column vector of the weight matrix is transported independently
    for (let c = 0; c < cols; c++) {
      const col = this.values.filter((_, i) => i % cols === c);
      const tCol = this.manifold.parallelTransport(col, this.basePoint, targetPoint);
      tCol.forEach((v, r) => { transported[r * cols + c] = v; });
    }
    return transported;
  }

  /**
   * Apply this weight at targetPoint to input vector.
   * 1. Transport weight to targetPoint
   * 2. Matrix-multiply transported weight × input
   * 3. Apply exp-map to project output back to manifold
   */
  applyAt(targetPoint, input) {
    const W      = this.parallelTransportTo(targetPoint);
    const rows   = this.shape[0], cols = this.shape[1];
    // W × input (flat matmul)
    const output = new Array(rows).fill(0);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        output[r] += W[r * cols + c] * (input[c] ?? 0);
    // Project result to tangent space at targetPoint, then exp-map to sphere
    return this.manifold.expMap(targetPoint, output.slice(0, 3)); // simplified 3D output
  }

  /** Accumulate gradient for update step. */
  accumulateGrad(gradValues, gradBase = null) {
    if (!this._gradValues) this._gradValues = new Array(this.values.length).fill(0);
    gradValues.forEach((g, i) => { this._gradValues[i] += g; });
    if (gradBase && this.learnBase) {
      if (!this._gradBase) this._gradBase = new Array(this.basePoint.length).fill(0);
      gradBase.forEach((g, i) => { this._gradBase[i] += g; });
    }
  }

  /** Riemannian SGD update. */
  update(lr = 0.001) {
    if (this._gradValues) {
      this.values = this.values.map((v, i) => v - lr * (this._gradValues[i] ?? 0));
      this._gradValues = null;
    }
    if (this._gradBase && this.learnBase) {
      // Move base point along sphere in gradient direction
      const step = this._gradBase.map(g => -lr * g);
      this.basePoint = this.manifold.expMap(this.basePoint, step);
      this._gradBase = null;
    }
  }

  snapshot() {
    return { id: this.id, basePoint: this.basePoint, shape: this.shape,
             manifold: this.manifold.type, curvature: this.manifold.curvature };
  }
}

// ─── GeodesicAttention ───────────────────────────────────────────────────────
//
// Attention that uses geodesic distance instead of dot product.
// Each token lives at a position on the semantic sphere.
// Attention weight = exp(-geodesic_dist / radius)

export class GeodesicAttention {
  constructor(dim, numHeads = 1, curvature = 0.1) {
    this.dim      = dim;
    this.numHeads = numHeads;
    this.manifold = new SphericalManifold(curvature);
    const size    = dim * dim;

    this.W_q = new GeodesicWeight(this.manifold, this.manifold.randomPoint(),
                                  new Array(size).fill(0).map(() => (Math.random()-0.5)/Math.sqrt(dim)),
                                  { id:'W_q', shape:[dim,dim] });
    this.W_k = new GeodesicWeight(this.manifold, this.manifold.randomPoint(),
                                  new Array(size).fill(0).map(() => (Math.random()-0.5)/Math.sqrt(dim)),
                                  { id:'W_k', shape:[dim,dim] });
    this.W_v = new GeodesicWeight(this.manifold, this.manifold.randomPoint(),
                                  new Array(size).fill(0).map(() => (Math.random()-0.5)/Math.sqrt(dim)),
                                  { id:'W_v', shape:[dim,dim] });
  }

  /** Compute attention for one query token against all key tokens.
   *  positions: array of sphere coords, one per token.
   *  Returns attention weights (softmax over geodesic distances). */
  attend(queryPos, keyPositions) {
    const scores = keyPositions.map(kp =>
      Math.exp(-this.manifold.geodesicDistance(queryPos, kp) / this.manifold.radius)
    );
    const sum = scores.reduce((a,b) => a+b, 0) || 1;
    return scores.map(s => s / sum);
  }

  /** Full forward pass: positions[i] = sphere location of token i.
   *  Returns output positions on sphere (one per token). */
  forward(tokenPositions) {
    const n = tokenPositions.length;
    const out = [];

    for (let i = 0; i < n; i++) {
      const qi = this.W_q.applyAt(tokenPositions[i], tokenPositions[i]);
      const weights = tokenPositions.map((kp, j) => {
        const kj = this.W_k.applyAt(kp, kp);
        return Math.exp(-this.manifold.geodesicDistance(qi, kj) / this.manifold.radius);
      });
      const sumW = weights.reduce((a,b)=>a+b,0)||1;
      const normW = weights.map(w=>w/sumW);

      // Weighted combination with parallel transport
      let combined = [0,0,0];
      for (let j = 0; j < n; j++) {
        const vj = this.W_v.applyAt(tokenPositions[j], tokenPositions[j]);
        const transported = this.manifold.parallelTransport(vj, tokenPositions[j], tokenPositions[i]);
        combined = combined.map((c, k) => c + normW[j] * transported[k]);
      }
      out.push(this.manifold.expMap(tokenPositions[i], combined));
    }
    return out;
  }

  weights() { return [this.W_q, this.W_k, this.W_v]; }
}

// ─── Map summary ──────────────────────────────────────────────────────────────
//
// The spherical map compiled by iGPU gives you:
//   distances[i][j]  = geodesic_dist(token_i, token_j)   → WHERE each concept is
//   expMap[i][dir]   = exp_p(tangent_dir)                 → HOW to move from i
//   metric[i]        = g_ij at point i                    → LOCAL curvature
//   transport[i→j]   = parallel_transport(v, i, j)        → HOW meaning shifts
//
// Together: a complete semantic coordinate system.
// GeodesicWeights read this map. They know:
//   - where they live (base_point)
//   - how far from everything else (geodesic_distance)
//   - how to move (parallel_transport)
//
// Without the map: Euclidean math on curved coordinates (wrong)
// With the map:    True geometric intelligence (respects curvature)

export function mapSummary(manifold, n = 8) {
  const pts = Array.from({ length: n }, () => manifold.randomPoint());
  const dists = pts.map((p,i) => pts.map((q,j) => i===j ? 0 : manifold.geodesicDistance(p,q)));
  const metrics = pts.map(p => manifold.metricAt(p));
  return {
    manifold: { type: manifold.type, curvature: manifold.curvature, radius: manifold.radius },
    sample_points: pts,
    distance_matrix: dists,
    metric_tensors: metrics,
    description: 'Spherical semantic map: each point is a concept location, distances are semantic distances, metric gives local curvature',
  };
}
