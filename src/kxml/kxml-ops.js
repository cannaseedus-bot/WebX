// kxml-ops.js — XCFE @ops[] dispatcher for KXML nodes
//
// Each op reads from / writes to the graph-scoped buffer map.
// Geometric ops (geodesic, parallel transport, Ricci flow) delegate to
// existing linalg.js and mayan-linalg.js where possible.
//
// Op contract: fn(op, buffers) → void (writes to buffers[op.dst])

import {
  matMul, matTranspose, matVecMul,
  vectorAdd, vectorScale, dotProduct,
  vectorNorm, vectorNormalize, softmaxInPlace,
  scaledDotAttention, tensorContract,
} from '../linalg/linalg.js';

// ─── Activation functions (all Lipschitz ≤ 1) ────────────────────────────────

const ACTIVATIONS = {
  relu:     x => Math.max(0, x),
  tanh:     x => Math.tanh(x),
  sigmoid:  x => 1 / (1 + Math.exp(-x)),
  gelu:     x => 0.5 * x * (1 + Math.tanh(Math.SQRT2 / Math.sqrt(Math.PI) * (x + 0.044715 * x ** 3))),
  softplus: x => Math.log(1 + Math.exp(x)),
  identity: x => x,
};

function applyActivation(kind, arr) {
  const fn = ACTIVATIONS[kind] ?? ACTIVATIONS.identity;
  const out = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = fn(arr[i]);
  return out;
}

// ─── Geodesic distance on a manifold with given curvature ────────────────────
//   κ > 0 → spherical,  κ < 0 → hyperbolic,  κ = 0 → Euclidean

export function geodesicDist(a, b, curvature = 0) {
  let sq = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) sq += (a[i] - b[i]) ** 2;
  const d = Math.sqrt(sq);
  if (curvature > 0) {
    const R = 1 / Math.sqrt(curvature);
    return R * Math.asin(Math.min(d / R, 1));
  }
  if (curvature < 0) {
    const R = 1 / Math.sqrt(-curvature);
    return R * Math.asinh(d / R);
  }
  return d;
}

// Parallel transport: for flat manifold → identity; for curved → scale by exp(-L*curvature)
export function parallelTransport(vec, length, curvature = 0) {
  const scale = Math.exp(-length * Math.abs(curvature));
  return vectorScale(vec instanceof Float64Array ? vec : new Float64Array(vec), scale);
}

// One step of Ricci flow: dg/dt = -2 Ric(g)  approximated as scaling
// For a 1D metric sequence: g_new = g - 2*κ*dt
export function ricciFlowStep(metric, curvature = 0.1, dt = 0.01) {
  const out = new Float64Array(metric.length);
  for (let i = 0; i < metric.length; i++) {
    out[i] = Math.max(0.01, metric[i] - 2 * curvature * dt);
  }
  return out;
}

// ─── Geometric attention ──────────────────────────────────────────────────────
// For each query position, compute exp(-geodesic_dist / radius) to each key,
// softmax-normalise, then weighted sum of values.
// positions: Float64Array[n*dim], values: Float64Array[n*dim]

export function geometricAttention(queryPositions, keyPositions, values, dim, curvature = 0, radius = 1) {
  const n = queryPositions.length / dim;
  const out = new Float64Array(n * dim);

  for (let i = 0; i < n; i++) {
    const q = queryPositions.subarray(i * dim, (i + 1) * dim);
    const weights = new Float64Array(n);
    let sumW = 0;

    for (let j = 0; j < n; j++) {
      const k = keyPositions.subarray(j * dim, (j + 1) * dim);
      const d = geodesicDist(q, k, curvature);
      weights[j] = Math.exp(-d / radius);
      sumW += weights[j];
    }
    if (sumW < 1e-12) sumW = 1;

    for (let j = 0; j < n; j++) {
      const v = values.subarray(j * dim, (j + 1) * dim);
      const w = weights[j] / sumW;
      for (let d = 0; d < dim; d++) out[i * dim + d] += w * v[d];
    }
  }
  return out;
}

// ─── Fold compression ─────────────────────────────────────────────────────────
// Keep top-k rows by L2 norm (attention weight proxy), return compressed tensor.

export function foldCompress(data, dim, ratio = 0.3) {
  const n = data.length / dim;
  const k = Math.max(1, Math.round(n * ratio));
  const norms = [];
  for (let i = 0; i < n; i++) {
    let sq = 0;
    for (let d = 0; d < dim; d++) sq += data[i * dim + d] ** 2;
    norms.push({ i, norm: Math.sqrt(sq) });
  }
  norms.sort((a, b) => b.norm - a.norm);
  const topK = norms.slice(0, k).sort((a, b) => a.i - b.i);
  const out = new Float64Array(k * dim);
  topK.forEach(({ i }, idx) => {
    for (let d = 0; d < dim; d++) out[idx * dim + d] = data[i * dim + d];
  });
  return out;
}

// ─── Cross-entropy loss ───────────────────────────────────────────────────────

export function crossEntropy(logits, targetIdx) {
  const max = logits.reduce((m, v) => Math.max(m, v), -Infinity);
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) sumExp += Math.exp(logits[i] - max);
  return Math.log(sumExp) + max - logits[targetIdx];
}

// ─── Op dispatch table ────────────────────────────────────────────────────────

function toF64(v) {
  if (v instanceof Float64Array) return v;
  if (Array.isArray(v)) return new Float64Array(v);
  if (typeof v === 'number') return new Float64Array([v]);
  return new Float64Array(0);
}

export const OPS = {

  '@load': (op, buf) => {
    const val = buf.get(op.src);
    if (val !== undefined) buf.set(op.dst, val);
  },

  '@store': (op, buf) => {
    const val = buf.get(op.src);
    if (val !== undefined) buf.set(op.dst, val);
  },

  '@input': (op, buf) => {
    if (!buf.has(op.name)) buf.set(op.name, new Float64Array(
      op.shape ? parseInt(op.shape.replace(/[\[\]]/g, '').split(',')[0], 10) : 1
    ));
  },

  '@add': (op, buf) => {
    const a = toF64(buf.get(op.src1 ?? op.src));
    const b = toF64(buf.get(op.src2 ?? op.src));
    buf.set(op.dst, vectorAdd(a, b));
  },

  '@mul': (op, buf) => {
    const a = toF64(buf.get(op.src1 ?? op.src));
    const b = buf.get(op.src2 ?? op.src);
    if (typeof b === 'number') {
      buf.set(op.dst, vectorScale(a, b));
    } else {
      const n = Math.sqrt(a.length) | 0;
      buf.set(op.dst, matMul(a, toF64(b), n, n, n));
    }
  },

  '@gemm': (op, buf) => {
    const A = toF64(buf.get(op.src1));
    const B = toF64(buf.get(op.src2));
    const rows = parseInt(op.rows ?? Math.sqrt(A.length), 10);
    const inner = parseInt(op.inner ?? rows, 10);
    const cols  = parseInt(op.cols ?? Math.sqrt(B.length), 10);
    buf.set(op.dst, matMul(A, B, rows, inner, cols));
  },

  '@linear': (op, buf) => {
    const x = toF64(buf.get(op.src));
    const W = toF64(buf.get(op.weight ?? (op.src + '_weight')));
    const rows = W.length / x.length | 0 || 1;
    buf.set(op.dst, matVecMul(W, x, rows, x.length));
  },

  '@scale': (op, buf) => {
    const src = toF64(buf.get(op.src));
    const factor = parseFloat(op.factor ?? '1');
    buf.set(op.dst, vectorScale(src, factor));
  },

  '@activation': (op, buf) => {
    const src = toF64(buf.get(op.src));
    const kind = op.kind ?? 'relu';
    if (kind === 'softmax') {
      const out = Float64Array.from(src);
      softmaxInPlace(out);
      buf.set(op.dst, out);
    } else {
      buf.set(op.dst, applyActivation(kind, src));
    }
  },

  '@softmax': (op, buf) => {
    const src = toF64(buf.get(op.src));
    const out = Float64Array.from(src);
    softmaxInPlace(out);
    buf.set(op.dst, out);
  },

  '@gelu': (op, buf) => {
    buf.set(op.dst, applyActivation('gelu', toF64(buf.get(op.src))));
  },

  '@geodesic_distance': (op, buf) => {
    const a = toF64(buf.get(op.src1 ?? op.src));
    const b = toF64(buf.get(op.src2));
    const curvature = parseFloat(op.curvature ?? '0');
    buf.set(op.dst, geodesicDist(a, b, curvature));
  },

  '@geometric_attention': (op, buf) => {
    const src  = toF64(buf.get(op.src ?? op.src1));
    const dim  = parseInt(op.dim ?? Math.sqrt(src.length), 10);
    const k    = buf.has(op.keys)   ? toF64(buf.get(op.keys))   : src;
    const v    = buf.has(op.values) ? toF64(buf.get(op.values)) : src;
    const curve = parseFloat(op.curvature ?? '0');
    const radius = parseFloat(op.radius ?? '1');
    buf.set(op.dst, geometricAttention(src, k, v, dim, curve, radius));
  },

  '@parallel_transport': (op, buf) => {
    const vec  = toF64(buf.get(op.src));
    const len  = parseFloat(op.length ?? '1');
    const curve = parseFloat(op.curvature ?? '0');
    buf.set(op.dst, parallelTransport(vec, len, curve));
  },

  '@ricci_flow': (op, buf) => {
    const metric   = toF64(buf.get(op.src));
    const steps    = parseInt(op.steps ?? '10', 10);
    const curve    = parseFloat(op.curvature ?? '0.1');
    let m = metric;
    for (let s = 0; s < steps; s++) m = ricciFlowStep(m, curve);
    buf.set(op.dst, m);
  },

  '@fold_compress': (op, buf) => {
    const data  = toF64(buf.get(op.src));
    const ratio = parseFloat(op.ratio ?? '0.3');
    const dim   = parseInt(op.dim ?? Math.sqrt(data.length), 10);
    buf.set(op.dst, foldCompress(data, dim, ratio));
  },

  '@loss': (op, buf) => {
    const pred   = toF64(buf.get(op.src1 ?? op.src));
    const target = buf.get(op.src2);
    if (typeof target === 'number') {
      buf.set(op.dst, crossEntropy(pred, target));
    } else {
      // MSE fallback
      const t = toF64(target ?? new Float64Array(pred.length));
      let mse = 0;
      for (let i = 0; i < pred.length; i++) mse += (pred[i] - t[i]) ** 2;
      buf.set(op.dst, mse / pred.length);
    }
  },

  '@metric': (op, buf) => {
    // Compute outer product as metric tensor approximation
    const x = toF64(buf.get(op.src));
    const n = x.length;
    const g = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        g[i * n + j] = i === j ? x[i] * x[i] + 1e-6 : 0;  // diagonal metric
    buf.set(op.dst, g);
  },

  '@combine_attention': (op, buf) => {
    const a = toF64(buf.get(op.src1));
    const b = toF64(buf.get(op.src2));
    const n = Math.min(a.length, b.length);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = (a[i] + b[i]) * 0.5;
    buf.set(op.dst, out);
  },

  '@create_folds': (op, buf) => {
    const data = toF64(buf.get(op.src));
    const half = data.length >> 1;
    buf.set(op.dst,        data.slice(0, half));
    buf.set(op.micro_dst ?? (op.dst + '_micro'), data.slice(half));
  },

  '@broadcast_to_micro': (op, buf) => {
    buf.set(op.dst, buf.get(op.src));
  },

  '@tangent_projection': (op, buf) => {
    const v  = toF64(buf.get(op.src));
    const base = toF64(buf.get(op.base ?? op.src));
    // P_x(v) = v - (x·v / ||x||^2) x
    const norm2 = dotProduct(base, base);
    if (norm2 < 1e-12) { buf.set(op.dst, v); return; }
    const scale = dotProduct(base, v) / norm2;
    const proj  = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) proj[i] = v[i] - scale * base[i];
    buf.set(op.dst, proj);
  },

  '@project_to_manifold': (op, buf) => {
    // Simple L2 normalisation onto unit sphere
    buf.set(op.dst, vectorNormalize(toF64(buf.get(op.src))));
  },

  '@geodesic_positional': (op, buf) => {
    // Add positional offset using geodesic distance from origin
    const x = toF64(buf.get(op.src));
    const positions = buf.get(op.positions);
    if (!positions) { buf.set(op.dst, x); return; }
    const n = Array.isArray(positions) ? positions.length : positions.length;
    const out = new Float64Array(x.length);
    const step = x.length / n;
    for (let i = 0; i < n; i++) {
      const d = geodesicDist([i], [0]);
      for (let j = 0; j < step; j++) out[i * step + j] = x[i * step + j] + d * 0.01;
    }
    buf.set(op.dst, out);
  },

  '@load_shard': (op, buf) => {
    // Stub — shard loading is handled by the KXMLGraph shard registry
    if (!buf.has(op.dst)) buf.set(op.dst, new Float64Array(0));
  },

  '@lookup': (op, buf) => {
    const table  = buf.get(op.table);
    const index  = buf.get(op.index);
    if (table && index != null) {
      const idx = typeof index === 'number' ? index : (index[0] | 0);
      const dim = Math.sqrt(table.length) | 0;
      buf.set(op.dst, toF64(table).slice(idx * dim, (idx + 1) * dim));
    } else {
      buf.set(op.dst, new Float64Array(0));
    }
  },

  '@exp_map': (op, buf) => {
    // Exponential map at base point: approximate as normalisation + scaling
    const base  = toF64(buf.get(op.base ?? op.src));
    const vec   = toF64(buf.get(op.src));
    const out   = new Float64Array(vec.length);
    const r     = vectorNorm(vec);
    if (r < 1e-12) { buf.set(op.dst, base); return; }
    for (let i = 0; i < vec.length; i++) out[i] = base[i] + Math.sin(r) / r * vec[i];
    buf.set(op.dst, out);
  },

  '@log_map': (op, buf) => {
    // Logarithmic map: approximate inverse of exp_map
    const base = toF64(buf.get(op.base ?? op.src));
    const x    = toF64(buf.get(op.src));
    const diff = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) diff[i] = x[i] - base[i];
    buf.set(op.dst, diff);
  },
};

// Dispatch a single op against the buffer map.
export function dispatchOp(op, buffers) {
  const fn = OPS[op.type];
  if (fn) { fn(op, buffers); return true; }
  return false;
}
