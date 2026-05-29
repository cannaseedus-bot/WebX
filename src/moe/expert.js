// SCX-MoE Expert Forward + Adapter Injection — CPU JS port (v3.1.0-scx-moe)
//
// Mirrors sxme_expert.hlsl CS_ExpertForward + AdapterScale().
// SwiGLU FFN: gate = silu(Wg·x), up = Wu·x, out = Wd·(gate*up)
// Expert 2 (amplify) reads addon adapter tensors: stride-4 mean-pool → cosine → scale.
//
// Root signature registers (informational — for GPU binding reference):
//   b0: ExpertParams, b1: AdapterParams
//   t0: hiddenIn, t1: expertIds, t2: expertWeights
//   t3: wGate, t4: wUp, t5: wDown
//   t6: adapterRouteEmb [adapterN × 256], t7: adapterRouteBias [adapterN]
//   u0: hiddenOut

export const AMPLIFY_EXPERT_ID = 2;

function silu(x) { return x / (1 + Math.exp(-x)); }

// AdapterScale: stride-4 mean-pool → cosine match → 1 + saturate(score)*bias*strength
// Matches AdapterScale() in sxme_expert.hlsl exactly.
function adapterScale(hidden, hiddenSize, adapterRouteEmb, adapterRouteBias, adapterN, adapterStrength) {
  if (adapterN === 0) return 1.0;

  const stride = Math.max((hiddenSize / 256) | 0, 1);
  const proj = new Float32Array(256);
  let pnorm = 0;
  for (let d = 0; d < 256; d++) {
    let s = 0;
    const base = d * stride;
    for (let k = 0; k < stride; k++) s += hidden[base + k] || 0;
    proj[d] = s / stride;
    pnorm += proj[d] * proj[d];
  }
  const invPnorm = 1 / Math.sqrt(Math.max(pnorm, 1e-8));

  let bestScore = -1e9, bestIdx = 0;
  for (let r = 0; r < adapterN; r++) {
    let dot = 0;
    const rBase = r * 256;
    for (let d = 0; d < 256; d++) dot += proj[d] * invPnorm * adapterRouteEmb[rBase + d];
    if (dot > bestScore) { bestScore = dot; bestIdx = r; }
  }

  const bias = adapterRouteBias[bestIdx];
  return 1 + Math.min(Math.max(bestScore, 0), 1) * bias * adapterStrength;
}

// expertForward: compute SwiGLU FFN output for one expert slot
// hidden: Float32Array [hiddenSize]
// wGate, wUp, wDown: Float32Array [hiddenSize × intermediateSize] (row-major)
// Returns Float32Array [hiddenSize]
export function expertForward(hidden, wGate, wUp, wDown, hiddenSize, intermediateSize) {
  // gate proj: g[i] = sum_d(hidden[d] * wGate[i*hiddenSize + d])
  const gate = new Float32Array(intermediateSize);
  const up   = new Float32Array(intermediateSize);
  for (let i = 0; i < intermediateSize; i++) {
    let g = 0, u = 0;
    for (let d = 0; d < hiddenSize; d++) {
      const idx = i * hiddenSize + d;
      g += hidden[d] * wGate[idx];
      u += hidden[d] * wUp[idx];
    }
    gate[i] = g; up[i] = u;
  }
  // SwiGLU: act[i] = silu(gate[i]) * up[i]
  const act = new Float32Array(intermediateSize);
  for (let i = 0; i < intermediateSize; i++) act[i] = silu(gate[i]) * up[i];
  // down proj
  const out = new Float32Array(hiddenSize);
  for (let d = 0; d < hiddenSize; d++) {
    let s = 0;
    for (let i = 0; i < intermediateSize; i++) s += act[i] * wDown[i * hiddenSize + d];
    out[d] = s;
  }
  return out;
}

// expertForwardBatch: process all token×slot pairs (CPU equivalent of CS_ExpertForward)
// hiddenIn: Float32Array [batchSize × hiddenSize]
// expertIds, expertWeights: outputs from routeTopK
// weights: {wGate, wUp, wDown} per expert [numExperts × 8 layers × hiddenSize × intermediateSize]
// adapter: {routeEmb, routeBias, adapterN, strength} — optional, for Expert 2
export function expertForwardBatch(hiddenIn, expertIds, expertWeights, weights, adapter = null, opts = {}) {
  const batchSize        = opts.batchSize        || (hiddenIn.length / (opts.hiddenSize || 1024) | 0);
  const hiddenSize       = opts.hiddenSize       || 1024;
  const intermediateSize = opts.intermediateSize || 2816;
  const topK             = opts.topK             || 2;
  const layerIdx         = opts.layerIdx         || 0;
  const numExperts       = opts.numExperts       || 8;

  const hiddenOut = new Float32Array(batchSize * topK * hiddenSize);

  const adapterN        = adapter?.adapterN        || 0;
  const adapterStrength = adapter?.adapterStrength || 0.1;
  const adapterRouteEmb  = adapter?.routeEmb  || new Float32Array(0);
  const adapterRouteBias = adapter?.routeBias || new Float32Array(0);

  for (let tok = 0; tok < batchSize; tok++) {
    const tokBase = tok * hiddenSize;
    const hidden  = hiddenIn.subarray(tokBase, tokBase + hiddenSize);

    for (let slot = 0; slot < topK; slot++) {
      const expertId = expertIds[tok * topK + slot];
      const w        = expertWeights[tok * topK + slot];

      const { wGate, wUp, wDown } = weights[expertId][layerIdx];
      let out = expertForward(hidden, wGate, wUp, wDown, hiddenSize, intermediateSize);

      // Expert 2 adapter injection
      if (expertId === AMPLIFY_EXPERT_ID && adapterN > 0) {
        const scale = adapterScale(hidden, hiddenSize, adapterRouteEmb, adapterRouteBias, adapterN, adapterStrength);
        for (let d = 0; d < hiddenSize; d++) out[d] *= scale;
      }

      const outBase = (tok * topK + slot) * hiddenSize;
      for (let d = 0; d < hiddenSize; d++) hiddenOut[outBase + d] = out[d] * w;
    }
  }

  return hiddenOut;
}

// expertReduce: sum top-K slot contributions per token
export function expertReduce(hiddenOut, batchSize, topK, hiddenSize) {
  const reduced = new Float32Array(batchSize * hiddenSize);
  for (let tok = 0; tok < batchSize; tok++) {
    for (let slot = 0; slot < topK; slot++) {
      const slotBase = (tok * topK + slot) * hiddenSize;
      const outBase  = tok * hiddenSize;
      for (let d = 0; d < hiddenSize; d++) reduced[outBase + d] += hiddenOut[slotBase + d];
    }
  }
  return reduced;
}

export const EXPERT_PARAMS_SCHEMA = Object.freeze({
  batchSize:        'uint32',
  hiddenSize:       'uint32',
  intermediateSize: 'uint32',
  topK:             'uint32',
  numExperts:       'uint32',
  layerIdx:         'uint32',
});
