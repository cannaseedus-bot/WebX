// SCX-MoE Router — CPU JavaScript port (v3.1.0-scx-moe)
//
// Mirrors sxme_router.hlsl CS_Router (one thread per token, NO WaveOps).
// TypedArray-based — browser + Node compatible.
//
// Architecture: 8 experts, top-2 routing, dot-product gate scores + softmax.

export const MOE_NUM_EXPERTS    = 8;
export const MOE_NUM_LAYERS     = 8;
export const MOE_HIDDEN_SIZE    = 1024;
export const MOE_INTERMEDIATE   = 2816;  // SwiGLU ×2.75
export const MOE_NUM_HEADS      = 16;
export const MOE_HEAD_DIM       = 64;
export const MOE_MAX_SEQ        = 2048;
export const MOE_VOCAB_SIZE     = 32000;
export const MOE_TOP_K          = 2;

// routerGate: Float32Array [numExperts × hiddenSize], row-major
// hiddenStates: Float32Array [batchSize × hiddenSize]
// Returns { expertIds: Uint32Array [batchSize × topK], expertWeights: Float32Array [batchSize × topK] }
export function routeTopK(hiddenStates, routerGate, opts = {}) {
  const batchSize  = opts.batchSize  || (hiddenStates.length / MOE_HIDDEN_SIZE | 0);
  const hiddenSize = opts.hiddenSize || MOE_HIDDEN_SIZE;
  const numExperts = opts.numExperts || MOE_NUM_EXPERTS;
  const topK       = opts.topK       || MOE_TOP_K;

  const expertIds     = new Uint32Array(batchSize * topK);
  const expertWeights = new Float32Array(batchSize * topK);

  for (let tok = 0; tok < batchSize; tok++) {
    const hidBase = tok * hiddenSize;
    const scores  = new Float32Array(numExperts);

    // Dot-product gate scores
    for (let e = 0; e < numExperts; e++) {
      const gateBase = e * hiddenSize;
      let dot = 0;
      for (let d = 0; d < hiddenSize; d++) {
        dot += hiddenStates[hidBase + d] * routerGate[gateBase + d];
      }
      scores[e] = dot;
    }

    // Top-K selection (serial, mirrors shader)
    const selId    = new Uint32Array(topK);
    const selScore = new Float32Array(topK);

    for (let k = 0; k < topK; k++) {
      let best = -1e30, bestE = 0;
      for (let e = 0; e < numExperts; e++) {
        let already = false;
        for (let j = 0; j < k; j++) if (selId[j] === e) { already = true; break; }
        if (!already && scores[e] > best) { best = scores[e]; bestE = e; }
      }
      selId[k] = bestE; selScore[k] = best;
    }

    // Softmax over top-K
    let maxS = selScore[0];
    for (let k = 1; k < topK; k++) if (selScore[k] > maxS) maxS = selScore[k];
    let sumExp = 0;
    for (let k = 0; k < topK; k++) sumExp += Math.exp(selScore[k] - maxS);

    for (let k = 0; k < topK; k++) {
      const outIdx = tok * topK + k;
      expertIds[outIdx]     = selId[k];
      expertWeights[outIdx] = Math.exp(selScore[k] - maxS) / (sumExp + 1e-9);
    }
  }

  return { expertIds, expertWeights };
}

// Convenience: route a single token hidden state
export function routeToken(hidden, routerGate, numExperts = MOE_NUM_EXPERTS, topK = MOE_TOP_K) {
  const { expertIds, expertWeights } = routeTopK(hidden, routerGate, {
    batchSize: 1, hiddenSize: hidden.length, numExperts, topK,
  });
  return {
    experts: Array.from(expertIds),
    weights: Array.from(expertWeights),
  };
}

export const ROUTER_PARAMS_SCHEMA = Object.freeze({
  batchSize:   'uint32',
  hiddenSize:  'uint32',
  numExperts:  'uint32',
  topK:        'uint32',
});
