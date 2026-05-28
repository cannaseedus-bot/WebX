// SMGM-16 — JavaScript architecture descriptor and inference runtime
// Port/description of smgm16.py from v0.1.0-xvm-cpu-thread-cluster.
//
// Architecture:
//   52 CardSlot modules (field_u[16×192], field_s[192], field_v[192×768])
//   6 attention blocks with phase_scale + sigma
//   Maya embedding (15→192→768), Phase embedding (1→96→768)
//   15 engineered token features per position
//   Multi-loss: task(1.0) + balance(0.1) + stage_balance(0.05) + entropy(0.02)

export const SMGM16_CONFIG = Object.freeze({
  vocabSize:     1024,
  hiddenSize:    768,
  maxPositions:  1024,
  numLayers:     6,
  numCards:      52,
  mayaDim:       15,
  mayaHidden:    192,
  phaseHidden:   96,
  ffnIntermediate: 3072,
});

export const LOSS_LAMBDAS = Object.freeze({
  balance:      0.1,
  stageBalance: 0.05,
  stageEntropy: 0.02,
});

// Describes the shape of one CardSlot's learnable parameters.
export function cardSlotShape() {
  return {
    field_u:           [16, 192],
    field_s:           [192],
    field_v:           [192, 768],
    amplitude:         [4],
    gradient:          [4],
    curvature:         [1],
    pi_mod:            [1],
    adjacency_strength:[8],
  };
}

// Compute the 15 engineered token features for a single (tokenId, posId, piTime) triple.
// Returns a Float32Array of length 15.
export function tokenFeatures(tokenId, posId, piTime = 0, vocabSize = 1024, maxPositions = 1024) {
  const tokFrac = tokenId / Math.max(vocabSize - 1, 1);
  const posFrac = posId  / Math.max(maxPositions - 1, 1);
  const PI = Math.PI;
  return new Float32Array([
    tokFrac,
    tokFrac * tokFrac,
    Math.sqrt(tokFrac + 1e-6),
    Math.sin(PI * tokFrac),
    Math.cos(PI * tokFrac),
    Math.sin(2 * PI * tokFrac),
    Math.cos(2 * PI * tokFrac),
    posFrac,
    posFrac * posFrac,
    Math.sin(PI * posFrac),
    Math.cos(PI * posFrac),
    (tokenId % 3) / 2,
    (tokenId % 5) / 4,
    (tokenId % 7) / 6,
    ((tokenId + posId + piTime) % 2),
  ]);
}

// Dot product of two Float32Arrays.
function dot(a, b, n) {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

// Softmax in-place over Float32Array.
function softmax(a) {
  let max = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > max) max = a[i];
  let sum = 0;
  for (let i = 0; i < a.length; i++) { a[i] = Math.exp(a[i] - max); sum += a[i]; }
  for (let i = 0; i < a.length; i++) a[i] /= sum;
  return a;
}

// JavaScript-native forward pass for inference with pre-loaded weight tensors.
// Weights must be plain typed-array objects matching cardSlotShape().
//
// For production use, load from .scxq2 via the SCX runtime and pass tensors here.
export class SMGM16Runtime {
  constructor(weights = null) {
    this.weights = weights;
    this.piTime = 0;
    this.config = { ...SMGM16_CONFIG };
  }

  // Compute card gate scores for a pooled hidden vector [hiddenSize].
  // Returns Float32Array of length numCards (softmax-normalized).
  cardGates(pooledHidden, cardWeights) {
    const { numCards } = this.config;
    const scores = new Float32Array(numCards);
    for (let c = 0; c < numCards; c++) {
      const cw = cardWeights[c];
      // summary_v = mean of field_v rows (192×768 → 768)
      const vMean = new Float32Array(768);
      const fv = cw.field_v;                 // Float32Array [192*768]
      for (let row = 0; row < 192; row++) {
        for (let col = 0; col < 768; col++) vMean[col] += fv[row * 768 + col] / 192;
      }
      scores[c] = dot(pooledHidden, vMean, 768) + cw.scalarBias;
    }
    return softmax(scores);
  }

  // Weighted sum of card_v vectors by gate scores → context vector [hiddenSize].
  cardContext(gates, cardWeights) {
    const ctx = new Float32Array(this.config.hiddenSize);
    for (let c = 0; c < this.config.numCards; c++) {
      const cw = cardWeights[c];
      const fv = cw.field_v;
      const g  = gates[c];
      // Add gate-weighted mean of field_v rows
      for (let row = 0; row < 192; row++) {
        for (let col = 0; col < 768; col++) ctx[col] += g * fv[row * 768 + col] / 192;
      }
    }
    return ctx;
  }

  // Advance pi_time (called each training step: +0.05)
  stepPiTime(delta = 0.05) {
    this.piTime += delta;
  }
}

// Describes which shard files map to which model component.
// Used by the adapter-registry to wire .scxq2 shards.
export const SHARD_MAP = Object.freeze({
  foundation: 'final_3way.scxq2',              // 52.95M params, INT4, 23.9MB
  adapters: {
    agents:      'agents_adapter.scxq2',       // rank-8 LoRA, 7.8MB
    commands:    'commands_adapter.scxq2',
    micronauts:  'micronauts_adapter.scxq2',
    tools:       'tools_adapter.scxq2',
  },
});

// Parameter count reference (from training history).
export const PARAM_COUNTS = Object.freeze({
  foundation:  52_950_000,
  adapterRank: 8,
  adapterParams: 295_000,
  adapterExportMB: 7.8,
});

export default SMGM16Runtime;
