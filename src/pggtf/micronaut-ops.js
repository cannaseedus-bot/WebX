// micronaut-ops.js — Micronaut Operators M_i: F → F'
//
// Micronauts are local tensor processors operating on the fold field.
// Each operator takes (F, G, P, Π) and returns a delta ΔF or mutates in-place.
//
// Four canonical operators from the PGGTF spec:
//   M_memory(F)     — updates memory lanes
//   M_route(G)      — mutates geodesic graph
//   M_compress(F)   — produces SCXQ2 lanes (Fibonacci fold)
//   M_infer(F,G,P)  — produces next fold prediction
//
// Layer 1→2 SafeTensors progression:
//   Operators accept projections from safetensors weights (flat Float64Array)
//   via FoldTensor.projectFromFlat() before operating.

import { SCXQ2_LANES } from './fold-tensor.js';
import { PHASE_INDEX }  from './phase-tensor.js';

// ─── Memory Micronaut ─────────────────────────────────────────────────────────
// M_memory(F): update MEMORY lane with exponential-decay write-through.
// Implements episodic replay by accumulating weighted history.

export function memoryMicronaut(F, activeFolds, opts = {}) {
  const decay  = opts.decay  ?? 0.9;    // retention factor
  const lane   = SCXQ2_LANES.MEMORY;
  const C      = F.n_channels;
  const deltaF = new Float64Array(F.n_folds * F.n_lanes * C);

  for (const f of activeFolds) {
    const cur  = F.lane(f, lane);
    const base = (f * F.n_lanes + lane) * C;
    // Decay existing memory and write current TENSOR lane as new entry
    const tensor = F.lane(f, SCXQ2_LANES.TENSOR);
    for (let c = 0; c < C; c++)
      deltaF[base + c] = (decay - 1) * cur[c] + (1 - decay) * tensor[c];
  }

  return { type: "memory", deltaF, activeFolds };
}

// ─── Routing Micronaut ────────────────────────────────────────────────────────
// M_route(G): update geodesic weights based on PHASE and GEO lanes.
// Adds edges where GEO lane shows strong correlation.

export function routingMicronaut(F, G, activeFolds, opts = {}) {
  const threshold = opts.threshold ?? 0.5;
  const additions = [];  // {i, j, weight} pairs for G.addEdge

  for (const f of activeFolds) {
    const geo_lane = F.lane(f, SCXQ2_LANES.GEO);
    // Find channels with high activation as target fold suggestions
    for (let c = 0; c < Math.min(geo_lane.length, F.n_folds); c++) {
      if (geo_lane[c] > threshold) {
        const target_fold = c;
        const weight = 1.0 / (1.0 + geo_lane[c]);  // inverse: stronger signal = shorter dist
        additions.push({ i: f, j: target_fold, weight });
      }
    }
  }

  // Apply edge additions to G
  for (const { i, j, weight } of additions) {
    if (i !== j) G.addEdge(i, j, weight);
  }

  return { type: "route", edges_added: additions.length };
}

// ─── Compression Micronaut ────────────────────────────────────────────────────
// M_compress(F): Fibonacci-fold the TENSOR lane, write to DICT lane.
// SCXQ2 encoding: compress each active fold's tensor to ~N/phi values.

const PHI = 1.6180339887498948482;

export function compressionMicronaut(F, activeFolds, opts = {}) {
  const src_lane  = SCXQ2_LANES.TENSOR;
  const dst_lane  = SCXQ2_LANES.DICT;
  const C         = F.n_channels;
  const deltaF    = new Float64Array(F.n_folds * F.n_lanes * C);

  for (const f of activeFolds) {
    const src = F.lane(f, src_lane);
    const dst_base = (f * F.n_lanes + dst_lane) * C;

    // Fibonacci windowing: accumulate windows 1,1,2,3,5,8...
    let a = 1, b = 1, i = 0, out_c = 0;
    while (i < C && out_c < C) {
      const w = Math.min(a, C - i);
      let sum = 0;
      for (let k = 0; k < w; k++) sum += src[i + k];
      deltaF[dst_base + out_c] = sum / w - F.lane(f, dst_lane)[out_c];
      const next = a + b; b = a; a = next;
      i += w; out_c++;
    }
  }

  const ratio = 1 / PHI;
  return { type: "compress", deltaF, activeFolds, ratio };
}

// ─── Inference Micronaut ──────────────────────────────────────────────────────
// M_infer(F, G, P): predicts next fold state P(F_{t+1} | F_t, G, P)
//
// Instead of next-token, predicts next field configuration:
//   For each active fold f:
//     H_f = Σ_j A_fj * F_j  (geodesic attention)
//   Then: F'[f] = H_f projected through FIELD lane weights
//
// This is the core shift from token prediction to field prediction.

export function inferenceMicronaut(F, G, P, activeFolds, opts = {}) {
  const temperature = opts.temperature ?? 1.0;
  const C           = F.n_channels;
  const deltaF      = new Float64Array(F.n_folds * F.n_lanes * C);

  for (const f of activeFolds) {
    // Geodesic attention weights for this fold
    const attn = G.attentionWeights(f, temperature);
    if (attn.length === 0) continue;

    // H_f = Σ_j A_fj * F_j[TENSOR lane]
    const h = new Float64Array(C);
    for (const { j, a } of attn) {
      const src = F.lane(j, SCXQ2_LANES.TENSOR);
      for (let c = 0; c < C; c++) h[c] += a * src[c];
    }

    // Phase modulation: scale by PHASE lane activation
    const phase_val = F.lane(f, SCXQ2_LANES.PHASE);
    const sek_activation = phase_val[PHASE_INDEX.Sek] ?? 1.0;

    // Write prediction delta: project H into TENSOR lane via FIELD weights
    const field_weights = F.lane(f, SCXQ2_LANES.FIELD);
    const dst_base = (f * F.n_lanes + SCXQ2_LANES.TENSOR) * C;
    const cur = F.lane(f, SCXQ2_LANES.TENSOR);
    for (let c = 0; c < C; c++) {
      const w = field_weights[c % field_weights.length];
      deltaF[dst_base + c] = sek_activation * (w * h[c] - cur[c]) * 0.01;
    }
  }

  return { type: "infer", deltaF, activeFolds };
}

// ─── Apply operator result to FoldTensor ──────────────────────────────────────

export function applyOperator(F, result) {
  if (!result.deltaF) return;
  const C = F.n_channels;
  for (const f of result.activeFolds) {
    for (let m = 0; m < F.n_lanes; m++) {
      const base  = (f * F.n_lanes + m) * C;
      const lane  = F.lane(f, m);
      for (let c = 0; c < C; c++) lane[c] += result.deltaF[base + c];
    }
  }
}
