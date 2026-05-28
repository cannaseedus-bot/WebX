// LoRA adapter loader — applies rank-8 delta weights at inference time.
// Port of pi_kuhul_adapter_to_scxq2.py loading logic.
//
// Architecture:
//   foundation model (52.95M params, frozen) + adapter delta (rank-8, ~295K params)
//   adapter.apply(base_activation, strength) → modulated activation

import { ADAPTER_REGISTRY, BASE_MODEL, ADAPTER_CONFIG, resolveAdapter } from './adapter-registry.js';

// INT4 nibble decode: [0..7] → +0..+7, [8..15] → -8..-1
function decodeINT4(nibble) {
  const n = nibble & 0xF;
  return n < 8 ? n : n - 16;
}

// Unpack a Uint8Array of nibble-packed INT4 into Float32Array with per-group dequant.
// groupSize = 64 (SCXQ2 Q4_BLOCK standard)
export function unpackINT4(packed, scales, groupSize = 64) {
  const count = packed.length * 2;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const byte   = packed[i >> 1];
    const nibble = (i & 1) ? (byte >> 4) : (byte & 0xF);
    const group  = Math.floor(i / groupSize);
    out[i] = decodeINT4(nibble) * (scales[group] || 1.0);
  }
  return out;
}

// Apply LoRA delta at inference time.
// adapter.lora_A [r, d_in] and adapter.lora_B [d_out, r] are the rank-8 decomposition.
// output = base + strength * (lora_B @ lora_A @ input)
export function applyLoRA(input, loraA, loraB, rank, dIn, dOut, strength = 1.0) {
  // intermediate = loraA @ input  [rank]
  const intermediate = new Float32Array(rank);
  for (let r = 0; r < rank; r++) {
    let s = 0;
    for (let d = 0; d < dIn; d++) s += loraA[r * dIn + d] * input[d];
    intermediate[r] = s;
  }
  // delta = loraB @ intermediate  [dOut]
  const delta = new Float32Array(dOut);
  for (let o = 0; o < dOut; o++) {
    let s = 0;
    for (let r = 0; r < rank; r++) s += loraB[o * rank + r] * intermediate[r];
    delta[o] = s * strength;
  }
  return delta;
}

// Adapter object returned by loadAdapter()
export class LoRAAdapter {
  constructor(domain, config = {}) {
    this.domain   = domain;
    this.rank     = config.rank  || ADAPTER_CONFIG.rank;
    this.strength = config.strength !== undefined ? config.strength : 1.0;
    this.loraA    = config.loraA || null;  // Float32Array [rank × dIn]
    this.loraB    = config.loraB || null;  // Float32Array [dOut × rank]
    this.dIn      = config.dIn  || 0;
    this.dOut     = config.dOut || 0;
    this.loaded   = !!(this.loraA && this.loraB);
  }

  // Apply to a single activation vector [dIn] → delta [dOut]
  apply(activation) {
    if (!this.loaded) throw new Error(`Adapter '${this.domain}' not loaded`);
    return applyLoRA(activation, this.loraA, this.loraB,
                     this.rank, this.dIn, this.dOut, this.strength);
  }

  // Merge delta into base activation (add in-place)
  modulate(baseActivation) {
    if (!this.loaded) return baseActivation;
    const delta = this.apply(baseActivation.slice(0, this.dIn));
    const out = baseActivation.slice();
    for (let i = 0; i < Math.min(delta.length, out.length); i++) {
      out[i] += delta[i];
    }
    return out;
  }
}

// Load an adapter from raw SCXQ2 binary data (ArrayBuffer).
// The SCXQ2 format stores lora_A and lora_B as consecutive INT4-packed weight blocks.
export function loadAdapterFromBuffer(domain, buffer, strength = 1.0) {
  const data = new Uint8Array(buffer);

  // Minimal SCXQ2 header: [version:1][flags:1][const_count:4][instr_count:4][crc:4] = 14B
  if (data.length < 14) throw new Error('Buffer too short for SCXQ2 header');
  const version    = data[0];
  const constCount = (data[2] << 24) | (data[3] << 16) | (data[4] << 8) | data[5];
  const instrCount = (data[6] << 24) | (data[7] << 16) | (data[8] << 8) | data[9];
  const constStart = 14;
  const instrStart = constStart + constCount;

  const { rank, params: totalParams } = ADAPTER_CONFIG;
  // rank-8 LoRA: loraA [8, dIn] + loraB [dOut, 8]
  // For hidden_size=768: dIn=dOut=768, loraA=8*768=6144, loraB=768*8=6144
  const dIn  = 768;
  const dOut = 768;
  const packed = data.slice(instrStart, instrStart + Math.ceil((dIn * rank + dOut * rank) / 2));
  const scale  = 0.001;  // default dequant scale when no per-group scales loaded
  const scales = new Float32Array(Math.ceil(packed.length * 2 / 64)).fill(scale);
  const weights = unpackINT4(packed, scales);

  const loraA = weights.slice(0, rank * dIn);
  const loraB = weights.slice(rank * dIn, rank * dIn + dOut * rank);

  return new LoRAAdapter(domain, { rank, dIn, dOut, loraA, loraB, strength });
}

// Create an unloaded adapter stub (for registries that lazy-load)
export function createAdapterStub(domain, strength = 1.0) {
  return new LoRAAdapter(domain, { strength });
}

export { ADAPTER_REGISTRY, BASE_MODEL, ADAPTER_CONFIG, resolveAdapter };
export default LoRAAdapter;
