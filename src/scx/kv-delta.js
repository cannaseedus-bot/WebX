// KVCacheDelta — INT4 delta encoding for KV cache (SCXRuntime.v1.0.0, include/runtime_state.h)
//
// KV cache deltas are stored as packed INT4 nibbles.
// Each delta encodes the difference between the new and previous cached value.
// Scale factor: nibble range [-8, 7] maps to float range [-1.0, +0.875] (step = 1/8)

const NIB_SCALE = 1 / 8; // one nibble step

export function kvDeltaEncode(v, prev) {
  const diff  = v - prev;
  const raw   = Math.round(diff / NIB_SCALE);
  const clamped = Math.max(-8, Math.min(7, raw));
  return clamped < 0 ? (clamped + 16) & 0xF : clamped & 0xF; // 4-bit two's complement
}

export function kvDeltaDecode(prev, nib) {
  const signed = nib >= 8 ? nib - 16 : nib; // 4-bit two's complement → signed
  return prev + signed * NIB_SCALE;
}

// Pack an array of 4-bit nibbles into Uint8Array (LSB-first per byte)
export function packNibs(nibs) {
  const out = new Uint8Array(Math.ceil(nibs.length / 2));
  for (let i = 0; i < nibs.length; i++) {
    if (i & 1) out[i >> 1] |= (nibs[i] & 0xF) << 4;
    else        out[i >> 1]  = nibs[i] & 0xF;
  }
  return out;
}

export function unpackNibs(packed, count) {
  const out = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const byte = packed[i >> 1];
    out[i] = i & 1 ? (byte >> 4) & 0xF : byte & 0xF;
  }
  return out;
}

export class KVCacheDelta {
  constructor(dim) {
    this.dim  = dim;
    this.data = new Uint8Array(Math.ceil(dim / 2));
  }

  encode(newVals, prevVals) {
    if (newVals.length !== this.dim) throw new Error('KVCacheDelta: dimension mismatch');
    const nibs = new Uint8Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      nibs[i] = kvDeltaEncode(newVals[i], prevVals[i]);
    }
    this.data = packNibs(nibs);
    return this;
  }

  decode(prevVals) {
    const nibs = unpackNibs(this.data, this.dim);
    const out  = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      out[i] = kvDeltaDecode(prevVals[i], nibs[i]);
    }
    return out;
  }
}
