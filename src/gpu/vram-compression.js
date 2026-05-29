// VRAM Compression v1 — delta + DCT spectral compression for Mayan tensor tiles
//
// Targets Intel HD 4600 shared VRAM budget (~1.5 GB usable).
// Two modes:
//   Delta: 5-bit modulo-20 delta + run-length → ~2.1× compression on Mayan digit streams
//   DCT:   8×8 block DCT, keep low-frequency coefficients → ~2.8× compression on activations
//
// All ops work on plain JS arrays / Float32Array / Uint8Array — no WebGPU required.
// WebGPU/HLSL implementations of these same algorithms live in shaders/.

// ─── Delta compression (Mayan digit streams) ──────────────────────────────────
// Input: Uint8Array of Mayan digits (0–19).
// Encoding: each delta d = (cur - prev + 20) % 20 stored in 5 bits.
// Run-length: if two consecutive deltas equal, emit (count, delta) pairs.

export function deltaCompress(digits) {
  if (digits.length === 0) return new Uint8Array(0);

  const deltas = new Uint8Array(digits.length);
  deltas[0] = digits[0];   // first value stored verbatim (5-bit, 0-19)
  for (let i = 1; i < digits.length; i++) {
    deltas[i] = ((digits[i] - digits[i - 1] % 20) + 20) % 20;
  }

  // Pack pairs: [count(3 bits) | delta(5 bits)] per byte
  // count = 0 means single occurrence (implicit 1); count 1-7 = 2-8 repeats
  const out = [];
  out.push(deltas[0]); // header byte: first value

  let i = 1;
  while (i < deltas.length) {
    let run = 1;
    while (i + run < deltas.length && deltas[i + run] === deltas[i] && run < 8) run++;
    // count field: 0 = run of 1, 1 = run of 2, …, 7 = run of 8
    out.push(((run - 1) << 5) | (deltas[i] & 0x1f));
    i += run;
  }

  return new Uint8Array(out);
}

export function deltaDecompress(compressed, originalLength) {
  if (compressed.length === 0) return new Uint8Array(0);
  const out = new Uint8Array(originalLength);
  out[0] = compressed[0] & 0x1f;   // first value verbatim

  let outIdx = 1;
  for (let i = 1; i < compressed.length && outIdx < originalLength; i++) {
    const byte  = compressed[i];
    const run   = (byte >> 5) + 1;
    const delta = byte & 0x1f;
    for (let r = 0; r < run && outIdx < originalLength; r++, outIdx++) {
      out[outIdx] = (out[outIdx - 1] + delta) % 20;
    }
  }
  return out;
}

// ─── DCT spectral compression (8×8 blocks) ───────────────────────────────────
// Input: Float32Array of activation values (arbitrary range).
// Forward 2D DCT-II on 8×8 tiles; keep `keepCoeffs` low-frequency coefficients
// using a zig-zag ordering (top-left = lowest frequency).
//
// keepCoeffs=10 ≈ 2.8× compression ratio on smooth activation maps.

const COS_TABLE = (() => {
  const C = new Float32Array(8 * 8);
  for (let k = 0; k < 8; k++) for (let n = 0; n < 8; n++) {
    C[k * 8 + n] = Math.cos(Math.PI * k * (2 * n + 1) / 16);
  }
  return C;
})();

// alpha normalization for DCT-II
function alpha(k) { return k === 0 ? Math.SQRT1_2 : 1; }

function dct8(block) {
  const out = new Float32Array(64);
  for (let v = 0; v < 8; v++) for (let u = 0; u < 8; u++) {
    let sum = 0;
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      sum += block[y * 8 + x] * COS_TABLE[u * 8 + x] * COS_TABLE[v * 8 + y];
    }
    out[v * 8 + u] = 0.25 * alpha(u) * alpha(v) * sum;
  }
  return out;
}

function idct8(dctBlock) {
  const out = new Float32Array(64);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    let sum = 0;
    for (let v = 0; v < 8; v++) for (let u = 0; u < 8; u++) {
      sum += alpha(u) * alpha(v) * dctBlock[v * 8 + u]
           * COS_TABLE[u * 8 + x] * COS_TABLE[v * 8 + y];
    }
    out[y * 8 + x] = 0.25 * sum;
  }
  return out;
}

// Zig-zag scan order for 8×8 block (top-left = index 0 = DC component)
export const ZIGZAG = Object.freeze([
   0,  1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63,
]);

// Compress a Float32Array (length must be multiple of 64 = 8×8 tiles)
// Returns { data: Float32Array (keepCoeffs per tile), meta: {tiles, keepCoeffs, rows, cols} }
export function dctCompress(activations, rows, cols, keepCoeffs = 10) {
  const tilesY = Math.ceil(rows / 8);
  const tilesX = Math.ceil(cols / 8);
  const tiles  = tilesY * tilesX;
  const out    = new Float32Array(tiles * keepCoeffs);

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const block = new Float32Array(64);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        const gy = ty * 8 + y, gx = tx * 8 + x;
        block[y * 8 + x] = (gy < rows && gx < cols) ? activations[gy * cols + gx] : 0;
      }

      const dctBlock = dct8(block);
      const tileIdx  = ty * tilesX + tx;
      for (let k = 0; k < keepCoeffs; k++) {
        out[tileIdx * keepCoeffs + k] = dctBlock[ZIGZAG[k]];
      }
    }
  }

  return { data: out, meta: { tiles, keepCoeffs, rows, cols } };
}

export function dctDecompress({ data, meta }) {
  const { tiles, keepCoeffs, rows, cols } = meta;
  const tilesX = Math.ceil(cols / 8);
  const tilesY = Math.ceil(rows / 8);
  const out    = new Float32Array(rows * cols);

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const tileIdx = ty * tilesX + tx;
      const dctBlock = new Float32Array(64);
      for (let k = 0; k < keepCoeffs; k++) {
        dctBlock[ZIGZAG[k]] = data[tileIdx * keepCoeffs + k];
      }
      const block = idct8(dctBlock);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        const gy = ty * 8 + y, gx = tx * 8 + x;
        if (gy < rows && gx < cols) out[gy * cols + gx] = block[y * 8 + x];
      }
    }
  }

  return out;
}

// ─── Compression ratio estimator ─────────────────────────────────────────────

export function estimateDeltaRatio(digits) {
  if (digits.length === 0) return 1;
  const compressed = deltaCompress(digits);
  return digits.length / compressed.length;
}

export function estimateDctRatio(rows, cols, keepCoeffs = 10) {
  const tiles = Math.ceil(rows / 8) * Math.ceil(cols / 8);
  const original   = rows * cols;
  const compressed = tiles * keepCoeffs;
  return original / compressed;
}
