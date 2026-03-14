/**
 * @fileoverview Core KUHUL glyph operations.
 *
 * Exports constants for each KUHUL glyph and a function implementing
 * each operation on flat Float32Array tensors (row-major storage).
 *
 * @module kuhul/stdlib/glyphs
 */

// ------------------------------------------------------------------ //
// Glyph constants
// ------------------------------------------------------------------ //

/** Tensor product / matrix multiplication `⊗` */
export const GLYPH_TENSOR_PRODUCT   = '⊗';
/** Element-wise addition / union `⊕` */
export const GLYPH_ADDITION         = '⊕';
/** Element-wise subtraction `⊖` */
export const GLYPH_SUBTRACTION      = '⊖';
/** Convolution `⊛` */
export const GLYPH_CONVOLUTION      = '⊛';
/** Equality / assign `⊜` */
export const GLYPH_EQUALITY         = '⊜';
/** Negation / complement `⊝` */
export const GLYPH_NEGATION         = '⊝';
/** Direct sum / concatenation `⊞` */
export const GLYPH_DIRECT_SUM       = '⊞';

/** All recognised KUHUL glyphs as a Set. */
export const GLYPHS = new Set([
  GLYPH_TENSOR_PRODUCT,
  GLYPH_ADDITION,
  GLYPH_SUBTRACTION,
  GLYPH_CONVOLUTION,
  GLYPH_EQUALITY,
  GLYPH_NEGATION,
  GLYPH_DIRECT_SUM,
]);

// ------------------------------------------------------------------ //
// Glyph operation implementations
// ------------------------------------------------------------------ //

/**
 * Matrix multiplication A (m×k) @ B (k×n) → C (m×n).
 * Tensors are flat Float32Arrays in row-major order.
 *
 * @param {Float32Array} a - Left matrix, shape [m, k]
 * @param {Float32Array} b - Right matrix, shape [k, n]
 * @param {number}       m
 * @param {number}       k
 * @param {number}       n
 * @returns {Float32Array}
 */
export function tensorProduct(a, b, m, k, n) {
  const c = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
      c[i * n + j] = sum;
    }
  }
  return c;
}

/**
 * Element-wise addition.  Arrays must be the same length.
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {Float32Array}
 */
export function addition(a, b) {
  const len = Math.min(a.length, b.length);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = a[i] + b[i];
  return out;
}

/**
 * Element-wise subtraction.  Arrays must be the same length.
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {Float32Array}
 */
export function subtraction(a, b) {
  const len = Math.min(a.length, b.length);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = a[i] - b[i];
  return out;
}

/**
 * 1-D discrete convolution (valid padding).
 *
 * @param {Float32Array} signal
 * @param {Float32Array} kernel
 * @returns {Float32Array}
 */
export function convolution(signal, kernel) {
  const outLen = signal.length - kernel.length + 1;
  if (outLen <= 0) return new Float32Array(0);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    let sum = 0;
    for (let j = 0; j < kernel.length; j++) sum += signal[i + j] * kernel[j];
    out[i] = sum;
  }
  return out;
}

/**
 * Element-wise equality (returns 1.0 where equal, 0.0 elsewhere).
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {Float32Array}
 */
export function equality(a, b) {
  const len = Math.min(a.length, b.length);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = a[i] === b[i] ? 1 : 0;
  return out;
}

/**
 * Element-wise negation.
 *
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function negation(a) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = -a[i];
  return out;
}

/**
 * Direct sum – concatenation of two tensors.
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {Float32Array}
 */
export function directSum(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

// ------------------------------------------------------------------ //
// Dispatch table  (glyph symbol → function)
// ------------------------------------------------------------------ //

/**
 * Execute a glyph operation on tensors.
 *
 * @param {string}       glyph
 * @param {Float32Array} a
 * @param {Float32Array} [b]
 * @param {object}       [opts]  - Extra options (e.g. matrix dimensions for ⊗)
 * @returns {Float32Array}
 */
export function executeGlyph(glyph, a, b, opts = {}) {
  switch (glyph) {
    case GLYPH_TENSOR_PRODUCT: {
      const { m = a.length, k = a.length, n = 1 } = opts;
      return tensorProduct(a, b, m, k, n);
    }
    case GLYPH_ADDITION:    return addition(a, b);
    case GLYPH_SUBTRACTION: return subtraction(a, b);
    case GLYPH_CONVOLUTION: return convolution(a, b);
    case GLYPH_EQUALITY:    return equality(a, b);
    case GLYPH_NEGATION:    return negation(a);
    case GLYPH_DIRECT_SUM:  return directSum(a, b);
    default:
      throw new Error(`Unknown glyph "${glyph}"`);
  }
}
