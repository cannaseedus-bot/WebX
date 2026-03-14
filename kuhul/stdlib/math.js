/**
 * @fileoverview Mathematical functions for the KUHUL standard library.
 *
 * Provides trigonometric, matrix, and tensor math utilities used by
 * KUHUL programs and the runtime.
 *
 * @module kuhul/stdlib/math
 */

// ------------------------------------------------------------------ //
// Trigonometric functions (operating on Float32Arrays)
// ------------------------------------------------------------------ //

/**
 * Element-wise sine.
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function sin(a) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.sin(a[i]);
  return out;
}

/**
 * Element-wise cosine.
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function cos(a) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.cos(a[i]);
  return out;
}

/**
 * Element-wise tangent.
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function tan(a) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.tan(a[i]);
  return out;
}

/**
 * Element-wise square root.
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function sqrt(a) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.sqrt(a[i]);
  return out;
}

/**
 * Element-wise exponential.
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function exp(a) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.exp(a[i]);
  return out;
}

/**
 * Element-wise natural logarithm.
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function log(a) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.log(a[i]);
  return out;
}

// ------------------------------------------------------------------ //
// Matrix operations (flat row-major Float32Arrays)
// ------------------------------------------------------------------ //

/**
 * Transpose a matrix of shape [rows × cols].
 *
 * @param {Float32Array} m
 * @param {number}       rows
 * @param {number}       cols
 * @returns {Float32Array} Shape [cols × rows]
 */
export function transpose(m, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[c * rows + r] = m[r * cols + c];
    }
  }
  return out;
}

/**
 * Compute the L2 norm of a vector.
 *
 * @param {Float32Array} v
 * @returns {number}
 */
export function norm(v) {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  return Math.sqrt(sum);
}

/**
 * Normalise a vector to unit length.
 *
 * @param {Float32Array} v
 * @returns {Float32Array}
 */
export function normalise(v) {
  const n   = norm(v);
  const out = new Float32Array(v.length);
  if (n === 0) return out;
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

/**
 * Dot product of two equal-length vectors.
 *
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number}
 */
export function dot(a, b) {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * Element-wise clamp.
 *
 * @param {Float32Array} a
 * @param {number}       min
 * @param {number}       max
 * @returns {Float32Array}
 */
export function clamp(a, min, max) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = Math.min(Math.max(a[i], min), max);
  return out;
}

/**
 * Element-wise softmax.
 *
 * @param {Float32Array} a
 * @returns {Float32Array}
 */
export function softmax(a) {
  const maxVal = a.reduce((m, v) => Math.max(m, v), -Infinity);
  const exps   = new Float32Array(a.length);
  let   sum    = 0;
  for (let i = 0; i < a.length; i++) { exps[i] = Math.exp(a[i] - maxVal); sum += exps[i]; }
  for (let i = 0; i < a.length; i++) exps[i] /= sum;
  return exps;
}
