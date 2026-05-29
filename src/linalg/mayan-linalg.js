// Mayan-Linear Algebra Hybrid v1
// Maps Long Count calendar ↔ Float64Array vectors and applies linear transforms.
// OLS regression predicts future Long Count dates from historical data.
//
// Convention: vector index order = [kin, uinal, tun, katun, baktun] (low→high)
// matching LONG_COUNT_POSITIONS indices 0..4.

import { bigIntToDigits, digitsToBase, LONG_COUNT_POSITIONS } from '../mayan/mayan-math.js';
import { matMul, matTranspose, matInverse, matVecMul, dotProduct } from './linalg.js';

// ─── Mayan date ↔ Float64Array ────────────────────────────────────────────────

// Uint8Array(5) digits → Float64Array(5): [kin, uinal, tun, katun, baktun]
export function mayanToVector(digits) {
  const v = new Float64Array(5);
  for (let i = 0; i < 5; i++) v[i] = digits[i];
  return v;
}

// Float64Array(5) → Uint8Array(5): clamp + round each component to [0,19]
export function vectorToMayanDigits(v) {
  const digits = new Uint8Array(5);
  for (let i = 0; i < 5; i++) digits[i] = Math.max(0, Math.min(19, Math.round(v[i])));
  return digits;
}

// Uint8Array(5) → number (scalar days from epoch, for regression X rows)
export function mayanDigitsToScalar(digits) {
  return Number(digitsToBase(digits));
}

// number (scalar) → Uint8Array(5)
export function scalarToMayanDigits(n) {
  return bigIntToDigits(BigInt(Math.round(n)));
}

// ─── Linear transform of a Mayan date ─────────────────────────────────────────
// M: Float64Array(25) — 5×5 row-major transform matrix
// Returns new Mayan digits after applying the linear map and re-normalizing.

export function mayanLinearTransform(M, digits) {
  const v   = mayanToVector(digits);
  const out = matVecMul(M, v, 5, 5);    // Float64Array(5)
  return vectorToMayanDigits(out);
}

// ─── Solve a Mayan calendar linear system: A·x = b ───────────────────────────
// A: Float64Array(n*n), b: Float64Array(n), n: dimension
// Returns solution vector x (Float64Array(n)) or null if singular.

export function mayanLinearSolve(A, b, n) {
  const Ainv = matInverse(A, n);
  if (!Ainv) return null;
  return matVecMul(Ainv, b, n, n);
}

// ─── OLS Linear Regression: β = (XᵀX)⁻¹ Xᵀy ───────────────────────────────
// X: Float64Array(m*n) — design matrix (m observations, n features)
// y: Float64Array(m)   — response vector (scalar day values)
// Returns Float64Array(n) β coefficients, or null if XᵀX is singular.

export function mayanLinearRegression(X, y, m, n) {
  const Xt    = matTranspose(X, m, n);        // n×m
  const XtX   = matMul(Xt, X, n, m, n);       // n×n
  const XtX_i = matInverse(XtX, n);
  if (!XtX_i) return null;
  const Xty   = matMul(Xt, y, n, m, 1);       // n×1 (y treated as m×1)
  return matMul(XtX_i, Xty, n, n, 1);         // n×1 beta
}

// ─── Predict scalar day-value for a new feature vector ───────────────────────

export function predictMayanDays(Xnew, beta, n) {
  return dotProduct(Xnew.slice(0, n), beta.slice(0, n));
}

// ─── Full calendar regression helper ─────────────────────────────────────────
// Given arrays of Mayan digit arrays (historical dates) + target scalars,
// builds the design matrix [days] and solves for β.
// Returns { beta, predict(newDigits) → { days: number, digits: Uint8Array } }

export function buildCalendarRegression(pastDates, futureDays) {
  const m = pastDates.length;
  const n = 1;                                  // single feature: days since epoch
  const X = new Float64Array(m * n);
  const y = new Float64Array(m);

  for (let i = 0; i < m; i++) {
    X[i * n] = mayanDigitsToScalar(pastDates[i]);
    y[i]     = futureDays[i];
  }

  const beta = mayanLinearRegression(X, y, m, n);
  if (!beta) return null;

  return {
    beta,
    predict(newDigits) {
      const x    = new Float64Array([mayanDigitsToScalar(newDigits)]);
      const days = predictMayanDays(x, beta, n);
      return { days, digits: scalarToMayanDigits(days) };
    },
  };
}

// ─── Mayan date arithmetic via linear algebra ─────────────────────────────────
// Represent Long Count as vector and apply a 5×5 transform.
// Useful for calendar round alignments, era shifts, and period synchronisation.

export const IDENTITY_TRANSFORM = Object.freeze(
  new Float64Array([1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,0,1,0, 0,0,0,0,1])
);

// Add a fixed offset [kin,uinal,tun,katun,baktun] via affine shift
export function mayanAffineShift(digits, offsetDigits) {
  const a = Number(digitsToBase(digits));
  const b = Number(digitsToBase(offsetDigits));
  return bigIntToDigits(BigInt(a + b));
}

// Calendar round alignment: find scalar k such that (date + k) aligns to tzolkin/haab joint
export function calendarRoundAlignment(digits, tzolkin = 260, haab = 365) {
  const current = Number(digitsToBase(digits));
  const lcm     = (tzolkin * haab) / gcd(tzolkin, haab);
  const k       = (lcm - (current % lcm)) % lcm;
  return { k, aligned: bigIntToDigits(BigInt(current + k)) };
}

function gcd(a, b) {
  while (b) { const t = b; b = a % b; a = t; }
  return a;
}
