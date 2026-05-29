// Linear Algebra v1 — atomic-style vector/matrix/tensor ops on Float64Array
//
// Storage convention: matrices are row-major Float64Array.
//   A[i,j] = flat[ i * cols + j ]
//
// All ops return new typed arrays; inputs are never mutated.
// "Atomic" framing: each op is deterministic given its inputs;
// caller may wrap with SharedArrayBuffer + Atomics if needed.

// ─── Vector ops ───────────────────────────────────────────────────────────────

export function vectorAdd(a, b) {
  const n = a.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i] + b[i];
  return out;
}

export function vectorSub(a, b) {
  const n = a.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i] - b[i];
  return out;
}

export function vectorScale(v, s) {
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * s;
  return out;
}

export function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function vectorNorm(v) {
  return Math.sqrt(dotProduct(v, v));
}

export function vectorNormalize(v) {
  const norm = vectorNorm(v);
  if (norm === 0) throw new Error('Cannot normalize zero vector');
  return vectorScale(v, 1 / norm);
}

// ─── Matrix ops (row-major) ───────────────────────────────────────────────────

// C = A·B  where A: m×k, B: k×n → C: m×n
export function matMul(A, B, m, k, n) {
  const C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) sum += A[i * k + l] * B[l * n + j];
      C[i * n + j] = sum;
    }
  }
  return C;
}

// Transpose: A: m×n → Aᵀ: n×m
export function matTranspose(A, m, n) {
  const T = new Float64Array(n * m);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j * m + i] = A[i * n + j];
  return T;
}

// Matrix-vector multiply: A: m×n, v: n → result: m
export function matVecMul(A, v, m, n) {
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += A[i * n + j] * v[j];
    out[i] = sum;
  }
  return out;
}

// Identity matrix: n×n
export function matIdentity(n) {
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  return I;
}

// Scale all elements of A: m×n by scalar s
export function matScale(A, s) {
  const out = new Float64Array(A.length);
  for (let i = 0; i < A.length; i++) out[i] = A[i] * s;
  return out;
}

// ─── Matrix inverse (Gauss-Jordan with partial pivoting) ──────────────────────
// Returns Float64Array(n×n) or null if singular.
export function matInverse(A, n) {
  // Augmented matrix [A | I], stored as Float64Array(n × 2n)
  const aug = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * 2 * n + j] = A[i * n + j];
    aug[i * 2 * n + n + i] = 1;
  }

  for (let col = 0; col < n; col++) {
    // Find pivot row (max abs value in column)
    let pivotRow = col;
    let maxVal   = Math.abs(aug[col * 2 * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(aug[r * 2 * n + col]);
      if (v > maxVal) { maxVal = v; pivotRow = r; }
    }

    if (maxVal < 1e-12) return null; // singular

    // Swap rows
    if (pivotRow !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const tmp = aug[col * 2 * n + j];
        aug[col * 2 * n + j]      = aug[pivotRow * 2 * n + j];
        aug[pivotRow * 2 * n + j] = tmp;
      }
    }

    // Scale pivot row so diagonal = 1
    const pivotVal = aug[col * 2 * n + col];
    for (let j = 0; j < 2 * n; j++) aug[col * 2 * n + j] /= pivotVal;

    // Eliminate all other rows
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r * 2 * n + col];
      for (let j = 0; j < 2 * n; j++) {
        aug[r * 2 * n + j] -= factor * aug[col * 2 * n + j];
      }
    }
  }

  // Extract right half
  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) inv[i * n + j] = aug[i * 2 * n + n + j];
  return inv;
}

// ─── Power iteration — largest eigenvalue ─────────────────────────────────────
// A: n×n, returns { eigenvalue: number, eigenvector: Float64Array }
export function powerIteration(A, n, tolerance = 1e-8, maxIter = 1000) {
  // Start with unit vector
  let v = new Float64Array(n);
  v[0] = 1;

  let lambda = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    const w        = matVecMul(A, v, n, n);
    const newLambda = dotProduct(v, w);

    if (Math.abs(newLambda - lambda) < tolerance) {
      return { eigenvalue: newLambda, eigenvector: v, iterations: iter + 1 };
    }

    lambda = newLambda;
    const norm = vectorNorm(w);
    if (norm < 1e-15) break;
    v = vectorScale(w, 1 / norm);
  }

  return { eigenvalue: lambda, eigenvector: v, iterations: maxIter };
}

// ─── Softmax ──────────────────────────────────────────────────────────────────
// In-place over a Float64Array slice of length n.
export function softmaxInPlace(arr, offset, n) {
  let maxV = -Infinity;
  for (let i = 0; i < n; i++) if (arr[offset + i] > maxV) maxV = arr[offset + i];

  let sumE = 0;
  for (let i = 0; i < n; i++) {
    arr[offset + i] = Math.exp(arr[offset + i] - maxV);
    sumE += arr[offset + i];
  }
  for (let i = 0; i < n; i++) arr[offset + i] /= sumE;
}

// ─── Batch matrix multiply ────────────────────────────────────────────────────
// A: [batch, m, k], B: [batch, k, p] → C: [batch, m, p]
export function batchMatMul(A, B, batch, m, k, p) {
  const C = new Float64Array(batch * m * p);
  for (let b = 0; b < batch; b++) {
    const Aoff = b * m * k;
    const Boff = b * k * p;
    const Coff = b * m * p;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let l = 0; l < k; l++) sum += A[Aoff + i * k + l] * B[Boff + l * p + j];
        C[Coff + i * p + j] = sum;
      }
    }
  }
  return C;
}

// ─── Tensor softmax — axis=-1 (last dimension) ───────────────────────────────
// scores: [batch, heads, seq, seq] → applies softmax over last dimension
export function tensorSoftmax(scores, batch, heads, seq) {
  const out = new Float64Array(scores);
  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < heads; h++) {
      for (let i = 0; i < seq; i++) {
        const offset = ((b * heads + h) * seq + i) * seq;
        softmaxInPlace(out, offset, seq);
      }
    }
  }
  return out;
}

// ─── Scaled dot-product attention ─────────────────────────────────────────────
// Q, K, V: [batch, heads, seq, dim] (flat Float64Array, row-major)
// Returns output: [batch, heads, seq, dim]
export function scaledDotAttention(Q, K, V, batch, heads, seq, dim) {
  const scale = 1 / Math.sqrt(dim);
  const out   = new Float64Array(batch * heads * seq * dim);

  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < heads; h++) {
      const bhOff = (b * heads + h) * seq * dim;

      // Scores[i,j] = Q[i] · K[j] * scale  →  [seq, seq]
      const scores = new Float64Array(seq * seq);
      for (let i = 0; i < seq; i++) {
        for (let j = 0; j < seq; j++) {
          let dot = 0;
          for (let d = 0; d < dim; d++) dot += Q[bhOff + i * dim + d] * K[bhOff + j * dim + d];
          scores[i * seq + j] = dot * scale;
        }
      }

      // Softmax over each row
      for (let i = 0; i < seq; i++) softmaxInPlace(scores, i * seq, seq);

      // Output[i] = Σ_j weights[i,j] · V[j]
      const outOff = bhOff;
      for (let i = 0; i < seq; i++) {
        for (let d = 0; d < dim; d++) {
          let sum = 0;
          for (let j = 0; j < seq; j++) sum += scores[i * seq + j] * V[bhOff + j * dim + d];
          out[outOff + i * dim + d] = sum;
        }
      }
    }
  }
  return out;
}

// ─── Tensor contraction (generic batch matmul alias) ─────────────────────────
// A: [batch, m, k], B: [batch, k, n] → C: [batch, m, n]
export const tensorContract = batchMatMul;

// ─── Utility ──────────────────────────────────────────────────────────────────

export function matFromArray(rows) {
  const m = rows.length, n = rows[0].length;
  const flat = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) flat[i * n + j] = rows[i][j];
  return { data: flat, m, n };
}

export function matToArray(flat, m, n) {
  const rows = [];
  for (let i = 0; i < m; i++) rows.push(Array.from(flat.slice(i * n, i * n + n)));
  return rows;
}
