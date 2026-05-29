// Mayan Tensor v1 — Float64Array n-dimensional tensors with CP/Tucker decomposition
//
// Storage: row-major flat Float64Array with an explicit shape array.
// All ops return new arrays; inputs never mutated.
//
// CP decomposition: rank-R approximation via ALS (Alternating Least Squares).
//   T ≈ Σ_r λ_r · a_r(1) ⊗ a_r(2) ⊗ … ⊗ a_r(N)
//
// Tucker decomposition: multi-linear rank-(R1,…,RN) approximation via HOSVD init.
//   T ≈ G ×_1 U(1) ×_2 U(2) ×_N U(N)

import { matMul, matTranspose, matInverse, matVecMul, vectorNorm, vectorScale } from '../linalg/linalg.js';

// ─── Shape utilities ──────────────────────────────────────────────────────────

export function tensorSize(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

export function strides(shape) {
  const s = new Array(shape.length);
  s[shape.length - 1] = 1;
  for (let i = shape.length - 2; i >= 0; i--) s[i] = s[i + 1] * shape[i + 1];
  return s;
}

// ─── Tensor unfold (matricization) ───────────────────────────────────────────
// Unfold along `mode` → 2D matrix of shape [shape[mode], prod(other dims)]
// Row i = all entries where the mode-n index equals i, ordered by the remaining indices.

export function tensorUnfold(data, shape, mode) {
  const N    = shape.length;
  const rows = shape[mode];
  const cols = tensorSize(shape) / rows;

  const out   = new Float64Array(rows * cols);
  const str   = strides(shape);
  const total = tensorSize(shape);

  for (let flatIdx = 0; flatIdx < total; flatIdx++) {
    // Decode flat index → multi-index
    let tmp = flatIdx;
    const midx = new Array(N);
    for (let n = 0; n < N; n++) {
      midx[n] = Math.floor(tmp / str[n]);
      tmp     = tmp % str[n];
    }

    const row = midx[mode];
    // Column: remaining indices packed in row-major order (skip `mode`)
    let col = 0;
    let colStride = 1;
    for (let n = N - 1; n >= 0; n--) {
      if (n === mode) continue;
      col  += midx[n] * colStride;
      colStride *= shape[n];
    }

    out[row * cols + col] = data[flatIdx];
  }

  return { data: out, rows, cols };
}

// ─── Khatri-Rao product ───────────────────────────────────────────────────────
// Column-wise Kronecker product of two matrices A (m×r) and B (n×r) → (mn×r).

export function khatriRao(A, m, B, n, r) {
  const out = new Float64Array(m * n * r);
  for (let col = 0; col < r; col++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        out[(i * n + j) * r + col] = A[i * r + col] * B[j * r + col];
      }
    }
  }
  return out;
}

// ─── CP decomposition (ALS) ───────────────────────────────────────────────────
// data: Float64Array, shape: number[], rank: number
// maxIter: ALS iterations, tol: convergence threshold on lambda change
// Returns { factors: Float64Array[] (one per mode, shape[n]×rank),
//           lambda: Float64Array(rank), iters: number }

export function cpDecompose(data, shape, rank, maxIter = 100, tol = 1e-6) {
  const N = shape.length;

  // Initialize factor matrices with normalized random columns
  const factors = shape.map(dim => {
    const F = new Float64Array(dim * rank);
    for (let i = 0; i < dim * rank; i++) F[i] = (Math.random() - 0.5) * 2;
    // Normalize each column
    for (let r = 0; r < rank; r++) {
      let norm = 0;
      for (let i = 0; i < dim; i++) norm += F[i * rank + r] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < dim; i++) F[i * rank + r] /= norm;
    }
    return F;
  });

  const lambda = new Float64Array(rank).fill(1);
  let prevLambdaNorm = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    for (let mode = 0; mode < N; mode++) {
      const { data: unfolded, rows, cols } = tensorUnfold(data, shape, mode);

      // Khatri-Rao product of all factors except this mode (reverse order)
      let kr = null;
      let krRows = 1;
      for (let n = N - 1; n >= 0; n--) {
        if (n === mode) continue;
        if (kr === null) {
          kr     = Float64Array.from(factors[n]);
          krRows = shape[n];
        } else {
          kr     = khatriRao(factors[n], shape[n], kr, krRows, rank);
          krRows = shape[n] * krRows;
        }
      }

      // ALS update: factor[mode] = unfolded · KR · (KRᵀ · KR)⁻¹
      // unfolded: rows×cols, KR: cols×rank
      const krT     = matTranspose(kr, krRows, rank);          // rank×cols
      const krTkr   = matMul(krT, kr, rank, krRows, rank);     // rank×rank
      const krTkrI  = matInverse(krTkr, rank);
      if (!krTkrI) continue;                                    // singular — skip update

      const unfKr   = matMul(unfolded, kr, rows, cols, rank);  // rows×rank
      const newF    = matMul(unfKr, krTkrI, rows, rank, rank); // rows×rank

      // Normalize columns → extract lambda
      for (let r = 0; r < rank; r++) {
        let norm = 0;
        for (let i = 0; i < rows; i++) norm += newF[i * rank + r] ** 2;
        norm = Math.sqrt(norm) || 1;
        lambda[r] = norm;
        for (let i = 0; i < rows; i++) factors[mode][i * rank + r] = newF[i * rank + r] / norm;
      }
    }

    // Convergence check on lambda norm
    let lNorm = 0;
    for (let r = 0; r < rank; r++) lNorm += lambda[r] ** 2;
    lNorm = Math.sqrt(lNorm);
    if (Math.abs(lNorm - prevLambdaNorm) < tol) {
      return { factors, lambda, iters: iter + 1 };
    }
    prevLambdaNorm = lNorm;
  }

  return { factors, lambda, iters: maxIter };
}

// ─── CP reconstruction ────────────────────────────────────────────────────────
// Reconstruct tensor from factors + lambda.
// Returns Float64Array of size tensorSize(shape).

export function reconstructCP(factors, lambda, shape) {
  const N     = shape.length;
  const rank  = lambda.length;
  const total = tensorSize(shape);
  const str   = strides(shape);
  const out   = new Float64Array(total);

  for (let flatIdx = 0; flatIdx < total; flatIdx++) {
    let tmp = flatIdx;
    const midx = new Array(N);
    for (let n = 0; n < N; n++) {
      midx[n] = Math.floor(tmp / str[n]);
      tmp     = tmp % str[n];
    }

    let val = 0;
    for (let r = 0; r < rank; r++) {
      let term = lambda[r];
      for (let n = 0; n < N; n++) term *= factors[n][midx[n] * rank + r];
      val += term;
    }
    out[flatIdx] = val;
  }
  return out;
}

// ─── Tucker decomposition (HOSVD init + HOOI refinement) ─────────────────────
// Truncated HOSVD: compute mode-n SVD and keep top-Rn left singular vectors.
// data: Float64Array, shape: number[], ranks: number[]
// Returns { core: Float64Array, factors: Float64Array[], coreShape: number[], iters: number }

export function tuckerDecompose(data, shape, ranks, maxIter = 20, tol = 1e-6) {
  const N = shape.length;

  // HOSVD initialization: mode-n unfolding → truncated SVD (power iteration per rank)
  const factors = shape.map((dim, mode) => {
    const { data: unf, rows, cols } = tensorUnfold(data, shape, mode);
    return truncatedSVD(unf, rows, cols, ranks[mode]);
  });

  let prevFit = Infinity;

  // HOOI: alternate least squares on Tucker factors
  for (let iter = 0; iter < maxIter; iter++) {
    for (let mode = 0; mode < N; mode++) {
      // Y = T ×_{n≠mode} U(n)ᵀ, then mode-n unfolding, then leading Rn left sing-vecs
      let Y = Float64Array.from(data);
      let Yshape = shape.slice();

      for (let n = N - 1; n >= 0; n--) {
        if (n === mode) continue;
        const { result, outShape } = modeProd(Y, Yshape, n, factors[n], ranks[n], true);
        Y      = result;
        Yshape = outShape;
      }

      const { data: unf, rows, cols } = tensorUnfold(Y, Yshape, mode);
      factors[mode] = truncatedSVD(unf, rows, cols, ranks[mode]);
    }

    // Compute core tensor G = T ×_1 U(1)ᵀ ×_2 U(2)ᵀ … ×_N U(N)ᵀ
    let G = Float64Array.from(data);
    let Gshape = shape.slice();
    for (let n = 0; n < N; n++) {
      const { result, outShape } = modeProd(G, Gshape, n, factors[n], ranks[n], true);
      G      = result;
      Gshape = outShape;
    }

    // Reconstruction fit (Frobenius norm of residual)
    const recon = tuckerReconstruct(G, factors, shape, ranks);
    let fit = 0;
    for (let i = 0; i < data.length; i++) fit += (data[i] - recon[i]) ** 2;
    fit = Math.sqrt(fit);

    if (Math.abs(fit - prevFit) < tol) {
      return { core: G, coreShape: Gshape, factors, iters: iter + 1 };
    }
    prevFit = fit;
  }

  let G = Float64Array.from(data);
  let Gshape = shape.slice();
  for (let n = 0; n < N; n++) {
    const { result, outShape } = modeProd(G, Gshape, n, factors[n], ranks[n], true);
    G      = result;
    Gshape = outShape;
  }

  return { core: G, coreShape: Gshape, factors, iters: maxIter };
}

// ─── Tucker reconstruction ────────────────────────────────────────────────────
// G ×_1 U(1) ×_2 U(2) … ×_N U(N)

export function tuckerReconstruct(core, factors, targetShape, ranks) {
  let T = Float64Array.from(core);
  let Tshape = ranks.slice();
  for (let n = 0; n < factors.length; n++) {
    const { result, outShape } = modeProd(T, Tshape, n, factors[n], targetShape[n], false);
    T      = result;
    Tshape = outShape;
  }
  return T;
}

// ─── Mode-n product: T ×_n M (or Mᵀ if transpose=true) ──────────────────────
// T: data with shape, M: (outDim×inDim) or (inDim×outDim) if transpose
// Returns { result: Float64Array, outShape: number[] }

function modeProd(data, shape, mode, M, outDim, transpose) {
  const N       = shape.length;
  const inDim   = shape[mode];
  const total   = tensorSize(shape);
  const str     = strides(shape);

  const newShape  = shape.slice();
  newShape[mode]  = outDim;
  const newTotal  = tensorSize(newShape);
  const newStr    = strides(newShape);
  const out       = new Float64Array(newTotal);

  // Iterate over all multi-indices with mode index free
  const sliceSize = total / inDim;
  // For each "slice" (fixing all non-mode indices), multiply by M
  for (let flatNew = 0; flatNew < newTotal; flatNew++) {
    let tmp = flatNew;
    const midxNew = new Array(N);
    for (let n = 0; n < N; n++) {
      midxNew[n] = Math.floor(tmp / newStr[n]);
      tmp        = tmp % newStr[n];
    }

    const outModeIdx = midxNew[mode];
    let val = 0;
    for (let k = 0; k < inDim; k++) {
      const mVal = transpose
        ? M[k * outDim + outModeIdx]   // Mᵀ: M[k,outModeIdx]
        : M[outModeIdx * inDim + k];   // M: M[outModeIdx,k]

      // Original flat index with mode = k
      let flatOrig = 0;
      for (let n = 0; n < N; n++) {
        flatOrig += (n === mode ? k : midxNew[n]) * str[n];
      }
      val += mVal * data[flatOrig];
    }
    out[flatNew] = val;
  }

  return { result: out, outShape: newShape };
}

// ─── Truncated SVD (power iteration per singular vector) ─────────────────────
// Returns U: rows×k (first k left singular vectors) as Float64Array(rows*k)

function truncatedSVD(A, rows, cols, k) {
  const U = new Float64Array(rows * k);
  // Deflation: subtract already-found components
  const Awork = Float64Array.from(A);

  for (let sv = 0; sv < k; sv++) {
    // Power iteration on A·Aᵀ to find the dominant left singular vector
    let u = new Float64Array(rows);
    u[sv % rows] = 1;

    const At = matTranspose(Awork, rows, cols);
    for (let iter = 0; iter < 50; iter++) {
      // w = A · (Aᵀ · u)
      const Atu = matVecMul(At, u, cols, rows);
      const w   = matVecMul(Awork, Atu, rows, cols);
      const norm = vectorNorm(w);
      if (norm < 1e-15) break;
      u = vectorScale(w, 1 / norm);
    }

    // Store singular vector
    for (let i = 0; i < rows; i++) U[i * k + sv] = u[i];

    // Deflate: A ← A - σ · u · vᵀ  (vᵀ = uᵀ·A / σ)
    const Atu    = matVecMul(At, u, cols, rows);
    const sigma  = vectorNorm(Atu);
    if (sigma > 1e-15) {
      const v = vectorScale(Atu, 1 / sigma);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          Awork[i * cols + j] -= sigma * u[i] * v[j];
        }
      }
    }
  }

  return U;
}

// ─── Mayan tensor wrapper ─────────────────────────────────────────────────────
// Thin wrapper associating a Float64Array with its shape metadata.

export class MayanTensor {
  constructor(data, shape) {
    this.data  = data instanceof Float64Array ? data : new Float64Array(data);
    this.shape = shape;
    this.ndim  = shape.length;
    this.size  = tensorSize(shape);
  }

  static zeros(shape) {
    return new MayanTensor(new Float64Array(tensorSize(shape)), shape);
  }

  static fromArray(nested) {
    const shape = inferShape(nested);
    const flat  = new Float64Array(tensorSize(shape));
    flattenInto(nested, flat, 0);
    return new MayanTensor(flat, shape);
  }

  get(indices) {
    const str = strides(this.shape);
    let idx = 0;
    for (let n = 0; n < this.ndim; n++) idx += indices[n] * str[n];
    return this.data[idx];
  }

  set(indices, value) {
    const str = strides(this.shape);
    let idx = 0;
    for (let n = 0; n < this.ndim; n++) idx += indices[n] * str[n];
    this.data[idx] = value;
    return this;
  }

  unfold(mode) { return tensorUnfold(this.data, this.shape, mode); }

  cpDecompose(rank, maxIter = 100, tol = 1e-6) {
    return cpDecompose(this.data, this.shape, rank, maxIter, tol);
  }

  tuckerDecompose(ranks, maxIter = 20, tol = 1e-6) {
    return tuckerDecompose(this.data, this.shape, ranks, maxIter, tol);
  }
}

function inferShape(nested) {
  const shape = [];
  let cur = nested;
  while (Array.isArray(cur)) { shape.push(cur.length); cur = cur[0]; }
  return shape;
}

function flattenInto(nested, out, offset) {
  if (!Array.isArray(nested)) { out[offset] = nested; return offset + 1; }
  for (const child of nested) offset = flattenInto(child, out, offset);
  return offset;
}
