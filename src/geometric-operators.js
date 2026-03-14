// Geometric operators for the K'uhul++ manifold M.
// These implement the algebraic operations that underpin the K++ geometric runtime.
// Each operator corresponds to a GPU compute-shader concept, executed here on
// Float32Array buffers (the browser's SharedArrayBuffer-backed "GPU memory").

/**
 * Symbolic constants for geometric operators.
 * Pass these to MindBinder.applyOperator() or use them directly with the
 * standalone functions below.
 */
export const GEOMETRIC_OPS = Object.freeze({
    PRODUCT:    '⊗', // Geometric product  — matrix / tensor multiply
    COMPOSE:    '⊕', // Manifold composition — additive overlay (translate)
    DIFFERENCE: '⊖', // Difference in M
    PROJECT:    '⊛', // Constraint projection — normalise onto valid subspace
    VALIDATE:   '⊜', // Constraint validation — assert invariants
    CLAMP:      '⊝', // Clamp / threshold (ReLU-like)
    FOLD:       '⊞', // Fold / accumulate — reduce across a dimension
});

// ------------------------------------------------------------------ //
// ⊗  Matrix multiply (Geometric product)
// ------------------------------------------------------------------ //

/**
 * ⊗ Multiply two row-major matrices A (aRows × aCols) and B (aCols × bCols).
 * Returns a new Float32Array of shape (aRows × bCols).
 *
 * @param {Float32Array} A
 * @param {Float32Array} B
 * @param {number} aRows
 * @param {number} aCols
 * @param {number} bCols
 * @returns {Float32Array}
 */
export function matMul(A, B, aRows, aCols, bCols) {
    const C = new Float32Array(aRows * bCols);
    for (let r = 0; r < aRows; r++) {
        for (let c = 0; c < bCols; c++) {
            let sum = 0;
            for (let k = 0; k < aCols; k++) {
                sum += A[r * aCols + k] * B[k * bCols + c];
            }
            C[r * bCols + c] = sum;
        }
    }
    return C;
}

// ------------------------------------------------------------------ //
// ⊕  Manifold composition (translate)
// ------------------------------------------------------------------ //

/**
 * ⊕ Translate a point cloud: T'[i] = T[i] + v.
 * Each point is a vec3 (stride = 3 floats).  v is a 3-element offset.
 *
 * @param {Float32Array} T  - Point cloud (stride 3)
 * @param {ArrayLike<number>} v - Translation vector [dx, dy, dz]
 * @returns {Float32Array}
 */
export function translate(T, v) {
    const out = new Float32Array(T.length);
    const dx = v[0] || 0, dy = v[1] || 0, dz = v[2] || 0;
    for (let i = 0; i + 2 < T.length; i += 3) {
        out[i]     = T[i]     + dx;
        out[i + 1] = T[i + 1] + dy;
        out[i + 2] = T[i + 2] + dz;
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊖  Difference
// ------------------------------------------------------------------ //

/**
 * ⊖ Element-wise subtraction: C[i] = A[i] - B[i].
 *
 * @param {Float32Array} A
 * @param {Float32Array} B
 * @returns {Float32Array}
 */
export function subtract(A, B) {
    const len = Math.min(A.length, B.length);
    const C = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        C[i] = A[i] - B[i];
    }
    return C;
}

// ------------------------------------------------------------------ //
// ⊛  Constraint projection (normalise)
// ------------------------------------------------------------------ //

/**
 * ⊛ Project each vec3 onto the sphere of the given radius (normalise + scale).
 * Zero-length vectors are left unchanged.
 *
 * @param {Float32Array} T      - Point cloud (stride 3)
 * @param {number}       [radius=1] - Target sphere radius
 * @returns {Float32Array}
 */
export function project(T, radius = 1) {
    const out = new Float32Array(T.length);
    for (let i = 0; i + 2 < T.length; i += 3) {
        const x = T[i], y = T[i + 1], z = T[i + 2];
        const len = Math.sqrt(x * x + y * y + z * z);
        const s = len === 0 ? 0 : radius / len;
        out[i]     = x * s;
        out[i + 1] = y * s;
        out[i + 2] = z * s;
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊜  Constraint validation
// ------------------------------------------------------------------ //

/**
 * ⊜ Validate each vec3 against axis-aligned bounding box constraints.
 * Returns a Uint8Array: 1 = point is valid, 0 = out of bounds.
 *
 * @param {Float32Array}      T         - Point cloud (stride 3)
 * @param {ArrayLike<number>} minBounds - [minX, minY, minZ] (default −∞)
 * @param {ArrayLike<number>} maxBounds - [maxX, maxY, maxZ] (default +∞)
 * @returns {Uint8Array}
 */
export function validate(T, minBounds = [], maxBounds = []) {
    const n = Math.floor(T.length / 3);
    const results = new Uint8Array(n);
    const minX = minBounds[0] !== undefined ? minBounds[0] : -Infinity;
    const minY = minBounds[1] !== undefined ? minBounds[1] : -Infinity;
    const minZ = minBounds[2] !== undefined ? minBounds[2] : -Infinity;
    const maxX = maxBounds[0] !== undefined ? maxBounds[0] : Infinity;
    const maxY = maxBounds[1] !== undefined ? maxBounds[1] : Infinity;
    const maxZ = maxBounds[2] !== undefined ? maxBounds[2] : Infinity;
    for (let i = 0; i < n; i++) {
        const x = T[i * 3], y = T[i * 3 + 1], z = T[i * 3 + 2];
        results[i] = (x >= minX && x <= maxX &&
                      y >= minY && y <= maxY &&
                      z >= minZ && z <= maxZ) ? 1 : 0;
    }
    return results;
}

// ------------------------------------------------------------------ //
// ⊝  Clamp / threshold (ReLU proxy)
// ------------------------------------------------------------------ //

/**
 * ⊝ Clamp each element of T to [min, max].
 * Default min=0, max=+∞ gives a ReLU activation.
 *
 * @param {Float32Array} T
 * @param {number} [min=0]
 * @param {number} [max=Infinity]
 * @returns {Float32Array}
 */
export function clamp(T, min = 0, max = Infinity) {
    const out = new Float32Array(T.length);
    for (let i = 0; i < T.length; i++) {
        out[i] = Math.max(min, Math.min(max, T[i]));
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊞  Fold / accumulate (reduce)
// ------------------------------------------------------------------ //

/**
 * ⊞ Reduce T by summing groups of `stride` consecutive elements.
 * Used to collapse a dimension (e.g. sum-pooling).
 *
 * @param {Float32Array} T
 * @param {number} [stride=3] - Elements per output value
 * @returns {Float32Array}
 */
export function foldReduce(T, stride = 3) {
    const n = Math.floor(T.length / stride);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < stride; j++) {
            sum += T[i * stride + j];
        }
        out[i] = sum;
    }
    return out;
}

// ------------------------------------------------------------------ //
// Softmax (used in the BERT / attention path)
// ------------------------------------------------------------------ //

/**
 * Compute softmax over the entire array (geometric normalisation).
 *
 * @param {Float32Array} T
 * @returns {Float32Array}
 */
export function softmax(T) {
    const out = new Float32Array(T.length);
    let max = -Infinity;
    for (let i = 0; i < T.length; i++) if (T[i] > max) max = T[i];
    let sum = 0;
    for (let i = 0; i < T.length; i++) {
        out[i] = Math.exp(T[i] - max);
        sum += out[i];
    }
    if (sum > 0) for (let i = 0; i < out.length; i++) out[i] /= sum;
    return out;
}
