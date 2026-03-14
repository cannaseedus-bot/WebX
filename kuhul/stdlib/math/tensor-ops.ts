// K'UHUL++ Standard Library — Tensor Operations
// Implements the core glyph arithmetic operators:
//   ⊗ (tensor product), ⊕ (translation), ⊖ (difference)
// Also provides higher-order ops like ⊛ (convolution) and ⊞ (union).

// ------------------------------------------------------------------ //
// ⊗  Tensor product (outer product for 1-D tensors, element-wise for same-shape)
// ------------------------------------------------------------------ //

/**
 * ⊗ — Tensor product.
 * - Same shape → element-wise multiply.
 * - 1-D vectors of different lengths → outer product.
 *
 * @param a - Left operand
 * @param b - Right operand
 * @returns Result tensor
 */
export function tensorProduct(a: Float32Array, b: Float32Array): Float32Array {
    if (a.length === b.length) {
        // Element-wise multiply
        const out = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
        return out;
    }
    // Outer product: result[i * b.length + j] = a[i] * b[j]
    const out = new Float32Array(a.length * b.length);
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < b.length; j++) {
            out[i * b.length + j] = a[i] * b[j];
        }
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊕  Translation / bias addition
// ------------------------------------------------------------------ //

/**
 * ⊕ — Element-wise addition (translation in manifold M).
 * If tensors differ in length, the shorter is broadcast (cyclic).
 *
 * @param tensor - Main tensor
 * @param bias   - Bias / translation vector
 * @returns Translated tensor
 */
export function tensorTranslate(tensor: Float32Array, bias: Float32Array): Float32Array {
    const out = new Float32Array(tensor.length);
    for (let i = 0; i < tensor.length; i++) {
        out[i] = tensor[i] + bias[i % bias.length];
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊖  Tensor difference
// ------------------------------------------------------------------ //

/**
 * ⊖ — Element-wise subtraction (difference in manifold M).
 * Shorter operand is broadcast cyclically.
 *
 * @param a - Minuend
 * @param b - Subtrahend
 * @returns Difference tensor
 */
export function tensorDifference(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
        out[i] = a[i] - b[i % b.length];
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊛  1-D Convolution
// ------------------------------------------------------------------ //

/**
 * ⊛ — Discrete 1-D convolution.
 *
 * @param signal - Input signal
 * @param kernel - Convolution kernel
 * @returns Convolved signal (same length as `signal`)
 */
export function tensorConvolution(signal: Float32Array, kernel: Float32Array): Float32Array {
    const out = new Float32Array(signal.length);
    const half = Math.floor(kernel.length / 2);
    for (let i = 0; i < signal.length; i++) {
        let acc = 0;
        for (let j = 0; j < kernel.length; j++) {
            const si = i - half + j;
            if (si >= 0 && si < signal.length) acc += signal[si] * kernel[j];
        }
        out[i] = acc;
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊞  Union / element-wise addition (alias for ⊕ with explicit broadcast)
// ------------------------------------------------------------------ //

/**
 * ⊞ — Union of two tensors via element-wise addition.
 * Output length equals the maximum of the two input lengths.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Union tensor
 */
export function tensorUnion(a: Float32Array, b: Float32Array): Float32Array {
    const len = Math.max(a.length, b.length);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        out[i] = (a[i] ?? 0) + (b[i] ?? 0);
    }
    return out;
}

// ------------------------------------------------------------------ //
// ⊝  Complement / negation
// ------------------------------------------------------------------ //

/**
 * ⊝ — Negate all elements of a tensor.
 *
 * @param a - Input tensor
 * @returns Negated tensor
 */
export function tensorComplement(a: Float32Array): Float32Array {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = -a[i];
    return out;
}

// ------------------------------------------------------------------ //
// Utility: reduce with summation
// ------------------------------------------------------------------ //

/**
 * Compute the L2 norm of a tensor.
 */
export function tensorNorm(a: Float32Array): number {
    let sum = 0;
    for (const v of a) sum += v * v;
    return Math.sqrt(sum);
}

/**
 * Normalize a tensor to unit L2 norm.
 */
export function tensorNormalize(a: Float32Array): Float32Array {
    const norm = tensorNorm(a);
    if (norm < 1e-9) return new Float32Array(a.length);
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] / norm;
    return out;
}

/**
 * Softmax of a tensor.
 */
export function tensorSoftmax(a: Float32Array): Float32Array {
    const maxVal = Math.max(...a);
    const exps = a.map(v => Math.exp(v - maxVal));
    const sum  = exps.reduce((s, v) => s + v, 0);
    return new Float32Array(exps.map(v => v / sum));
}
