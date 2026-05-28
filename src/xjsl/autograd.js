// XJSL autograd rules — backward definitions for all supported ops.
// Each rule specifies how to propagate ∂L/∂out back to ∂L/∂inputs.
// Format: { inputs, output, backward: description string, grad_fn }
//
// grad_fn(grad_out, ...saved_tensors) → { [input_name]: grad }
// All tensors are Float32Array; shapes are tracked by caller.

// ─── Elementwise ops ─────────────────────────────────────────────────────────

function grad_add(grad_out) {
  return { x: grad_out.slice(), y: grad_out.slice() };
}

function grad_mul(grad_out, x, y) {
  const gx = new Float32Array(grad_out.length);
  const gy = new Float32Array(grad_out.length);
  for (let i = 0; i < grad_out.length; i++) {
    gx[i] = grad_out[i] * y[i];
    gy[i] = grad_out[i] * x[i];
  }
  return { x: gx, y: gy };
}

function grad_clamp(grad_out, x, lo, hi) {
  const gx = new Float32Array(grad_out.length);
  for (let i = 0; i < grad_out.length; i++) {
    gx[i] = (x[i] >= lo && x[i] <= hi) ? grad_out[i] : 0;
  }
  return { x: gx };
}

function grad_relu(grad_out, x) {
  const gx = new Float32Array(grad_out.length);
  for (let i = 0; i < grad_out.length; i++) {
    gx[i] = x[i] > 0 ? grad_out[i] : 0;
  }
  return { x: gx };
}

function grad_sigmoid(grad_out, x) {
  const gx = new Float32Array(grad_out.length);
  for (let i = 0; i < grad_out.length; i++) {
    const s = 1 / (1 + Math.exp(-x[i]));
    gx[i] = grad_out[i] * s * (1 - s);
  }
  return { x: gx };
}

function grad_tanh(grad_out, x) {
  const gx = new Float32Array(grad_out.length);
  for (let i = 0; i < grad_out.length; i++) {
    const t = Math.tanh(x[i]);
    gx[i] = grad_out[i] * (1 - t * t);
  }
  return { x: gx };
}

// GELU backward: d/dx [0.5x(1+tanh(√(2/π)(x+0.044715x³)))]
function grad_gelu(grad_out, x) {
  const gx = new Float32Array(grad_out.length);
  const K  = Math.sqrt(2 / Math.PI);
  const C  = 0.044715;
  for (let i = 0; i < grad_out.length; i++) {
    const xi   = x[i];
    const arg  = K * (xi + C * xi * xi * xi);
    const t    = Math.tanh(arg);
    const sech2 = 1 - t * t;
    const darg = K * (1 + 3 * C * xi * xi);
    gx[i] = grad_out[i] * (0.5 * (1 + t) + 0.5 * xi * sech2 * darg);
  }
  return { x: gx };
}

// ─── Softmax backward ────────────────────────────────────────────────────────

// Numerically stable softmax backward.
// s is the softmax output (saved forward output); grad_out is ∂L/∂s.
// ∂L/∂x_i = s_i * (∂L/∂s_i - ∑_j ∂L/∂s_j * s_j)
function grad_softmax(grad_out, s) {
  const gx  = new Float32Array(grad_out.length);
  let dot = 0;
  for (let i = 0; i < s.length; i++) dot += grad_out[i] * s[i];
  for (let i = 0; i < s.length; i++) gx[i] = s[i] * (grad_out[i] - dot);
  return { x: gx };
}

// ─── Normalization backward ───────────────────────────────────────────────────

// RMSNorm backward: y = x / rms(x) * w, rms = sqrt(mean(x²) + eps)
// ∂L/∂x and ∂L/∂w computed from saved x, rms, and w.
function grad_rmsnorm(grad_out, x, w, eps = 1e-6) {
  const n   = x.length;
  const rms = Math.sqrt(x.reduce((s, v) => s + v * v, 0) / n + eps);
  const gw  = new Float32Array(n);
  const gx  = new Float32Array(n);
  // ∂L/∂w_i = ∂L/∂y_i * (x_i / rms)
  // ∂L/∂x_i = (∂L/∂y_i * w_i / rms) - (∑_j ∂L/∂y_j * w_j * x_j/rms) * x_i / (n*rms²)
  let sumDot = 0;
  for (let i = 0; i < n; i++) {
    gw[i]   = grad_out[i] * x[i] / rms;
    sumDot += grad_out[i] * w[i] * x[i];
  }
  const scale = sumDot / (n * rms * rms * rms);
  for (let i = 0; i < n; i++) {
    gx[i] = grad_out[i] * w[i] / rms - scale * x[i];
  }
  return { x: gx, w: gw };
}

// LayerNorm backward: y = (x - mean) / std * w + b, std = sqrt(var + eps)
function grad_layernorm(grad_out, x, w, eps = 1e-5) {
  const n    = x.length;
  let   mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  variance /= n;
  const std  = Math.sqrt(variance + eps);
  const xhat = new Float32Array(n);
  for (let i = 0; i < n; i++) xhat[i] = (x[i] - mean) / std;

  const gw = new Float32Array(n);
  const gb = new Float32Array(n);
  const gx = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    gw[i] = grad_out[i] * xhat[i];
    gb[i] = grad_out[i];
  }

  // ∂L/∂x: project out mean and variance from the scaled gradient
  let sum1 = 0, sum2 = 0;
  for (let i = 0; i < n; i++) {
    sum1 += grad_out[i] * w[i];
    sum2 += grad_out[i] * w[i] * xhat[i];
  }
  for (let i = 0; i < n; i++) {
    gx[i] = (grad_out[i] * w[i] - sum1 / n - xhat[i] * sum2 / n) / std;
  }
  return { x: gx, w: gw, b: gb };
}

// ─── Reduction backward ───────────────────────────────────────────────────────

function grad_reduce_sum(grad_scalar, n) {
  return { x: new Float32Array(n).fill(grad_scalar) };
}

function grad_reduce_mean(grad_scalar, n) {
  return { x: new Float32Array(n).fill(grad_scalar / n) };
}

// grad_max: only the max element gets the gradient (ties: first occurrence).
function grad_reduce_max(grad_scalar, x) {
  let maxVal = -Infinity, maxIdx = 0;
  for (let i = 0; i < x.length; i++) if (x[i] > maxVal) { maxVal = x[i]; maxIdx = i; }
  const gx = new Float32Array(x.length);
  gx[maxIdx] = grad_scalar;
  return { x: gx };
}

// ─── Matmul backward ─────────────────────────────────────────────────────────

// C = A @ B, shapes: A=[M,K], B=[K,N], C=[M,N]
// ∂L/∂A = ∂L/∂C @ B^T  [M,K]
// ∂L/∂B = A^T @ ∂L/∂C  [K,N]
function grad_matmul(grad_C, A, B, M, K, N) {
  const gA = new Float32Array(M * K);
  const gB = new Float32Array(K * N);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += grad_C[i * N + j] * B[k * N + j];
      gA[i * K + k] = s;
    }
  }
  for (let k = 0; k < K; k++) {
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let i = 0; i < M; i++) s += A[i * K + k] * grad_C[i * N + j];
      gB[k * N + j] = s;
    }
  }
  return { A: gA, B: gB };
}

// dot(a, b) = ∑ a_i * b_i
// ∂L/∂a_i = ∂L/∂out * b_i, ∂L/∂b_i = ∂L/∂out * a_i
function grad_dot(grad_scalar, a, b) {
  const ga = new Float32Array(a.length);
  const gb = new Float32Array(b.length);
  for (let i = 0; i < a.length; i++) {
    ga[i] = grad_scalar * b[i];
    gb[i] = grad_scalar * a[i];
  }
  return { a: ga, b: gb };
}

// ─── Rule table ───────────────────────────────────────────────────────────────

export const AUTOGRAD_RULES = Object.freeze({
  add: {
    inputs:   ['x', 'y'],
    output:   'out',
    backward: '∂L/∂x = ∂L/∂out, ∂L/∂y = ∂L/∂out',
    grad_fn:  grad_add,
  },
  mul: {
    inputs:   ['x', 'y'],
    output:   'out',
    backward: '∂L/∂x = ∂L/∂out * y, ∂L/∂y = ∂L/∂out * x',
    grad_fn:  grad_mul,
  },
  clamp: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x = ∂L/∂out if lo≤x≤hi else 0',
    grad_fn:  grad_clamp,
  },
  relu: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x = ∂L/∂out * (x > 0)',
    grad_fn:  grad_relu,
  },
  sigmoid: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x = ∂L/∂out * s * (1 - s), s = sigmoid(x)',
    grad_fn:  grad_sigmoid,
  },
  tanh: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x = ∂L/∂out * (1 - tanh²(x))',
    grad_fn:  grad_tanh,
  },
  gelu: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x via d/dx[0.5x(1+tanh(√(2/π)(x+0.044715x³)))]',
    grad_fn:  grad_gelu,
  },
  softmax: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x_i = s_i*(∂L/∂s_i - ∑_j ∂L/∂s_j*s_j)',
    grad_fn:  grad_softmax,
  },
  rmsnorm: {
    inputs:   ['x', 'w'],
    output:   'out',
    backward: '∂L/∂x = (∂L/∂y*w/rms) - dot_term*x/(n*rms³); ∂L/∂w = ∂L/∂y*x/rms',
    grad_fn:  grad_rmsnorm,
  },
  layernorm: {
    inputs:   ['x', 'w', 'b'],
    output:   'out',
    backward: 'project-out mean + variance from scaled gradient, per PyTorch convention',
    grad_fn:  grad_layernorm,
  },
  reduce_sum: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x_i = ∂L/∂out (broadcast)',
    grad_fn:  grad_reduce_sum,
  },
  reduce_mean: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x_i = ∂L/∂out / n',
    grad_fn:  grad_reduce_mean,
  },
  reduce_max: {
    inputs:   ['x'],
    output:   'out',
    backward: '∂L/∂x_i = ∂L/∂out if x_i == max else 0 (first max wins)',
    grad_fn:  grad_reduce_max,
  },
  matmul: {
    inputs:   ['A', 'B'],
    output:   'C',
    backward: '∂L/∂A = ∂L/∂C @ B^T; ∂L/∂B = A^T @ ∂L/∂C',
    grad_fn:  grad_matmul,
  },
  dot: {
    inputs:   ['a', 'b'],
    output:   'out',
    backward: '∂L/∂a_i = ∂L/∂out * b_i; ∂L/∂b_i = ∂L/∂out * a_i',
    grad_fn:  grad_dot,
  },
});

export default AUTOGRAD_RULES;
