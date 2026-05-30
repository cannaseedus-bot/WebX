// math-layer.js — @mathml / @fibonacci / @pi / @matmul / @linalg
//                  @loop / @biginteger / @formula / @zero / @vigesimal
//
// Mathematical foundation stack (bottom → top):
//
//   @zero / @vigesimal   — identity and base representations
//   @pi / @fibonacci     — transcendental constants + sequences
//   @biginteger          — arbitrary precision arithmetic
//   @loop                — iteration primitives
//   @matmul              — core matrix multiply kernel
//   @linalg              — linear algebra suite (decomp, eigen, solve)
//   @formula             — symbolic expression engine
//   @mathml              — MathML parse + LaTeX render
//
// Mayan connection: @vigesimal (base-20) underlies the Long Count calendar.
// @fibonacci golden ratio φ = (1+√5)/2 is the geodesic curvature constant
// in the Mayan orchestrator. @zero is the Mayan invention: the dot-bar system.

// ─── @zero — additive identity ────────────────────────────────────────────────

export const ZERO = Object.freeze({
  int:    0,
  float:  0.0,
  vector: (n) => new Float64Array(n),
  matrix: (m, n) => new Float64Array(m * n),
  poly:   [],
  modular:(p)  => 0,
  properties: ['a + 0 = a', 'a * 0 = 0', '0 / a = 0 (a≠0)', '0^0 = 1 (convention)'],
});

// ─── @vigesimal — base-20 (Mayan) ────────────────────────────────────────────

export const VIGESIMAL_DIGITS = '0123456789ABCDEFGHIJ';

export class Vigesimal {
  static fromDecimal(n) {
    if (n === 0) return '0';
    let result = '';
    let v = Math.abs(Math.trunc(n));
    while (v > 0) {
      result = VIGESIMAL_DIGITS[v % 20] + result;
      v = Math.floor(v / 20);
    }
    return (n < 0 ? '-' : '') + result;
  }

  static toDecimal(s) {
    return [...s.toUpperCase()].reduce((acc, c) => {
      const d = VIGESIMAL_DIGITS.indexOf(c);
      if (d < 0) throw new Error(`Invalid vigesimal digit: ${c}`);
      return acc * 20 + d;
    }, 0);
  }

  static bitsPerDigit() { return Math.log2(20); }  // ≈ 4.32 bits

  // Mayan long-count: coefficients [baktun, katun, tun, uinal, kin]
  // Not pure base-20: uinal position = 18, rest = 20
  static longCount(days) {
    const kin   =  days % 20;        const r1 = Math.floor(days / 20);
    const uinal = r1 % 18;           const r2 = Math.floor(r1 / 18);
    const tun   = r2 % 20;           const r3 = Math.floor(r2 / 20);
    const katun = r3 % 20;
    const baktun= Math.floor(r3 / 20);
    return { baktun, katun, tun, uinal, kin };
  }
}

// ─── @pi — transcendental constants ──────────────────────────────────────────

export const PI_CONSTANTS = Object.freeze({
  pi:    Math.PI,                        // 3.14159265358979...
  tau:   2 * Math.PI,                    // 6.28318530717958...
  e:     Math.E,                         // 2.71828182845904...
  phi:   (1 + Math.sqrt(5)) / 2,         // 1.61803398874989... golden ratio
  psi:   (1 - Math.sqrt(5)) / 2,         // -0.6180339887...
  sqrt2: Math.SQRT2,                     // 1.41421356...
  sqrt3: Math.sqrt(3),
  sqrt5: Math.sqrt(5),
  pi_sq: Math.PI ** 2,                   // 9.8696...
  sqrt_pi: Math.sqrt(Math.PI),           // 1.7724...
  ln2:   Math.LN2,
  ln10:  Math.LN10,
  log2e: Math.LOG2E,
  log10e:Math.LOG10E,
});

export class PiCompute {
  // Leibniz series (slow convergence, educational)
  static leibniz(terms = 1000) {
    let pi = 0;
    for (let k = 0; k < terms; k++) pi += (k % 2 === 0 ? 1 : -1) / (2*k + 1);
    return 4 * pi;
  }

  // BBP (Bailey-Borwein-Plouffe) — computes hex digits of pi
  static bbp(terms = 50) {
    let pi = 0;
    for (let k = 0; k < terms; k++) {
      const k8 = 8 * k;
      pi += (1 / 16**k) * (
        4/(k8+1) - 2/(k8+4) - 1/(k8+5) - 1/(k8+6)
      );
    }
    return pi;
  }

  // Monte Carlo estimate of π
  static monteCarlo(samples = 1_000_000) {
    let inside = 0;
    for (let i = 0; i < samples; i++) {
      const x = Math.random(), y = Math.random();
      if (x*x + y*y <= 1) inside++;
    }
    return 4 * inside / samples;
  }
}

// ─── @fibonacci — sequence + golden ratio ────────────────────────────────────

export class Fibonacci {
  // Iterative (O(n), double-precision safe up to F(78))
  static iter(n) {
    if (n <= 0) return 0n;
    if (n === 1) return 1n;
    let [a, b] = [0n, 1n];
    for (let i = 2; i <= n; i++) [a, b] = [b, a + b];
    return b;
  }

  // Matrix exponentiation — O(log n), uses BigInt
  static matrix(n) {
    if (n <= 0) return 0n;
    const pow = (m, e) => {
      if (e === 1n) return m;
      if (e % 2n === 0n) { const h = pow(m, e/2n); return mul(h, h); }
      return mul(m, pow(m, e - 1n));
    };
    const mul = ([a,b,c,d], [e,f,g,h]) => [
      a*e+b*g, a*f+b*h, c*e+d*g, c*f+d*h
    ];
    const [,r] = pow([1n,1n,1n,0n], BigInt(n));
    return r;
  }

  // Fast doubling (O(log n), BigInt)
  static fast(n) {
    const _fd = (k) => {
      if (k === 0) return [0n, 1n];
      const [a, b] = _fd(Math.floor(k/2));
      const c = a * (2n*b - a);
      const d = a*a + b*b;
      return k % 2 === 0 ? [c, d] : [d, c+d];
    };
    return _fd(n)[0];
  }

  // Generate n Fibonacci numbers
  static series(n) {
    const result = [0n, 1n];
    for (let i = 2; i < n; i++) result.push(result[i-1] + result[i-2]);
    return result.slice(0, n);
  }

  // Binet's formula (float approximation)
  static binet(n) {
    const { phi, psi, sqrt5 } = PI_CONSTANTS;
    return Math.round((phi**n - psi**n) / sqrt5);
  }

  // Zeckendorf decomposition (sum of non-consecutive Fibonacci numbers)
  static zeckendorf(n) {
    const fibs = [];
    let f = 1n, prev = 0n;
    while (f <= BigInt(n)) { fibs.push(f); [prev,f] = [f, prev+f]; }
    fibs.reverse();
    let rem = BigInt(n);
    const terms = [];
    for (const fib of fibs) {
      if (fib <= rem) { terms.push(Number(fib)); rem -= fib; }
    }
    return terms;
  }
}

// ─── @biginteger — arbitrary precision ───────────────────────────────────────

export class BigInteger {
  // Use native BigInt — just wrap with utility methods
  static add(a, b)      { return BigInt(a) + BigInt(b); }
  static mul(a, b)      { return BigInt(a) * BigInt(b); }
  static pow(a, e)      { return BigInt(a) ** BigInt(e); }
  static mod(a, m)      { return ((BigInt(a) % BigInt(m)) + BigInt(m)) % BigInt(m); }
  static modpow(base, exp, mod) {
    let b = BigInt(base), e = BigInt(exp), m = BigInt(mod), r = 1n;
    b = b % m;
    while (e > 0n) {
      if (e % 2n === 1n) r = r * b % m;
      e = e >> 1n;
      b = b * b % m;
    }
    return r;
  }

  // Karatsuba multiplication (divide-and-conquer, O(n^1.585))
  static karatsuba(x, y) {
    x = BigInt(x); y = BigInt(y);
    if (x < 1000n || y < 1000n) return x * y;
    const n  = BigInt(Math.max(x.toString().length, y.toString().length));
    const m  = n / 2n;
    const base = 10n ** m;
    const [a, b] = [x / base, x % base];
    const [c, d] = [y / base, y % base];
    const z0 = BigInteger.karatsuba(a, c);
    const z1 = BigInteger.karatsuba(b, d);
    const z2 = BigInteger.karatsuba(a + b, c + d) - z0 - z1;
    return z0 * (base * base) + z2 * base + z1;
  }

  // Miller-Rabin primality test
  static isPrime(n, witnesses = 40) {
    n = BigInt(n);
    if (n < 2n) return false;
    if (n < 4n) return true;
    if (n % 2n === 0n) return false;
    let d = n - 1n, r = 0n;
    while (d % 2n === 0n) { d /= 2n; r++; }
    const SMALL = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];
    for (const a of SMALL.slice(0, witnesses)) {
      if (a >= n) continue;
      let x = BigInteger.modpow(a, d, n);
      if (x === 1n || x === n - 1n) continue;
      let composite = true;
      for (let i = 0n; i < r - 1n; i++) {
        x = x * x % n;
        if (x === n - 1n) { composite = false; break; }
      }
      if (composite) return false;
    }
    return true;
  }
}

// ─── @loop — iteration control ────────────────────────────────────────────────

export class Loop {
  // for (i in range(start,end,step))
  static *range(start, end, step = 1) {
    for (let i = start; step > 0 ? i < end : i > end; i += step) yield i;
  }

  // Converging loop: iterate until |value - prev| < epsilon
  static converge(fn, init, epsilon = 1e-10, maxIter = 10_000) {
    let prev = init, curr = fn(prev), iters = 0;
    while (Math.abs(curr - prev) > epsilon && iters++ < maxIter) {
      prev = curr;
      curr = fn(prev);
    }
    return { value: curr, iterations: iters, converged: iters < maxIter };
  }

  // Binary exponentiation (square-and-multiply)
  static binpow(base, exp, mul) {
    let result = null, b = base;
    while (exp > 0) {
      if (exp & 1) result = result === null ? b : mul(result, b);
      b = mul(b, b);
      exp >>= 1;
    }
    return result;
  }

  // Unroll factor-4 vectorization hint
  static unrolled4(n, fn) {
    let i = 0;
    for (; i <= n - 4; i += 4) { fn(i); fn(i+1); fn(i+2); fn(i+3); }
    for (; i < n; i++) fn(i);
  }
}

// ─── @matmul — matrix multiply kernel ────────────────────────────────────────

export class MatMul {
  // Naive O(MNK) — reference implementation
  static naive(A, B, M, K, N) {
    const C = new Float64Array(M * N);
    for (let i = 0; i < M; i++)
      for (let k = 0; k < K; k++) {
        const a = A[i*K+k];
        for (let j = 0; j < N; j++) C[i*N+j] += a * B[k*N+j];
      }
    return C;
  }

  // Tiled / blocked O(MNK) with better cache reuse
  static tiled(A, B, M, K, N, tile = 32) {
    const C = new Float64Array(M * N);
    for (let ii = 0; ii < M; ii += tile)
      for (let kk = 0; kk < K; kk += tile)
        for (let jj = 0; jj < N; jj += tile)
          for (let i = ii; i < Math.min(ii+tile,M); i++)
            for (let k = kk; k < Math.min(kk+tile,K); k++) {
              const a = A[i*K+k];
              for (let j = jj; j < Math.min(jj+tile,N); j++) C[i*N+j] += a * B[k*N+j];
            }
    return C;
  }

  // Strassen O(n^2.807) for square n×n (n must be power of 2)
  static strassen(A, B, n) {
    if (n <= 64) return MatMul.naive(A, B, n, n, n);
    const h = n >> 1;
    const split = (M, r, c) => {
      const S = new Float64Array(h*h);
      for (let i=0;i<h;i++) for (let j=0;j<h;j++) S[i*h+j]=M[(i+r)*n+(j+c)];
      return S;
    };
    const add  = (X,Y) => X.map((v,i)=>v+Y[i]);
    const sub  = (X,Y) => X.map((v,i)=>v-Y[i]);
    const mul  = (X,Y) => MatMul.strassen(X,Y,h);

    const [A11,A12,A21,A22] = [split(A,0,0),split(A,0,h),split(A,h,0),split(A,h,h)];
    const [B11,B12,B21,B22] = [split(B,0,0),split(B,0,h),split(B,h,0),split(B,h,h)];

    const P1=mul(add(A11,A22),add(B11,B22)), P2=mul(add(A21,A22),B11);
    const P3=mul(A11,sub(B12,B22)),          P4=mul(A22,sub(B21,B11));
    const P5=mul(add(A11,A12),B22),          P6=mul(sub(A21,A11),add(B11,B12));
    const P7=mul(sub(A12,A22),add(B21,B22));

    const C = new Float64Array(n*n);
    for (let i=0;i<h;i++) for (let j=0;j<h;j++) {
      C[i*n+j]         = P1[i*h+j]+P4[i*h+j]-P5[i*h+j]+P7[i*h+j];
      C[i*n+j+h]       = P3[i*h+j]+P5[i*h+j];
      C[(i+h)*n+j]     = P2[i*h+j]+P4[i*h+j];
      C[(i+h)*n+j+h]   = P1[i*h+j]-P2[i*h+j]+P3[i*h+j]+P6[i*h+j];
    }
    return C;
  }
}

// ─── @linalg — linear algebra suite ──────────────────────────────────────────

export class LinAlg {
  // Dot product
  static dot(a, b)  { return a.reduce((s,v,i) => s + v*b[i], 0); }

  // L2 norm
  static norm2(v)   { return Math.sqrt(LinAlg.dot(v, v)); }

  // Vector addition / scale
  static add(a, b)  { return a.map((v,i) => v + b[i]); }
  static scale(a,s) { return a.map(v => v * s); }

  // LU decomposition (Doolittle, in-place pivot)
  static lu(A, n) {
    const L = new Float64Array(n*n), U = new Float64Array(n*n);
    for (let i=0;i<n;i++) L[i*n+i] = 1;
    for (let i=0;i<n;i++) U[i*n+i] = A[i*n+i];
    for (let k=0;k<n;k++) {
      for (let i=k+1;i<n;i++) {
        const m = A[i*n+k] / A[k*n+k];
        L[i*n+k] = m;
        for (let j=k;j<n;j++) A[i*n+j] -= m * A[k*n+j];
      }
      for (let j=k;j<n;j++) U[k*n+j] = A[k*n+j];
    }
    return { L, U };
  }

  // Forward substitution Ly = b
  static forwardSub(L, b, n) {
    const y = new Float64Array(n);
    for (let i=0;i<n;i++) {
      y[i] = b[i];
      for (let j=0;j<i;j++) y[i] -= L[i*n+j] * y[j];
      y[i] /= L[i*n+i];
    }
    return y;
  }

  // Back substitution Ux = y
  static backSub(U, y, n) {
    const x = new Float64Array(n);
    for (let i=n-1;i>=0;i--) {
      x[i] = y[i];
      for (let j=i+1;j<n;j++) x[i] -= U[i*n+j] * x[j];
      x[i] /= U[i*n+i];
    }
    return x;
  }

  // Solve Ax = b via LU
  static solve(A, b, n) {
    const Ac = Float64Array.from(A);
    const { L, U } = LinAlg.lu(Ac, n);
    const y = LinAlg.forwardSub(L, b, n);
    return LinAlg.backSub(U, y, n);
  }

  // Power iteration for dominant eigenvalue
  static powerIter(A, n, maxIter = 100, tol = 1e-10) {
    let v = new Float64Array(n).fill(1);
    let lambda = 0;
    for (let iter = 0; iter < maxIter; iter++) {
      const Av  = new Float64Array(n);
      for (let i=0;i<n;i++) for (let j=0;j<n;j++) Av[i] += A[i*n+j]*v[j];
      const newLambda = LinAlg.norm2(Av);
      v = LinAlg.scale(Av, 1/newLambda);
      if (Math.abs(newLambda - lambda) < tol) { lambda = newLambda; break; }
      lambda = newLambda;
    }
    return { eigenvalue: lambda, eigenvector: Array.from(v) };
  }
}

// ─── @formula — symbolic expression engine ───────────────────────────────────

export class Formula {
  // Symbolic derivative of an AST (subset: +, -, *, /, ^, constant, variable)
  static diff(ast, wrt = 'x') {
    if (!ast) return { val: 0 };
    if ('val' in ast) return { val: 0 };
    if ('var' in ast) return ast.var === wrt ? { val: 1 } : { val: 0 };
    const { op, left, right } = ast;
    switch (op) {
      case '+': return { op: '+', left: Formula.diff(left, wrt), right: Formula.diff(right, wrt) };
      case '-': return { op: '-', left: Formula.diff(left, wrt), right: Formula.diff(right, wrt) };
      case '*': return { op: '+',
        left:  { op: '*', left: Formula.diff(left, wrt), right },
        right: { op: '*', left, right: Formula.diff(right, wrt) } };
      case '/': return { op: '/',
        left:  { op: '-',
          left:  { op: '*', left: Formula.diff(left, wrt), right },
          right: { op: '*', left, right: Formula.diff(right, wrt) } },
        right: { op: '*', left: right, right } };
      case '^': // simple case: f(x)^n → n*f(x)^(n-1)*f'(x)
        return { op: '*',
          left: { op: '*', left: right, right: { op: '^', left, right: { op:'-', left:right, right:{val:1} } } },
          right: Formula.diff(left, wrt) };
      default: return { val: 0 };
    }
  }

  // Simplify constant-folding (val op val)
  static simplify(ast) {
    if (!ast || 'val' in ast || 'var' in ast) return ast;
    const l = Formula.simplify(ast.left);
    const r = Formula.simplify(ast.right);
    if ('val' in l && 'val' in r) {
      const OPS = { '+': (a,b)=>a+b, '-': (a,b)=>a-b, '*': (a,b)=>a*b, '/': (a,b)=>a/b, '^': (a,b)=>a**b };
      return { val: OPS[ast.op]?.(l.val, r.val) ?? 0 };
    }
    // Identity rules
    if (ast.op==='+' && 'val' in r && r.val===0) return l;
    if (ast.op==='+' && 'val' in l && l.val===0) return r;
    if (ast.op==='*' && 'val' in r && r.val===1) return l;
    if (ast.op==='*' && 'val' in l && l.val===1) return r;
    if (ast.op==='*' && (('val' in r && r.val===0) || ('val' in l && l.val===0))) return { val: 0 };
    return { op: ast.op, left: l, right: r };
  }

  // Riemann sum (numerical integration)
  static integrate(fn, a, b, steps = 1000) {
    const dx = (b - a) / steps;
    let sum = 0;
    for (let i = 0; i < steps; i++) sum += fn(a + (i + 0.5) * dx);
    return sum * dx;
  }

  // Newton's method root finding
  static root(fn, dfn, x0 = 1, tol = 1e-12, maxIter = 100) {
    let x = x0;
    for (let i = 0; i < maxIter; i++) {
      const fx = fn(x), dfx = dfn(x);
      if (Math.abs(fx) < tol) break;
      x -= fx / dfx;
    }
    return x;
  }
}

// ─── @mathml — MathML parse + LaTeX render ────────────────────────────────────

export class MathML {
  // Parse simple content MathML XML to AST (subset of MathML spec)
  static parse(xml) {
    // Very simplified: extracts <cn> (number), <ci> (identifier), <plus/> etc.
    const numRe  = /<cn>([^<]+)<\/cn>/g;
    const varRe  = /<ci>([^<]+)<\/ci>/g;
    const opMap  = { 'plus': '+', 'times': '*', 'minus': '-', 'divide': '/', 'power': '^' };

    let m;
    const nums = []; while ((m = numRe.exec(xml))  !== null) nums.push(parseFloat(m[1]));
    const vars = []; while ((m = varRe.exec(xml))  !== null) vars.push(m[1].trim());

    const ops = [];
    for (const [tag, sym] of Object.entries(opMap)) {
      if (xml.includes(`<${tag}/>`)) ops.push(sym);
    }

    return { nums, vars, ops, raw: xml };
  }

  // Render AST to LaTeX string
  static toLatex(ast) {
    if (!ast) return '';
    if ('val' in ast) return String(ast.val);
    if ('var' in ast) return ast.var;
    const l = MathML.toLatex(ast.left);
    const r = MathML.toLatex(ast.right);
    switch (ast.op) {
      case '+': return `${l} + ${r}`;
      case '-': return `${l} - ${r}`;
      case '*': return `${l} \\cdot ${r}`;
      case '/': return `\\frac{${l}}{${r}}`;
      case '^': return `{${l}}^{${r}}`;
      default:  return `(${l} ${ast.op} ${r})`;
    }
  }
}

// ─── Register all math namespaces into XCFENodeRuntime ────────────────────────

export function registerMathNamespaces(rt) {
  rt._handlers.set('@zero', (val, ctx) => {
    const type = Object.keys(val)[0] ?? 'int';
    const n    = val.dimensions ?? val.n ?? 1;
    ctx['_zero'] = type === 'vector' ? ZERO.vector(n)
                 : type === 'matrix' ? ZERO.matrix(n, n)
                 : ZERO.int;
    if (val['@store']) ctx[val['@store']] = ctx['_zero'];
  });

  rt._handlers.set('@vigesimal', (val, ctx) => {
    const n = val['@decimal_to_vigesimal'] ?? val.n ?? 0;
    const r = Vigesimal.fromDecimal(Number(n));
    if (val['@store']) ctx[val['@store']] = r;
    return r;
  });

  rt._handlers.set('@pi', (val, ctx) => {
    const key = val['@constant'] ?? 'pi';
    const c   = PI_CONSTANTS[key] ?? PI_CONSTANTS.pi;
    if (val['@store']) ctx[val['@store']] = c;
    return c;
  });

  rt._handlers.set('@fibonacci', (val, ctx) => {
    const n      = val.n ?? val['@n'] ?? 10;
    const method = val['@method'] ?? 'fast';
    const result = method === 'matrix' ? Fibonacci.matrix(n)
                 : method === 'series' ? Fibonacci.series(n)
                 : Fibonacci.fast(n);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@biginteger', (val, ctx) => {
    const op  = Object.keys(val).find(k => !k.startsWith('@')) ?? 'identity';
    const a   = BigInt(val.a ?? 0);
    const b   = BigInt(val.b ?? 1);
    const r   = op === 'add' ? BigInteger.add(a,b)
              : op === 'mul' ? BigInteger.mul(a,b)
              : op === 'pow' ? BigInteger.pow(a,b)
              : op === 'mod' ? BigInteger.mod(a,b)
              : op === 'prime' ? BigInteger.isPrime(a)
              : a;
    if (val['@store']) ctx[val['@store']] = r;
    return r;
  });

  rt._handlers.set('@loop', (val, ctx) => {
    const type = Object.keys(val).find(k => !k.startsWith('@')) ?? 'range';
    if (type === 'converge') {
      const fn = val.function ? Function('x', `return(${val.function})`) : x => x;
      const r  = Loop.converge(fn, val.init ?? 1, val.epsilon ?? 1e-10);
      if (val['@store']) ctx[val['@store']] = r;
      return r;
    }
    const { start=0, end=10, step=1 } = val[type] ?? val;
    const items = [...Loop.range(start, end, step)];
    if (val['@store']) ctx[val['@store']] = items;
    return items;
  });

  rt._handlers.set('@matmul', (val, ctx) => {
    const A = ctx[val.A] ?? val.A ?? [];
    const B = ctx[val.B] ?? val.B ?? [];
    const M = val.M ?? Math.sqrt(A.length);
    const K = val.K ?? M;
    const N = val.N ?? M;
    const method = val['@method'] ?? 'tiled';
    const FA = Float64Array.from(A), FB = Float64Array.from(B);
    const C  = method === 'strassen' ? MatMul.strassen(FA, FB, M)
             : method === 'naive'    ? MatMul.naive(FA, FB, M, K, N)
             : MatMul.tiled(FA, FB, M, K, N);
    if (val['@store']) ctx[val['@store']] = C;
    return C;
  });

  rt._handlers.set('@linalg', (val, ctx) => {
    const op = Object.keys(val).find(k => !k.startsWith('@')) ?? 'dot';
    const a  = ctx[val[op]?.a] ?? val[op]?.a ?? [];
    const b  = ctx[val[op]?.b] ?? val[op]?.b ?? [];
    const r  = op === 'dot'  ? LinAlg.dot(a, b)
             : op === 'norm' ? LinAlg.norm2(a)
             : op === 'add'  ? LinAlg.add(a, b)
             : null;
    if (val['@store']) ctx[val['@store']] = r;
    return r;
  });

  rt._handlers.set('@formula', (val, ctx) => {
    if (val['@integrate']) {
      const { a, b, steps } = val['@integrate'];
      const fn = Function('x', `return(${val['@integrate'].fn})`);
      const r  = Formula.integrate(fn, a, b, steps);
      if (val['@store']) ctx[val['@store']] = r;
      return r;
    }
    if (val['@root']) {
      const { fn, dfn, x0 } = val['@root'];
      const f  = Function('x', `return(${fn})`);
      const df = Function('x', `return(${dfn})`);
      const r  = Formula.root(f, df, x0);
      if (val['@store']) ctx[val['@store']] = r;
      return r;
    }
  });

  rt._handlers.set('@mathml', (val, ctx) => {
    const xml  = val['@expression'] ?? val.expression ?? '';
    const ast  = MathML.parse(xml);
    if (val['@store']) ctx[val['@store']] = ast;
    return ast;
  });
}

// ─── Math @ opcode alignment ──────────────────────────────────────────────────

export const MATH_OPCODE_MAP = Object.freeze({
  '@zero':       { kuhul: '⟁Stop⟁ 0x33', description: 'identity element / zero tensor' },
  '@vigesimal':  { kuhul: '⟁Tok⟁ 0x22',  description: 'base-20 encoding (Mayan)' },
  '@pi':         { kuhul: '⟁Wey⟁ 0x21',  description: 'transcendental weight constant' },
  '@fibonacci':  { kuhul: '⟁Act⟁ 0x2D',  description: 'activation via golden ratio' },
  '@biginteger': { kuhul: '⟁Ten⟁! 0x24', description: 'mutable arbitrary-precision tensor' },
  '@loop':       { kuhul: '⟁Wo⟁each 0x0B',description: 'iterator opcode' },
  '@matmul':     { kuhul: '⟁Log⟁ 0x23',  description: 'matrix multiply logic node' },
  '@linalg':     { kuhul: '⟁Attn⟁ 0x2A', description: 'attention = QK^T/sqrt(d) matmul' },
  '@formula':    { kuhul: '⟁Fwd⟁ 0x26',  description: 'forward symbolic evaluation' },
  '@mathml':     { kuhul: '⟁Norm⟁ 0x2B', description: 'normalize expression' },
});
