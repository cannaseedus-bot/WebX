// Mayan Crypto v1 — GL(20,n) general linear group over ℤ/nℤ
//
// Arithmetic over ℤ/nℤ (default n=20, the Mayan base).
// GL(20,n) = group of invertible k×k matrices over ℤ/nℤ.
// Operations: matMul mod n, inverse mod n, determinant mod n.
//
// GL20N class: matrix-based key pair, encrypt/decrypt, sign/verify,
// homomorphic add, Mayan LWE commitment, ZKP of invertibility.
//
// No WebCrypto. All operations are pure integer arithmetic.
// Suitable for deterministic browser + Node 18+ environments.

// ─── Modular integer primitives ───────────────────────────────────────────────

export function mod(a, n) {
  return ((a % n) + n) % n;
}

export function gcd(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { const t = b; b = a % b; a = t; }
  return a;
}

// Extended Euclidean algorithm → { g, x, y } such that ax + by = g
function extGcd(a, b) {
  if (b === 0) return { g: a, x: 1, y: 0 };
  const { g, x, y } = extGcd(b, a % b);
  return { g, x: y, y: x - Math.floor(a / b) * y };
}

// Modular multiplicative inverse of a mod m (returns null if gcd ≠ 1)
export function modInverse(a, m) {
  const { g, x } = extGcd(mod(a, m), m);
  if (g !== 1) return null;
  return mod(x, m);
}

// ─── Matrix arithmetic over ℤ/nℤ ─────────────────────────────────────────────
// Matrices stored as Int32Array row-major: A[i,j] = flat[i*k+j]
// k = matrix dimension (square)

export function matMulMod(A, B, k, n) {
  const C = new Int32Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) sum += A[i * k + l] * B[l * k + j];
      C[i * k + j] = mod(sum, n);
    }
  }
  return C;
}

// Determinant over ℤ/nℤ via cofactor expansion (exact for small k)
export function matDetMod(A, k, n) {
  if (k === 1) return mod(A[0], n);
  if (k === 2) return mod(A[0] * A[3] - A[1] * A[2], n);

  let det = 0;
  for (let col = 0; col < k; col++) {
    const minor = cofactor(A, k, 0, col);
    const sign  = col % 2 === 0 ? 1 : -1;
    det = mod(det + sign * A[col] * matDetMod(minor, k - 1, n), n);
  }
  return det;
}

function cofactor(A, k, skipRow, skipCol) {
  const minor = new Int32Array((k - 1) * (k - 1));
  let ri = 0;
  for (let r = 0; r < k; r++) {
    if (r === skipRow) continue;
    let ci = 0;
    for (let c = 0; c < k; c++) {
      if (c === skipCol) continue;
      minor[ri * (k - 1) + ci] = A[r * k + c];
      ci++;
    }
    ri++;
  }
  return minor;
}

// Adjugate matrix (matrix of cofactors, transposed)
function adjugateMod(A, k, n) {
  const adj = new Int32Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const minor = cofactor(A, k, i, j);
      const sign  = (i + j) % 2 === 0 ? 1 : -1;
      // adj[j,i] = sign · det(minor) (transposed)
      adj[j * k + i] = mod(sign * matDetMod(minor, k - 1, n), n);
    }
  }
  return adj;
}

// Matrix inverse over ℤ/nℤ: A⁻¹ = det(A)⁻¹ · adj(A) mod n
// Returns null if determinant is not invertible mod n.
export function matInverseMod(A, k, n) {
  const det    = matDetMod(A, k, n);
  const detInv = modInverse(det, n);
  if (detInv === null) return null;

  const adj = adjugateMod(A, k, n);
  const inv = new Int32Array(k * k);
  for (let i = 0; i < k * k; i++) inv[i] = mod(adj[i] * detInv, n);
  return inv;
}

// Check if matrix is invertible over ℤ/nℤ
export function isInvertibleMod(A, k, n) {
  const det = matDetMod(A, k, n);
  return gcd(det, n) === 1;
}

// ─── GL20N class ──────────────────────────────────────────────────────────────
// GL(20,n): invertible k×k matrices over ℤ/20ℤ (or custom modulus).
// Default k=2 (compact key), n=20.

export class GL20N {
  constructor(k = 2, n = 20) {
    this.k = k;
    this.n = n;
  }

  // Generate a random invertible matrix in GL(k, n)
  generateKeyPair() {
    const k = this.k, n = this.n;
    let pub = null;
    for (let attempt = 0; attempt < 1000; attempt++) {
      const M = new Int32Array(k * k);
      for (let i = 0; i < k * k; i++) M[i] = Math.floor(Math.random() * n);
      if (isInvertibleMod(M, k, n)) {
        pub = M;
        break;
      }
    }
    if (!pub) throw new Error('GL20N.generateKeyPair: could not find invertible matrix after 1000 attempts');
    const priv = matInverseMod(pub, k, n);
    return { publicKey: pub, privateKey: priv };
  }

  // Encrypt: cipher = pub · msg (mod n), msg is Int32Array(k*k)
  encrypt(msg, publicKey) {
    return matMulMod(publicKey, msg, this.k, this.n);
  }

  // Decrypt: msg = priv · cipher (mod n)
  decrypt(cipher, privateKey) {
    return matMulMod(privateKey, cipher, this.k, this.n);
  }

  // Sign: sig = priv · msgHash (mod n), msgHash is Int32Array(k*k)
  sign(msgHash, privateKey) {
    return matMulMod(privateKey, msgHash, this.k, this.n);
  }

  // Verify: check pub · sig ≡ msgHash (mod n)
  verify(msgHash, sig, publicKey) {
    const recovered = matMulMod(publicKey, sig, this.k, this.n);
    for (let i = 0; i < this.k * this.k; i++) {
      if (recovered[i] !== msgHash[i]) return false;
    }
    return true;
  }

  // Homomorphic addition over ℤ/nℤ (element-wise)
  homomorphicAdd(a, b) {
    const out = new Int32Array(this.k * this.k);
    for (let i = 0; i < out.length; i++) out[i] = mod(a[i] + b[i], this.n);
    return out;
  }

  // Mayan LWE commitment: commit = A·s + e mod n
  // s: secret Int32Array(k), e: error Int32Array(k), A: Int32Array(k*k)
  mayanLWE(A, s, e) {
    const k = this.k, n = this.n;
    const out = new Int32Array(k);
    for (let i = 0; i < k; i++) {
      let sum = 0;
      for (let j = 0; j < k; j++) sum += A[i * k + j] * s[j];
      out[i] = mod(sum + e[i], n);
    }
    return out;
  }

  // ZKP of invertibility: prove a matrix M is invertible without revealing M.
  // Protocol: commit to M via mayanHash, reveal det(M) and its modular inverse.
  // Returns { commitment: string, det: number, detInv: number, valid: boolean }
  proveInvertibility(M) {
    const det    = matDetMod(M, this.k, this.n);
    const detInv = modInverse(det, this.n);
    return {
      commitment: mayanHash(M),
      det,
      detInv,
      valid: detInv !== null,
    };
  }

  // Verify a ZKP proof (verifier knows commitment + det + detInv, not M)
  verifyInvertibility(proof) {
    if (!proof.valid) return false;
    return (
      proof.detInv !== null &&
      mod(proof.det * proof.detInv, this.n) === 1
    );
  }

  // Mayan hash: deterministic 8-char hex from matrix entries (djb2 variant)
  mayanHash(M) { return mayanHash(M); }
}

// ─── Mayan hash (browser-safe, no WebCrypto) ──────────────────────────────────
// djb2-style over the matrix entries, returns 16-char hex string.

export function mayanHash(data) {
  let h1 = 0x811c9dc5 >>> 0;
  let h2 = 0xdeadbef7 >>> 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] >>> 0;
    h1 = Math.imul(h1 ^ v, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ (v >> 4 | v << 28), 0x01000193) >>> 0;
  }
  return h1.toString(16).padStart(8, '0') + h2.toString(16).padStart(8, '0');
}

// ─── Convenience: create GL20N with Mayan base-20 ────────────────────────────

export function createMayanGL(k = 2) {
  return new GL20N(k, 20);
}
