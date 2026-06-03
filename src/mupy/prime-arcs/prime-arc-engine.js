// prime-arc-engine.js — Prime Number ARC (Arithmetic-Riemannian Coordinates)
//
// 168 primes up to 1000 = the spectral basis for the complete geometric intelligence.
// Extends the 9-band SH system (l=0..2) to 168 prime harmonic dimensions.
//
// Unified tensor: 3 (sphere pos) + 100 (features) + 168 (prime ARCs) = 271D
// This IS the SCXQ7 SCO/1 compressed_state address space.
//
// Connection to π-KUHUL RAG engine:
//   Token.position (100D Mayan vigesimal) → prime_arc_coordinates()
//   Prime ARCs encode where the token lives in the prime harmonic spectrum
//   prime_geodesic_distance() replaces simple Euclidean token distance
//
// Connection to optical processor:
//   OpticalNode.sh[9] = first 9 prime harmonic coefficients (l=0..2)
//   Full prime ARC = all 168 harmonic bands
//   prime_spectral_decomposition() → which primes dominate this node's field
//
// Connection to geodesic attention:
//   arccos(q·k^T) = geodesic on S^(d-1)
//   prime_arc_distance() = geodesic on the prime harmonic manifold
//   Both measure the same thing: distance between semantic positions
//
// SCXQ2 domain: @π (math context) — all prime ARC ops live in math domain

const PI = Math.PI;

// ─── Sieve of Eratosthenes — 168 primes up to 1000 ───────────────────────────

function generatePrimes(limit = 1000) {
  const sieve = new Uint8Array(limit + 1).fill(1);
  sieve[0] = sieve[1] = 0;
  for (let p = 2; p * p <= limit; p++) {
    if (sieve[p]) for (let i = p*p; i <= limit; i += p) sieve[i] = 0;
  }
  const primes = [];
  for (let i = 2; i <= limit; i++) if (sieve[i]) primes.push(i);
  return primes;  // 168 primes
}

export const PRIMES = generatePrimes(1000);  // [2, 3, 5, 7, 11, ..., 997]
export const PRIME_COUNT = PRIMES.length;    // 168

// ─── Prime harmonic frequencies ω_p = p·ln(p) / 2π ──────────────────────────

export const PRIME_FREQUENCIES = PRIMES.map(p => p * Math.log(p) / (2 * PI));

// ─── Prime ARC coordinate computation ────────────────────────────────────────

/**
 * Geometric component of ARC: spherical harmonic modulated by prime.
 * point = [x,y,z] on unit sphere
 */
function geometricARC(point, prime) {
  const [x, y, z] = point;
  const theta = Math.atan2(y, x);
  const phi   = Math.acos(Math.max(-1, Math.min(1, z)));
  return Math.sin(prime * theta) * Math.cos(prime * phi) / Math.sqrt(prime);
}

/**
 * Harmonic component of ARC: Fourier projection of feature vector onto prime freq.
 * features = Float32Array of length D
 */
function harmonicARC(features, prime) {
  const D = features.length;
  let sum = 0;
  for (let i = 0; i < D; i++) {
    sum += features[i] * Math.sin(prime * i * PI / D);
  }
  return sum / Math.sqrt(prime);
}

/**
 * Full prime ARC coordinates for a point on S² with feature vector.
 * Returns Float32Array of length PRIME_COUNT (168).
 */
export function primeArcCoordinates(point3d, features100d) {
  const arcs = new Float32Array(PRIME_COUNT);
  for (let i = 0; i < PRIME_COUNT; i++) {
    const p = PRIMES[i];
    arcs[i] = geometricARC(point3d, p) * harmonicARC(features100d, p);
  }
  return arcs;
}

// ─── Prime Riemannian metric g_ij = (p_i × p_j) / (p_i + p_j) ────────────────
// Stored as diagonal only (full 168×168 is 112KB — too large for runtime)

export const PRIME_METRIC_DIAG = new Float32Array(PRIME_COUNT).map((_, i) =>
  PRIMES[i] / 2.0
);

/**
 * Geodesic distance in prime ARC space (using diagonal metric approximation).
 * arc1, arc2: Float32Array[PRIME_COUNT]
 */
export function primeArcDistance(arc1, arc2) {
  let sum = 0;
  for (let i = 0; i < PRIME_COUNT; i++) {
    const d = arc2[i] - arc1[i];
    sum += d * d * PRIME_METRIC_DIAG[i];
  }
  return Math.sqrt(sum);
}

// ─── Prime spectral decomposition ─────────────────────────────────────────────

/**
 * Which primes dominate this ARC signature?
 * Returns top-k {prime, magnitude} pairs — these are the "resonant primes"
 * for this token/node position in the semantic field.
 */
export function primeSpectralDecomposition(arcs, topK = 10) {
  const scored = PRIMES.map((p, i) => ({ prime: p, magnitude: Math.abs(arcs[i]) }));
  scored.sort((a, b) => b.magnitude - a.magnitude);
  return scored.slice(0, topK);
}

// ─── Unified 271D tensor ──────────────────────────────────────────────────────
// 3 (sphere pos) + 100 (features) + 168 (prime ARCs) = 271D

export const UNIFIED_DIM = 3 + 100 + PRIME_COUNT;  // 271

/**
 * Build unified 271D tensor for a point.
 * Concatenates position + features + prime ARCs.
 */
export function buildUnifiedTensor(point3d, features100d) {
  const arcs = primeArcCoordinates(point3d, features100d);
  const unified = new Float32Array(UNIFIED_DIM);
  unified.set(point3d, 0);
  unified.set(features100d, 3);
  unified.set(arcs, 103);
  return unified;
}

/**
 * Geodesic distance in unified 271D manifold (block diagonal metric).
 * Block 1 [0:3]:   Euclidean (sphere position)
 * Block 2 [3:103]: L2 (feature space)
 * Block 3 [103:]:  Prime metric diagonal
 */
export function unifiedGeodesicDistance(t1, t2) {
  let sum = 0;
  // Block 1: position (weight 1.0)
  for (let i = 0; i < 3; i++) { const d = t2[i]-t1[i]; sum += d*d; }
  // Block 2: features (weight 0.1 — lower scale)
  for (let i = 3; i < 103; i++) { const d = t2[i]-t1[i]; sum += d*d*0.1; }
  // Block 3: prime ARCs (prime diagonal metric)
  for (let i = 103; i < UNIFIED_DIM; i++) {
    const d = t2[i]-t1[i];
    sum += d*d*PRIME_METRIC_DIAG[i-103];
  }
  return Math.sqrt(sum);
}

// ─── PrimeArcEngine ───────────────────────────────────────────────────────────

export class PrimeArcEngine {
  constructor() {
    this.nodes = [];  // unified 271D tensors
    this._arcCache = new Map();  // cache ARC coords by node index
  }

  /**
   * Add an optical node (from ComputeOpticalMesh or π-KUHUL toOpticalNodes).
   * node.position [3], node.sh [18] → extended to 100D features, then 271D unified.
   */
  addOpticalNode(node, nodeIdx) {
    // Extend 9-band SH (18 floats) to 100D features via prime harmonic padding
    const features = new Float32Array(100);
    const sh = node.sh instanceof Float32Array ? node.sh : new Float32Array(node.sh);
    features.set(sh.slice(0, Math.min(18, sh.length)), 0);
    // Pad remaining 82 dims using prime harmonics of existing SH state
    for (let i = 18; i < 100; i++) {
      const p = PRIMES[i % PRIME_COUNT];
      features[i] = sh[(i - 18) % 18] * Math.sin(p * i * PI / 100) * 0.1;
    }
    const pos = node.pos ?? node.position ?? [0, 0, 1];
    const unified = buildUnifiedTensor(pos, features);
    this.nodes.push(unified);
    this._arcCache.set(nodeIdx, unified.slice(103));  // cache ARC portion
    return unified;
  }

  /**
   * Prime ARC attention: for query node, score all key nodes by prime ARC distance.
   * Returns softmax-normalised weights (same role as arccos geodesic attention
   * but using the 168D prime harmonic manifold).
   */
  primeArcAttention(queryIdx, keyIndices, temperature = 1.0) {
    const qArcs = this._arcCache.get(queryIdx);
    if (!qArcs) return null;
    const scores = keyIndices.map(ki => {
      const kArcs = this._arcCache.get(ki);
      if (!kArcs) return 0;
      return Math.exp(-primeArcDistance(qArcs, kArcs) / temperature);
    });
    const sum = scores.reduce((a, b) => a + b, 0) || 1;
    return scores.map(s => s / sum);
  }

  /**
   * Find the dominant prime for a node — which harmonic resonates most strongly.
   * This is the "prime signature" of the node's semantic position.
   */
  dominantPrime(nodeIdx) {
    const arcs = this._arcCache.get(nodeIdx);
    if (!arcs) return null;
    const top = primeSpectralDecomposition(arcs, 3);
    return top;
  }

  /**
   * Convert π-KUHUL token states to prime ARC nodes.
   * engine = PiKuhulFieldEngine instance
   */
  fromPiKuhul(engine) {
    this.nodes = []; this._arcCache.clear();
    engine.tokens.forEach((tok, i) => {
      const pos = (() => {
        const n = Math.hypot(...tok.pos.slice(0,3)) || 1;
        return tok.pos.slice(0,3).map(v => v/n);
      })();
      const features = new Float32Array(100);
      // Vigesimal encoding: Mayan base-20 position → first 6 features
      for (let d = 0; d < Math.min(6, tok.pos.length); d++) features[d] = tok.pos[d] / 20.0;
      // Phase encoding: π-KUHUL phase → next 2 features
      features[6] = Math.cos(tok.phase);
      features[7] = Math.sin(tok.phase);
      // Coherence → feature 8
      features[8] = tok.coherence;
      const unified = buildUnifiedTensor(pos, features);
      this.nodes.push(unified);
      this._arcCache.set(i, unified.slice(103));
    });
    return this;
  }

  stats() {
    return { nodes: this.nodes.length, dims: UNIFIED_DIM,
             primeCount: PRIME_COUNT, primes: `${PRIMES[0]}..${PRIMES.at(-1)}` };
  }
}
