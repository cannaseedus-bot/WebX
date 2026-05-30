// tensor-algebra.js — XCFE as Grammatical Tensor
//
// Theorem: XCFE is a rank-R sparse tensor over @-space.
//
//   Axes       : @-namespace prefixes (basis vectors)
//   Values     : xJSON block payloads
//   Rank       : indentation depth of the deepest @-chain
//   Sparsity   : grammatical constraints (most @a × @b are structurally invalid)
//   Contraction: dataflow (@store, @wait_for, →) binding free tensor indices
//   Outer prod : parallel blocks (sibling @-keys at the same indent level)
//   Tensor prod: nested @-blocks (@agent ⊗ @skill ⊗ @command ⊗ @tool)
//
// Proof sketch:
//   Any XCFE block B can be encoded as:
//     B ∈ Tensor(ℕ^R → 𝔹 × Value)
//   where:
//     R     = max nesting depth of @ prefixes
//     ℕ^R   = multi-index over (semantic, temporal, spatial, numerical, comm) axes
//     𝔹     = {valid, invalid} (grammatical constraint)
//     Value = xJSON payload at that position
//
// Canonical XCFE tensor dimensions:
//   D0 semantic  : @agent @skill @micronaut @command @tool               (dim ≈ 5)
//   D1 temporal  : @tick @step @batch @flux @round @mark                 (dim ≈ 6)
//   D2 spatial   : @thread @graph @node @fold @horizontal_folds          (dim ≈ 5)
//   D3 numerical : @pi @fibonacci @matmul @linalg @biginteger @formula   (dim ≈ 6)
//   D4 comms     : @protocol @jsonl @notation @ngram @map                (dim ≈ 5)
//   ... (open set, grows with each new @ namespace)
//
// Key isomorphisms:
//   Block nesting       ↔  tensor product     ⊗
//   Parallel @-siblings ↔  outer product      ⊕
//   Dataflow contraction↔  Einstein summation Σ_i A^i B_i
//   @store / @wait_for  ↔  index binding      (free → bound index)
//   Indentation depth   ↔  tensor rank        R
//   Grammar rules       ↔  tensor sparsity    pattern

// ─── @-space basis ────────────────────────────────────────────────────────────

export const AT_AXES = Object.freeze({
  semantic:    ['@agent','@skill','@micronaut','@command','@tool'],
  temporal:    ['@tick','@step','@batch','@flux','@round','@mark'],
  spatial:     ['@thread','@graph','@node','@fold','@horizontal_folds','@micro_folds'],
  numerical:   ['@pi','@fibonacci','@matmul','@linalg','@biginteger','@formula','@zero','@vigesimal'],
  comms:       ['@protocol','@jsonl','@notation','@ngram','@map'],
  execution:   ['@opcodes','@context','@semantics','@folds'],
  node_runtime:['@node','@ipc','@stdio','@fs','@http'],
});

// Total unique basis vectors (@ prefixes)
export const AT_BASIS_DIMENSION = Object.values(AT_AXES).flat().length;

// ─── XCFE Tensor representation ───────────────────────────────────────────────

export class XCFETensor {
  constructor() {
    // Sparse storage: multi-index string → value
    this._data     = new Map();
    this._rank     = 0;
    this._shape    = [];   // [dim0, dim1, ...]
  }

  // Set a value at multi-index (array of @-namespace strings)
  set(indices, value) {
    const key  = indices.join(':');
    this._data.set(key, value);
    this._rank = Math.max(this._rank, indices.length);
    return this;
  }

  get(indices) { return this._data.get(indices.join(':')); }

  // Tensor product of two XCFE blocks (nesting them)
  // (a ⊗ b) produces a higher-rank tensor
  static product(A, B) {
    const C = new XCFETensor();
    for (const [ka, va] of A._data) {
      for (const [kb, vb] of B._data) {
        C.set([...ka.split(':'), ...kb.split(':')], { a: va, b: vb });
      }
    }
    return C;
  }

  // Outer product (parallel @ siblings at same indent)
  // Concatenates indices rather than nesting them
  static outer(tensors) {
    const result = new XCFETensor();
    let offset = 0;
    for (const T of tensors) {
      for (const [k, v] of T._data) {
        result.set([`d${offset}:${k}`], v);
      }
      offset++;
    }
    return result;
  }

  // Contraction: bind index `axis` to value `binding`
  // (eliminates one degree of freedom = one @-dimension)
  contract(axis, binding) {
    const contracted = new XCFETensor();
    for (const [k, v] of this._data) {
      const parts = k.split(':');
      const idx   = parts.indexOf(axis);
      if (idx !== -1 && (binding === '*' || parts[idx] === binding)) {
        const newKey = parts.filter((_, i) => i !== idx);
        contracted.set(newKey, v);
      }
    }
    return contracted;
  }

  // Density: fraction of non-zero entries vs total possible
  density(totalPossible = null) {
    const total = totalPossible ?? Math.pow(AT_BASIS_DIMENSION, this._rank);
    return this._data.size / Math.max(1, total);
  }

  get rank()       { return this._rank; }
  get nnz()        { return this._data.size; }  // non-zero entries
  get entries()    { return [...this._data.entries()]; }
}

// ─── XCFE block → tensor conversion ──────────────────────────────────────────

export function blockToTensor(block, prefix = [], depth = 0) {
  const T = new XCFETensor();
  for (const [key, val] of Object.entries(block)) {
    if (!key.startsWith('@')) {
      // Leaf value — tensor entry at this multi-index
      T.set([...prefix, `data:${key}`], val);
    } else {
      // @-key = new tensor axis
      const ns     = key.slice(1).split('.')[0];  // namespace part
      const coords = [...prefix, ns];
      T.set(coords, typeof val !== 'object' ? val : null);
      if (val && typeof val === 'object') {
        // Recurse: tensor product of nested block
        const sub = blockToTensor(val, coords, depth + 1);
        for (const [k, v] of sub._data) T.set(k.split(':'), v);
      }
    }
  }
  return T;
}

// ─── Tucker decomposition (compact grammar extraction) ───────────────────────
//
// Tucker(T, r) = G ×₁ U₁ ×₂ U₂ ... ×ₙ Uₙ
// Core tensor G captures the grammar skeleton.
// Factor matrices Uᵢ map @-namespace indices to latent grammar factors.
// Simplified: extract frequent sub-patterns as "factors".

export class TuckerDecomposition {
  constructor(tensor, maxFactors = 4) {
    this._tensor     = tensor;
    this._maxFactors = maxFactors;
    this._factors    = [];
    this._core       = null;
  }

  // Extract dominant @ co-occurrence patterns
  decompose() {
    const cooccurrence = new Map();
    for (const [key] of this._tensor._data) {
      const parts = key.split(':');
      for (let i = 0; i < parts.length; i++) {
        for (let j = i + 1; j < parts.length; j++) {
          const pair = `${parts[i]}:${parts[j]}`;
          cooccurrence.set(pair, (cooccurrence.get(pair) ?? 0) + 1);
        }
      }
    }

    // Top-k co-occurring pairs = tensor factors
    this._factors = [...cooccurrence.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, this._maxFactors)
      .map(([pair, count]) => ({ pair: pair.split(':'), count }));

    // Core = unique axes after factoring out top patterns
    const usedAxes = new Set(this._factors.flatMap(f => f.pair));
    const allAxes  = new Set();
    for (const [key] of this._tensor._data) key.split(':').forEach(p => allAxes.add(p));
    this._core = [...allAxes].filter(a => !usedAxes.has(a));

    return this;
  }

  get factors()  { return this._factors; }
  get coreAxes() { return this._core; }
}

// ─── Grammatical similarity (cosine over @-space) ─────────────────────────────

export function semanticSimilarity(nsA, nsB) {
  // Embed each @-prefix into axis-membership space
  const axes = Object.values(AT_AXES);
  const embed = (ns) => {
    const clean = ns.startsWith('@') ? ns.slice(1) : ns;
    return axes.map(axis => axis.some(a => a.slice(1) === clean || a.slice(1).startsWith(clean)) ? 1 : 0);
  };
  const vA = embed(nsA), vB = embed(nsB);
  const dot = vA.reduce((s, v, i) => s + v * vB[i], 0);
  const mA  = Math.sqrt(vA.reduce((s, v) => s + v*v, 0));
  const mB  = Math.sqrt(vB.reduce((s, v) => s + v*v, 0));
  return mA && mB ? dot / (mA * mB) : 0;
}

// ─── Predefined semantic clusters ────────────────────────────────────────────

export const SEMANTIC_CLUSTERS = Object.freeze({
  execution:    ['@agent','@skill','@command'],
  temporal:     ['@tick','@step','@flux'],
  communication:['@protocol','@jsonl','@notation'],
  computation:  ['@matmul','@linalg','@formula'],
  concurrency:  ['@thread','@batch','@graph'],
});

// ─── XCFE tensor rank estimator ───────────────────────────────────────────────

export function estimateRank(block, depth = 0) {
  if (typeof block !== 'object' || block === null) return depth;
  let maxDepth = depth;
  for (const [k, v] of Object.entries(block)) {
    if (k.startsWith('@') && v && typeof v === 'object') {
      maxDepth = Math.max(maxDepth, estimateRank(v, depth + 1));
    }
  }
  return maxDepth;
}

// The rank-11 example from the spec:
// @agent > @skill > @flux > @tick > @batch > @matmul > @round > @thread > @map > @graph > @protocol > @jsonl
// = 12 nested @-blocks = rank-12 tensor

export const RANK_11_EXAMPLE = {
  '@agent.ml_engineer': {
    '@skill.model_optimization': {
      '@flux.stream': {
        '@tick.interval': {
          '@batch.size': {
            '@matmul.sparse': {
              '@round.quantize_to_int8': {
                '@thread.pool': {
                  '@map.parallel': {
                    '@graph.sink': {
                      '@protocol.grpc': {
                        '@jsonl.serialize': {}
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

// ─── Theorem summary (formal statement) ───────────────────────────────────────

export const XCFE_TENSOR_THEOREM = Object.freeze({
  name:    'XCFE Grammatical Tensor Theorem',
  statement: [
    'XCFE is a rank-R sparse tensor T over @-space where:',
    '  T ∈ Tensor(ℕ^R → 𝔹 × Value)',
    '  @-prefixes are basis vectors (currently ' + AT_BASIS_DIMENSION + ')',
    '  Block nesting    ↔ tensor product   ⊗',
    '  Parallel siblings↔ outer product    ⊕',
    '  @store/@wait_for ↔ index contraction Σᵢ',
    '  Indentation depth↔ tensor rank      R',
    '  Grammar rules    ↔ tensor sparsity',
  ],
  corollaries: [
    'XCFE execution = tensor contraction sequence',
    'XCFE composition = tensor product',
    'XCFE transformation = tensor operation',
    'Grammar = sparsity pattern of the @-space tensor',
  ],
  practical: [
    'Tucker decomposition extracts reusable grammar patterns',
    'Semantic similarity = cosine in @-embedding space',
    'Parallel @-siblings = SIMD lane dispatch (HorizontalFold)',
    'Nested @-chain = sequential pipeline (pipeline depth = rank)',
  ],
});
