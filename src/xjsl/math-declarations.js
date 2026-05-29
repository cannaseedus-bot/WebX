// Math Declarations v0.1 — JSONL as declarative math layer
// Maps: natural language → math semantics → XJSL kernel → XVM execution.
// Spec: pattern→operation→semantic_fold→math_equivalent→xjsl_kernel pipeline.

// ─── Schema ───────────────────────────────────────────────────────────────────

export const MATH_DECL_SCHEMA = Object.freeze({
  required: ['pattern', 'operation', 'semantic_fold', 'math_equivalent', 'xjsl_kernel'],
  optional: ['math_technique', 'kuhul_phase', 'constraints', 'meta'],
  fields: Object.freeze({
    pattern:        'regex capturing natural language input',
    operation:      'canonical abstract math action (snake_case)',
    semantic_fold:  'domain.intent — e.g. neural.relate, calculus.rate_of_change',
    math_equivalent:'canonical math expression or equation string',
    math_technique: 'optional: FlashAttention, symbolic_diff, numeric_eval, etc.',
    xjsl_kernel:    'XJSL primitive or fused op to emit',
    kuhul_phase:    'optional: Pop | Wo | Yax | Sek | Ch\'en | Xul (with sub-label)',
    constraints:    'optional: shape rules, dtype, domain requirements, capture_map',
    meta:           'optional: extra params (num_heads, d_model, epsilon, etc.)',
  }),
});

// ─── Declaration contract (5 rules) ──────────────────────────────────────────

export const MATH_DECL_CONTRACT = Object.freeze([
  {
    id: 1,
    rule: 'Semantic determinism',
    description: 'pattern → operation must be unambiguous — one pattern maps to exactly one operation',
  },
  {
    id: 2,
    rule: 'Mathematical equivalence',
    description: 'math_equivalent must fully describe the operation with no hidden semantics',
  },
  {
    id: 3,
    rule: 'Computational binding',
    description: 'xjsl_kernel must be a valid XJSL primitive or registered fused op',
  },
  {
    id: 4,
    rule: 'Temporal legality',
    description: 'kuhul_phase (if present) must be a valid K\'uhul temporal fold or sub-label',
  },
  {
    id: 5,
    rule: 'Executability',
    description: 'The declaration must be sufficient to generate: XJSL node, XVM opcode, math technique selection, and autograd rule if needed',
  },
]);

// ─── Kuhul temporal fold table ────────────────────────────────────────────────

export const KUHUL_PHASES = Object.freeze({
  Pop:     'ingest — raw input capture',
  Wo:      'classify — domain + semantic_fold assignment',
  Yax:     'decompose — structural + temporal decomposition',
  Sek:     'compute — actual math / XJSL kernel execution',
  "Ch'en": 'render — explanation / output formatting',
  Xul:     'seal — codex storage + replay record',
});

// ─── Domain table ─────────────────────────────────────────────────────────────

export const MATH_DECL_DOMAINS = Object.freeze({

  // Arithmetic
  add: {
    operation:      'add',
    semantic_fold:  'arithmetic.combine',
    math_equivalent:'a + b',
    xjsl_kernel:    'add',
  },
  subtract: {
    operation:      'subtract',
    semantic_fold:  'arithmetic.combine',
    math_equivalent:'a - b',
    xjsl_kernel:    'add',
  },
  multiply: {
    operation:      'multiply',
    semantic_fold:  'arithmetic.scale',
    math_equivalent:'a * b',
    xjsl_kernel:    'mul',
  },
  divide: {
    operation:      'divide',
    semantic_fold:  'arithmetic.scale',
    math_equivalent:'a / b',
    xjsl_kernel:    'mul',
  },

  // Algebra
  solve_linear: {
    operation:      'solve_linear',
    semantic_fold:  'algebra.isolate',
    math_equivalent:'ax + b = c → x = (c - b) / a',
    xjsl_kernel:    'solve_linear',
  },
  matmul: {
    operation:      'matrix_multiply',
    semantic_fold:  'algebra.transform',
    math_equivalent:'C = A @ B',
    xjsl_kernel:    'matmul',
  },

  // Calculus
  derivative: {
    operation:      'derivative',
    semantic_fold:  'calculus.rate_of_change',
    math_equivalent:'d/dx f(x)',
    math_technique: 'symbolic_diff',
    xjsl_kernel:    'diff',
    kuhul_phase:    'Sek.gradient',
  },
  integral: {
    operation:      'integral',
    semantic_fold:  'calculus.accumulate',
    math_equivalent:'∫ f(x) dx',
    math_technique: 'numeric_quad',
    xjsl_kernel:    'int',
    kuhul_phase:    'Sek.gradient',
  },
  gradient: {
    operation:      'gradient',
    semantic_fold:  'calculus.rate_of_change',
    math_equivalent:'∇f(x) — partial derivatives w.r.t. all inputs',
    math_technique: 'symbolic_diff',
    xjsl_kernel:    'diff',
    kuhul_phase:    'Sek.gradient',
  },

  // Geometry
  area: {
    operation:      'area',
    semantic_fold:  'geometry.area',
    math_equivalent:'∫ width · height',
    xjsl_kernel:    'area',
  },
  distance: {
    operation:      'distance',
    semantic_fold:  'geometry.measure',
    math_equivalent:'√(∑(xᵢ - yᵢ)²)',
    xjsl_kernel:    'add',
  },

  // Neural / ML
  fused_mlp: {
    operation:      'fused_mlp',
    semantic_fold:  'neural.transform',
    math_equivalent:'h=xW₁+b₁; h\'=act(h); y=h\'W₂+b₂',
    math_technique: 'fused_matmul',
    xjsl_kernel:    'fused_mlp',
    kuhul_phase:    'Sek.compute',
  },
  fused_attention: {
    operation:      'fused_attention',
    semantic_fold:  'neural.relate',
    math_equivalent:'softmax(QKᵀ/√d_head)V·Wo',
    math_technique: 'FlashAttention',
    xjsl_kernel:    'fused_attention',
    kuhul_phase:    'Sek.compute',
  },
  fused_norm: {
    operation:      'fused_norm',
    semantic_fold:  'neural.stabilize',
    math_equivalent:'y=(x-μ)/√(σ²+ε)·w+b+residual',
    math_technique: 'online_layernorm',
    xjsl_kernel:    'fused_norm',
    kuhul_phase:    'Sek.compute',
  },
  softmax: {
    operation:      'softmax',
    semantic_fold:  'neural.normalize',
    math_equivalent:'softmax(x)_i = e^x_i / ∑_j e^x_j',
    xjsl_kernel:    'softmax',
    kuhul_phase:    'Sek.compute',
  },
});

// ─── Canonical example declaration (spec-compliant) ──────────────────────────

export const MATH_DECL_EXAMPLE = Object.freeze({
  pattern:        'derivative of (.*?) with respect to (\\w+)',
  operation:      'derivative',
  semantic_fold:  'calculus.rate_of_change',
  math_equivalent:'d/dx f(x)',
  math_technique: 'symbolic_diff',
  xjsl_kernel:    'diff',
  kuhul_phase:    'Sek.gradient',
  constraints:    Object.freeze({ requires_variable: true }),
});

// ─── Validate a math declaration against the schema ──────────────────────────

export function validateMathDecl(decl) {
  const errors = [];
  for (const field of MATH_DECL_SCHEMA.required) {
    if (!decl[field]) errors.push(`missing required field: ${field}`);
  }
  if (decl.kuhul_phase && !Object.keys(KUHUL_PHASES).some(p => decl.kuhul_phase.startsWith(p))) {
    errors.push(`unknown kuhul_phase: ${decl.kuhul_phase}`);
  }
  return { ok: errors.length === 0, errors };
}

// ─── Pipeline description (frozen reference) ──────────────────────────────────

export const MATH_DECL_PIPELINE = Object.freeze([
  'JSONL (Math Declaration)',
  '  semantic_fold → domain resolution',
  '  math_equivalent → math technique selection',
  '  xjsl_kernel → kernel IR emission',
  '  kuhul_phase → temporal execution placement',
  'XJSL node',
  'XVM opcode',
  'GPU kernel',
]);
