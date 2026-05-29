// XJSL Fused Ops v0.1 — canonical node schemas
// Each schema is the exact XJSL kernel node template for fused_mlp, fused_attention, fused_norm.
// Forward semantics and backward rules are described in meta.math.
// Lowering: each schema produces a single WGSL/HLSL kernel (no intermediate CPU readback).

// ─── Op registry (inputs/outputs by kernel name) ──────────────────────────────
// Used by the JSONL→XJSL compiler to build node IO without re-declaring per call.

export const XJSL_OP_REGISTRY = Object.freeze({
  fused_mlp: {
    inputs:  ['x', 'W1', 'b1', 'W2', 'b2'],
    outputs: ['y'],
    meta_defaults: { activation: 'gelu', dtype: 'f16' },
  },
  fused_attention: {
    inputs:  ['x', 'Wq', 'Wk', 'Wv', 'Wo'],
    outputs: ['y'],
    meta_defaults: { causal: true, dropout: 0.0, dtype: 'f16' },
  },
  fused_norm: {
    inputs:  ['x', 'w', 'b', 'residual'],
    outputs: ['y'],
    meta_defaults: { kind: 'layernorm', epsilon: 1e-5, use_affine: true, use_residual: true, dtype: 'f32' },
  },
  // Primitive ops (used by pattern compiler)
  add:           { inputs: ['x', 'y'],    outputs: ['out'], meta_defaults: {} },
  mul:           { inputs: ['x', 'y'],    outputs: ['out'], meta_defaults: {} },
  matmul:        { inputs: ['A', 'B'],    outputs: ['C'],   meta_defaults: {} },
  diff:          { inputs: ['x'],         outputs: ['out'], meta_defaults: {} },
  solve_linear:  { inputs: ['a','b','c'], outputs: ['x'],   meta_defaults: {} },
  area:          { inputs: ['w','h'],     outputs: ['out'], meta_defaults: {} },
});

// ─── fused_mlp ────────────────────────────────────────────────────────────────
// h = x @ W1 + b1  [m, h]
// h'= activation(h)
// y = h' @ W2 + b2  [m, d]

export const FUSED_MLP_SCHEMA = Object.freeze({
  kind:    'kernel',
  kernel:  'fused_mlp',
  inputs:  ['x', 'W1', 'b1', 'W2', 'b2'],
  outputs: ['y'],
  meta: Object.freeze({
    dtype:      'f16',
    activation: 'gelu',   // 'relu' | 'gelu' | 'tanh'
    // shape params — caller fills:
    // m: batch * seq_len
    // h: intermediate dim (hidden expansion)
    // d: model dim
  }),
  bindings_template: Object.freeze({
    x:  'buffers.x',
    W1: 'params.W1',
    b1: 'params.b1',
    W2: 'params.W2',
    b2: 'params.b2',
    y:  'buffers.y',
  }),
  math: Object.freeze({
    forward:  'h=xW₁+b₁; h\'=act(h); y=h\'W₂+b₂',
    backward: '∂L/∂W₂=h\'ᵀ∂L/∂y; ∂L/∂h\'=∂L/∂y·W₂ᵀ; ∂L/∂h=∂L/∂h\'·act\'(h); ∂L/∂W₁=xᵀ∂L/∂h; ∂L/∂x=∂L/∂h·W₁ᵀ',
    technique: 'fused single-pass — no intermediate CPU readback',
  }),
});

// ─── fused_attention ──────────────────────────────────────────────────────────
// Q=xWq, K=xWk, V=xWv
// S=QKᵀ/√d_head  → causal mask → softmax → dropout → P
// y=PV·Wo

export const FUSED_ATTENTION_SCHEMA = Object.freeze({
  kind:    'kernel',
  kernel:  'fused_attention',
  inputs:  ['x', 'Wq', 'Wk', 'Wv', 'Wo'],
  outputs: ['y'],
  meta: Object.freeze({
    dtype:     'f16',
    causal:    true,
    dropout:   0.0,
    // caller fills: num_heads, d_model, d_head
  }),
  bindings_template: Object.freeze({
    x:  'buffers.x',
    Wq: 'params.Wq',
    Wk: 'params.Wk',
    Wv: 'params.Wv',
    Wo: 'params.Wo',
    y:  'buffers.y',
  }),
  math: Object.freeze({
    forward:  'Q=xWq; K=xWk; V=xWv; S=QKᵀ/√d_head; P=softmax(mask(S)); y=PV·Wo',
    backward: '∂L/∂Wo=PVᵀ∂L/∂y; ∂L/∂V=Pᵀ∂L/∂yWoᵀ; ∂L/∂P=∂L/∂yWoᵀVᵀ; ∂L/∂S=P⊙(∂L/∂P-P·∂L/∂P); ∂L/∂Q=∂L/∂S·K/√d; ∂L/∂K=∂L/∂Sᵀ·Q/√d',
    technique: 'FlashAttention-compatible — tiles QK in SRAM, no N² softmax materialization',
  }),
});

// ─── fused_norm ───────────────────────────────────────────────────────────────
// layernorm: μ=mean(x); σ²=mean((x-μ)²); n=(x-μ)/√(σ²+ε); n'=n·w+b; y=n'+residual
// rmsnorm:   n=x/√(mean(x²)+ε); n'=n·w+b; y=n'+residual

export const FUSED_NORM_SCHEMA = Object.freeze({
  kind:    'kernel',
  kernel:  'fused_norm',
  inputs:  ['x', 'w', 'b', 'residual'],
  outputs: ['y'],
  meta: Object.freeze({
    dtype:        'f32',
    kind:         'layernorm',  // 'layernorm' | 'rmsnorm'
    epsilon:      1e-5,
    use_affine:   true,
    use_residual: true,
  }),
  bindings_template: Object.freeze({
    x:        'buffers.x',
    w:        'params.w',
    b:        'params.b',
    residual: 'buffers.residual',
    y:        'buffers.y',
  }),
  math: Object.freeze({
    forward_layernorm: 'μ=mean(x); σ²=mean((x-μ)²); n=(x-μ)/√(σ²+ε); n\'=n·w+b; y=n\'+residual',
    forward_rmsnorm:   'n=x/√(mean(x²)+ε); n\'=n·w+b; y=n\'+residual',
    backward: '∂L/∂x via project-out mean+variance; ∂L/∂w=∂L/∂y·n̂; ∂L/∂b=∂L/∂y; ∂L/∂residual=∂L/∂y',
    technique: 'fused affine + residual in one kernel pass',
  }),
});

// Convenience map: kernel name → schema
export const FUSED_OP_SCHEMAS = Object.freeze({
  fused_mlp:       FUSED_MLP_SCHEMA,
  fused_attention: FUSED_ATTENTION_SCHEMA,
  fused_norm:      FUSED_NORM_SCHEMA,
});

// Build a concrete XJSL node from a schema + caller-supplied meta + binding overrides.
export function buildFusedNode(kernelName, meta = {}, bindings = {}) {
  const schema = FUSED_OP_SCHEMAS[kernelName];
  if (!schema) throw new Error(`Unknown fused op: ${kernelName}`);
  return {
    kind:     schema.kind,
    kernel:   schema.kernel,
    inputs:   schema.inputs.slice(),
    outputs:  schema.outputs.slice(),
    meta:     { ...schema.meta, ...meta },
    bindings: { ...schema.bindings_template, ...bindings },
  };
}
