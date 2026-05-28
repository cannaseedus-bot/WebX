// D3D11 GPT-2 trainer architecture descriptor.
// Documents the buffer model, dispatch pipeline, and shader inventory
// from v0.1.0-igpu-trainer (D3D11/cs_5_0 — NOT DXIL; Intel HD 4600 compatible).
//
// This module is a descriptor, not a D3D11 runtime. Actual dispatch happens in native C++.
// Use TrainerConfig from src/xjsl/index.js for JS-side training configuration.

// ─── XVM fiber buffer layout (D3D11 UAV-backed) ──────────────────────────────
// Mirrors D3D11Engine member layout in src/d3d11_engine.h

export const XVM_D3D11_BUFFERS = Object.freeze({
  code:       { type: 'SRV', format: 'R32_UINT',  description: 'XVM bytecode (read-only)' },
  fibers:     { type: 'UAV', format: 'structured', stride: 40, description: 'XVMFiber[n] — pc/sp/phase/flags/r0-r3/entropy/pressure' },
  shared:     { type: 'UAV', format: 'R32_UINT',  description: 'cluster shared memory (Uint32)' },
  stack:      { type: 'UAV', format: 'R32_UINT',  description: 'per-fiber call stack' },
  trace:      { type: 'UAV', format: 'R32_UINT',  description: 'execution trace ring buffer' },
  traceIndex: { type: 'UAV', format: 'R32_UINT',  description: 'trace write head' },
});

export const XVM_TRACE_CAPACITY_WORDS = 131072; // 512 KB trace buffer

// ─── GPT-2 trainer working buffer inventory ───────────────────────────────────
// Mirrors GPT2Trainer member variables. All are float32 GPU buffers.

export const TRAINER_BUFFERS = Object.freeze({
  // Activations (per-batch, discarded after backward)
  hidden:       'float[B*S, E]         — current hidden state',
  logits:       'float[V]              — LM head output logits',
  targets:      'int32[B*S]            — token target indices',
  dlogits:      'float[V]              — dL/dlogits from cross-entropy',
  ln_out:       'float[B*S, E]         — final layer-norm output (pre-LM head)',
  ln_xhat:      'float[B*S, E]         — layer-norm normalized (x_hat, for bwd)',
  qkv:          'float[B*S, 3*E]       — packed QKV projection output',
  q:            'float[S, head_dim]    — single-head query (per head loop)',
  k:            'float[S, head_dim]    — single-head key',
  v:            'float[S, head_dim]    — single-head value',
  attn_out:     'float[S, head_dim]    — single-head attention output',
  attn_P:       'float[S, S]           — softmax weights (saved for bwd)',
  ffn_hidden:   'float[B*S, d_ff]      — FFN post-GELU activations',
  ffn_pre_gelu: 'float[B*S, d_ff]      — FFN pre-GELU (saved for GELU bwd)',
  dhidden:      'float[B*S, E]         — accumulated hidden-state gradient',
  // Scalars
  loss:         'float[1]              — scalar cross-entropy loss (GPU-side)',
});

// ─── Forward + backward shader pipeline ──────────────────────────────────────
// cs_5_0 DXBC only. All shaders located in shaders/ directory.

export const FORWARD_PIPELINE = [
  { shader: 'gpt2_embed',          dispatch: 'Dispatch(seq_len/64, 1, 1)',  note: 'token+position embedding lookup' },
  { shader: 'gpt2_layernorm_fwd',  dispatch: 'Dispatch(seq_len, 1, 1)',      note: 'LN before attention' },
  { shader: 'gpt2_qkv_split',      dispatch: 'Dispatch(seq_len, n_head, 1)', note: 'split QKV for each head' },
  { shader: 'gpt2_attn_fwd',       dispatch: 'Dispatch(seq_len, 1, 1)',      note: 'causal self-attention per head' },
  { shader: 'gpt2_residual',       dispatch: 'Dispatch(numel/256, 1, 1)',    note: 'residual add after attention' },
  { shader: 'gpt2_matmul',         dispatch: 'Dispatch(M, N/64, 1)',          note: 'MLP fc1 projection' },
  { shader: 'gpt2_gelu_fwd',       dispatch: 'Dispatch(numel/256, 1, 1)',    note: 'GELU activation (saves pre-GELU)' },
  { shader: 'gpt2_matmul',         dispatch: 'Dispatch(M, N/64, 1)',          note: 'MLP fc2 projection' },
  { shader: 'gpt2_layernorm_fwd',  dispatch: 'Dispatch(seq_len, 1, 1)',      note: 'LN after MLP (×6 layers)' },
  { shader: 'gpt2_layernorm_fwd',  dispatch: 'Dispatch(seq_len, 1, 1)',      note: 'final layer norm (LN_f)' },
  { shader: 'gpt2_lm_head',        dispatch: 'Dispatch(ceil(V/64), 1, 1)',   note: 'logits = hidden[last_pos] @ wte.T' },
];

export const BACKWARD_PIPELINE = [
  { shader: 'gpt2_loss',           dispatch: 'Dispatch(1, 1, 1)',            note: 'cross-entropy loss + dlogits' },
  { shader: 'gpt2_layernorm_bwd',  dispatch: 'Dispatch(seq_len, 1, 1)',      note: 'LN_f backward (2-pass reduction)' },
  { shader: 'gpt2_gelu_bwd',       dispatch: 'Dispatch(numel/256, 1, 1)',    note: 'GELU backward (uses saved pre-GELU)' },
  { shader: 'gpt2_attn_bwd',       dispatch: 'Dispatch(seq_len, 1, 1)',      note: 'attention backward (uses saved P)' },
  { shader: 'gpt2_layernorm_bwd',  dispatch: 'Dispatch(seq_len, 1, 1)',      note: 'LN backward per layer' },
];

export const OPTIMIZER_PIPELINE = [
  { shader: 'gpt2_adam',    dispatch: 'Dispatch(ceil(numel/256), 1, 1)', note: 'Adam for all non-wte params' },
  // gpt2_adam_wte.hlsl (from v0.1.1-igpu-trainer-xjsl): separate shader for wte
];

// ─── Known NaN bugs and their fixes (Bug 6 in docs/BUGS.md) ─────────────────

export const D3D11_NAN_BUGS = Object.freeze([
  {
    id:   'layernorm-srvaav-alias',
    desc: 'SRV/UAV aliasing in final LN backward: binding same buffer as both input (t-slot) and gradient output (u-slot) causes undefined behavior',
    fix:  'Insert a buffer copy before dispatching LN_f backward; use separate ln_out_buf_ (SRV) and dhidden_buf_ (UAV)',
  },
  {
    id:   'gelu-srvaav-alias',
    desc: 'SRV/UAV aliasing in GELU backward: binding ffn_pre_gelu as both read source and gradient destination',
    fix:  'Use in-place gradient multiply workaround: accumulate ∂gelu into ffn_hidden, do not re-read pre_gelu as UAV',
  },
  {
    id:   'layernorm-dgamma-dbeta-race',
    desc: 'Race condition in LayerNorm dgamma/dbeta accumulation: multiple threads writing += to same dgamma[i] position',
    fix:  'Split into two entry points — one strided loop per thread (each thread covers distinct i values: i = tid + k*256) guarantees no collision at cs_5_0 without atomics',
  },
]);

export default { XVM_D3D11_BUFFERS, TRAINER_BUFFERS, FORWARD_PIPELINE, BACKWARD_PIPELINE, OPTIMIZER_PIPELINE, D3D11_NAN_BUGS };
