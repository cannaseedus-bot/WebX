// XJSL — Cross-platform Shader JSON Language
// Entry point for the full XJSL module suite.
// Covers: lowering (WGSL/HLSL), WebGPU runtime, schema validation,
//         autograd rules, fused ops catalog, GPT-2 config, TrainerConfig.

export { generateWGSL, generateHLSL, lowerXJSLDoc } from './lowering.js';
export { XJSLWGPURuntime }                           from './wgpu-runtime.js';
export { validateXJSLDoc, validateXJSLDocOrThrow }   from './validate.js';
export { AUTOGRAD_RULES }                            from './autograd.js';

// ─── GPT-2 / DistilGPT-2 shape constants ─────────────────────────────────────
// Matches trainer/gpt2_config.h from v0.1.1-igpu-trainer-xjsl.

export const GPT2_CONFIG = Object.freeze({
  vocab_size: 50260,   // 50257 base + 3 special tokens added during fine-tune
  n_ctx:      1024,
  n_embd:     768,
  n_head:     12,
  n_layer:    6,       // DistilGPT-2 = 6 (GPT-2 117M = 12)
  d_head:     64,      // n_embd / n_head
  d_ff:       3072,    // 4 * n_embd
  attn_scale: 0.125,   // 1 / sqrt(64)
});

// ─── Trainer configuration ────────────────────────────────────────────────────
// Hybrid Adam: GPU Adam for wte, CPU Adam for all other params.
// Introduced in v0.1.1-igpu-trainer-xjsl alongside gpt2_adam_wte.hlsl.

export class TrainerConfig {
  constructor(opts = {}) {
    // Adam hyperparams
    this.lr           = opts.lr           ?? 1e-4;
    this.beta1        = opts.beta1        ?? 0.9;
    this.beta2        = opts.beta2        ?? 0.999;
    this.eps          = opts.eps          ?? 1e-8;
    this.weight_decay = opts.weight_decay ?? 0.01;
    this.grad_clip    = opts.grad_clip    ?? 1.0;   // global gradient norm clamp

    // Batch / sequence geometry
    this.batch_size   = opts.batch_size   ?? 8;
    this.seq_len      = opts.seq_len      ?? 1024;

    // Hybrid Adam flags
    this.gpu_adam_wte = opts.gpu_adam_wte ?? true;  // dispatch gpt2_adam_wte.hlsl for wte
    this.cpu_adam     = opts.cpu_adam     ?? true;  // CPU Adam for all other params

    // Training steps
    this.warmup_steps  = opts.warmup_steps  ?? 100;
    this.total_steps   = opts.total_steps   ?? 10_000;
    this.save_interval = opts.save_interval ?? 500;

    // Device (DXBC cs_5_0 only — NOT DXIL; Intel HD 4600 incompatible with DXIL)
    this.shader_model  = opts.shader_model  ?? 'cs_5_0';
  }

  // Compute bias-correction scalars for Adam step t (1-indexed).
  biasCorrectionFactors(step) {
    return {
      bias_corr1: 1 / (1 - Math.pow(this.beta1, step)),
      bias_corr2: 1 / (1 - Math.pow(this.beta2, step)),
    };
  }

  // Cosine-decay learning rate schedule with warmup.
  lrAtStep(step) {
    if (step < this.warmup_steps) return this.lr * step / this.warmup_steps;
    const progress = (step - this.warmup_steps) / Math.max(1, this.total_steps - this.warmup_steps);
    return this.lr * 0.5 * (1 + Math.cos(Math.PI * progress));
  }
}

// ─── Fused ops catalog ────────────────────────────────────────────────────────
// Describes composite operations that are dispatched as a single shader or
// a tightly fused sequence — no intermediate buffer reads back to CPU.

export const FUSED_OPS = Object.freeze({
  fused_mlp: {
    description: 'Linear → GELU → Linear: x @ W1 + b1 → gelu → @ W2 + b2',
    inputs:  ['x', 'W1', 'b1', 'W2', 'b2'],
    output:  'out',
    forward: (x, W1, b1, W2, b2, M, K, N1, N2) => {
      // h = gelu(x @ W1 + b1)  [M, N1]
      // out = h @ W2 + b2       [M, N2]
      // Implemented by caller via matmul + gelu + matmul sequence on GPU.
      throw new Error('fused_mlp: implement via GPU dispatch using KLSL/XJSL kernel');
    },
  },
  fused_attention: {
    description: 'QKV projection + scaled dot-product attention (single pass)',
    inputs:  ['x', 'Wq', 'Wk', 'Wv', 'Wo'],
    output:  'out',
    config:  { scale: GPT2_CONFIG.attn_scale, n_head: GPT2_CONFIG.n_head },
    forward: () => {
      throw new Error('fused_attention: implement via GPU dispatch (gpt2_qkv_split + gpt2_attn_fwd)');
    },
  },
  fused_norm: {
    description: 'Normalize + optional affine (w, b) + optional residual add',
    inputs:  ['x', 'w', 'b'],
    flags:   { affine: true, residual: false },
    output:  'out',
    forward: () => {
      throw new Error('fused_norm: implement via GPU dispatch (gpt2_layernorm_fwd)');
    },
  },
});

// ─── XJSL canonical schema ────────────────────────────────────────────────────
// Documents the expected shape of an XJSL document for reference / tooling.

export const XJSL_SCHEMA = Object.freeze({
  required:    ['@xjson_version', '@paradigm', '@shader_language', '@shaders'],
  shaderKeys:  ['@type', '@workgroup', '@inputs', '@outputs', '@uniforms', '@kernel'],
  bufferKeys:  ['@type', '@layout'],
  memoryModel: {
    DICT: 'JSON objects — symbol tables / named field maps',
    LANE: 'JSON arrays — contiguous numeric streams / uniform arrays',
    TILE: 'hex/base64/zstd blob strings — binary payloads (weights, tokens)',
  },
  concurrency: {
    batch:   'unit of work — one forward+backward pass',
    thread:  'parallel lane — one fiber or GPU thread',
    process: 'isolated context owning all state (params, grads, opt)',
    expert:  'specialized subgraph with own params, grads, optimizer state',
  },
  bufferClasses: {
    PARAM: 'weight tensors (frozen during inference; updated during training)',
    GRAD:  '∂L/∂W — gradient accumulator, zeroed after each optimizer step',
    ACT:   'activations — batch-scoped, discarded after backward',
    OPT:   'Adam m/v states — persist across steps',
    VEC:   'logits, loss, routing scores — output of forward, input to loss',
  },
  trainingPhases: ['forward: PARAM → ACT + VEC', 'backward: ACT + VEC → GRAD', 'optimize: PARAM + GRAD + OPT → PARAM'],
});
