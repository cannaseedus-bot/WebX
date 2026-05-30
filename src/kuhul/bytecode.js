// bytecode.js — K'UHUL bytecode format + opcode → bytecode mapping
//
// Each instruction is 3 bytes: [opcode u8][flags u8][data u8]
// For data > 255 bytes, a variable-length extension follows:
//   [length u8][opcode u8][data bytes...]
//
// WASM target: each 3-byte instruction maps to 1-3 WASM instructions.
// Compilation pipeline: ngram → sugar expand → opcode → bytecode → WASM
//
// Flag bits (byte 1):
//   0x01 IMMUTABLE   0x02 ASYNC    0x04 PARALLEL  0x08 OPTIONAL
//   0x10 FORCE       0x20 META     0x40 INLINE     0x80 reserved

export const FLAGS = Object.freeze({
  IMMUTABLE: 0x01,
  ASYNC:     0x02,
  PARALLEL:  0x04,
  OPTIONAL:  0x08,
  FORCE:     0x10,
  META:      0x20,
  INLINE:    0x40,
});

// ─── Core bytecodes (0x01-0x15) ───────────────────────────────────────────────

export const CORE_BYTECODE = {
  'POP':      { opcode: 0x01, bytes: [0x01,0x00,0x00], stack: {in:0, out:1}, description: 'Begin block — push frame' },
  'XUL':      { opcode: 0x02, bytes: [0x02,0x00,0x00], stack: {in:1, out:0}, description: 'End block — pop frame' },
  'SEK':      { opcode: 0x03, bytes: [0x03,0x00,0x00], stack: {in:2, out:1}, description: 'Set / assign' },
  'YAX':      { opcode: 0x04, bytes: [0x04,0x00,0x00], stack: {in:1, out:1}, description: 'Get / access' },
  'WO':       { opcode: 0x05, bytes: [0x05,0x00,0x00], stack: {in:'var', out:1}, description: 'Call / invoke' },
  'CHEN':     { opcode: 0x06, bytes: [0x06,0x00,0x00], stack: {in:2, out:0}, description: 'Store / persist' },
  'YAX_OPT':  { opcode: 0x07, bytes: [0x07,0x08,0x00], stack: {in:1, out:1}, description: 'Optional get' },
  'SEK_FORCE':{ opcode: 0x08, bytes: [0x08,0x10,0x00], stack: {in:2, out:1}, description: 'Force assign' },
  'WO_META':  { opcode: 0x09, bytes: [0x09,0x20,0x00], stack: {in:'var', out:1}, description: 'Meta call' },
  'IF':       { opcode: 0x0A, bytes: [0x0A,0x00,0x00], stack: {in:1, out:0}, description: 'Conditional', format: 'offset16' },
  'EACH':     { opcode: 0x0B, bytes: [0x0B,0x00,0x00], stack: {in:2, out:0}, description: 'Iterator', format: 'offset16' },
  'MATCH':    { opcode: 0x0C, bytes: [0x0C,0x00,0x00], stack: {in:2, out:1}, description: 'Pattern match', format: 'table32' },
  'RET':      { opcode: 0x0D, bytes: [0x0D,0x00,0x00], stack: {in:1, out:0}, description: 'Return' },
  'BRK':      { opcode: 0x0E, bytes: [0x0E,0x00,0x00], stack: {in:0, out:0}, description: 'Break loop' },
  'CONT':     { opcode: 0x0F, bytes: [0x0F,0x00,0x00], stack: {in:0, out:0}, description: 'Continue loop' },
  'TRY':      { opcode: 0x10, bytes: [0x10,0x00,0x00], stack: {in:0, out:0}, description: 'Try-catch', format: 'offset16' },
  'PIPE':     { opcode: 0x11, bytes: [0x11,0x00,0x00], stack: {in:2, out:1}, description: 'Pipe / flow' },
  'PAR':      { opcode: 0x12, bytes: [0x12,0x04,0x00], stack: {in:'var', out:'var'}, description: 'Parallel' },
  'BRANCH':   { opcode: 0x13, bytes: [0x13,0x00,0x00], stack: {in:1, out:2}, description: 'Branch' },
  'LOOP':     { opcode: 0x14, bytes: [0x14,0x00,0x00], stack: {in:0, out:0}, description: 'Loop back', format: 'offset16' },
  'BREAKF':   { opcode: 0x15, bytes: [0x15,0x00,0x00], stack: {in:0, out:0}, description: 'Break flow' },
};

// ─── ML / Tensor bytecodes (0x20-0x37) ───────────────────────────────────────

// Dtype codes for tensor definitions
export const DTYPE = Object.freeze({
  float32: 0x00, float64: 0x01, int8: 0x02, int16: 0x03,
  int32:   0x04, int64:   0x05, uint8: 0x06, uint16: 0x07,
  uint32:  0x08, uint64:  0x09, bool: 0x0A, bfloat16: 0x0B, float16: 0x0C,
});

// Activation types for ⟁Act⟁ / ⟁Fwd⟁
export const ACTIVATION = Object.freeze({
  relu: 0x00, leaky_relu: 0x01, elu: 0x02, selu: 0x03, gelu: 0x04,
  swish: 0x05, sigmoid: 0x06, tanh: 0x07, softmax: 0x08,
  log_softmax: 0x09, softplus: 0x0A, softsign: 0x0B, mish: 0x0C,
});

// Optimizer types for ⟁Opt⟁
export const OPTIMIZER = Object.freeze({
  sgd: 0x00, adam: 0x01, adamw: 0x02, rmsprop: 0x03,
  adagrad: 0x04, adadelta: 0x05, nadam: 0x06, lamb: 0x07,
});

// Loss types for ⟁Loss⟁
export const LOSS_TYPE = Object.freeze({
  mse: 0x00, mae: 0x01, huber: 0x02, cross_entropy: 0x03,
  binary_crossentropy: 0x04, kl_divergence: 0x05, hinge: 0x06,
  squared_hinge: 0x07, poisson: 0x08, cosine_similarity: 0x09,
});

// Normalization types for ⟁Norm⟁
export const NORM_TYPE = Object.freeze({
  layer_norm: 0x00, batch_norm: 0x01, instance_norm: 0x02,
  group_norm: 0x03, rms_norm: 0x04, spectral_norm: 0x05, weight_norm: 0x06,
});

// Attention types for ⟁Attn⟁
export const ATTENTION_TYPE = Object.freeze({
  self:    0x00, cross:   0x01, multi_head: 0x02, sparse:  0x03,
  linear:  0x04, flash:   0x05, sliding_window: 0x06, axial: 0x07,
});

export const ML_BYTECODE = {
  // Tensor/Weight definition
  'TENSOR':       { opcode: 0x20, bytes: [0x20,0x00,0x00], stack: {in:3,out:1}, format: 'shape_dtype' },
  'WEIGHT':       { opcode: 0x21, bytes: [0x21,0x00,0x00], stack: {in:2,out:1}, format: 'shape_init' },
  'TOKEN':        { opcode: 0x22, bytes: [0x22,0x00,0x00], stack: {in:2,out:1}, format: 'vocab_maxlen' },
  'LOGIC':        { opcode: 0x23, bytes: [0x23,0x00,0x00], stack: {in:3,out:1}, format: 'op_inputs' },
  'TENSOR_MUT':   { opcode: 0x24, bytes: [0x24,0x00,0x00], stack: {in:3,out:1}, format: 'view_slice' },
  'WEIGHT_TRAIN': { opcode: 0x25, bytes: [0x25,0x00,0x00], stack: {in:2,out:1}, format: 'hyperparams' },
  // Neural network
  'FORWARD':      { opcode: 0x26, bytes: [0x26,0x00,0x00], stack: {in:2,out:1}, format: 'layer_config' },
  'BACKWARD':     { opcode: 0x27, bytes: [0x27,0x00,0x00], stack: {in:2,out:1}, format: 'grad_config' },
  'LOSS':         { opcode: 0x28, bytes: [0x28,0x00,0x00], stack: {in:2,out:1}, format: 'loss_config' },
  'OPTIM':        { opcode: 0x29, bytes: [0x29,0x00,0x00], stack: {in:3,out:1}, format: 'opt_config' },
  'ATTN':         { opcode: 0x2A, bytes: [0x2A,0x00,0x00], stack: {in:3,out:1}, format: 'attn_config' },
  'NORM':         { opcode: 0x2B, bytes: [0x2B,0x00,0x00], stack: {in:2,out:1}, format: 'norm_config' },
  // Data ops
  'DROP':         { opcode: 0x2C, bytes: [0x2C,0x00,0x00], stack: {in:2,out:1}, format: 'rate' },
  'ACT':          { opcode: 0x2D, bytes: [0x2D,0x00,0x00], stack: {in:2,out:1}, format: 'act_type' },
  'LOAD':         { opcode: 0x2E, bytes: [0x2E,0x00,0x00], stack: {in:1,out:1}, format: 'path' },
  'SAVE':         { opcode: 0x2F, bytes: [0x2F,0x00,0x00], stack: {in:2,out:0}, format: 'path_fmt' },
  'EVAL':         { opcode: 0x30, bytes: [0x30,0x00,0x00], stack: {in:2,out:1}, format: 'metrics' },
  'PRED':         { opcode: 0x31, bytes: [0x31,0x00,0x00], stack: {in:2,out:1}, format: 'temp_topk' },
  'GRAD':         { opcode: 0x32, bytes: [0x32,0x00,0x00], stack: {in:2,out:1}, format: 'variables' },
  'STOP':         { opcode: 0x33, bytes: [0x33,0x00,0x00], stack: {in:1,out:1} },
  'BATCH':        { opcode: 0x34, bytes: [0x34,0x00,0x00], stack: {in:2,out:1}, format: 'batch_size' },
  'SHUF':         { opcode: 0x35, bytes: [0x35,0x00,0x00], stack: {in:1,out:1}, format: 'seed' },
  'BATCH_NORM':   { opcode: 0x36, bytes: [0x36,0x00,0x00], stack: {in:2,out:1}, format: 'gamma_beta' },
  'AUG':          { opcode: 0x37, bytes: [0x37,0x00,0x00], stack: {in:2,out:1}, format: 'aug_type' },
};

// ─── Distributed bytecodes (0x40-0x4B) ───────────────────────────────────────

export const DIST_BYTECODE = {
  'CLUSTER':     { opcode: 0x40, bytes: [0x40,0x00,0x00], stack: {in:1,out:1} },
  'NODE':        { opcode: 0x41, bytes: [0x41,0x00,0x00], stack: {in:1,out:1} },
  'CLUSTER_DYN': { opcode: 0x42, bytes: [0x42,0x10,0x00], stack: {in:1,out:1} },
  'NODE_DYN':    { opcode: 0x43, bytes: [0x43,0x10,0x00], stack: {in:1,out:1} },
  'DIST':        { opcode: 0x44, bytes: [0x44,0x00,0x00], stack: {in:2,out:1} },
  'GATH':        { opcode: 0x45, bytes: [0x45,0x00,0x00], stack: {in:1,out:1} },
  'SYNC':        { opcode: 0x46, bytes: [0x46,0x00,0x00], stack: {in:1,out:0} },
  'PART':        { opcode: 0x47, bytes: [0x47,0x00,0x00], stack: {in:2,out:1} },
  'REDUCE':      { opcode: 0x48, bytes: [0x48,0x00,0x00], stack: {in:2,out:1} },
  'REPL':        { opcode: 0x49, bytes: [0x49,0x00,0x00], stack: {in:2,out:1} },
  'FAIL':        { opcode: 0x4A, bytes: [0x4A,0x00,0x00], stack: {in:1,out:0} },
  'REC':         { opcode: 0x4B, bytes: [0x4B,0x00,0x00], stack: {in:1,out:1} },
};

// ─── XCFE bytecodes (0x60-0x69) ───────────────────────────────────────────────

export const XCFE_BYTECODE = {
  'XCFE':  { opcode: 0x60, bytes: [0x60,0x00,0x00], stack: {in:2,out:1} },
  'VAL':   { opcode: 0x61, bytes: [0x61,0x00,0x00], stack: {in:2,out:1} },
  'MON':   { opcode: 0x62, bytes: [0x62,0x00,0x00], stack: {in:2,out:0} },
  'ENF':   { opcode: 0x63, bytes: [0x63,0x00,0x00], stack: {in:2,out:0} },
  'DEC':   { opcode: 0x64, bytes: [0x64,0x00,0x00], stack: {in:2,out:1} },
  'PATH':  { opcode: 0x65, bytes: [0x65,0x00,0x00], stack: {in:2,out:1} },
  'RULE':  { opcode: 0x66, bytes: [0x66,0x00,0x00], stack: {in:2,out:0} },
  'STATE': { opcode: 0x67, bytes: [0x67,0x00,0x00], stack: {in:2,out:1} },
  'TRANS': { opcode: 0x68, bytes: [0x68,0x00,0x00], stack: {in:2,out:1} },
  'CHECK': { opcode: 0x69, bytes: [0x69,0x00,0x00], stack: {in:1,out:1} },
};

// ─── Tensor memory layout (WASM linear memory header) ─────────────────────────
// Every tensor starts with a 32-byte header at its allocation offset.

export const TENSOR_HEADER = Object.freeze({
  MAGIC:          0x544E5352,  // "TNSR"
  VERSION:        0x00010000,
  // Field offsets within header
  OFFSET_MAGIC:   0,
  OFFSET_VERSION: 4,
  OFFSET_ID:      8,
  OFFSET_RANK:    12,
  OFFSET_NELEMS:  16,
  OFFSET_ESIZE:   20,
  OFFSET_FLAGS:   24,
  OFFSET_DATA:    28,
  SIZE:           32,
});

// ─── Bytecode validator ────────────────────────────────────────────────────────

const VALID_RANGES = [
  [0x01, 0x15],  // core
  [0x20, 0x37],  // ml
  [0x40, 0x4B],  // distributed
  [0x60, 0x69],  // xcfe
];

export function validateBytecode(buf) {
  const errors = [];
  for (let i = 0; i < buf.length; i += 3) {
    const op = buf[i];
    const inRange = VALID_RANGES.some(([lo, hi]) => op >= lo && op <= hi);
    if (!inRange) errors.push(`Unknown opcode 0x${op.toString(16).padStart(2,'0')} at offset ${i}`);
  }
  return { valid: errors.length === 0, errors, instructions: Math.floor(buf.length / 3) };
}

// ─── Simple bytecode assembler ─────────────────────────────────────────────────

export class BytecodeAssembler {
  constructor() { this._buf = []; }

  emit(opcode, flags = 0x00, data = 0x00) {
    this._buf.push(opcode & 0xFF, flags & 0xFF, data & 0xFF);
    return this;
  }

  get buffer() { return new Uint8Array(this._buf); }

  // Convenience methods
  sek(data = 0)    { return this.emit(0x03, 0x00, data); }
  wo(fn = 0)       { return this.emit(0x05, 0x00, fn); }
  woAsync(fn = 0)  { return this.emit(0x05, FLAGS.ASYNC, fn); }
  tensor(id = 0)   { return this.emit(0x20, 0x00, id); }
  tensorPar(id = 0){ return this.emit(0x20, FLAGS.PARALLEL, id); }
  forward(layer=0) { return this.emit(0x26, 0x00, layer); }
  backward()       { return this.emit(0x27); }
  loss(type = 3)   { return this.emit(0x28, 0x00, type); }  // default: cross_entropy
  optim(type = 2)  { return this.emit(0x29, 0x00, type); }  // default: adamw
  pipe()           { return this.emit(0x11); }
  trainStep() {
    return this.forward().loss().backward().optim().pipe();
  }
}
