// Micronaut Expert Registry + SCXQ2 ISA (v3.0.0-agentic-micronaut)
//
// 8-expert MoE shader pipeline mapped to micronaut expert roles.
// SCXQ2 ISA: 9 opcode categories, 11 domains, 11 lanes, 16 flags.
// Source: experts/micronaut_expert_registry.json + specs/scxq2-isa.json

// ── SCXQ2 ISA ────────────────────────────────────────────────────────────────

export const SCXQ2_OPCODES = Object.freeze({
  // MEMORY_OPS 0x00-0x07
  MEM_LOAD:         0x00,
  MEM_STORE:        0x01,
  MEM_MOVE:         0x02,
  MEM_SWAP:         0x03,
  MEM_QUANTIZE:     0x04,
  MEM_DEQUANT:      0x05,
  MEM_PIN:          0x06,
  MEM_STREAM:       0x07,
  // COMPUTE_OPS 0x10-0x17
  COMPUTE_MATMUL:   0x10,
  COMPUTE_ATTENTION:0x11,
  COMPUTE_SOFTMAX:  0x12,
  COMPUTE_FFN:      0x13,
  COMPUTE_SILU:     0x14,
  COMPUTE_NORM:     0x15,
  COMPUTE_ENTROPY:  0x16,
  COMPUTE_SIGNAL:   0x17,
  // FLOW_OPS 0x30-0x34
  FLOW_SIGNAL:      0x30,
  FLOW_ROUTE:       0x31,
  FLOW_BARRIER:     0x32,
  FLOW_SYNC:        0x33,
  FLOW_DISPATCH:    0x34,
  // CACHE_OPS 0x40-0x44
  CACHE_KV_STORE:   0x40,
  CACHE_KV_LOAD:    0x41,
  CACHE_EVICT:      0x42,
  CACHE_QUANTIZE:   0x43,
  CACHE_PREFETCH:   0x44,
  // REPLAY_OPS 0x50-0x53
  REPLAY_CAPTURE:   0x50,
  REPLAY_RUN:       0x51,
  REPLAY_COMPRESS:  0x52,
  REPLAY_VERIFY:    0x53,
  // ADAPTER_OPS 0x60-0x63
  ADAPTER_APPLY:    0x60,
  ADAPTER_MERGE:    0x61,
  ADAPTER_QUANT:    0x62,
  ADAPTER_STREAM:   0x63,
  // STREAM_OPS 0x70-0x73
  STREAM_TILE:      0x70,
  STREAM_MIP:       0x71,
  STREAM_EVICT:     0x72,
  STREAM_PIN:       0x73,
  // FIELD_OPS 0x80-0x83
  FIELD_GEODESIC:   0x80,
  FIELD_PROPAGATE:  0x81,
  FIELD_STRESS:     0x82,
  FIELD_ROUTE:      0x83,
  // VERIFY_OPS 0x90-0x93
  VERIFY_HASH:      0x90,
  VERIFY_BARRIER:   0x91,
  VERIFY_ATTEST:    0x92,
  VERIFY_CHAIN:     0x93,
});

export const SCXQ2_DOMAINS = Object.freeze({
  COGNITION:  { id: 0x01, residency: 'HOT',       max_entropy: 0.85 },
  MEMORY:     { id: 0x02, residency: 'WARM',      max_entropy: 0.70 },
  GPU:        { id: 0x03, residency: 'HOT',       max_entropy: 0.90 },
  STREAM:     { id: 0x04, residency: 'STREAMING', max_entropy: 0.60 },
  SECURITY:   { id: 0x05, residency: 'PINNED',    max_entropy: 0.95 },
  MESH:       { id: 0x06, residency: 'WARM',      max_entropy: 0.75 },
  TRAINING:   { id: 0x07, residency: 'COLD',      max_entropy: 0.65 },
  SESSION:    { id: 0x08, residency: 'WARM',      max_entropy: 0.80 },
  PROJECTION: { id: 0x09, residency: 'COLD',      max_entropy: 0.55 },
  ADAPTER:    { id: 0x0A, residency: 'WARM',      max_entropy: 0.70 },
  SHARD:      { id: 0x0B, residency: 'STREAMING', max_entropy: 0.50 },
});

export const SCXQ2_LANES = Object.freeze({
  DICT:   { id: 0x01, purpose: 'metadata/control',  quant: 'NONE'   },
  FIELD:  { id: 0x02, purpose: 'tensors/weights',   quant: 'INT8/INT16' },
  EDGE:   { id: 0x03, purpose: 'fold relations',    quant: 'NONE'   },
  LANE:   { id: 0x04, purpose: 'execution schedule',quant: 'NONE'   },
  CACHE:  { id: 0x05, purpose: 'KV + replay',       quant: 'INT8'   },
  GRAM:   { id: 0x06, purpose: 'token programs',    quant: 'NONE'   },
  TRACE:  { id: 0x07, purpose: 'event traces',      quant: 'SCXQ2'  },
  MESH:   { id: 0x08, purpose: 'distributed shards',quant: 'INT8'   },
  STREAM: { id: 0x09, purpose: 'GPU streaming',     quant: 'INT4'   },
  DELTA:  { id: 0x0A, purpose: 'adapter updates',   quant: 'INT8'   },
  PROOF:  { id: 0x0B, purpose: 'verification',      quant: 'NONE'   },
});

// instruction_words: base_64=[opcode:u8,domain:u8,fold_id:u16,offset:u32]
// extended_128 adds [length:u16,stride:u16,lane_id:u16,flags:u16]
export const SCXQ2_INSTR_BYTES = Object.freeze({ base: 8, extended: 16 });

export const SCXQ2_FLAGS = Object.freeze({
  BARRIER_REQUIRED: 1 << 0,
  SYNC_REQUIRED:    1 << 1,
  QUANTIZED_IN:     1 << 2,
  QUANTIZED_OUT:    1 << 3,
  STREAMING:        1 << 4,
  ASYNC:            1 << 5,
  REPLAY_CAPTURE:   1 << 6,
  VERIFY_HASH:      1 << 7,
  MUTATION_ALLOWED: 1 << 8,
  GPU_RESIDENT:     1 << 9,
  PERSISTENT:       1 << 10,
  EPHEMERAL:        1 << 11,
});

// ── Expert Registry ───────────────────────────────────────────────────────────

export const MICRONAUT_EXPERTS = Object.freeze([
  { id: 0, name: 'geometry',  expert: 'FOLD_TOPOLOGY_EXPERT',    domain: 'COGNITION', lane: 'FIELD', folds: ['MICRONAUT_TOPOLOGY_FOLD', 'DDS_SHARD_*'] },
  { id: 1, name: 'temporal',  expert: 'SESSION_TEMPORAL_EXPERT', domain: 'SESSION',   lane: 'TRACE', folds: ['SESSION_FOLD', 'MICRONAUT_ADAPTER_FOLD'] },
  { id: 2, name: 'amplify',   expert: 'AGENTIC_AMPLIFY_EXPERT',  domain: 'ADAPTER',   lane: 'DELTA', folds: ['ADAPTER_FOLD', 'MICRONAUT_AGENT_FOLD', 'MICRONAUT_ADAPTER_FOLD'] },
  { id: 3, name: 'compress',  expert: 'SCXQ2_COMPRESS_EXPERT',   domain: 'MEMORY',    lane: 'CACHE', folds: ['KV_CACHE_FOLD', 'DDS_SHARD_*'] },
  { id: 4, name: 'focus',     expert: 'ATTENTION_FOCUS_EXPERT',  domain: 'GPU',       lane: 'FIELD', folds: ['ATTENTION_FOLD_L*'] },
  { id: 5, name: 'integrate', expert: 'FOLD_INTEGRATE_EXPERT',   domain: 'GPU',       lane: 'LANE',  folds: ['EMBEDDING_FOLD', 'ATTENTION_FOLD_L*', 'FFN_FOLD_L*', 'KV_CACHE_FOLD'] },
  { id: 6, name: 'pattern',   expert: 'MICRONAUT_PATTERN_EXPERT',domain: 'COGNITION', lane: 'DICT',  folds: ['MICRONAUT_KNOWLEDGE_FOLD', 'MICRONAUT_AGENT_FOLD'] },
  { id: 7, name: 'novelty',   expert: 'NOVELTY_ROUTING_EXPERT',  domain: 'COGNITION', lane: 'EDGE',  folds: ['MICRONAUT_AGENT_FOLD', 'SESSION_FOLD', 'REPLAY_FOLD'] },
]);

export const EXPERT_COUNT = 8;

export const ADAPTATION_SHADERS = Object.freeze({
  reward:    { stage: 1, role: 'compute_bounded_reward_for_adapter_candidates',  writes: ['reward_scores', 'reward_log'] },
  mutation:  { stage: 2, role: 'emit_bounded_adapter_deltas',                   writes: ['adapter_out', 'mutation_log'] },
  evolution: { stage: 3, role: 'select_and_recombine_adapter_candidates',       writes: ['evolved_out', 'evolution_log'] },
});

export const FOLD_BARRIER_LAWS = Object.freeze([
  'session_cannot_mutate_until_replay_finalized',
  'training_cannot_update_until_attention_verified',
  'kv_cache_persists_with_hash',
]);

export const RECOGNITION_RULE = 'Every shader ExpertId 0-7 must resolve to at least one physical fold or Micronaut adapter fold before dispatch.';

export function getExpertById(id) {
  return MICRONAUT_EXPERTS.find(e => e.id === id) || null;
}

export function getExpertByName(name) {
  return MICRONAUT_EXPERTS.find(e => e.name === name) || null;
}
