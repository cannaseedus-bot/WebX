// Micronaut Coding DAG + 12-core temperature map (v3.0.0-agentic-micronaut)
//
// Deterministic coding workflow: intent → context → plan → generate →
//   validate → patch → test → reward → mutate → evolve → done | blocked
// Source: experts/coding_dag.json + specs/agentic-bridge-gpt2.json

// ── 12-core temperature-controlled micronaut types ────────────────────────────

export const CORE_MICRONAUTS = Object.freeze([
  { name: 'TOOL-µ',  temperature: 0.2, role: 'Tool Execution & Management' },
  { name: 'AGENT-µ', temperature: 0.6, role: 'Autonomous Agent Orchestration' },
  { name: 'SKILL-µ', temperature: 0.4, role: 'Skill Acquisition & Execution' },
  { name: 'CMD-µ',   temperature: 0.2, role: 'Command Processing & Validation' },
  { name: 'OP-µ',    temperature: 0.1, role: 'Low-Level Operation Codes' },
  { name: 'FILE-µ',  temperature: 0.2, role: 'File System Management' },
  { name: 'TASK-µ',  temperature: 0.5, role: 'Task Planning & Management' },
  { name: 'THINK-µ', temperature: 0.7, role: 'Deep Reasoning & Reflection' },
  { name: 'VALID-µ', temperature: 0.2, role: 'Code & Action Validation' },
  { name: 'PLAN-µ',  temperature: 0.5, role: 'Strategic Planning' },
  { name: 'LLM-µ',   temperature: 0.7, role: 'Language Model Bridge' },
  { name: 'BOSS-µ',  temperature: 0.4, role: 'System Orchestration' },
]);

export const CORE_MICRONAUT_COUNT = 12;

export function getMicronaught(name) {
  return CORE_MICRONAUTS.find(m => m.name === name) || null;
}

// ── Coding DAG ────────────────────────────────────────────────────────────────

export const DAG_ENTRY_NODE    = 'intent';
export const DAG_TERMINAL_NODES = Object.freeze(['done', 'blocked']);

export const CODING_DAG_NODES = Object.freeze([
  { id: 'intent',   kind: 'classifier',          expert: 'MICRONAUT_PATTERN_EXPERT',   folds: ['MICRONAUT_KNOWLEDGE_FOLD', 'MICRONAUT_AGENT_FOLD'],                              outputs: ['task_type', 'risk', 'requires_code_change'] },
  { id: 'context',  kind: 'retrieval',            expert: 'FOLD_TOPOLOGY_EXPERT',       folds: ['MEMORY_FOLD', 'MICRONAUT_TOPOLOGY_FOLD', 'DDS_SHARD_*'],                        outputs: ['files', 'symbols', 'local_contracts'] },
  { id: 'plan',     kind: 'planner',              expert: 'AGENTIC_AMPLIFY_EXPERT',     folds: ['MICRONAUT_AGENT_FOLD', 'SESSION_FOLD'],                                         outputs: ['steps', 'write_scope', 'verification_plan'] },
  { id: 'generate', kind: 'codegen',              expert: 'FOLD_INTEGRATE_EXPERT',      folds: ['EMBEDDING_FOLD', 'ATTENTION_FOLD_L*', 'FFN_FOLD_L*', 'MICRONAUT_ADAPTER_FOLD'], outputs: ['candidate_patch', 'trace'] },
  { id: 'validate', kind: 'static_validation',    expert: 'SCXQ2_COMPRESS_EXPERT',      folds: ['KV_CACHE_FOLD', 'REPLAY_FOLD', 'SECURITY_FOLD'],                                 outputs: ['syntax_ok', 'contract_ok', 'violations'] },
  { id: 'patch',    kind: 'apply_patch',          expert: 'ATTENTION_FOCUS_EXPERT',     folds: ['SESSION_FOLD', 'REPLAY_FOLD'],                                                  outputs: ['changed_files', 'patch_hash'] },
  { id: 'test',     kind: 'runtime_validation',   expert: 'SESSION_TEMPORAL_EXPERT',    folds: ['SESSION_FOLD', 'REPLAY_FOLD'],                                                  outputs: ['test_pass', 'test_failures'] },
  { id: 'reward',   kind: 'adaptation_shader',    expert: 'AGENTIC_AMPLIFY_EXPERT',     folds: ['MICRONAUT_ADAPTER_FOLD'],                                                       outputs: ['reward_score'] },
  { id: 'mutate',   kind: 'adaptation_shader',    expert: 'AGENTIC_AMPLIFY_EXPERT',     folds: ['MICRONAUT_ADAPTER_FOLD'],                                                       outputs: ['delta'] },
  { id: 'evolve',   kind: 'adaptation_shader',    expert: 'NOVELTY_ROUTING_EXPERT',     folds: ['MICRONAUT_ADAPTER_FOLD', 'MICRONAUT_AGENT_FOLD'],                               outputs: ['evolved_adapter'] },
  { id: 'done',     kind: 'terminal' },
  { id: 'blocked',  kind: 'terminal' },
]);

export function getDagNode(id) {
  return CODING_DAG_NODES.find(n => n.id === id) || null;
}

// ── Coding JSONL record types ─────────────────────────────────────────────────

export const CODING_RECORD_TYPES = Object.freeze([
  'coding_role_turn',
  'coding_response',
  'coding_dag_event',
  'coding_validation',
  'coding_reward',
  'coding_mutation',
  'coding_evolution',
  'coding_replay',
]);

export const CODING_ROLES = Object.freeze([
  'user', 'assistant', 'system', 'tool', 'agent',
  'validator', 'shader', 'reward', 'mutation', 'evolution',
]);

export function createCodingRecord(type, opts = {}) {
  if (!CODING_RECORD_TYPES.includes(type)) throw new Error(`Unknown coding record type: ${type}`);
  return {
    type,
    version:   '3.0.0',
    record_id: opts.record_id || `cod_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    session_id: opts.session_id || '',
    turn_id:    opts.turn_id   || '',
    parent_record_id: opts.parent_record_id || null,
    timestamp:  opts.timestamp || new Date().toISOString(),
    role:       opts.role      || 'assistant',
    dag_node:   opts.dag_node  || DAG_ENTRY_NODE,
    expert:     opts.expert    || null,
    folds:      opts.folds     || [],
    content:    opts.content   || null,
  };
}

// ── Fold types ────────────────────────────────────────────────────────────────

export const CORE_FOLDS = Object.freeze([
  'TOKEN_FOLD', 'EMBEDDING_FOLD', 'ATTENTION_FOLD', 'FFN_FOLD',
  'KV_CACHE_FOLD', 'MEMORY_FOLD', 'REPLAY_FOLD', 'TRAINING_FOLD',
  'PI_FIELD_FOLD', 'SESSION_FOLD_*',
]);

export const FOLD_TYPES = Object.freeze({
  DICT:  'metadata/control',
  FIELD: 'tensors/weights',
  EDGE:  'fold relations',
  LANE:  'execution schedule',
  CACHE: 'KV + replay',
  GRAM:  'token programs',
  TRACE: 'event traces',
  MESH:  'distributed shards',
});
