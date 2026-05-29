// SCX Control Delta — surface authority + replay integrity (v3.3.0-scx-control-flow)
//
// Mirrors control/core.js createDelta + surface tables.
// Pure in-memory — no fs/path/crypto Node deps.
// Hash function: djb2 over stable JSON (same scheme as PhaseArray).
// For production use, inject SubtleCrypto SHA256 via hashFn option.

export const SURFACE_FOLDS = Object.freeze({
  file:      'FILE_EXEC_FOLD',
  tool:      'TOOL_EXEC_FOLD',
  command:   'COMMAND_EXEC_FOLD',
  agent:     'AGENT_EXEC_FOLD',
  micronaut: 'MICRONAUT_EXEC_FOLD',
  skill:     'SKILL_EXEC_FOLD',
  action:    'ACTION_EXEC_FOLD',
  program:   'PROGRAM_EXEC_FOLD',
  function:  'FUNCTION_EXEC_FOLD',
});

export const SURFACE_EFFECTS = Object.freeze({
  file:      ['read', 'write', 'execute', 'replay'],
  tool:      ['read', 'write', 'execute', 'network', 'replay'],
  command:   ['read', 'write', 'execute', 'spawn', 'replay'],
  agent:     ['read', 'write', 'execute', 'memory', 'replay'],
  micronaut: ['read', 'write', 'execute', 'network', 'memory', 'replay'],
  skill:     ['read', 'write', 'execute', 'memory', 'replay'],
  action:    ['read', 'write', 'execute', 'replay'],
  program:   ['read', 'write', 'execute', 'spawn', 'gpu', 'model', 'replay'],
  function:  ['read', 'write', 'execute', 'gpu', 'model', 'memory', 'replay'],
});

export const COMPOUND_LOOP = Object.freeze([
  '/explore', '/specs', '/plan', '/work', '/review', '/compound', '/housekeeping',
]);

export const CONTROL_TRANSITIONS = Object.freeze(['@next', '@accept', '@reject']);

// Deterministic stable JSON (matches stableJson in core.js)
export function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${stableJson(value[k])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

// djb2 hash over a string → 8-char hex (browser-safe, sync)
export function djb2Hex(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = (((h << 5) + h) + str.charCodeAt(i)) >>> 0;
  // extend to 64-bit-like: hash the hash twice and combine
  let h2 = 0;
  for (let i = 0; i < str.length; i++) h2 = (((h2 << 5) + h2) ^ str.charCodeAt(i)) >>> 0;
  return h.toString(16).padStart(8, '0') + h2.toString(16).padStart(8, '0');
}

export function createDelta(opts = {}) {
  const { surface, target, operation = 'invoke', inputs = {}, result = {}, effects = ['read'], parent_delta_id = null, authority, base_hash } = opts;
  if (!SURFACE_FOLDS[surface]) throw new Error(`Unknown SCX surface: ${surface}`);
  const allowed = new Set(SURFACE_EFFECTS[surface]);
  for (const eff of effects) {
    if (!allowed.has(eff)) throw new Error(`Effect ${eff} is not allowed on surface ${surface}`);
  }
  const body = {
    record_type: 'scx_control_delta',
    delta_id: '',
    parent_delta_id: parent_delta_id || null,
    surface,
    target,
    fold_id: SURFACE_FOLDS[surface],
    authority: authority || { domain: surface.toUpperCase(), lane: 'CONTROL', effects },
    base_hash: base_hash || djb2Hex(target),
    operation,
    inputs,
    effects,
    result,
    replay_hash: '',
  };
  body.delta_id    = djb2Hex(stableJson({ ...body, delta_id: '', replay_hash: '' }));
  body.replay_hash = djb2Hex(stableJson(body));
  return body;
}

export function validateDelta(delta) {
  const errors = [];
  if (!delta.record_type)    errors.push('missing record_type');
  if (!SURFACE_FOLDS[delta.surface]) errors.push(`unknown surface: ${delta.surface}`);
  if (!delta.delta_id)       errors.push('missing delta_id');
  if (!delta.replay_hash)    errors.push('missing replay_hash');
  if (!delta.fold_id)        errors.push('missing fold_id');
  return { ok: errors.length === 0, errors };
}
