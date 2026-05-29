// JSONL → XJSL Compiler v0.1
// Transforms a Math Declaration + match context into one or more XJSL kernel nodes.
// Pure: no IO, no GPU dispatch. Output is a plain object ready for XJSL lowering.

import { XJSL_OP_REGISTRY } from './fused-op-schemas.js';

// ─── Compiler algorithm ───────────────────────────────────────────────────────
//
// compile_jsonl_to_xjsl(declaration, match, env):
//   1. Verify text matches declaration.pattern
//   2. Extract captures from regex match
//   3. Resolve kernel_name = xjsl_kernel || operation
//   4. Lookup IO spec from XJSL_OP_REGISTRY[kernel_name]
//   5. Build node (kind, kernel, inputs, outputs)
//   6. Build meta (copy decl.meta + inject env + map captures + attach kuhul_phase)
//   7. Build bindings skeleton (all nulls — filled by model-binding pass)
//   8. Return { [node_id]: node }

const ACTIVATION_MAP = Object.freeze({
  gelu:  'gelu',
  relu:  'relu',
  tanh:  'tanh',
  silu:  'silu',
  swish: 'silu',
});

function resolveCaptures(captures, captureMap) {
  if (!captureMap || !Array.isArray(captures)) return {};
  const out = {};
  for (const [idx, fieldName] of Object.entries(captureMap)) {
    const val = captures[Number(idx)];
    if (val !== undefined) {
      // Coerce to number if the string looks numeric
      out[fieldName] = /^\d+(\.\d+)?$/.test(val) ? Number(val) : val;
    }
  }
  return out;
}

function buildBindingsSkeleton(inputs, outputs) {
  const b = {};
  for (const k of inputs)  b[k] = null;
  for (const k of outputs) b[k] = null;
  return b;
}

function nextNodeId(kernel) {
  return `${kernel}_${Math.floor(Math.random() * 0xffff).toString(16).padStart(4, '0')}`;
}

/**
 * Compile one JSONL math declaration + match context into a XJSL node map.
 *
 * @param {object} declaration  - Math Declaration v0.1 object
 * @param {object} [match]      - { text?, captures?: string[], env?: object }
 * @param {object} [env]        - extra environment (dtype, dims, etc.)
 * @returns {{ [nodeId: string]: object }} - XJSL node map
 */
export function compileJsonlToXjsl(declaration, match = {}, env = {}) {
  const { text = '', captures = [], env: matchEnv = {} } = match;
  const merged_env = { ...matchEnv, ...env };

  // 1. Verify pattern match (warn, don't throw — caller may pre-filter)
  if (text && declaration.pattern) {
    const re = new RegExp(declaration.pattern, 'i');
    if (!re.test(text)) {
      console.warn(`[jsonl-compiler] pattern mismatch: "${declaration.pattern}" did not match "${text}"`);
    }
  }

  // 3. Resolve kernel
  const kernel_name = declaration.xjsl_kernel || declaration.operation;
  if (!kernel_name) throw new Error('declaration must have xjsl_kernel or operation');

  // 4. Lookup IO
  const spec = XJSL_OP_REGISTRY[kernel_name];
  const inputs  = spec ? spec.inputs.slice()  : [];
  const outputs = spec ? spec.outputs.slice() : ['y'];

  // 6. Build meta
  const meta = {
    // From registry defaults
    ...(spec ? spec.meta_defaults : {}),
    // From declaration.meta
    ...(declaration.meta || {}),
    // From environment
    ...(merged_env.dtype  !== undefined ? { dtype:    merged_env.dtype  } : {}),
    ...(merged_env.d_model !== undefined? { d_model:  merged_env.d_model} : {}),
    // From captures via constraints.capture_map
    ...resolveCaptures(captures, declaration.constraints?.capture_map),
    // kuhul_phase tagging
    ...(declaration.kuhul_phase ? { kuhul_phase: declaration.kuhul_phase } : {}),
    // semantic_fold for runtime routing
    ...(declaration.semantic_fold ? { semantic_fold: declaration.semantic_fold } : {}),
  };

  // Normalize activation field
  if (meta.activation) meta.activation = ACTIVATION_MAP[meta.activation] || meta.activation;

  // d_head inference for attention
  if (kernel_name === 'fused_attention' && meta.d_model && meta.num_heads && !meta.d_head) {
    meta.d_head = Math.floor(meta.d_model / meta.num_heads);
  }

  // 5 + 7. Build node
  const node_id = nextNodeId(kernel_name);
  const node = {
    kind:     'kernel',
    kernel:   kernel_name,
    inputs,
    outputs,
    meta,
    bindings: buildBindingsSkeleton(inputs, outputs),
    // Compiler provenance (stripped before GPU dispatch)
    _decl: {
      operation:      declaration.operation,
      semantic_fold:  declaration.semantic_fold,
      math_equivalent:declaration.math_equivalent,
      math_technique: declaration.math_technique || null,
    },
  };

  return { [node_id]: node };
}

/**
 * Compile multiple declarations (or text→declarations resolved externally).
 * Returns a merged XJSL node map.
 */
export function compileMany(declarations, match = {}, env = {}) {
  const out = {};
  for (const decl of declarations) {
    Object.assign(out, compileJsonlToXjsl(decl, match, env));
  }
  return out;
}

/**
 * Strip compiler provenance fields before lowering or dispatch.
 */
export function stripCompilerMeta(nodeMap) {
  const clean = {};
  for (const [id, node] of Object.entries(nodeMap)) {
    const { _decl, ...rest } = node;
    void _decl;
    clean[id] = rest;
  }
  return clean;
}

/**
 * Fill binding skeleton from a model-parameter map.
 * paramMap: { 'Wq': 'params.transformer.attn.Wq', ... }
 */
export function fillBindings(nodeMap, paramMap) {
  const out = {};
  for (const [id, node] of Object.entries(nodeMap)) {
    const bindings = { ...node.bindings };
    for (const key of Object.keys(bindings)) {
      if (paramMap[key] !== undefined) bindings[key] = paramMap[key];
    }
    out[id] = { ...node, bindings };
  }
  return out;
}
