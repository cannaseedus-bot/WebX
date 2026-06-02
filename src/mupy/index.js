// mupy/index.js — µPY: the Python evolutionary layer of the µMODEL system
//
// µMODEL hierarchy:
//   µMODEL  — Schema/TOML/YAML/XML/MD as behavioral spec (Driver/Kernel)
//   µPY     — Python trainer + evolver that turns specs into running models
//   SMGM-16 — The underlying MoE architecture (52 CardSlots, 6 layers)
//
// µPY is the evolutionary step because Python is where:
//   - Specs are read (SemanticReader → topology)
//   - Weights are trained (finetune_toolcall_pt.py / smgm16 train.py)
//   - Models are evolved  (mutate.py — breed/prune/score candidates)
//   - Bins are built      (build_math_tokens.py, build_coder_tokens.py)
//
// A µPY model is defined by three files:
//   <domain>.spec.toml   — behavioral spec (what the µMODEL knows)
//   <domain>.train.py    — training script (µPY trainer)
//   <domain>.ckpt/       — checkpoint directory (evolving weights)
//
// The JS runtime (this file) reads the spec and routes to the correct µPY model.

import { SemanticReader } from '../xcfe/semantic-reader.js';

// ─── Spec loader ──────────────────────────────────────────────────────────────

/** Parse a µPY spec TOML into a normalized descriptor. */
export function parseMupySpec(tomlText, sourceName = 'spec.toml') {
  const lines = tomlText.split(/\r?\n/);
  const spec  = { source: sourceName, domain: '', phase: 'Sek', gravity: 'Normal',
                   training: {}, routing: {}, capabilities: [] };

  let section = '';
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const secM = line.match(/^\[([^\]]+)\]$/);
    if (secM) { section = secM[1]; continue; }
    const kv = line.match(/^([A-Za-z0-9_.]+)\s*=\s*(.+)$/);
    if (!kv) continue;
    const [, key, val] = kv;
    const v = val.replace(/^["']|["']$/g, '');
    if (!section) {
      if (key === 'domain')    spec.domain  = v;
      if (key === 'phase')     spec.phase   = v;
      if (key === 'gravity')   spec.gravity = v;
    } else if (section === 'training') {
      spec.training[key] = isNaN(v) ? v : Number(v);
    } else if (section === 'routing') {
      spec.routing[key] = v;
    } else if (section === 'capabilities') {
      if (key === 'tools') spec.capabilities = v.split(',').map(s => s.trim()).filter(Boolean);
    }
  }
  return spec;
}

/** Build a µPY model descriptor from a TOML spec + optional XML KXML node. */
export function buildMupyDescriptor(tomlText, kxmlText = '', sourceName = 'spec.toml') {
  const spec     = parseMupySpec(tomlText, sourceName);
  const topology = kxmlText
    ? SemanticReader.read(kxmlText, sourceName.replace('.toml', '.kxml')).topology
    : null;

  const folds = topology?.folds ?? [];
  const pressure = Math.min(1.0,
    0.2 +
    spec.capabilities.length * 0.05 +
    folds.length * 0.08
  );

  return {
    ok:           true,
    type:         'mupy_descriptor.v1',
    domain:       spec.domain,
    phase:        spec.phase,
    gravity:      spec.gravity,
    capabilities: spec.capabilities,
    training:     spec.training,
    routing:      spec.routing,
    topology_folds: folds.map(f => f.id),
    activation: {
      pressure,
      active: pressure >= 0.5,
    },
    trainer_cmd: [
      'python', `tools/trainers/train_${spec.domain}_micronaut.py`,
      '--steps',   String(spec.training.steps  ?? 3000),
      '--batch',   String(spec.training.batch  ?? 4),
      '--lr',      String(spec.training.lr     ?? '2e-5'),
    ],
    invariants: {
      spec_is_source_of_truth:   true,
      weights_are_derived:       true,
      evolution_preserves_spec:  true,
    },
  };
}

// ─── µMODEL registry ─────────────────────────────────────────────────────────

const _registry = new Map();

/** Register a µPY model descriptor under its domain name. */
export function registerMupy(descriptor) {
  _registry.set(descriptor.domain, descriptor);
}

/** Retrieve a registered µPY descriptor by domain. */
export function getMupy(domain) {
  return _registry.get(domain) ?? null;
}

/** List all registered µPY domains. */
export function listMupy() {
  return [..._registry.values()].map(d => ({
    domain:       d.domain,
    phase:        d.phase,
    gravity:      d.gravity,
    capabilities: d.capabilities,
    active:       d.activation.active,
    pressure:     d.activation.pressure,
  }));
}

// ─── Built-in µPY model specs (canonical µMODELS) ────────────────────────────

export const MUPY_MATH_SPEC = `
[model]
domain   = "math_tool"
phase    = "Sek"
gravity  = "Normal"

[training]
steps  = 3000
batch  = 4
lr     = 2e-5
block  = 256
seqs   = 56707

[routing]
trigger  = "arithmetic,calculus,linear_algebra,statistics,geometry,proof"
fallback = "base_gpt2"

[capabilities]
tools = "math_tool,fibonacci_fold,pi_field,linalg_solver,mayan_fold"
`;

export const MUPY_CODER_SPEC = `
[model]
domain   = "coder_tool"
phase    = "Sek"
gravity  = "Heavy"

[training]
steps  = 5000
batch  = 8
lr     = 1e-5
block  = 256
seqs   = 30822

[routing]
trigger  = "code,function,class,algorithm,debug,refactor,test,script"
fallback = "base_gpt2"

[capabilities]
tools = "coder_tool,shell_run,file_read,file_write,git_status,kuhul_agent"
`;

// Register built-ins at module load
for (const spec of [MUPY_MATH_SPEC, MUPY_CODER_SPEC]) {
  const d = buildMupyDescriptor(spec);
  registerMupy(d);
}

export { parseMupySpec as parse, buildMupyDescriptor as build,
         registerMupy as register, getMupy as get, listMupy as list };
