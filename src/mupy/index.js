// mupy/index.js — µPY: the Python evolutionary layer of the µMODEL system
//
// µMODEL hierarchy:
//   µMODEL  — Schema/TOML/YAML/XML/MD as behavioral spec (Driver/Kernel)
//   µPY     — Python trainer + evolver that turns specs into running models
//   SMGM-16 — The underlying MoE architecture (52 CardSlots, 6 layers)
//
// µPY is the evolutionary step because Python is where:
//   - Specs are read (SemanticReader → topology + CDATA capsules)
//   - Weights are trained (finetune_toolcall_pt.py / smgm16 train.py)
//   - Models are evolved  (mutate.py — breed/prune/score candidates)
//   - Bins are built      (build_math_tokens.py, build_coder_tokens.py)
//
// A µPY model is defined by three files:
//   <domain>.spec.kxml   — KXML spec with CDATA inclusions
//   <domain>.train.py    — µPY trainer
//   <domain>.ckpt/       — checkpoint directory (evolving weights)
//
// CDATA inclusions are first-class in µMODEL specs. A CDATA block inside a
// µMODEL's KXML spec can carry:
//   kind="kuhul"        — K'UHUL phase program (Pop/Wo/Sek/Ch'en/Xul)
//   kind="projection"   — HLSL/WGSL shader code for GPU execution
//   kind="semantic_grams" — dotted n-grams for semantic pre-screening
//   kind="policy"       — what the µMODEL is and is not allowed to compute
//
// SemanticReader.read() extracts and classifies all CDATA capsules automatically.
// buildMupyDescriptor() folds them into the descriptor so the µPY trainer
// and the runtime both see the same CDATA surface.

import { SemanticReader } from '../xcfe/semantic-reader.js';

// ─── CDATA helpers ────────────────────────────────────────────────────────────

/** Extract all CDATA capsules by kind from a SemanticReader topology. */
function extractCdataByKind(topology) {
  const byKind = { kuhul: [], projection: [], semantic_grams: [], policy: [] };
  for (const cap of (topology?.cdata_capsules ?? [])) {
    for (const kind of cap.kinds) {
      if (byKind[kind]) byKind[kind].push(cap.payload);
    }
  }
  return byKind;
}

/** Merge CDATA semantic grams with TOML capabilities to produce unified tool list. */
function mergeCdataTools(capabilityList, semanticGramCdata) {
  const tools = new Set(capabilityList);
  // Extract dotted grams that look like tool names from CDATA payloads
  const gramRe = /([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){0,3})/g;
  for (const payload of semanticGramCdata) {
    for (const m of payload.matchAll(gramRe)) {
      const g = m[1];
      // Only include short, snake_case grams as tool candidates
      if (g.length < 40 && /^[a-z][a-z0-9_]*$/.test(g)) tools.add(g);
    }
  }
  return [...tools];
}

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

/**
 * Build a µPY model descriptor from a TOML spec + optional KXML document.
 *
 * The KXML document is the primary surface for CDATA inclusions:
 *   - <![CDATA[ Pop / Sek / Xul phase programs ]]>  → kuhul programs
 *   - <![CDATA[ shader hlsl ... ]]>                 → projection targets
 *   - <![CDATA[ tool.name gram.list ]]>             → semantic grams
 *   - <![CDATA[ policy: math_only ]]>               → policy gates
 *
 * All four kinds are extracted by SemanticReader and surfaced in the descriptor
 * under `cdata`. The µPY trainer receives them via the descriptor so policy is
 * enforced at training time and projection shaders are preloaded.
 */
export function buildMupyDescriptor(tomlText, kxmlText = '', sourceName = 'spec.toml') {
  const spec     = parseMupySpec(tomlText, sourceName);
  const result   = kxmlText
    ? SemanticReader.read(kxmlText, sourceName.replace('.toml', '.kxml'))
    : null;
  const topology = result?.topology ?? null;

  // CDATA surface — this is where µMODEL behavior programs live
  const cdata    = extractCdataByKind(topology);

  const folds    = topology?.folds    ?? [];
  const grams    = topology?.grams    ?? { semantic: [], coarse: [] };
  const policies = topology?.policies ?? [];
  const kuhulPrograms    = topology?.kuhul_programs    ?? [];
  const projectionTargets = topology?.projection_targets ?? [];

  // Capabilities: TOML list + anything that appeared in semantic_grams CDATA
  const allTools = mergeCdataTools(spec.capabilities, cdata.semantic_grams);

  const pressure = Math.min(1.0,
    0.2 +
    allTools.length               * 0.04 +
    folds.length                  * 0.08 +
    (topology?.cdata_capsules?.length ?? 0) * 0.06 +
    kuhulPrograms.length          * 0.10
  );

  return {
    ok:     true,
    type:   'mupy_descriptor.v1',
    domain: spec.domain,
    phase:  spec.phase,
    gravity: spec.gravity,

    capabilities: allTools,
    training:     spec.training,
    routing:      spec.routing,

    // CDATA inclusions — the µMODEL's embedded programs
    cdata: {
      // K'UHUL phase programs (Pop/Wo/Sek/Ch'en/Xul sequences)
      kuhul_programs:     kuhulPrograms,
      // HLSL/WGSL shader payloads for GPU projection
      projection_targets: projectionTargets,
      // Raw CDATA payloads by kind
      payloads: cdata,
      // Policy gates derived from CDATA + KXML policy nodes
      policies: policies.map(p => p.id),
      // Semantic grams extracted from CDATA (dotted tool/domain identifiers)
      semantic_grams: [...new Set([...grams.semantic, ...grams.coarse])],
    },

    topology_folds: folds.map(f => ({
      id: f.id, domain: f.domain, phase: f.phase, gravity: f.gravity,
    })),

    activation: {
      pressure,
      active: pressure >= 0.5,
      // CDATA presence boosts confidence: a µMODEL with phase programs
      // is more fully specified than one with only TOML fields
      cdata_boost: kuhulPrograms.length > 0 || cdata.projection.length > 0,
    },

    trainer_cmd: [
      'python', `tools/trainers/train_${spec.domain}_micronaut.py`,
      '--steps',   String(spec.training.steps  ?? 3000),
      '--batch',   String(spec.training.batch  ?? 4),
      '--lr',      String(spec.training.lr     ?? '2e-5'),
    ],

    invariants: {
      spec_is_source_of_truth:         true,
      weights_are_derived:             true,
      evolution_preserves_spec:        true,
      // CDATA invariants — same as SemanticReader
      cdata_preserved:                 true,
      policy_during_traversal:         true,
      cdata_policy_enforced_at_train:  true,
      projection_shaders_preloaded:    cdata.projection.length > 0,
    },
  };
}

// ─── µMODEL registry ─────────────────────────────────────────────────────────

const _registry = new Map();

export function registerMupy(descriptor)       { _registry.set(descriptor.domain, descriptor); }
export function getMupy(domain)                { return _registry.get(domain) ?? null; }
export function listMupy() {
  return [..._registry.values()].map(d => ({
    domain:       d.domain,
    phase:        d.phase,
    gravity:      d.gravity,
    capabilities: d.capabilities,
    active:       d.activation.active,
    pressure:     d.activation.pressure,
    cdata_boost:  d.activation.cdata_boost,
    kuhul_programs: d.cdata.kuhul_programs.length,
    policies:     d.cdata.policies,
  }));
}

// ─── Canonical µMODEL specs with CDATA inclusions ────────────────────────────
//
// These are the KXML documents that define the two built-in µMODELS.
// CDATA blocks embed K'UHUL phase programs and policy directly in the spec.

export const MUPY_MATH_KXML = `<?xml version="1.0" encoding="utf-8"?>
<kxml domain="math_tool" phase="Sek" gravity="Normal">

  <fold id="arithmetic"    domain="arithmetic"/>
  <fold id="calculus"      domain="calculus"/>
  <fold id="linalg"        domain="linear_algebra"/>
  <fold id="statistics"    domain="statistics"/>

  <geodesic from="arithmetic" to="calculus"   cost="0.3"/>
  <geodesic from="calculus"   to="linalg"     cost="0.4"/>
  <geodesic from="linalg"     to="statistics" cost="0.2"/>

  <lane id="compute" type="math_compute" permission="math_only"/>

  <policy id="math_only">
    <directive type="restrict" domain="code_execution" permission="deny"/>
    <directive type="allow"    domain="symbolic_math"  permission="grant"/>
  </policy>

  <![CDATA[
    Pop: load input expression, validate numeric domain
    Wo:  declare symbolic intent, bind MathML resolver
    Sek: execute — route to fibonacci_fold / pi_field / linalg_solver / mayan_fold
    Ch'en: accumulate gradient across math folds
    Xul: emit MathML output, verify Lipschitz soft-landing
  ]]>

  <![CDATA[
    math_tool.fibonacci_fold math_tool.pi_field math_tool.linalg_solver
    math_tool.mayan_fold math_tool.matmul_kernel math_tool.geodesic_router
    policy=math_only required
  ]]>

  <shader type="math_projection">
    <![CDATA[
      // field_optimizer.hlsl stub — math fold projection
      // Applies attraction_well gravity toward loss minimum
      // scroll_inertia preserves Adam momentum across math folds
      float gravity_scale = 1.0; // Normal
      float attraction = attraction_well(loss, gravity_scale);
    ]]>
  </shader>

</kxml>`;

export const MUPY_CODER_KXML = `<?xml version="1.0" encoding="utf-8"?>
<kxml domain="coder_tool" phase="Sek" gravity="Heavy">

  <fold id="syntax"    domain="syntax_analysis"/>
  <fold id="semantics" domain="code_semantics"/>
  <fold id="codegen"   domain="code_generation"/>
  <fold id="debug"     domain="debugging"/>

  <geodesic from="syntax" to="semantics" cost="0.2"/>
  <geodesic from="semantics" to="codegen" cost="0.3"/>
  <geodesic from="codegen"   to="debug"  cost="0.5"/>

  <lane id="code_compute" type="code_execution" permission="inherit"/>

  <![CDATA[
    Pop: load code prompt, detect language and intent
    Wo:  declare coder intent, bind shell_run / file_read / kuhul_agent
    Sek: execute — generate code via coder_tool specialist weights
    Ch'en: backprop through syntax + semantics folds
    Xul: emit code block, verify syntax validity
  ]]>

  <![CDATA[
    coder_tool.shell_run coder_tool.file_read coder_tool.file_write
    coder_tool.git_status coder_tool.kuhul_agent coder_tool.kxml_run
  ]]>

  <shader type="code_projection">
    <![CDATA[
      // Heavy gravity — coder stays close to training distribution
      float gravity_scale = 2.0; // Heavy
      float attraction = attraction_well(loss, gravity_scale);
      float inertia    = scroll_inertia(adam_m1, adam_m2);
    ]]>
  </shader>

</kxml>`;

// TOML side-by-side with KXML (TOML provides numeric training params,
// KXML provides CDATA programs + topology)
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

// Register all built-ins — TOML + KXML together so CDATA capsules are live
import { MUPY_ATOMIC_BRAIN_SPEC, MUPY_ATOMIC_BRAIN_KXML } from '../xcfe/atomic-brain.js';
import { BASE_MUMODELS } from './base-models.js';
import { MUPY_BRAIN_SPEC, MUPY_BRAIN_KXML } from './mu-brain.js';
import { MUPY_MAYAN_SPEC, MUPY_MAYAN_KXML } from './flux-tensor.js';

// Specialist trained models + µBRAIN cognitive architecture
for (const [spec, kxml] of [
  [MUPY_MATH_SPEC,         MUPY_MATH_KXML],
  [MUPY_CODER_SPEC,        MUPY_CODER_KXML],
  [MUPY_ATOMIC_BRAIN_SPEC, MUPY_ATOMIC_BRAIN_KXML],
  [MUPY_BRAIN_SPEC,        MUPY_BRAIN_KXML],
  [MUPY_MAYAN_SPEC,        MUPY_MAYAN_KXML],
]) {
  registerMupy(buildMupyDescriptor(spec, kxml));
}

// Base µMODELS — every K'UHUL runtime ships with these
for (const { spec, kxml } of BASE_MUMODELS) {
  registerMupy(buildMupyDescriptor(spec, kxml));
}

export { parseMupySpec as parse, buildMupyDescriptor as build,
         registerMupy as register, getMupy as get, listMupy as list };
