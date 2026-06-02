// atomic-brain.js — Atomic Blocks + Brain Hypergraph runtime
//
// Implements the KuhulAtomicBrain ACU (Atomic Cognitive Unit) model from
// examples/atomic-blocks/atomic-brain-hypergraph.html.
//
// CDATA inclusions this module produces (for µMODEL consumption):
//   kind=kuhul       Pop/Wo/Sek/Ch'en phase program per brain node
//   kind=projection  96-glyph dispatch table mapped to HLSL fold categories
//   kind=semantic_grams  dotted glyph.opcode.fold n-grams
//   kind=policy      lane permission table (agent/skills/experts/runtime/router)
//
// Pi token base (from glyph-spec pi_token_base):
//   Digits of pi drive geodesic phase rotation and matrix series coefficients.
//   theta_n = theta_{n-1} + (2PI/10)*d_n  mod 2PI
//   S = sum (d_n/10^n) * RotationMatrix(theta_n)
//   This is the SMGM-16 pi_mod scalar expressed at system scale.

import GLYPH_SPEC from '../../model/atomic-brain.glyph-spec.json' assert { type: 'json' };

// ─── Fibonacci ────────────────────────────────────────────────────────────────
// Opcode sugar: 0x34 TENSOR_INT ⨻ (intersection), 0x36 TRI_SUM ⨹ (triangular)

/** First n Fibonacci numbers. F(0)=0, F(1)=1. */
export function fibonacci(n) {
  const seq = [0, 1];
  for (let i = 2; i < n; i++) seq.push(seq[i-1] + seq[i-2]);
  return seq.slice(0, n);
}

/** Zeckendorf decomposition: every positive integer as sum of non-consecutive Fibonaccis. */
export function zeckendorf(n) {
  const fibs = fibonacci(87).filter(f => f > 0 && f <= n);
  const terms = [];
  let rem = n;
  for (let i = fibs.length - 1; i >= 0 && rem > 0; i--) {
    if (fibs[i] <= rem) { terms.push(fibs[i]); rem -= fibs[i]; }
  }
  return terms;
}

/** Fibonacci windowing fold: compress array by averaging over Fibonacci-sized windows.
 *  Opcode 0x34 TENSOR_INT ⨻ — tensor intersection via golden-ratio windowing. */
export function fibonacciFold(arr) {
  const fibs = fibonacci(20).filter(f => f >= 2 && f <= arr.length);
  const windows = fibs.map(w => {
    const out = [];
    for (let i = 0; i + w <= arr.length; i += w)
      out.push(arr.slice(i, i + w).reduce((a, b) => a + b, 0) / w);
    return out;
  });
  return { windows, ratio: (1 + Math.sqrt(5)) / 2 };
}

/** Golden ratio. */
export const PHI = (1 + Math.sqrt(5)) / 2;

// ─── Mayan base-20 (vigesimal) ────────────────────────────────────────────────
// Opcode sugar: 0x67 TRIPLE_GEO ⫵ (barycentric triple geodesic)

/** Encode integer to Mayan vigesimal digit array (least-significant first). */
export function mayanEncode(n) {
  if (n === 0) return [0];
  const digits = [];
  let rem = n;
  while (rem > 0) { digits.push(rem % 20); rem = Math.floor(rem / 20); }
  return digits;
}

/** Decode Mayan vigesimal digit array back to integer. */
export function mayanDecode(digits) {
  return digits.reduce((acc, d, i) => acc + d * Math.pow(20, i), 0);
}

/** Mayan Long Count: 5-position (kin, uinal, tun, katun, baktun). */
export function longCount(totalDays) {
  return {
    kin:    totalDays % 20,
    uinal:  Math.floor(totalDays / 20)    % 18,
    tun:    Math.floor(totalDays / 360)   % 20,
    katun:  Math.floor(totalDays / 7200)  % 20,
    baktun: Math.floor(totalDays / 144000),
  };
}

// ─── Pi token base ────────────────────────────────────────────────────────────

const PI_DIGITS = [1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6]; // first 20 decimal places

/** Geodesic phase sequence driven by pi digits. */
export function piGeodesicSequence(steps = PI_DIGITS.length, theta0 = 0) {
  const seq = [];
  let theta = theta0;
  for (let n = 0; n < steps; n++) {
    theta = (theta + (2 * Math.PI / 10) * PI_DIGITS[n % PI_DIGITS.length]) % (2 * Math.PI);
    seq.push({ step: n, digit: PI_DIGITS[n % PI_DIGITS.length], theta, deg: theta * 180 / Math.PI });
  }
  return seq;
}

/** Pi-driven 2x2 rotation matrix series: S = sum (d_n/10^n) * R(theta_n). */
export function piMatrixSeries(terms = 10) {
  const seq = piGeodesicSequence(terms);
  let s = [[0, 0], [0, 0]];
  for (let n = 0; n < terms; n++) {
    const scale = PI_DIGITS[n] / Math.pow(10, n + 1);
    const c = Math.cos(seq[n].theta), si = Math.sin(seq[n].theta);
    s[0][0] += scale * c;   s[0][1] += scale * -si;
    s[1][0] += scale * si;  s[1][1] += scale * c;
  }
  return s;
}

// ─── Deep Thinking Index ──────────────────────────────────────────────────────

/** D = N * G * T * sum(kappa_i).  G=4 (SIMD-128), T=8 (8D tensor). */
export function deepThinkingIndex(nBrains = 8, kappas = null) {
  const G = 4, T = 8;
  const ksum = kappas ? kappas.reduce((a, b) => a + b, 0) : nBrains * 1.0;
  return nBrains * G * T * ksum;
}

// ─── Opcode / glyph lookup ────────────────────────────────────────────────────

const _opcodes = GLYPH_SPEC.kuhl.runtime.opcodes;
const _layout  = GLYPH_SPEC.kuhl.glyphs.layout;

/** Resolve a glyph character or opcode string to its spec entry. */
export function resolveGlyph(glyphOrOpcode) {
  for (const [op, entry] of Object.entries(_opcodes)) {
    if (op === glyphOrOpcode || entry.glyph === glyphOrOpcode || entry.mnemonic === glyphOrOpcode)
      return { opcode: op, ...entry };
  }
  return _layout[glyphOrOpcode] ? { opcode: glyphOrOpcode, ..._layout[glyphOrOpcode] } : null;
}

/** All opcodes for a given fold category. */
export function opcodesForFold(fold) {
  return Object.entries(_opcodes)
    .filter(([, e]) => e.fold === fold)
    .map(([op, e]) => ({ opcode: op, ...e }));
}

// ─── KuhulAtomicBrain ─────────────────────────────────────────────────────────
//
// JS port of the KuhulAtomicBrain class from atomic-brain-hypergraph.html.
// Implements the ACU law:
//   ATOMIC_BLOCK = symbol + state + operator + routing_node + learning_rule

export class KuhulAtomicBrain {
  constructor(nBrains = 8) {
    this.routing_bias   = { css_shader: 0, html_shader: 0, glyph_dispatch: 0, scx_token: 0 };
    this.interaction_count = 0;
    this.phase_idx      = 0;
    this.phases         = ['Pop', 'Wo', 'Sek', "Ch'en"];
    this.brainChain     = GLYPH_SPEC.kuhl.brain_fleet.nodes.slice(0, nBrains).map(b => b.name);
    this.currentBrain   = 0;
    this.curvatureLog   = [];
  }

  get phase() { return this.phases[this.phase_idx]; }

  /** Reinforce selected expert, decay others (A·x update rule). */
  reinforce(expert, signal = 0.08, decay = 0.02) {
    for (const k in this.routing_bias) {
      this.routing_bias[k] = k === expert
        ? Math.min(1, this.routing_bias[k] + signal)
        : Math.max(0, this.routing_bias[k] - decay);
    }
    this.interaction_count++;
  }

  /** Advance one K'UHUL phase. */
  tick() {
    this.phase_idx = (this.phase_idx + 1) % this.phases.length;
    return this.phase;
  }

  /** Run 8-brain deep chain. Returns DTI and curvature log. */
  runDeepChain() {
    this.curvatureLog = [];
    for (let i = 0; i < this.brainChain.length; i++) {
      const kappa = 0.8 + Math.random() * 0.4;
      this.curvatureLog.push({ brain: this.brainChain[i], kappa });
    }
    return {
      dti:        deepThinkingIndex(this.brainChain.length, this.curvatureLog.map(c => c.kappa)),
      curvatures: this.curvatureLog,
      pi_matrix:  piMatrixSeries(8),
    };
  }

  /** Emit a CDATA descriptor for this brain's current state (for µMODEL consumption). */
  toCdata() {
    const kuhulProgram = this.phases.map((p, i) => {
      const descs = [
        'x <- BSON.load(snapshot)',
        'x_skills <- intent_mask(x, input)',
        'x_next <- A*x ; top_k(experts, k=2)',
        'A[i,j] += dA ; BSON.store(biases)',
      ];
      return `${p}: ${descs[i]}`;
    }).join('\n');

    const semanticGrams = [
      'atomic_brain.routing_bias', 'atomic_brain.deep_chain',
      'atomic_brain.phase_cycle', 'atomic_brain.pi_token',
      'glyph.BLOCK_OPEN', 'glyph.TENSOR_CORE', 'glyph.GEODESIC',
    ];

    const projectionStub = [
      '// field_optimizer.hlsl — atomic brain projection',
      `// DTI = ${deepThinkingIndex(this.brainChain.length)} | ${this.brainChain.length} brains`,
      'float attraction = attraction_well(loss, gravity_scale);',
      'float inertia    = scroll_inertia(adam_m1, adam_m2);',
    ].join('\n');

    const policy = 'policy=lane_permission required\n' +
      'agent=allow skills=allow experts=allow runtime=allow router=allow';

    return {
      kuhul:          kuhulProgram,
      semantic_grams: semanticGrams.join(' '),
      projection:     projectionStub,
      policy,
    };
  }
}

// ─── µMODEL KXML spec for the atomic brain ───────────────────────────────────
//
// This KXML document carries four CDATA capsules — one per kind.
// µPY's buildMupyDescriptor() will read it and populate descriptor.cdata.

export const MUPY_ATOMIC_BRAIN_KXML = `<?xml version="1.0" encoding="utf-8"?>
<kxml domain="atomic_brain" phase="Sek" gravity="Normal">

  <fold id="control_flow"  domain="COMPUTE_FOLD"/>
  <fold id="tensor_core"   domain="TENSOR_CORE"/>
  <fold id="tensor_net"    domain="TENSOR_NETWORK"/>
  <fold id="phase_quantum" domain="QUANTUM_GATE"/>
  <fold id="geodesic_wave" domain="OPTICAL_FLOW"/>

  <geodesic from="control_flow"  to="tensor_core"   cost="0.3"/>
  <geodesic from="tensor_core"   to="tensor_net"    cost="0.2"/>
  <geodesic from="tensor_net"    to="phase_quantum" cost="0.4"/>
  <geodesic from="phase_quantum" to="geodesic_wave" cost="0.3"/>

  <lane id="agent"   type="agent"   permission="allow"/>
  <lane id="skills"  type="skills"  permission="allow"/>
  <lane id="experts" type="experts" permission="allow"/>
  <lane id="runtime" type="runtime" permission="allow"/>
  <lane id="router"  type="router"  permission="allow"/>

  <policy id="lane_permission">
    <directive type="allow" domain="all_lanes" permission="grant"/>
  </policy>

  <![CDATA[
    Pop:   x <- BSON.load(snapshot) — load BSON state for all 8 brain nodes
    Wo:    x_skills <- intent_mask(x, input) — bind routing intent
    Sek:   x_next <- A*x ; top_k(experts, k=2) — execute 8D tensor chain B1->B8
    Ch'en: A[i,j] += dA ; BSON.store(biases) — update routing_bias, store
    Pi token base: theta_n = theta_{n-1} + (2PI/10)*d_n mod 2PI
    Deep Thinking Index: D = N * G * T * sum(kappa_i) = 2048 for 8 brains
  ]]>

  <![CDATA[
    atomic_brain.routing_bias atomic_brain.deep_chain atomic_brain.phase_cycle
    atomic_brain.pi_token glyph.BLOCK_OPEN glyph.TENSOR_CORE glyph.GEODESIC
    glyph.layout.row glyph.layout.col glyph.layout.3col glyph.layout.surround
    glyph.control_flow glyph.tensor_network glyph.phase_quantum glyph.geodesic_wave
    policy=lane_permission required
  ]]>

  <shader type="atomic_brain_projection">
    <![CDATA[
      // field_optimizer.hlsl — atomic brain tensor projection
      // 8D tensor -> 3D PCA projection via DirectXMath SIMD 128-bit
      // DTI = 2048 (N=8 brains, G=4 SIMD, T=8 tensor rank, kappa~1.0)
      float gravity_scale = 1.0; // Normal
      float attraction    = attraction_well(loss, gravity_scale);
      float inertia       = scroll_inertia(adam_m1, adam_m2);
      // Pi-driven rotation matrix S applied to expert routing weights
      float2x2 S = pi_matrix_series(8); // sum (d_n/10^n) * R(theta_n)
    ]]>
  </shader>

</kxml>`;

export const MUPY_ATOMIC_BRAIN_SPEC = `
[model]
domain   = "atomic_brain"
phase    = "Sek"
gravity  = "Normal"

[training]
steps  = 4000
batch  = 4
lr     = 1e-5
block  = 256

[routing]
trigger  = "glyph,opcode,layout,tensor,geodesic,atomic,brain,fold,phase"
fallback = "base_gpt2"

[capabilities]
tools = "kxml_run,gpu_dispatch,fibonacci_fold,pi_field,kuhul_agent,micronaut_dispatch"
`;

// ─── Linear algebra helpers ───────────────────────────────────────────────────
// Opcode sugar: 0x42 TENSOR_CORE ⨀, 0x44 TENSOR_PROD ⨂, 0x40 DOT ⨰

/** Dot product of two equal-length arrays. Opcode ⨰ DOT 0x40. */
export function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

/** Matrix multiply: A(m×k) · B(k×n) → C(m×n). Opcode ⨀ TENSOR_CORE 0x42. */
export function matmul(A, B) {
  const m = A.length, k = B.length, n = B[0].length;
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      dot(A[i], B.map(row => row[j]))
    )
  );
}

/** Outer product of two vectors. Opcode ⨂ TENSOR_PROD 0x44. */
export function outer(a, b) {
  return a.map(ai => b.map(bi => ai * bi));
}

/** L2 norm of a vector. */
export function norm(v) { return Math.sqrt(dot(v, v)); }

/** Normalize vector to unit length. */
export function normalize(v) { const n = norm(v); return v.map(x => x / n); }

/** Softmax. Opcode ⨁ TENSOR_SUM 0x43. */
export function softmax(v) {
  const max = Math.max(...v);
  const exps = v.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

/** Cross product of two 3D vectors. Opcode ⨯ CROSS 0x3F. */
export function cross3(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}

/** Geodesic distance on unit sphere (great-circle). Opcode ⫵ TRIPLE_GEO 0x67. */
export function geodesicDist(a, b) {
  return Math.acos(Math.min(1, Math.max(-1, dot(normalize(a), normalize(b)))));
}

// ─── FLUX IR module for atomic brain (complete, runnable) ─────────────────────
// Demonstrates the full Math→FLUX IR→Runtime pipeline.
// Import and pass to FluxIRInterpreter.compile() to get a live runtime.

import { moduleDef, storeDef, actionDef, reduceDef,
         queryDef, effectCreatorDef, pureFn } from '../runtime/flux-ir.js';

export const ATOMIC_BRAIN_IR = moduleDef('atomic_brain', [
  // Pure math functions (Layer 0 — timeless)
  pureFn('fibonacci',      ['n'],    'fibonacci(n)'),
  pureFn('fibonacciFold',  ['arr'],  'fibonacciFold(arr)'),
  pureFn('deepThinkingIndex', ['n'], 'deepThinkingIndex(n)'),
  pureFn('piGeodesicStep', ['t','d'],'(t + (2*Math.PI/10)*d) % (2*Math.PI)'),
  pureFn('softmax',        ['v'],    'softmax(v)'),
  pureFn('mayanEncode',    ['n'],    'mayanEncode(n)'),
  pureFn('dot',            ['a','b'],'dot(a,b)'),
  pureFn('norm',           ['v'],    'norm(v)'),

  // Store (Layer 1 — state across time)
  storeDef('AtomicBrain', [
    { name: 'phase',          defaultValue: 'Pop' },
    { name: 'routing_bias',   defaultValue: { css_shader:0, html_shader:0, glyph_dispatch:0, scx_token:0 } },
    { name: 'interaction',    defaultValue: 0 },
    { name: 'dti',            defaultValue: 2048 },
    { name: 'pi_theta',       defaultValue: 0 },
    { name: 'current_brain',  defaultValue: 0 },
    { name: 'fibonacci_fold', defaultValue: null },
    { name: 'mayan_ticks',    defaultValue: 0 },
  ]),

  // Actions
  actionDef('REINFORCE',       [{ name: 'expert' }, { name: 'signal' }]),
  actionDef('TICK_PHASE'),
  actionDef('RUN_DEEP_CHAIN'),
  actionDef('PI_STEP',         [{ name: 'digit' }]),
  actionDef('FIBONACCI_FOLD',  [{ name: 'arr' }]),
  actionDef('MAYAN_TICK',      [{ name: 'days' }]),
  actionDef('CHAIN_RESULT',    [{ name: 'dti' }, { name: 'curvatures' }]),

  // Reducer (pure math function over time — Layer 1)
  reduceDef('AtomicBrain', {
    REINFORCE: (state, { expert, signal = 0.08 }) => {
      const bias = { ...state.routing_bias };
      for (const k in bias) bias[k] = k === expert
        ? Math.min(1, bias[k] + signal)
        : Math.max(0, bias[k] - 0.02);
      return { ...state, routing_bias: bias, interaction: state.interaction + 1 };
    },
    TICK_PHASE: state => {
      const phases = ['Pop', 'Wo', 'Sek', "Ch'en"];
      const idx = (phases.indexOf(state.phase) + 1) % phases.length;
      return { ...state, phase: phases[idx] };
    },
    PI_STEP: (state, { digit }) => ({
      ...state,
      pi_theta: (state.pi_theta + (2 * Math.PI / 10) * digit) % (2 * Math.PI),
    }),
    FIBONACCI_FOLD: (state, { arr }) => ({
      ...state,
      fibonacci_fold: fibonacciFold(arr),
    }),
    MAYAN_TICK: (state, { days = 1 }) => ({
      ...state,
      mayan_ticks: state.mayan_ticks + days,
    }),
    CHAIN_RESULT: (state, { dti, curvatures }) => ({
      ...state,
      dti,
      curvatures,
      current_brain: 0,
    }),
  }),

  // Queries
  queryDef('currentPhase',  'AtomicBrain', s => s.phase),
  queryDef('topExpert',     'AtomicBrain', s =>
    Object.entries(s.routing_bias).sort((a,b) => b[1]-a[1])[0]),
  queryDef('piPosition',    'AtomicBrain', s => ({ theta: s.pi_theta, deg: s.pi_theta * 180/Math.PI })),
  queryDef('mayanLongCount','AtomicBrain', s => longCount(s.mayan_ticks)),
  queryDef('fibFold',       'AtomicBrain', s => s.fibonacci_fold),

  // Effect creator: run deep chain async
  effectCreatorDef('runDeepChain', ['nBrains'], 'timer', (nBrains = 8) => ({
    type:       'timer',
    delayMs:    nBrains * 50,
    onComplete: () => {
      const kappas = Array.from({ length: nBrains }, () => 0.8 + Math.random() * 0.4);
      return { type: 'CHAIN_RESULT', payload: {
        dti: deepThinkingIndex(nBrains, kappas), curvatures: kappas,
      }};
    },
  })),
]);

export { GLYPH_SPEC, PI_DIGITS,
         fibonacci, fibonacciFold, zeckendorf, PHI,
         mayanEncode, mayanDecode, longCount,
         dot, matmul, outer, norm, normalize, softmax, cross3, geodesicDist };
