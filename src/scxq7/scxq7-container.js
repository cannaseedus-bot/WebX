// scxq7-container.js — SCXQ7 Ω Sovereign Computational Object
//
// SCXQ7 is the container for the entire WebX-3D system.
// Everything built in this codebase lives inside the SCXQ7 Ω SCO/1 object.
//
// MAGIC: 0x53 0x43 0x58 0x51 0x37 0x00 0xAA 0x55  ("SCXQ7\0ªU")
//
// SCO/1 structure:
//   header          → package.json + CHANGELOG.md
//   kernel          → this file + 256-byte immutable law
//   compressed_state → safetensors checkpoints (math µMODEL etc)
//   μ-op_methods    → K'UHUL phases + SCXQ2 symbol dispatch
//   projection_config → Win2D canvas + optical sphere shader
//   persistence_rules → ARC weights (append-only, hash-chained trace)
//   trace           → geodesic_cache/arc_bias_*.npy (execution facts)
//
// SCXQ7 → WebX-3D implementation mapping:
//
//   KERNEL LAW (immutable)
//     "The file contains all authority"
//     → SCXQ7 magic + 256-byte law defines the object boundary
//     → No host (browser, OS, Claude) may extend execution
//
//   SCO/1 (authority container)
//     header:           WebX-3D v3.5.0, package.json
//     kernel:           scxq7-container.js (this file)
//     compressed_state: gpt2_medium_ft_toolcall.safetensors (math µMODEL)
//                       + ARC bias sparse COO (268K pairs)
//     μ-op_methods:     K'UHUL Pop→Wo→Sek→Ch'en→Xul dispatch
//                       SCXQ2 symbols: ? : & | ! @π @λ @∇ @∑ @@ <- ->
//     projection_config: Win2D CanvasSvgDocument + vector_green_screen.hlsl
//                        + optical_sphere.hlsl
//     persistence_rules: geodesic cache E:\models\GPT2\geodesic_cache\
//                        arc_bias sparse, append-only, monotonic ticks
//     trace:            ARC weight accumulation (hash-chained per step)
//
//   SCXQ2-IA/1 (inference algebra = µ-op execution over compressed state)
//     @π  → geodesic attention in math domain (arccos(q·k^T))
//     @λ  → parallel transport (GeodesicWeight.parallelTransportTo())
//     @∇  → gradient + gravity (KuhulPhysicsSolver, epoch loop)
//     @∑  → SH wave aggregation (OpticalProcessor.propagate())
//     @@  → K'UHUL phase gate (Pop/Wo/Sek/Ch'en/Xul)
//     <-  → ARC weight write (record_batch → _pairs dict)
//     ->  → SVG3D compute dispatch (SVG3DComputeGraph.dispatch())
//
//   SVG-3D TENSOR SUBSTRATE (geometry is computation)
//     node  = OpticalNode (pos + sh[9] + neighbors)
//     edge  = geodesic arc (ARC trajectory between clusters)
//     group = Fold type (COMPUTE/TOOL/THINK/AGENT/META)
//     weight = SH coefficient (cos/sin phase pair per band)
//
//   TRACE/1 (execution facts, not explanations)
//     monotonic: tick counter per training step
//     append-only: arc_bias dict only grows
//     hash-chained: (future: BLAKE3 footer per ARC batch)
//     delta-only: sparse COO saves only changed pairs
//     replay-sufficient: ARC weights reconstruct attention bias
//
//   LEARNING (bounded)
//     substrate: optical mesh as SVG-3D tensor
//     update: @∇ → AdamW + KuhulPhysicsSolver gravity
//     objective: @∑ → mean geodesic loss (arccos attention)
//     constraints:
//       - no new μ-ops: SCXQ2 symbol set is fixed
//       - no kernel mutation: MAGIC + 256-byte law immutable
//       - append-only trace: ARC weights never shrink
//       - state delta only: sparse COO saves only nonzero pairs
//
//   π-KUHUL RAG (SCXQ2-IA/1 over compressed token states)
//     Corpus    = Token states on 100D Mayan vigesimal manifold
//     Retrieval = Geodesic phase interference (@π domain)
//     Augment   = Shard transitions (@λ domain)
//     Generate  = SH wave propagation (@∑ domain)
//     Calendar  = Haab months = K'UHUL phases (Pop→Ch'en→Xul)
//
// SEMANTIC MICROCODE PIPELINE:
//   FETCH_SYMBOL    → K'UHUL phase dispatch (Pop gate check)
//   DECODE_μOP      → route to @π/@λ/@∇/@∑ domain handler
//   EXECUTE_SEM_UNIT → geodesic_arc_attention / propagate / Adam step
//   WRITE_DELTA     → ARC weight dict update (sparse, <-)
//   APPEND_TRACE    → tick++, save sparse COO to E:\geodesic_cache
//
// "Intelligence fits in a file." — SCXQ7 Ω final collapse

export const SCXQ7_MAGIC = new Uint8Array([0x53,0x43,0x58,0x51,0x37,0x00,0xAA,0x55]);
export const SCXQ7_VERSION = 'Ω';
export const KERNEL_LAW = Object.freeze({
  version:     'SCXQ7_KERNEL_LAW.v1',
  sections: {
    I:   'The file contains all authority. No host may extend execution. Execution is internal and complete.',
    II:  'SCXQ2 symbols are μ-ops. Symbol dictionary declared in header. No hidden execution paths.',
    III: { π:'math', λ:'transform', '∇':'optimization', '∑':'aggregation', m:'matrix' },
    IV:  'Data contains its methods. Configuration is executable. Projection ≠ authority.',
  },
  invariants: [
    'Symbolic Closure',
    'Domain Legality',
    'Compression-Native Execution',
    'Deterministic Replay',
    'No Epistemic Leakage',
  ],
});

// SCXQ2 μ-op symbol table
export const SCXQ2_OPS = Object.freeze({
  '?':   { domain: 'control',   desc: 'branch if condition true' },
  ':':   { domain: 'control',   desc: 'else branch' },
  '&':   { domain: 'logic',     desc: 'logical AND — geodesic intersection' },
  '|':   { domain: 'logic',     desc: 'logical OR — geodesic union' },
  '!':   { domain: 'logic',     desc: 'logical NOT — phase inversion' },
  '@π':  { domain: 'math',      desc: 'geodesic attention, arccos distance, SH evaluation' },
  '@λ':  { domain: 'transform', desc: 'parallel transport, fold pressure mapping' },
  '@∇':  { domain: 'gradient',  desc: 'Adam + KuhulPhysicsSolver gravity constraint' },
  '@∑':  { domain: 'aggregate', desc: 'SH wave aggregation, optical propagation' },
  '@@':  { domain: 'authority', desc: 'K\'UHUL phase gate (Pop/Wo/Sek/Ch\'en/Xul)' },
  '<-':  { domain: 'state',     desc: 'ARC weight write (sparse dict update)' },
  '->':  { domain: 'dispatch',  desc: 'SVG3D compute dispatch, shard transition' },
});

// SCO/1 manifest — maps SCXQ7 layers to WebX-3D files
export const SCO1_MANIFEST = Object.freeze({
  kernel:           'src/scxq7/scxq7-container.js',
  compressed_state: 'E:/models/GPT2/math_micronaut/gpt2_medium_ft_toolcall.safetensors',
  mu_op_methods:    ['src/xcfe/geo-weights.js', 'src/xcfe/optical-mesh.js',
                     'src/xcfe/svg3d-compute.js', 'src/mupy/pi-kuhul/pi-kuhul-engine.js'],
  projection_config:['native/shaders/optical/optical_sphere.hlsl',
                     'native/shaders/optical/vector_green_screen.hlsl',
                     'native/win2d/Win2DGreenScreenCanvas.h'],
  persistence_rules:'E:/models/GPT2/geodesic_cache/',
  trace:            'tools/trainers/geodesic_attention_bridge.py → arc_bias_*.npy',
  svg3d_substrate:  'src/xcfe/optical-mesh.js → ComputeOpticalMesh()',
  learning:         'tools/trainers/finetune_toolcall_pt.py --geodesic',
  rag_engine:       'src/mupy/pi-kuhul/pi-kuhul-engine.js → PiKuhulFieldEngine',
  blueprints:       'src/mupy/blueprints/*.kuhul',
});

// Verify SCXQ7 magic bytes
export function verifyMagic(bytes) {
  return SCXQ7_MAGIC.every((b, i) => bytes[i] === b);
}

// Execute a SCXQ2 μ-op (dispatch to domain handler)
export function executeOp(op, context = {}) {
  const def = SCXQ2_OPS[op];
  if (!def) throw new Error(`Unknown μ-op: ${op}`);
  // Domain boundary enforcement (Invariant 2: Domain Legality)
  if (context.currentDomain && context.currentDomain !== def.domain &&
      def.domain !== 'control' && def.domain !== 'authority') {
    throw new Error(`Domain violation: op ${op} requires domain ${def.domain}, ` +
                    `currently in ${context.currentDomain}. ` +
                    `Explicit bridge required.`);
  }
  return { op, domain: def.domain, tick: (context.tick ?? 0) + 1 };
}
