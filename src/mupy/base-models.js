// base-models.js — Canonical base µMODEL registry
//
// Every K'UHUL runtime ships with these µMODELS pre-registered.
// Each is a Driver/Kernel: TOML spec (training config) + KXML (CDATA programs).
//
// Base set covers all major capability domains:
//
//   Numeric micronauts (KXML governance):
//     fibonacci_fold   golden-ratio windowing + Zeckendorf encoding
//     pi_field         transcendental constants + geodesic phase rotation
//     mayan_fold       base-20 vigesimal + Long Count calendar
//     linalg_solver    matmul / dot / norm / eigenvalue / OLS
//     geodesic_router  hyperbolic routing + great-circle distance
//
//   Glyph opcode µMODELS (96-glyph spec):
//     tensor_ops       TENSOR_CORE/PROD/SUM/INT (0x42-0x51)
//     geodesic_ops     GEODESIC/OPTICAL wave (0x62-0x71)
//     logic_ops        LOGICAL AND/OR/XOR/NAND/NOR (0x5C-0x61)
//     phase_ops        PHASE intersection/union/cap (0x57-0x5B)
//
//   Runtime µMODELS:
//     flux_runtime     FLUX state machine — pure reducer host
//     semantic_reader  KXML/JSONL document topology extraction
//
//   Specialist µMODELS (trained GPT-2 specialists):
//     math_tool        arithmetic / calculus / statistics (training: step ~300/3000)
//     coder_tool       code generation / shell / file I/O (training: GPU D3D11)
//     atomic_brain     8D tensor mesh / glyph ops / brain hypergraph

// ─── helpers ──────────────────────────────────────────────────────────────────

const kxml = (domain, phase, gravity, folds, geodesics, lanes, policies, cdata) => `<?xml version="1.0" encoding="utf-8"?>
<kxml domain="${domain}" phase="${phase}" gravity="${gravity}">
${folds.map(f => `  <fold id="${f.id}" domain="${f.domain}"/>`).join('\n')}
${geodesics.map(g => `  <geodesic from="${g.from}" to="${g.to}" cost="${g.cost}"/>`).join('\n')}
${lanes.map(l => `  <lane id="${l.id}" type="${l.type}" permission="${l.perm}"/>`).join('\n')}
${policies.map(p => `  <policy id="${p.id}"><directive type="${p.type}" domain="${p.domain}" permission="${p.perm}"/></policy>`).join('\n')}
${cdata.map(c => `  <![CDATA[\n${c.split('\n').map(l => '    '+l).join('\n')}\n  ]]>`).join('\n')}
</kxml>`;

const toml = (domain, phase, gravity, steps, batch, lr, trigger, tools) =>
`[model]
domain   = "${domain}"
phase    = "${phase}"
gravity  = "${gravity}"

[training]
steps  = ${steps}
batch  = ${batch}
lr     = ${lr}
block  = 256

[routing]
trigger  = "${trigger}"
fallback = "base_gpt2"

[capabilities]
tools = "${tools}"
`;

// ─── Numeric micronauts ───────────────────────────────────────────────────────

export const FIBONACCI_FOLD_KXML = kxml(
  'fibonacci_fold', 'Sek', 'Normal',
  [{ id:'phi_compress', domain:'fibonacci' }, { id:'zeckendorf', domain:'encoding' }],
  [{ from:'phi_compress', to:'zeckendorf', cost:'0.4' }],
  [{ id:'compute', type:'math_compute', perm:'math_only' }],
  [{ id:'math_only', type:'allow', domain:'symbolic_math', perm:'grant' }],
  [
    `Pop:   load tensor array, validate numeric domain\nWo:    bind fibonacci_fold opcode 0x34 TENSOR_INT\nSek:   compress via golden-ratio windows  phi=${((1+Math.sqrt(5))/2).toFixed(6)}\nCh'en: emit compressed windows + zeckendorf decomposition`,
    `fibonacci_fold.compress fibonacci_fold.zeckendorf fibonacci_fold.phi\nfibonacci_fold.window fibonacci_fold.golden_ratio\nglyph.TENSOR_INT glyph.TRI_SUM policy=math_only required`,
    `// field_optimizer.hlsl stub — fibonacci fold\nfloat phi = 1.618034;\nfloat attraction = attraction_well(loss, phi);`,
  ]
);
export const FIBONACCI_FOLD_SPEC = toml(
  'fibonacci_fold','Sek','Normal', 2000, 4, '2e-5',
  'fibonacci,compress,golden,ratio,fold,window,zeckendorf',
  'fibonacci_fold,zeckendorf_encode,math_tool,pi_field'
);

export const PI_FIELD_KXML = kxml(
  'pi_field', 'Wo', 'Float',
  [{ id:'transcendental', domain:'pi_digits' }, { id:'geodesic_phase', domain:'rotation' }],
  [{ from:'transcendental', to:'geodesic_phase', cost:'0.3' }],
  [{ id:'compute', type:'math_compute', perm:'math_only' }],
  [{ id:'math_only', type:'allow', domain:'transcendental_math', perm:'grant' }],
  [
    `Pop:   load digit index + theta_0, validate base\nWo:    declare pi token intent, bind digit sequence\nSek:   theta_n = theta_{n-1} + (2PI/10)*d_n  mod 2PI\nCh'en: emit rotation matrix S = sum(d_n/10^n * R(theta_n))`,
    `pi_field.geodesic pi_field.phase_rotation pi_field.matrix_series\npi_field.token_base pi_field.leibniz glyph.PARALLEL_GEO glyph.OPTIC_AXIS\npolicy=math_only required`,
    `// pi field shader\nfloat theta = pi_geodesic_step(state.theta, digit);\nfloat2x2 S  = pi_matrix_series(8);`,
  ]
);
export const PI_FIELD_SPEC = toml(
  'pi_field','Wo','Float', 2000, 4, '2e-5',
  'pi,geodesic,rotation,phase,transcendental,constant,series',
  'pi_field,fibonacci_fold,linalg_solver,geodesic_router'
);

export const MAYAN_FOLD_KXML = kxml(
  'mayan_fold', 'Sek', 'Normal',
  [{ id:'vigesimal', domain:'base20' }, { id:'long_count', domain:'calendar' }],
  [{ from:'vigesimal', to:'long_count', cost:'0.2' }],
  [{ id:'compute', type:'math_compute', perm:'math_only' }],
  [{ id:'math_only', type:'allow', domain:'calendar_math', perm:'grant' }],
  [
    `Pop:   load integer or day count, validate positive\nWo:    declare mayan intent, bind vigesimal encoder\nSek:   encode base-20: digits = [n%20, n//20 %20, ...]\n       long count: kin/uinal/tun/katun/baktun\nCh'en: emit encoded form + long count struct`,
    `mayan_fold.vigesimal mayan_fold.long_count mayan_fold.base20\nmayan_fold.kin mayan_fold.uinal mayan_fold.tun mayan_fold.katun\nglyph.TRIPLE_GEO glyph.TETRA_GEO policy=math_only required`,
    `// mayan fold shader\nfloat score = vigesimal_score(n, base20_weights);\nfloat route = geodesic_router(score, topology);`,
  ]
);
export const MAYAN_FOLD_SPEC = toml(
  'mayan_fold','Sek','Normal', 2000, 4, '2e-5',
  'mayan,vigesimal,base20,calendar,long_count,kin,baktun',
  'mayan_fold,fibonacci_fold,pi_field,linalg_solver'
);

export const LINALG_SOLVER_KXML = kxml(
  'linalg_solver', 'Sek', 'Heavy',
  [
    { id:'matmul',    domain:'matrix_multiply' },
    { id:'decomp',    domain:'lu_svd_eigen' },
    { id:'regression',domain:'ols_ridge' },
  ],
  [
    { from:'matmul', to:'decomp',     cost:'0.3' },
    { from:'decomp', to:'regression', cost:'0.4' },
  ],
  [{ id:'compute', type:'math_compute', perm:'math_only' }],
  [{ id:'math_only', type:'allow', domain:'linear_algebra', perm:'grant' }],
  [
    `Pop:   load matrix/vector inputs, validate shapes\nWo:    declare linalg intent — matmul/decomp/solve\nSek:   execute: matmul A*B, LU decomp, eigenvalue, OLS\nCh'en: emit result tensor + condition number`,
    `linalg_solver.matmul linalg_solver.dot linalg_solver.norm\nlinalg_solver.svd linalg_solver.eigenvalue linalg_solver.ols\nglyph.TENSOR_CORE glyph.TENSOR_PROD glyph.DOT glyph.CROSS\npolicy=math_only required`,
    `// linalg shader — opcode 0x42 TENSOR_CORE\nfloat4x4 C = mul(A, B); // WO_MATMUL 0x10\nfloat attraction = attraction_well(norm(C), gravity_scale);`,
  ]
);
export const LINALG_SOLVER_SPEC = toml(
  'linalg_solver','Sek','Heavy', 2000, 8, '1e-5',
  'matrix,vector,dot,cross,norm,svd,eigen,decompose,solve,regression',
  'linalg_solver,matmul_kernel,math_tool,fibonacci_fold'
);

export const GEODESIC_ROUTER_KXML = kxml(
  'geodesic_router', 'Wo', 'Float',
  [
    { id:'hyperbolic', domain:'hyperbolic_routing' },
    { id:'great_circle', domain:'spherical_geodesic' },
    { id:'moe_route', domain:'expert_routing' },
  ],
  [
    { from:'hyperbolic',   to:'great_circle', cost:'0.3' },
    { from:'great_circle', to:'moe_route',    cost:'0.2' },
  ],
  [{ id:'route', type:'routing', perm:'inherit' }],
  [{ id:'route_policy', type:'allow', domain:'all_routing', perm:'grant' }],
  [
    `Pop:   load token vector + expert weight matrix\nWo:    declare routing intent, bind geodesic cost fn\nSek:   compute great-circle distances, top-K selection\nCh'en: emit route table + selected expert indices`,
    `geodesic_router.hyperbolic geodesic_router.great_circle geodesic_router.top_k\ngeodesic_router.moe geodesic_router.cost_matrix\nglyph.PARALLEL_GEO glyph.TRIPLE_GEO glyph.CIRCLE_GEO glyph.QUAD_DASH`,
    `// geodesic router shader — opcode 0x64 PARALLEL_GEO\nfloat dist = geodesic_dist(normalize(q), normalize(k));\nfloat route_score = 1.0 / (1.0 + dist);`,
  ]
);
export const GEODESIC_ROUTER_SPEC = toml(
  'geodesic_router','Wo','Float', 2000, 4, '2e-5',
  'route,geodesic,hyperbolic,great_circle,distance,moe,expert,top_k',
  'geodesic_router,dxsk_route,linalg_solver,pi_field'
);

// ─── Glyph opcode µMODELS ─────────────────────────────────────────────────────

export const TENSOR_OPS_KXML = kxml(
  'tensor_ops', 'Sek', 'Normal',
  [
    { id:'core',    domain:'TENSOR_CORE' },
    { id:'product', domain:'TENSOR_PROD' },
    { id:'network', domain:'TENSOR_NETWORK' },
  ],
  [
    { from:'core', to:'product', cost:'0.2' },
    { from:'product', to:'network', cost:'0.3' },
  ],
  [{ id:'tensor', type:'tensor_compute', perm:'inherit' }],
  [{ id:'tensor_policy', type:'allow', domain:'tensor_ops', perm:'grant' }],
  [
    `Pop:   load tensor operands, validate ranks\nWo:    declare tensor intent — core/product/sum/union\nSek:   execute tensor operation via glyph opcode\nCh'en: emit result tensor + contraction trace`,
    `tensor_ops.core tensor_ops.product tensor_ops.sum tensor_ops.union\ntensor_ops.intersection tensor_ops.contraction tensor_ops.outer\nglyph.TENSOR_CORE glyph.TENSOR_PROD glyph.TENSOR_SUM glyph.TENSOR_INT\nglyph.TENSOR_UNION glyph.SMASH_PROD glyph.INTERIOR`,
    `// tensor ops shader — opcodes 0x42-0x56\nRWStructuredBuffer<float> tensorOut;\ntensorOut[tid] = dot(A[tid], B[tid]); // 0x40 DOT`,
  ]
);
export const TENSOR_OPS_SPEC = toml(
  'tensor_ops','Sek','Normal', 1500, 4, '2e-5',
  'tensor,matrix,product,sum,union,intersection,contraction,outer,core',
  'tensor_ops,linalg_solver,matmul_kernel,gpu_dispatch'
);

export const GEODESIC_OPS_KXML = kxml(
  'geodesic_ops', 'Wo', 'Float',
  [
    { id:'optical',  domain:'OPTICAL_FLOW' },
    { id:'parallel', domain:'PARALLEL_GEO' },
    { id:'wave',     domain:'WAVE_OPS' },
  ],
  [
    { from:'optical', to:'parallel', cost:'0.3' },
    { from:'parallel', to:'wave',    cost:'0.4' },
  ],
  [{ id:'geodesic', type:'geodesic_compute', perm:'inherit' }],
  [{ id:'geo_policy', type:'allow', domain:'geodesic_ops', perm:'grant' }],
  [
    `Pop:   load field vectors, validate dimensions\nWo:    declare geodesic intent — optical/parallel/wave\nSek:   compute optical axis alignment, parallel transport\nCh'en: emit field result + curvature trace`,
    `geodesic_ops.optical geodesic_ops.parallel geodesic_ops.wave\ngeodesic_ops.curvature geodesic_ops.axis geodesic_ops.plane\nglyph.OPTIC_AXIS glyph.OPTIC_PLANE glyph.PARALLEL_GEO glyph.PARALLEL_WAVE\nglyph.CIRCLE_GEO glyph.TRIPLE_GEO glyph.TETRA_GEO glyph.QUAD_DASH`,
    `// geodesic ops shader — opcodes 0x62-0x71\nfloat3 axis   = optical_axis(field, normal);\nfloat  curvature = riemann_curvature(axis, metric);`,
  ]
);
export const GEODESIC_OPS_SPEC = toml(
  'geodesic_ops','Wo','Float', 1500, 4, '2e-5',
  'geodesic,optical,parallel,wave,curvature,axis,plane,transport,field',
  'geodesic_ops,geodesic_router,pi_field,linalg_solver'
);

export const LOGIC_OPS_KXML = kxml(
  'logic_ops', 'Sek', 'Normal',
  [
    { id:'boolean',  domain:'LOGIC' },
    { id:'phase_logic', domain:'PHASE_LOGIC' },
  ],
  [{ from:'boolean', to:'phase_logic', cost:'0.2' }],
  [{ id:'logic', type:'logic_compute', perm:'inherit' }],
  [{ id:'logic_policy', type:'allow', domain:'logic_ops', perm:'grant' }],
  [
    `Pop:   load boolean/phase operands\nWo:    declare logic intent — and/or/xor/nand/nor/equiv\nSek:   evaluate logical expression, propagate phase\nCh'en: emit boolean result + phase trajectory`,
    `logic_ops.and logic_ops.or logic_ops.xor logic_ops.nand logic_ops.nor\nlogic_ops.equiv logic_ops.phase_and logic_ops.phase_or\nglyph.LOGICAL_AND glyph.LOGICAL_OR glyph.LOGICAL_XOR\nglyph.LOGICAL_NAND glyph.LOGICAL_NOR glyph.LOGICAL_EQUIV`,
    `// logic ops shader — opcodes 0x5C-0x61\nbool result = (a & b); // LOGICAL_AND 0x5C`,
  ]
);
export const LOGIC_OPS_SPEC = toml(
  'logic_ops','Sek','Normal', 1000, 4, '2e-5',
  'logic,boolean,and,or,xor,nand,nor,equivalent,phase',
  'logic_ops,tensor_ops,flux_runtime'
);

export const PHASE_OPS_KXML = kxml(
  'phase_ops', 'Wo', 'Normal',
  [
    { id:'intersection', domain:'PHASE_INT' },
    { id:'union',        domain:'PHASE_UNION' },
    { id:'quantum',      domain:'QUANTUM_GATE' },
  ],
  [
    { from:'intersection', to:'union',   cost:'0.2' },
    { from:'union',        to:'quantum', cost:'0.4' },
  ],
  [{ id:'phase', type:'phase_compute', perm:'inherit' }],
  [{ id:'phase_policy', type:'allow', domain:'phase_quantum', perm:'grant' }],
  [
    `Pop:   load phase operands + quantum state\nWo:    declare phase intent — intersect/union/cap/cup\nSek:   compute phase intersection/union, quantum gate\nCh'en: emit phase result + quantum amplitude`,
    `phase_ops.intersection phase_ops.union phase_ops.cap phase_ops.cup\nphase_ops.quantum phase_ops.smash phase_ops.interior\nglyph.PHASE_INT glyph.PHASE_UNION glyph.PHASE_CAP glyph.PHASE_CUP\nglyph.QUANTUM_GATE glyph.SMASH_PROD glyph.LEFT_ANGLE glyph.RIGHT_ANGLE`,
    `// phase ops shader — opcodes 0x52-0x5B\ncomplex phase_result = phase_intersect(psi_a, psi_b);\nfloat  amplitude     = abs(phase_result);`,
  ]
);
export const PHASE_OPS_SPEC = toml(
  'phase_ops','Wo','Normal', 1000, 4, '2e-5',
  'phase,quantum,intersection,union,cap,cup,amplitude,gate,smash',
  'phase_ops,tensor_ops,geodesic_ops,atomic_brain'
);

// ─── Runtime µMODELS ──────────────────────────────────────────────────────────

export const FLUX_RUNTIME_KXML = kxml(
  'flux_runtime', 'Pop', 'Normal',
  [
    { id:'action_queue', domain:'FIFO_QUEUE' },
    { id:'store_registry', domain:'STATE_CONTAINERS' },
    { id:'effect_engine',  domain:'ASYNC_EFFECTS' },
    { id:'time_traveler',  domain:'ACTION_LOG' },
  ],
  [
    { from:'action_queue',  to:'store_registry', cost:'0.1' },
    { from:'store_registry',to:'effect_engine',  cost:'0.2' },
    { from:'effect_engine', to:'time_traveler',  cost:'0.3' },
  ],
  [
    { id:'pop',  type:'action_queue',   perm:'inherit' },
    { id:'wo',   type:'dispatcher',     perm:'inherit' },
    { id:'sek',  type:'reducer',        perm:'inherit' },
    { id:'chen', type:'commit_notify',  perm:'inherit' },
  ],
  [{ id:'flux_policy', type:'allow', domain:'runtime_ops', perm:'grant' }],
  [
    `Pop:   dequeue next action, snapshot state_before\nWo:    route action to all store reducers\nSek:   reducers compute new state (pure, no side effects)\nCh'en: commit new state, notify subscribers, tick effects\nTime travel: action_log + state_snapshots => replay`,
    `flux_runtime.action_queue flux_runtime.store_registry flux_runtime.effects\nflux_runtime.time_travel flux_runtime.dispatch flux_runtime.subscribe\nflux_runtime.reducer flux_runtime.snapshot flux_runtime.restore\nflux_ir.pure flux_ir.store flux_ir.action flux_ir.reduce flux_ir.query`,
    `// No GPU shader needed — runtime is pure JS state machine\n// K'UHUL phase cycle IS the FLUX action processing loop\nfloat time_cost = action_queue.size * reducer_complexity;`,
  ]
);
export const FLUX_RUNTIME_SPEC = toml(
  'flux_runtime','Pop','Normal', 1000, 4, '2e-5',
  'flux,runtime,action,reducer,store,dispatch,subscribe,state,time_travel',
  'flux_runtime,kuhul_agent,micronaut_dispatch,kxml_run'
);

export const SEMANTIC_READER_KXML = kxml(
  'semantic_reader', 'Pop', 'Normal',
  [
    { id:'containment',  domain:'XML_NODES' },
    { id:'cdata',        domain:'CDATA_CAPSULES' },
    { id:'folds',        domain:'TOPOLOGY_FOLDS' },
    { id:'grams',        domain:'SEMANTIC_GRAMS' },
    { id:'activation',   domain:'PRESSURE_FIELD' },
  ],
  [
    { from:'containment', to:'cdata',      cost:'0.1' },
    { from:'cdata',       to:'folds',      cost:'0.2' },
    { from:'folds',       to:'grams',      cost:'0.2' },
    { from:'grams',       to:'activation', cost:'0.3' },
  ],
  [{ id:'read', type:'document_reader', perm:'inherit' }],
  [
    { id:'no_destructive_flattening',     type:'restrict', domain:'flatten', perm:'deny' },
    { id:'cdata_preserved',              type:'allow',    domain:'cdata',   perm:'grant' },
    { id:'policy_during_traversal',      type:'allow',    domain:'policy',  perm:'grant' },
  ],
  [
    `Pop:   read XML/JSONL document, preserve CDATA verbatim\nWo:    collect containment nodes + CDATA capsules by kind\nSek:   extract folds, geodesics, lanes, policies, grams\nCh'en: compute activation pressure, build execution plan\nInvariants: no_destructive_flattening, causal_replay_required`,
    `semantic_reader.cdata semantic_reader.folds semantic_reader.grams\nsemantic_reader.geodesics semantic_reader.pressure semantic_reader.activation\nsemantic_reader.kuhul_programs semantic_reader.projection semantic_reader.policy\nglyph.BLOCK_OPEN glyph.NEURAL_PATH glyph.STABLE_TENSOR`,
    `// Semantic reader pressure field\nfloat pressure = 0.15 + folds*0.08 + cdata*0.04 + grams*0.025;\nfloat active   = pressure >= threshold ? 1.0 : 0.0;`,
  ]
);
export const SEMANTIC_READER_SPEC = toml(
  'semantic_reader','Pop','Normal', 1000, 4, '2e-5',
  'semantic,reader,xml,cdata,topology,fold,gram,activation,pressure',
  'semantic_reader,kxml_run,micronaut_dispatch,flux_runtime'
);

// ─── Full base set ────────────────────────────────────────────────────────────

export const BASE_MUMODELS = [
  // Numeric micronauts
  { spec: FIBONACCI_FOLD_SPEC, kxml: FIBONACCI_FOLD_KXML },
  { spec: PI_FIELD_SPEC,        kxml: PI_FIELD_KXML        },
  { spec: MAYAN_FOLD_SPEC,      kxml: MAYAN_FOLD_KXML      },
  { spec: LINALG_SOLVER_SPEC,   kxml: LINALG_SOLVER_KXML   },
  { spec: GEODESIC_ROUTER_SPEC, kxml: GEODESIC_ROUTER_KXML },
  // Glyph opcode µMODELS
  { spec: TENSOR_OPS_SPEC,      kxml: TENSOR_OPS_KXML      },
  { spec: GEODESIC_OPS_SPEC,    kxml: GEODESIC_OPS_KXML    },
  { spec: LOGIC_OPS_SPEC,       kxml: LOGIC_OPS_KXML       },
  { spec: PHASE_OPS_SPEC,       kxml: PHASE_OPS_KXML       },
  // Runtime µMODELS
  { spec: FLUX_RUNTIME_SPEC,    kxml: FLUX_RUNTIME_KXML    },
  { spec: SEMANTIC_READER_SPEC, kxml: SEMANTIC_READER_KXML },
];
