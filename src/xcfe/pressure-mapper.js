// pressure-mapper.js — 2D Fold Pressure Mapper + Pressure Reserves
//
// FOLD PRESSURE = how tightly a tensor is compressed in a given fold
//   COMPUTE   high  (8x at rank-2) tight packing for GPU
//   STORAGE   med   (4x at rank-2) balanced
//   META      low   (1x at rank-2) semantic expansion
//   ROUTING   var   fan-out dependent
//   UI        min   (0.0625x rank-2) human-readable expansion
//
// BALLOONING = tensor expands moving from high-pressure to low-pressure fold
//   balloon_factor = source_pressure / target_pressure
//   COMPUTE(8x) → UI(0.0625x) = 128x expansion at rank-2
//
// PRESSURE RESERVE = stored excess gravity that a fold draws on before
//   live constraints are tightened by KuhulPhysicsSolver.
//   Accumulates during stable orbit (Rule 4), depletes during instability (Rule 1-6).
//   A fold with a full reserve can absorb one shock without tightening constraints.
//
// Connection to K'UHUL physics:
//   reserve drains  → gravity_scale decreases → KuhulPhysicsSolver.observe() sees it
//   reserve fills   → stable orbit → antigravity_float increases
//   reserve empty   → hard constraint tightening (logit_bound, grad_clip)
//
// Connection to math µMODEL training (step 700, loss=10.03):
//   loss near EVENT_HORIZON → physics solver fires Rule 1
//   BEFORE tightening live bounds → drain from COMPUTE_FOLD reserve first
//   If reserve exhausted → then tighten logit_bound 20→18

import { G } from './gravity.js';

// ─── Fold types ────────────────────────────────────────────────────────────────
//
// Base execution folds (geometry / physics):
//   COMPUTE  STORAGE  META  ROUTING  UI
//
// Agentic tool folds (GPT-2 Mini + Micronauts → Agentic System):
//   TOOL    0.8  @Tool-annotated methods — phase-gated, validated
//   SKILL   0.6  learned procedures — read/write, evolves
//   CMD     0.6  command schemas — parse/validate/exec
//   OPCODE  0.9  VM ops / stack machine — tightest bounds
//   AGENT   0.3  autonomous decisions — orchestrator only
//   FILE    0.5  sandboxed file IO — read/write/watch
//   TASK    0.3  plans + TODO lists — read/update
//   THINK   0.2  CoT/ToT reasoning traces — write only, antigravity

export const FOLD = Object.freeze({
  // Base execution folds
  COMPUTE:  'COMPUTE_FOLD',
  STORAGE:  'STORAGE_FOLD',
  META:     'META_FOLD',
  ROUTING:  'ROUTING_FOLD',
  UI:       'UI_FOLD',
  // Agentic tool folds
  TOOL:     'TOOL_FOLD',
  SKILL:    'SKILL_FOLD',
  CMD:      'CMD_FOLD',
  OPCODE:   'OPCODE_FOLD',
  AGENT:    'AGENT_FOLD',
  FILE:     'FILE_FOLD',
  TASK:     'TASK_FOLD',
  THINK:    'THINK_FOLD',
});

// ─── Pressure table: pressure[fold][rank 0..4] ────────────────────────────────

export const PRESSURE_TABLE = Object.freeze({
  // Base execution folds
  [FOLD.COMPUTE]: [8.0,  4.0,   2.0,    1.0,     0.5    ],
  [FOLD.STORAGE]: [4.0,  2.0,   1.0,    0.5,     0.25   ],
  [FOLD.META]:    [1.0,  0.5,   0.25,   0.125,   0.0625 ],
  [FOLD.ROUTING]: [2.0,  1.0,   1.0,    2.0,     4.0    ],
  [FOLD.UI]:      [0.25, 0.125, 0.0625, 0.03125, 0.015625],
  // Agentic tool folds — pressure by tensor rank 0..4
  //                      0D    1D    2D    3D    4D
  [FOLD.TOOL]:    [0.8,  0.8,   0.8,   0.8,   0.8   ],  // uniform — tool schemas are rank-agnostic
  [FOLD.SKILL]:   [0.6,  0.6,   0.6,   0.6,   0.6   ],  // learned procedures, medium constraint
  [FOLD.CMD]:     [0.6,  0.6,   0.6,   0.6,   0.6   ],  // command parse/exec, same as skill
  [FOLD.OPCODE]:  [0.9,  0.9,   0.9,   0.9,   0.9   ],  // tightest — VM ops, stack machine
  [FOLD.AGENT]:   [0.3,  0.3,   0.3,   0.3,   0.3   ],  // autonomous, low-pressure decisions
  [FOLD.FILE]:    [0.5,  0.5,   0.5,   0.5,   0.5   ],  // sandboxed IO — balanced
  [FOLD.TASK]:    [0.3,  0.3,   0.3,   0.3,   0.3   ],  // planning/TODO — low pressure
  [FOLD.THINK]:   [0.2,  0.2,   0.2,   0.2,   0.2   ],  // CoT/ToT — minimal, near antigravity
});

export function getPressure(fold, rank = 2) {
  const row = PRESSURE_TABLE[fold];
  if (!row) return 1.0;
  return row[Math.min(rank, 4)];
}

// balloon_factor = source_pressure / target_pressure
export function balloonFactor(sourceFold, targetFold, rank = 2) {
  return getPressure(sourceFold, rank) / getPressure(targetFold, rank);
}

// ─── PressureReserve ──────────────────────────────────────────────────────────
//
// A buffer of stored gravity pressure per fold node.
// Accumulates during stable orbit, depletes when physics solver would tighten bounds.
// When reserve > 0, the solver draws from it BEFORE tightening live constraints.

export class PressureReserve {
  /**
   * @param {number} capacity  max reserve (default: 3 × normal gravity)
   * @param {number} initial   starting fill level (0 = empty, capacity = full)
   */
  constructor(capacity = 3.0, initial = 0.5) {
    this._capacity  = capacity;
    this._level     = Math.min(initial, capacity);
    this._totalIn   = 0;
    this._totalOut  = 0;
  }

  get level()    { return this._level; }
  get capacity() { return this._capacity; }
  get ratio()    { return this._level / this._capacity; }
  get empty()    { return this._level <= 0; }
  get full()     { return this._level >= this._capacity; }

  /** Accumulate reserve during stable orbit. Returns amount actually added. */
  charge(amount) {
    const before = this._level;
    this._level  = Math.min(this._capacity, this._level + amount);
    const added  = this._level - before;
    this._totalIn += added;
    return added;
  }

  /**
   * Discharge reserve to absorb a physics shock.
   * Returns {discharged, satisfied} — if discharged == requested, the shock is
   * fully absorbed and live constraints do NOT need to tighten.
   */
  discharge(requested) {
    const discharged = Math.min(this._level, requested);
    this._level  -= discharged;
    this._totalOut += discharged;
    return { discharged, satisfied: discharged >= requested };
  }

  /** Transfer pressure to another reserve (propagate along edge). */
  transfer(target, amount) {
    const { discharged } = this.discharge(amount);
    target.charge(discharged);
    return discharged;
  }

  summary() {
    return { level: this._level, capacity: this._capacity, ratio: this.ratio,
             totalIn: this._totalIn, totalOut: this._totalOut };
  }
}

// ─── FoldNode ─────────────────────────────────────────────────────────────────

export class FoldNode {
  constructor(id, fold, rank = 2, opts = {}) {
    this.id       = id;
    this.fold     = fold;
    this.rank     = rank;
    this.pressure = getPressure(fold, rank);
    this.reserve  = new PressureReserve(
      opts.reserveCapacity ?? this.pressure * 3,
      opts.reserveInitial  ?? this.pressure * 0.5
    );
    this.antigravity = fold === FOLD.UI || fold === FOLD.META || opts.antigravity === true;
    this.gravity_scale = this.antigravity ? G.FLOAT : G.NORMAL;
    this._tensors = [];
  }

  /** True if a shock of `amount` can be fully absorbed by the reserve. */
  canAbsorb(amount) { return this.reserve.level >= amount; }

  /** Absorb a shock: draw from reserve first, return remainder (live tightening needed). */
  absorb(shock) {
    if (this.antigravity) return 0;        // antigravity nodes never tighten
    const { discharged, satisfied } = this.reserve.discharge(shock);
    return satisfied ? 0 : shock - discharged;  // 0 = fully absorbed
  }

  /** Accumulate reserve (called when in stable orbit). */
  stabilize(amount = 0.1) {
    if (this.antigravity) return;           // antigravity nodes don't accumulate
    this.reserve.charge(amount * this.pressure);
  }
}

// ─── Fold2DMapper ─────────────────────────────────────────────────────────────

export class Fold2DMapper {
  constructor() {
    this._nodes    = new Map();  // id → FoldNode
    this._edges    = [];         // [{from, to}]
  }

  addNode(id, fold, rank = 2, opts = {}) {
    const node = new FoldNode(id, fold, rank, opts);
    this._nodes.set(id, node);
    return node;
  }

  addEdge(fromId, toId) {
    this._edges.push({ from: fromId, to: toId });
    return this;
  }

  getNode(id) { return this._nodes.get(id); }

  /** Compute balloon factor for a tensor moving between two fold nodes. */
  balloon(fromId, toId) {
    const src = this._nodes.get(fromId);
    const dst = this._nodes.get(toId);
    if (!src || !dst) return 1.0;
    return balloonFactor(src.fold, dst.fold, src.rank);
  }

  /**
   * Apply a physics shock across the graph.
   * Each COMPUTE/STORAGE node absorbs from its reserve first.
   * Returns {absorbed, propagated, tightened} per node.
   */
  applyShock(shockMagnitude) {
    const report = {};

    // Phase 1: each constrained node tries to absorb from its reserve
    for (const [id, node] of this._nodes) {
      if (node.antigravity) {
        report[id] = { absorbed: 0, remaining: 0, antigravity: true };
        continue;
      }
      const remaining = node.absorb(shockMagnitude * node.pressure);
      report[id] = {
        absorbed:  shockMagnitude * node.pressure - remaining,
        remaining,
        reserve:   node.reserve.summary(),
        tightened: remaining > 0,
      };
    }

    // Phase 2: propagate residual shock along edges (pressure wave)
    for (const { from, to } of this._edges) {
      const srcNode = this._nodes.get(from);
      const dstNode = this._nodes.get(to);
      if (!srcNode || !dstNode) continue;
      const residual = report[from]?.remaining ?? 0;
      if (residual > 0 && !dstNode.antigravity) {
        const extra = dstNode.absorb(residual * 0.5);  // attenuate by 50%
        report[to] = { ...(report[to] ?? {}), pressure_wave_absorbed: residual * 0.5 - extra };
      }
    }

    return report;
  }

  /** Stabilize reserves for all nodes in stable orbit. */
  stabilize(stableNodeIds = null) {
    const ids = stableNodeIds ?? [...this._nodes.keys()];
    for (const id of ids) {
      const node = this._nodes.get(id);
      if (node) node.stabilize();
    }
  }

  /** Full pressure map: fold × rank → pressure + balloon_factor to UI. */
  pressureMap() {
    const out = {};
    for (const [fold, row] of Object.entries(PRESSURE_TABLE)) {
      out[fold] = row.map((p, rank) => ({
        rank, pressure: p,
        balloon_to_ui: p / PRESSURE_TABLE[FOLD.UI][rank],
        balloon_to_meta: p / PRESSURE_TABLE[FOLD.META][rank],
      }));
    }
    return out;
  }

  summary() {
    return [...this._nodes.entries()].map(([id, n]) => ({
      id, fold: n.fold, rank: n.rank, pressure: n.pressure,
      antigravity: n.antigravity, reserve: n.reserve.summary(),
    }));
  }
}

// ─── Math µMODEL pressure configuration ─────────────────────────────────────
//
// Wires the math_tool fold topology into the 2D mapper with reserves.

// ─── Agentic tool fold mapper ─────────────────────────────────────────────────
//
// GPT-2 Mini + Micronauts + Tool Folds + K'UHUL Physics = Agentic System
//
// Micronaut → Fold mapping:
//   LLM-µ     → TOOL_FOLD    (augments GPT-2 with tool definitions)
//   PLAN-µ    → TASK_FOLD    (hierarchical task decomposition)
//   TOOL-µ    → TOOL_FOLD    (@Tool execution)
//   SKILL-µ   → SKILL_FOLD   (learned procedures, self-improves)
//   CMD-µ     → CMD_FOLD     (parse/validate commands)
//   OP-µ      → OPCODE_FOLD  (low-level VM ops)
//   FILE-µ    → FILE_FOLD    (sandboxed IO)
//   TASK-µ    → TASK_FOLD    (TODO management)
//   THINK-µ   → THINK_FOLD   (CoT/ToT reasoning — antigravity)
//   VALID-µ   → CMD_FOLD     (pre/post-condition validation)
//
// Gravity of THINK_FOLD = 0.2 → near antigravity (G.THRESHOLD = 1e-3)
// This means THINK nodes observe without constraining training — like [dbg] lines.

export function buildAgenticMapper() {
  const m = new Fold2DMapper();

  // Execution folds (⟁Grav⟁ — constrained)
  m.addNode('tool_execute',   FOLD.TOOL,   2, { reserveInitial: 1.2 });
  m.addNode('skill_apply',    FOLD.SKILL,  2, { reserveInitial: 0.9 });
  m.addNode('cmd_parse',      FOLD.CMD,    1, { reserveInitial: 0.9 });
  m.addNode('opcode_vm',      FOLD.OPCODE, 1, { reserveInitial: 1.5 });
  m.addNode('file_io',        FOLD.FILE,   1, { reserveInitial: 0.75 });

  // Planning folds (low-pressure)
  m.addNode('task_plan',      FOLD.TASK,   2, { reserveInitial: 0.45 });
  m.addNode('agent_decide',   FOLD.AGENT,  2, { reserveInitial: 0.45 });

  // Antigravity folds (⟁AntiGrav⟁ — float, near 0.2 pressure)
  m.addNode('think_cot',      FOLD.THINK,  2, { antigravity: false });  // 0.2 — near float
  m.addNode('think_reflect',  FOLD.THINK,  1, { antigravity: false });

  // Flow edges
  m.addEdge('task_plan',    'tool_execute');
  m.addEdge('task_plan',    'cmd_parse');
  m.addEdge('tool_execute', 'skill_apply');
  m.addEdge('cmd_parse',    'opcode_vm');
  m.addEdge('opcode_vm',    'file_io');
  m.addEdge('think_cot',    'task_plan');   // THINK → TASK: balloon 0.2/0.3 = 0.67× (slight compression)
  m.addEdge('agent_decide', 'tool_execute'); // AGENT → TOOL: balloon 0.3/0.8 = 0.375×

  return m;
}

// ─── KXML math node builder ───────────────────────────────────────────────────
//
// Generates a KXML <node> for a DirectXMath SIMD math operation.
// Connects KXML phase gating to DirectXMath dispatch + XMHALF4 compression.

export function buildKXMLMathNode(opts = {}) {
  const {
    id       = 'fib_batch_simd',
    op       = 'fibonacci',
    batchSize = 256,
    simdWidth = 4,
    pack      = 'half4',
    phase     = 'Sek',
    device    = 'gpu',
  } = opts;

  const threads = batchSize / simdWidth;  // e.g. 256/4 = 64

  return `<node id="${id}" phase="${phase}" domain="compute"
       fold="COMPUTE_FOLD" device="${device}"
       gravity="1.0">

  <description>
    DirectXMath SIMD ${op} — ${simdWidth}-wide XMVECTOR, ${pack} compression.
    Gravity = 1.0 (Normal): logit_bound=20, grad_clip=1.0.
    Each thread computes ${simdWidth} values (one XMVECTOR instruction).
  </description>

  <inputs>
    <input name="n_indices" type="uint[]" min="0" max="65535"/>
    <input name="batch_size" type="uint" default="${batchSize}"/>
  </inputs>

  <outputs>
    <output name="${op}_values" type="${pack === 'half4' ? 'half4[]' : 'float4[]'}"/>
    <output name="execution_time_ms" type="float"/>
  </outputs>

  <!-- MathML specification — verified by Lipschitz soft-landing at Xul -->
  <mathml>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/>
        <apply><ci>F</ci><ci>n</ci></apply>
        <apply><floor/>
          <apply><plus/>
            <apply><divide/>
              <apply><power/><mi>&#x3C6;</mi><ci>n</ci></apply>
              <apply><sqrt/><cn>5</cn></apply>
            </apply>
            <cn>0.5</cn>
          </apply>
        </apply>
      </apply>
    </math>
    <annotation encoding="asx/lipschitz">L=phi=${((1+Math.sqrt(5))/2).toFixed(6)}</annotation>
    <annotation encoding="asx/simd_width">${simdWidth}</annotation>
    <annotation encoding="asx/compression">${pack}</annotation>
  </mathml>

  <dx:implementation>
    <dx:kernel>${op}_simd.hlsl</dx:kernel>
    <dx:dispatch>${threads},1,1</dx:dispatch>
    <dx:group_size>${threads},1,1</dx:group_size>
    <dx:memory_layout>interleaved</dx:memory_layout>
    <dx:compression>${pack === 'half4' ? 'f32tof16 (50% saving)' : 'none'}</dx:compression>
  </dx:implementation>

  <soft_landing lipschitz="${((1+Math.sqrt(5))/2).toFixed(6)}"/>

  <!-- Phase sequence: Pop init → Wo bind → Sek dispatch → Ch'en pack → Xul close -->
  <phase_sequence>
    <step phase="Pop"><action>dx:create_buffer size="${batchSize * 2 * 4}" type="structured"/></step>
    <step phase="Wo"><action>dx:bind_uav slot="0" + dx:bind_cbv slot="0"</action></step>
    <step phase="Sek" duration="compute_bound">
      <action>dx:dispatch ${threads},1,1</action>
      <action>dx:sync type="uav_barrier"</action>
    </step>
    <step phase="Ch'en">
      <action>dx:readback + dx:unpack_${pack}</action>
    </step>
    <step phase="Xul">
      <action>dx:release_buffer + dx:record_metrics</action>
    </step>
  </phase_sequence>

</node>`;
}

export function buildMathMupyMapper() {
  const m = new Fold2DMapper();

  // Compute folds (⟁Grav⟁ — constrained, phase-gated)
  m.addNode('arithmetic',   FOLD.COMPUTE, 2, { reserveInitial: 1.5 });
  m.addNode('calculus',     FOLD.COMPUTE, 2, { reserveInitial: 1.5 });
  m.addNode('linalg',       FOLD.COMPUTE, 3, { reserveInitial: 2.0 });
  m.addNode('statistics',   FOLD.COMPUTE, 2, { reserveInitial: 1.5 });

  // Output/loss folds (⟁HeavyGrav⟁)
  m.addNode('loss',         FOLD.META, 0, { reserveCapacity: 5.0, reserveInitial: 2.0 });
  m.addNode('logit_head',   FOLD.COMPUTE, 1, { reserveCapacity: 4.0, reserveInitial: 1.0 });

  // Antigravity telemetry folds (⟁AntiGrav⟁ — the [dbg] lines)
  m.addNode('dbg_embed',   FOLD.UI, 0, { antigravity: true });
  m.addNode('dbg_logits',  FOLD.UI, 1, { antigravity: true });
  m.addNode('dbg_loss',    FOLD.UI, 0, { antigravity: true });

  // Edges (data flow + pressure wave propagation)
  m.addEdge('arithmetic', 'calculus');
  m.addEdge('calculus',   'linalg');
  m.addEdge('linalg',     'statistics');
  m.addEdge('statistics', 'logit_head');
  m.addEdge('logit_head', 'loss');

  // Debug edges (antigravity — balloon from COMPUTE to UI)
  m.addEdge('arithmetic', 'dbg_embed');   // balloon: 8x → 0.25x = 32x expansion
  m.addEdge('logit_head', 'dbg_logits');  // balloon: 4x → 0.125x = 32x expansion
  m.addEdge('loss',       'dbg_loss');    // balloon: 1x → 0.25x = 4x expansion

  return m;
}
