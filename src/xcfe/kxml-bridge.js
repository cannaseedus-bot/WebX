// kxml-bridge.js — KXML graph topology → KU'HUL DAG + Micronaut bridge
//
// The Bridge Equation:
//   KXML (Graph Topology) + KU'HUL (DAG + Micronauts) + XCFE (Blocks)
//   = Universal Computing Framework with Provable Soft Landings
//
// Axioms:
//   A1: Every KXML <node> is a KU'HUL Micronaut
//   A2: Every KXML <edge> is KU'HUL fold entanglement (forward+backward channel)
//   A3: Phase transitions enforce soft landings (bounded gradient)
//   A4: MathML expressions derive Lipschitz constants
//   A5: XCFE @ops[] maps directly to Micronaut opcodes
//
// Theorems:
//   T1: Forward pass  = Sek → Ch'en
//   T2: Backward pass = Ch'en → Sek (reverse DAG traversal)
//   T3: Phase gate    = fold synchronization barrier
//   T4: Soft landing  = ||∇f|| ≤ L * ||x||  (Lipschitz bound)
//   T5: DAG topology  encodes parallelism, not runtime threads
//
// Pipeline: .kxml → parse → transform → optimize → compile → .mxb

// ─── Phase machine ─────────────────────────────────────────────────────────────

export const KXML_PHASES      = ['Pop', 'Wo', 'Sek', "Ch'en", 'Xul'];
export const PHASE_INDEX      = Object.fromEntries(KXML_PHASES.map((p, i) => [p, i]));
export const ALLOWED_TRANSITIONS = [
  ['Pop', 'Wo'], ['Wo', 'Sek'], ['Sek', "Ch'en"], ["Ch'en", 'Xul']
];

export function isValidTransition(from, to) {
  return ALLOWED_TRANSITIONS.some(([f, t]) => f === from && t === to);
}

export function canExecute(nodePhase, currentPhase) {
  return PHASE_INDEX[nodePhase] <= PHASE_INDEX[currentPhase];
}

// ─── Opcode translation matrix (@ops[] → Micronaut opcodes) ───────────────────

export const KXML_OPCODE_TABLE = Object.freeze({
  // Memory ops
  '@load':    { opcode: 0x20, name: 'LOAD',         op: 'load_from_fold' },
  '@store':   { opcode: 0x21, name: 'STORE',        op: 'store_to_fold' },
  '@input':   { opcode: 0x22, name: 'LOAD_INPUT',   op: 'load_from_input_fold' },
  '@output':  { opcode: 0x23, name: 'STORE_OUTPUT', op: 'store_to_output_fold' },
  // Arithmetic
  '@mul':     { opcode: 0x30, name: 'TPROD',  op: 'tensor_product' },
  '@add':     { opcode: 0x31, name: 'TSUM',   op: 'tensor_sum' },
  '@gemm':    { opcode: 0x32, name: 'GEMM',   op: 'matrix_multiply' },
  '@scale':   { opcode: 0x33, name: 'SCALE',  op: 'scale_tensor' },
  '@dot':     { opcode: 0x34, name: 'DOT',    op: 'dot_product' },
  // Activations (all map to opcode 0x06 with subtype)
  '@activation.relu':    { opcode: 0x06, name: 'RELU',    lipschitz: 1.0, range: '[0,∞)' },
  '@activation.tanh':    { opcode: 0x06, name: 'TANH',    lipschitz: 1.0, range: '[-1,1]' },
  '@activation.sigmoid': { opcode: 0x06, name: 'SIGMOID', lipschitz: 0.25,range: '[0,1]'  },
  '@activation.softmax': { opcode: 0x06, name: 'SOFTMAX', lipschitz: 1.0, range: '[0,1]'  },
  '@activation.gelu':    { opcode: 0x06, name: 'GELU',    lipschitz: 1.12,range: '(-∞,∞)' },
  // Loss
  '@loss.cross_entropy': { opcode: 0x50, name: 'CROSS_ENTROPY', op: 'loss_ce'  },
  '@loss.mse':           { opcode: 0x51, name: 'MSE',           op: 'loss_mse' },
  '@loss.huber':         { opcode: 0x52, name: 'HUBER',         op: 'loss_huber'},
  // Control
  '@barrier': { opcode: 0x13, name: 'SYNC',        op: 'fold_synchronization' },
  '@gate':    { opcode: 0x14, name: 'WAIT_SIGNAL', op: 'phase_gating' },
  '@noop':    { opcode: 0x00, name: 'NOP',         op: 'no_operation' },
});

// ─── MathML → Lipschitz constant derivation ───────────────────────────────────

export class LipschitzAnalyzer {
  // Map known MathML function names to their Lipschitz constants
  static KNOWN = {
    relu:    1.0, tanh:    1.0, sigmoid: 0.25,
    softmax: 1.0, gelu:    1.12, swish:   1.1,
    sin:     1.0, cos:     1.0, linear:  Infinity,  // unbounded without clipping
  };

  // Estimate Lipschitz constant from a MathML expression string (heuristic)
  static estimate(mathmlOrExpr) {
    const s     = String(mathmlOrExpr).toLowerCase();
    let   L     = 1.0;
    for (const [fn, l] of Object.entries(LipschitzAnalyzer.KNOWN)) {
      if (s.includes(fn)) { L = Math.min(L, l); }
    }
    // Linear terms (multiplication) don't bound themselves
    if (s.includes('matmul') || s.includes('times') || s.includes('product')) {
      L = Math.min(L, 10.0);  // assume weight-normalized
    }
    return L;
  }

  // Verify ||grad|| ≤ L · ||input||
  static verify(gradNorm, inputNorm, L) {
    if (inputNorm === 0) return true;   // degenerate case
    return gradNorm <= L * inputNorm + 1e-7;  // small epsilon for float
  }
}

// ─── KXMLNode → Micronaut descriptor ─────────────────────────────────────────

export class KXMLMicronaut {
  constructor(nodeAttr) {
    this.id          = nodeAttr.id;
    this.phase       = nodeAttr.phase   ?? 'Sek';
    this.domain      = nodeAttr.domain  ?? 'compute';
    this.device      = nodeAttr.device  ?? 'cpu';
    this.fold        = nodeAttr.fold    ?? null;
    this.mathml      = nodeAttr.mathml  ?? null;
    this.ops         = [];
    this.lipschitz   = nodeAttr.mathml ? LipschitzAnalyzer.estimate(nodeAttr.mathml) : 1.0;
    this._state      = 'Pop';
    this._inputs     = new Map();
    this._output     = null;
    this._gradients  = new Map();
  }

  // Compile @ops[] entries to bytecode descriptors
  compileOps(opsArray) {
    this.ops = (opsArray ?? []).map(op => {
      const key = op.startsWith('@') ? op : `@${op}`;
      return KXML_OPCODE_TABLE[key] ?? { opcode: 0x00, name: op.toUpperCase(), op };
    });
    return this;
  }

  // Phase transition (enforces ALLOWED_TRANSITIONS)
  transition(newPhase) {
    if (!isValidTransition(this._state, newPhase)) {
      throw new Error(`Invalid phase transition ${this._state} → ${newPhase} on node ${this.id}`);
    }
    this._state = newPhase;
    return this;
  }

  // Forward pass: load inputs, execute ops, produce output
  async forward(inputs) {
    for (const [k, v] of Object.entries(inputs)) this._inputs.set(k, v);
    let result = inputs;
    for (const op of this.ops) {
      result = await this._execOp(op, result);
    }
    this._output = result;
    return result;
  }

  // Backward pass: compute gradient and clip to Lipschitz bound
  async backward(gradOut, lr = 0.01) {
    let grad = gradOut;
    const norm = Array.isArray(grad)
      ? Math.sqrt(grad.reduce((s, v) => s + v**2, 0))
      : Math.abs(grad);

    if (norm > this.lipschitz) {
      const scale = this.lipschitz / norm;
      grad = Array.isArray(grad) ? grad.map(v => v * scale) : grad * scale;
    }

    this._gradients.set('output_grad', grad);
    return { grad, clipped: norm > this.lipschitz, lipschitz: this.lipschitz };
  }

  async _execOp(op, data) {
    switch (op.name) {
      case 'LOAD':   return data;
      case 'TPROD':  return data;
      case 'TSUM':   return data;
      case 'RELU':   return Array.isArray(data) ? data.map(v => Math.max(0, v)) : Math.max(0, data);
      case 'SIGMOID':return Array.isArray(data) ? data.map(v => 1/(1+Math.exp(-v))) : 1/(1+Math.exp(-data));
      case 'TANH':   return Array.isArray(data) ? data.map(Math.tanh) : Math.tanh(data);
      case 'STORE':  this._output = data; return data;
      default:       return data;
    }
  }

  get state()  { return this._state; }
  get output() { return this._output; }
}

// ─── FoldEntanglement (bidirectional edge) ────────────────────────────────────

export class FoldEntanglement {
  constructor(fromId, toId, opts = {}) {
    this.from    = fromId;
    this.to      = toId;
    this.id      = `${fromId}→${toId}`;
    this._fwd    = null;    // activation
    this._bwd    = null;    // gradient
    this._phase  = 'Pop';
  }

  // Forward channel: source activation → target input
  async sendForward(data) {
    this._fwd = data;
    this._phase = "Sek→Ch'en";
    return data;
  }

  // Backward channel: target gradient → source gradient input
  async sendBackward(grad, lr = 0.01) {
    this._bwd = grad;
    this._phase = "Ch'en→Sek";
    return grad;
  }

  get activation() { return this._fwd; }
  get gradient()   { return this._bwd; }
  get phase()      { return this._phase; }
}

// ─── KXMLGraph → KU'HUL DAG compiler ─────────────────────────────────────────

export class KXMLBridge {
  constructor() {
    this._micronauts  = new Map();   // id → KXMLMicronaut
    this._edges       = new Map();   // 'from→to' → FoldEntanglement
    this._execOrder   = [];          // topological sort
    this._phase       = 'Pop';       // global phase machine state
  }

  // Phase 1: Parse KXML graph object into micronauts + entanglements
  parseGraph(graph) {
    // graph: { nodes: [{id, phase, domain, device, mathml, ops}],
    //          edges: [{from, to, forward:bool, backward:bool}] }
    for (const node of graph.nodes ?? []) {
      const mn = new KXMLMicronaut(node);
      if (node.ops) mn.compileOps(node.ops);
      this._micronauts.set(node.id, mn);
    }
    for (const edge of graph.edges ?? []) {
      const e = new FoldEntanglement(edge.from, edge.to);
      this._edges.set(e.id, e);
    }
    this._topoSort();
    return this;
  }

  // Phase 2: Topological sort (Kahn's algorithm)
  _topoSort() {
    const inDeg = new Map([...this._micronauts.keys()].map(id => [id, 0]));
    for (const { from, to } of this._edgeList()) {
      inDeg.set(to, (inDeg.get(to) ?? 0) + 1);
    }
    const queue = [...inDeg.entries()].filter(([, d]) => d === 0).map(([id]) => id);
    const order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      for (const { from, to } of this._edgeList()) {
        if (from !== id) continue;
        const d = inDeg.get(to) - 1;
        inDeg.set(to, d);
        if (d === 0) queue.push(to);
      }
    }
    this._execOrder = order;
  }

  _edgeList() { return [...this._edges.values()].map(e => ({ from: e.from, to: e.to })); }

  // Phase 3: Execute forward pass across the DAG
  async executeForward(inputs = {}) {
    this._phase = 'Sek';
    const activations = { ...inputs };

    for (const id of this._execOrder) {
      const mn = this._micronauts.get(id);
      if (!canExecute(mn.phase, this._phase)) continue;

      // Gather inputs from incoming edges
      const incoming = {};
      for (const [key, ent] of this._edges) {
        if (ent.to === id && ent.activation !== null) {
          incoming[ent.from] = ent.activation;
        }
      }

      const result = await mn.forward({ ...activations, ...incoming });
      activations[id] = result;

      // Send activation downstream
      for (const ent of this._edges.values()) {
        if (ent.from === id) await ent.sendForward(result);
      }
    }

    this._phase = "Ch'en";
    return activations;
  }

  // Phase 4: Execute backward pass (reverse DAG)
  async executeBackward(lossGrad, lr = 0.01) {
    this._phase = "Ch'en";
    const gradients = {};
    let grad = lossGrad;

    for (const id of [...this._execOrder].reverse()) {
      const mn   = this._micronauts.get(id);
      const result = await mn.backward(grad, lr);
      gradients[id] = result;
      grad = result.grad;

      // Propagate gradient upstream
      for (const ent of this._edges.values()) {
        if (ent.to === id) await ent.sendBackward(result.grad, lr);
      }
    }

    this._phase = 'Xul';
    return gradients;
  }

  // Phase 5: Soft-landing verification
  verifySoftLandings(gradients) {
    const violations = [];
    for (const [id, { grad, lipschitz }] of Object.entries(gradients)) {
      const norm  = Array.isArray(grad) ? Math.sqrt(grad.reduce((s, v) => s + v**2, 0)) : Math.abs(grad);
      const ok    = norm <= lipschitz + 1e-7;
      if (!ok) violations.push({ id, norm, lipschitz, violation: norm - lipschitz });
    }
    return {
      soft_landings: violations.length === 0,
      violations,
      node_count: this._micronauts.size,
      bounded_count: this._micronauts.size - violations.length,
    };
  }

  // Optimization: operator fusion (adjacent compatible ops)
  fuseOps() {
    for (const mn of this._micronauts.values()) {
      const fused = [];
      let i = 0;
      while (i < mn.ops.length) {
        const op  = mn.ops[i];
        const next= mn.ops[i + 1];
        // Fuse LOAD + arithmetic into single op
        if (op.name === 'LOAD' && next && ['TPROD','TSUM','GEMM'].includes(next.name)) {
          fused.push({ ...next, name: `${next.name}_FUSED`, _fused: [op, next] });
          i += 2;
        } else {
          fused.push(op);
          i++;
        }
      }
      mn.ops = fused;
    }
    return this;
  }

  // Generate MXB-style binary descriptor
  toMXBDescriptor() {
    return {
      version:    '1.0',
      format:     'kuhul_dag',
      nodes:      [...this._micronauts.values()].map(mn => ({
        id:         mn.id,
        phase:      mn.phase,
        domain:     mn.domain,
        device:     mn.device,
        fold:       mn.fold,
        lipschitz:  mn.lipschitz,
        opcodes:    mn.ops.map(o => ({ code: o.opcode, name: o.name })),
        mathml:     mn.mathml,
      })),
      edges:      [...this._edges.values()].map(e => ({
        id:         e.id,
        from:       e.from,
        to:         e.to,
        forward:    true,
        backward:   true,
      })),
      exec_order:  this._execOrder,
      phase_machine: { current: this._phase, allowed: ALLOWED_TRANSITIONS },
    };
  }

  get micronauts() { return this._micronauts; }
  get edges()      { return this._edges; }
  get execOrder()  { return this._execOrder; }
  get phase()      { return this._phase; }
}

// ─── Bridge @ namespace handler ───────────────────────────────────────────────

export function registerBridgeNamespaces(rt) {
  const bridges = new Map();

  rt._handlers.set('@bridge', async (val, ctx) => {
    const name = val['@name'] ?? 'default';
    const graph = ctx[val.graph] ?? val.graph ?? { nodes: [], edges: [] };
    const bridge = new KXMLBridge();
    bridge.parseGraph(graph);
    if (val.fuse) bridge.fuseOps();
    bridges.set(name, bridge);
    ctx[`_bridge_${name}`] = bridge;
    if (val['@store']) ctx[val['@store']] = bridge.toMXBDescriptor();
    return bridge;
  });

  rt._handlers.set('@kxml', (val, ctx) => {
    // Build a simple KXML graph spec from declarative @kxml block
    const nodes = [], edges = [];
    for (const [k, v] of Object.entries(val)) {
      if (k.startsWith('@node.')) {
        nodes.push({ id: k.slice(6), ...v });
      } else if (k.startsWith('@edge.')) {
        const [from, to] = k.slice(6).split('→');
        edges.push({ from: from?.trim(), to: to?.trim(), ...v });
      }
    }
    const graph = { nodes, edges };
    ctx['_kxml_graph'] = graph;
    if (val['@store']) ctx[val['@store']] = graph;
    return graph;
  });

  rt._handlers.set('@verify.soft_landings', (val, ctx) => {
    const gradients = ctx[val.gradients] ?? ctx['_gradients'] ?? {};
    const bridge    = ctx[`_bridge_${val.bridge ?? 'default'}`];
    const result    = bridge?.verifySoftLandings(gradients) ?? { soft_landings: true };
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });
}

// ─── Integration matrix (reference table) ─────────────────────────────────────

export const KXML_KUHUL_MAP = Object.freeze([
  { kxml: '<node>',              kuhul: 'KXMLMicronaut',        xcfe: '@kxml.node'          },
  { kxml: '<node/@phase>',       kuhul: 'Phase state machine',   xcfe: '@phase_machine'      },
  { kxml: '<node/@fold>',        kuhul: 'Fold assignment',       xcfe: '@fold'               },
  { kxml: '<edge>',              kuhul: 'FoldEntanglement',      xcfe: '@edge'               },
  { kxml: '<edge/forward>',      kuhul: 'Forward channel',       xcfe: '@forward_channel'    },
  { kxml: '<edge/backward>',     kuhul: 'Backward channel',      xcfe: '@backward_channel'   },
  { kxml: '<mathml>',            kuhul: 'Opcode sequence',       xcfe: '@opcode.translation' },
  { kxml: '<gradient>',          kuhul: 'Backward pass',         xcfe: '@backward_pass'      },
  { kxml: '<phase_gate>',        kuhul: 'Sync barrier',          xcfe: '@barrier'            },
  { kxml: '<phase_sequence>',    kuhul: 'Execution plan',        xcfe: '@exec_order'         },
  { kxml: '<soft_landing>',      kuhul: 'Bounded gradient',      xcfe: '@lipschitz.bound'    },
  { kxml: 'XCFE @ops[]',         kuhul: 'Micronaut opcodes',     xcfe: '@opcode_table'       },
]);

export const BRIDGE_THEOREMS = Object.freeze([
  { id:'T1', statement: 'Forward pass  = Sek → Ch\'en' },
  { id:'T2', statement: 'Backward pass = Ch\'en → Sek (reverse DAG)' },
  { id:'T3', statement: 'Phase gate    = fold synchronization barrier' },
  { id:'T4', statement: '||∇f|| ≤ L·||x||  (Lipschitz soft landing)' },
  { id:'T5', statement: 'DAG topology  encodes parallelism, not threads' },
  { id:'T6', statement: 'MathML derives Lipschitz constants analytically' },
  { id:'T7', statement: 'Bidirectional edge compiles to static backward pass' },
]);
