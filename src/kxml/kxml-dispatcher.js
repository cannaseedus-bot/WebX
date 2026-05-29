// kxml-dispatcher.js — PhaseGatedDispatcher for KXML v7.2
//
// Executes nodes in phase order. Within a phase, nodes whose dependencies
// are satisfied are dispatched. Bidirectional edges propagate tensors from
// the completed-node buffer map to dependent nodes' input slots.
//
// Rules (from spec):
//   1. node.phase must be <= currentPhase (in phase order)
//   2. All depends_on entries must be in completedNodes at the required phase
//   3. device="gpu" requires tinyXValid
//   4. Fold must be active
//   5. Forward phase gate: predecessor must have reached forwardRequiresFrom before
//      this node can read its output

import { dispatchOp } from './kxml-ops.js';

export const PHASE_ORDER = Object.freeze(['Pop', 'Wo', 'Sek', "Ch'en", 'Xul']);

function phaseIndex(phase) {
  const i = PHASE_ORDER.indexOf(phase);
  return i < 0 ? 0 : i;
}

const FOLD_NAMES = ['COMPUTE_FOLD','STORAGE_FOLD','META_FOLD','UI_FOLD','ROUTING_FOLD'];

// ─── PhaseGatedDispatcher ────────────────────────────────────────────────────

export class PhaseGatedDispatcher {
  constructor(graph, opts = {}) {
    this._graph      = graph;
    this._tinyXValid = opts.tinyXValid ?? false;
    this._completed  = new Map();  // nodeId → { phase: string, outputs: Map }
    this._buffers    = new Map();  // graph-scoped buffer name → value
    this._foldStates = Object.fromEntries(FOLD_NAMES.map(f => [f, { isActive: true }]));
    this._currentPhase = 'Pop';
    this._log        = opts.log ?? false;
  }

  // ── Pre-populate known input buffers ────────────────────────────────────────
  setBuffer(name, value) { this._buffers.set(name, value); return this; }
  getBuffer(name)        { return this._buffers.get(name); }

  // ── Dispatch eligibility check ──────────────────────────────────────────────
  canDispatch(nodeId, currentPhase) {
    const node = this._graph.nodes.get(nodeId);
    if (!node) return false;

    // Rule 1: node phase <= current phase
    if (phaseIndex(node.phase) > phaseIndex(currentPhase)) return false;

    // Rule 2: all depends_on satisfied
    for (const dep of node.dependsOn) {
      const comp = this._completed.get(dep.nodeId);
      if (!comp) return false;
      if (dep.phase && phaseIndex(comp.phase) < phaseIndex(dep.phase)) return false;
    }

    // Rule 3: GPU requires tiny.x
    if (node.device === 'gpu' && !this._tinyXValid) return false;

    // Rule 4: fold active
    if (!this._foldStates[node.fold]?.isActive) return false;

    // Rule 5: incoming forward-channel phase gates
    for (const edge of this._graph.edges) {
      if (edge.to !== nodeId || !edge.forward) continue;
      const gate = edge.phaseGate;
      if (gate.forwardRequiresFrom) {
        const fromComp = this._completed.get(edge.from);
        if (!fromComp) return false;
        if (phaseIndex(fromComp.phase) < phaseIndex(gate.forwardRequiresFrom)) return false;
      }
    }

    return true;
  }

  // ── Execute a single node ────────────────────────────────────────────────────
  dispatchNode(nodeId, currentPhase) {
    if (!this.canDispatch(nodeId, currentPhase)) {
      throw new Error(`KXMLDispatcher: cannot dispatch "${nodeId}" at phase ${currentPhase}`);
    }

    const node = this._graph.nodes.get(nodeId);

    // Propagate forward-channel inputs from predecessor outputs into buffers
    for (const edge of this._graph.edges) {
      if (edge.to !== nodeId || !edge.forward?.data) continue;
      const comp = this._completed.get(edge.from);
      if (comp?.outputs?.has(edge.forward.data)) {
        this._buffers.set(edge.forward.data, comp.outputs.get(edge.forward.data));
      }
    }

    // Execute ops
    const nodeOutputs = new Map();
    for (const op of node.ops) {
      dispatchOp(op, this._buffers);
      if (op.dst) nodeOutputs.set(op.dst, this._buffers.get(op.dst));
    }

    if (this._log) console.log(`[KXML] dispatched "${nodeId}" phase=${currentPhase} ops=${node.ops.length}`);

    // Mark completed
    this._completed.set(nodeId, { phase: currentPhase, outputs: nodeOutputs });

    // Propagate backward-channel gradients (if predecessor already completed Ch'en)
    this._propagateBackward(nodeId);

    return nodeOutputs;
  }

  _propagateBackward(nodeId) {
    for (const edge of this._graph.edges) {
      if (edge.from !== nodeId || !edge.backward?.data) continue;
      const gate = edge.phaseGate;
      const reqFrom = gate.backwardRequiresFrom;
      if (reqFrom) {
        const comp = this._completed.get(nodeId);
        if (!comp || phaseIndex(comp.phase) < phaseIndex(reqFrom)) continue;
      }
      // Scale gradient by edge scale factor
      const grad = this._buffers.get(edge.backward.data);
      if (grad != null) {
        const scale = edge.backward.scale ?? 1;
        if (typeof grad === 'object' && grad.length) {
          const scaled = new Float64Array(grad.length);
          for (let i = 0; i < grad.length; i++) scaled[i] = grad[i] * scale;
          this._buffers.set(edge.backward.data + '_scaled', scaled);
        }
      }
    }
  }

  // ── Run all nodes for a given phase in topological order ────────────────────
  runPhase(phase) {
    this._currentPhase = phase;
    const order = this._topoOrder(phase);
    const dispatched = [];

    for (const nodeId of order) {
      if (this._completed.has(nodeId)) continue;
      if (this.canDispatch(nodeId, phase)) {
        this.dispatchNode(nodeId, phase);
        dispatched.push(nodeId);
      }
    }
    return dispatched;
  }

  // ── Run the full phase sequence ──────────────────────────────────────────────
  run() {
    const results = {};
    for (const phase of PHASE_ORDER) {
      results[phase] = this.runPhase(phase);
    }
    return results;
  }

  // ── Topological order within a phase (Kahn's algorithm) ─────────────────────
  _topoOrder(phase) {
    const nodes   = [...this._graph.nodes.values()]
      .filter(n => phaseIndex(n.phase) <= phaseIndex(phase))
      .map(n => n.id);

    // Build in-degree and adjacency from depends_on + forward edge gates
    const inDegree = Object.fromEntries(nodes.map(id => [id, 0]));
    const adj      = Object.fromEntries(nodes.map(id => [id, []]));

    for (const node of this._graph.nodes.values()) {
      if (!nodes.includes(node.id)) continue;
      for (const dep of node.dependsOn) {
        if (nodes.includes(dep.nodeId)) {
          adj[dep.nodeId].push(node.id);
          inDegree[node.id]++;
        }
      }
    }

    // Also add forward-edge ordering
    for (const edge of this._graph.edges) {
      if (nodes.includes(edge.from) && nodes.includes(edge.to) && edge.forward) {
        if (!adj[edge.from].includes(edge.to)) {
          adj[edge.from].push(edge.to);
          inDegree[edge.to]++;
        }
      }
    }

    const queue  = nodes.filter(id => inDegree[id] === 0);
    const result = [];

    while (queue.length > 0) {
      const id = queue.shift();
      result.push(id);
      for (const next of (adj[id] ?? [])) {
        inDegree[next]--;
        if (inDegree[next] === 0) queue.push(next);
      }
    }

    // Append any remaining (cycles — shouldn't occur in valid KXML)
    for (const id of nodes) {
      if (!result.includes(id)) result.push(id);
    }

    return result;
  }

  // ── State inspection ────────────────────────────────────────────────────────
  get completedCount() { return this._completed.size; }
  get totalNodes()     { return this._graph.nodes.size; }
  get currentPhase()   { return this._currentPhase; }

  isCompleted(nodeId) { return this._completed.has(nodeId); }

  snapshot() {
    return {
      phase: this._currentPhase,
      completed: [...this._completed.entries()].map(([id, c]) => ({
        id, phase: c.phase, outputKeys: [...c.outputs.keys()],
      })),
      bufferKeys: [...this._buffers.keys()],
    };
  }

  // Activate or deactivate a fold
  setFoldActive(foldName, active) {
    if (this._foldStates[foldName]) this._foldStates[foldName].isActive = active;
  }
}
