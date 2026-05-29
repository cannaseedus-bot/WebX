// kxml-graph.js — KXMLGraph: static bidirectional computation graph
//
// Responsibilities:
//   - Load KXML from string via parseKXML
//   - compile(): build forward index, backward index, phase batches
//   - merkleRoot(): SCXQ2-style per-node hash → Merkle tree root
//   - dispatch via PhaseGatedDispatcher
//
// The graph is compiled ONCE (static). No runtime expansion of backward pass.

import { parseKXML } from './kxml-parser.js';
import { PhaseGatedDispatcher, PHASE_ORDER } from './kxml-dispatcher.js';

// ─── Tiny hash (djb2 variant, browser-safe) ───────────────────────────────────

function hashStr(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

function nodeHash(node) {
  // Hash the deterministic content: id + phase + ops + mathml
  const content = [
    node.id, node.phase, node.fold, node.device,
    node.ops.map(o => JSON.stringify(o)).join('|'),
    node.mathmlRaw ?? '',
    node.gradientRaw ?? '',
  ].join('\x00');
  return hashStr(content);
}

function merkleStep(hashes) {
  if (hashes.length === 0) return hashStr('empty');
  if (hashes.length === 1) return hashes[0];
  const next = [];
  for (let i = 0; i < hashes.length; i += 2) {
    const a = hashes[i];
    const b = i + 1 < hashes.length ? hashes[i + 1] : a;
    next.push(hashStr(a + b));
  }
  return merkleStep(next);
}

// ─── KXMLGraph ────────────────────────────────────────────────────────────────

export class KXMLGraph {
  constructor(graphObj) {
    this._g        = graphObj;
    this._compiled = false;
    this._fwdIndex = new Map();  // nodeId → [edge, ...]  outgoing forward edges
    this._bwdIndex = new Map();  // nodeId → [edge, ...]  incoming backward edges
    this._phases   = new Map();  // phase → [nodeId, ...]
    this._merkle   = null;
  }

  static fromString(xml) {
    return new KXMLGraph(parseKXML(xml));
  }

  get id()       { return this._g.id; }
  get type()     { return this._g.type; }
  get metadata() { return this._g.metadata; }
  get nodes()    { return this._g.nodes; }
  get edges()    { return this._g.edges; }
  get softLanding() { return this._g.softLanding; }

  // ── compile(): build forward/backward indexes and phase batches ──────────────
  compile() {
    if (this._compiled) return this;

    // Forward index: for each node, which edges flow out (forward)
    for (const node of this._g.nodes.values()) {
      this._fwdIndex.set(node.id, []);
      this._bwdIndex.set(node.id, []);
    }

    for (const edge of this._g.edges) {
      if (edge.forward) {
        const arr = this._fwdIndex.get(edge.from) ?? [];
        arr.push(edge);
        this._fwdIndex.set(edge.from, arr);
      }
      if (edge.backward) {
        const arr = this._bwdIndex.get(edge.to) ?? [];
        arr.push(edge);
        this._bwdIndex.set(edge.to, arr);
      }
    }

    // Phase batches: group nodes by their declared phase
    for (const phase of PHASE_ORDER) {
      this._phases.set(phase, []);
    }
    for (const node of this._g.nodes.values()) {
      const phase = node.phase ?? 'Pop';
      if (!this._phases.has(phase)) this._phases.set(phase, []);
      this._phases.get(phase).push(node.id);
    }

    // Compute Merkle root over sorted node hashes
    const sorted = [...this._g.nodes.values()].sort((a, b) => a.id < b.id ? -1 : 1);
    this._merkle  = merkleStep(sorted.map(nodeHash));

    this._compiled = true;
    return this;
  }

  // ── Merkle root (SCXQ2 lane integrity) ──────────────────────────────────────
  merkleRoot() {
    if (!this._compiled) this.compile();
    return this._merkle;
  }

  // Per-node hash (for lane verification)
  nodeHash(nodeId) {
    const node = this._g.nodes.get(nodeId);
    return node ? nodeHash(node) : null;
  }

  // ── Phase-level accessors ────────────────────────────────────────────────────
  nodesForPhase(phase) {
    if (!this._compiled) this.compile();
    return this._phases.get(phase) ?? [];
  }

  forwardEdgesFrom(nodeId) {
    if (!this._compiled) this.compile();
    return this._fwdIndex.get(nodeId) ?? [];
  }

  backwardEdgesTo(nodeId) {
    if (!this._compiled) this.compile();
    return this._bwdIndex.get(nodeId) ?? [];
  }

  // ── Create a dispatcher for this graph ───────────────────────────────────────
  createDispatcher(opts = {}) {
    if (!this._compiled) this.compile();
    return new PhaseGatedDispatcher(this._g, opts);
  }

  // ── Validate graph consistency ───────────────────────────────────────────────
  validate() {
    if (!this._compiled) this.compile();
    const errors = [];

    // Every edge endpoint must reference a known node
    for (const edge of this._g.edges) {
      if (!this._g.nodes.has(edge.from))
        errors.push(`edge.from "${edge.from}" not found`);
      if (!this._g.nodes.has(edge.to))
        errors.push(`edge.to "${edge.to}" not found`);
    }

    // Every depends_on must reference a known node
    for (const node of this._g.nodes.values()) {
      for (const dep of node.dependsOn) {
        if (!this._g.nodes.has(dep.nodeId))
          errors.push(`node "${node.id}" depends_on unknown "${dep.nodeId}"`);
      }
    }

    // Lipschitz contracts: all declared lipschitz must be <= 1 for stability
    for (const node of this._g.nodes.values()) {
      if (node.lipschitz != null && node.lipschitz > 1)
        errors.push(`node "${node.id}" Lipschitz=${node.lipschitz} > 1 — not a contraction`);
    }

    // Soft landing bounded ops
    if (this._g.softLanding) {
      for (const op of this._g.softLanding.boundedOps) {
        if (op.lipschitz > 1)
          errors.push(`soft_landing op "${op.type}" Lipschitz=${op.lipschitz} > 1`);
      }
    }

    return { valid: errors.length === 0, errors };
  }

  // ── Summary ───────────────────────────────────────────────────────────────────
  summary() {
    if (!this._compiled) this.compile();
    const phaseCounts = Object.fromEntries(
      [...this._phases.entries()].map(([p, ids]) => [p, ids.length])
    );
    return {
      id:          this._g.id,
      type:        this._g.type,
      nodeCount:   this._g.nodes.size,
      edgeCount:   this._g.edges.length,
      phaseCounts,
      merkleRoot:  this._merkle,
      hasSoftLanding: this._g.softLanding != null,
    };
  }
}
