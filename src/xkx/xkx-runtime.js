// xkx-runtime.js — XKX Unified Runtime: XCFE + KXML + XJSL
//
// Three orthogonal concerns, one language:
//   XCFE  — Execution semantics + flow control (for/if/switch/parallel/try-catch)
//   KXML  — Graph topology + phase gates + bidirectional edges
//   XJSL  — GPU compute shaders + WGSL/HLSL lowering
//
// FINAL TRUTH:
//   KXML is the skeleton. XJSL is the muscle.
//   Nodes = XJSL shaders. Edges = data + gradient channels.
//   Pop→Wo→Sek→Ch'en→Xul gates ALL dispatch.
//   Soft landing: phase sequence + Lipschitz bounds in MathML.
//
// K'uhul phase ordering (strictly enforced):
//   Pop     allocate / load snapshot
//   Wo      bind resources / declare intent
//   Sek     execute compute (GPU dispatch)
//   Ch'en   store results / backward pass
//   Xul     release / emit metrics
//
// Edge channels:
//   forward  channel=activation  — tensor flows from source to target
//   backward channel=gradient    — error signal flows in reverse
//   phase_gate: forward(Sek→Sek), backward(Ch'en→Sek)

export const PHASE_ORDER = ['Pop', 'Wo', 'Sek', "Ch'en", 'Xul'];

// ─── GeoTensor type tags (from geo-ir.js, without import cycle) ───────────────

export const FOLD = Object.freeze({
  COMPUTE:  'COMPUTE_FOLD',
  STORAGE:  'STORAGE_FOLD',
  META:     'META_FOLD',
  UI:       'UI_FOLD',
  ROUTING:  'ROUTING_FOLD',
});

// ─── XKXNode ──────────────────────────────────────────────────────────────────

export class XKXNode {
  constructor(id, phase, domain, fold, device = 'auto') {
    this.id           = id;
    this.phase        = phase;
    this.domain       = domain;
    this.fold         = fold;
    this.device       = device;
    this.xjslShader   = null;   // compiled shader spec
    this.mathml       = null;
    this.gradient     = null;
    this.dependencies = [];     // [{nodeId, requiresPhase}]
    this.inEdges      = [];
    this.outEdges     = [];
    this.inputs       = {};     // pending tensor inputs
    this.outputs      = {};     // completed outputs
    this.completedPhase = null;
  }

  canDispatch(currentPhase, completedMap, tinyX = true) {
    const pi = PHASE_ORDER.indexOf(currentPhase);
    if (PHASE_ORDER.indexOf(this.phase) > pi) return false;
    if (this.completedPhase) return false; // already ran

    // Dependency check
    for (const dep of this.dependencies) {
      const done = completedMap.get(dep.nodeId);
      if (!done) return false;
      if (PHASE_ORDER.indexOf(done) < PHASE_ORDER.indexOf(dep.requiresPhase)) return false;
    }

    // In-edge phase gate
    for (const edge of this.inEdges) {
      if (!edge.phaseGate?.forward) continue;
      const fromDone = completedMap.get(edge.from);
      if (!fromDone) return false;
      if (PHASE_ORDER.indexOf(fromDone) < PHASE_ORDER.indexOf(edge.phaseGate.forward.fromPhase)) return false;
      if (pi < PHASE_ORDER.indexOf(edge.phaseGate.forward.toPhase)) return false;
    }

    if (this.device === 'gpu' && !tinyX) return false;
    return true;
  }
}

// ─── XKXEdge ──────────────────────────────────────────────────────────────────

export class XKXEdge {
  constructor(id, from, to) {
    this.id      = id;
    this.from    = from;
    this.to      = to;
    this.forward  = null;  // { source, target, transform }
    this.backward = null;  // { source, target, scale }
    this.phaseGate = null; // { forward:{fromPhase,toPhase}, backward:{...} }
    this.transformations = {};
  }
}

// ─── XKXGraph ─────────────────────────────────────────────────────────────────

export class XKXGraph {
  constructor(name, type = 'bidirectional') {
    this.name    = name;
    this.type    = type;
    this.phase   = 'Pop';
    this.tinyX   = false;
    this.nodes   = new Map();
    this.edges   = new Map();
    this.phaseSequence = [];
    this.softLanding   = null;
  }

  addNode(node) { this.nodes.set(node.id, node); return this; }
  addEdge(edge) {
    this.edges.set(edge.id, edge);
    const from = this.nodes.get(edge.from);
    const to   = this.nodes.get(edge.to);
    if (from) from.outEdges.push(edge);
    if (to)   to.inEdges.push(edge);
    return this;
  }

  hasGradients() {
    for (const e of this.edges.values()) if (e.backward) return true;
    return false;
  }
}

// ─── XJSL Shader stub ─────────────────────────────────────────────────────────
//
// In production this delegates to WebGPU or D3D11 pipeline.
// Here we provide a scalar JS fallback for testing.

export class XJSLEngine {
  constructor() { this._shaders = new Map(); }

  compileShader(spec) {
    this._shaders.set(spec.name, spec);
    return true;
  }

  async executeShader(name, inputs, outputSpec, dispatchSize, uniforms) {
    const spec = this._shaders.get(name);
    if (!spec) throw new Error(`XJSLEngine: shader not found: ${name}`);
    // JS scalar fallback — real impl goes to WebGPU / D3D11
    const out = {};
    for (const outName of Object.keys(outputSpec ?? {})) out[outName] = new Float32Array(1);
    return out;
  }
}

// ─── KXMLXJSLRuntime (KXML parser + execution engine) ─────────────────────────

export class KXMLXJSLRuntime {
  constructor(xjslEngine = new XJSLEngine()) {
    this._graphs   = new Map();
    this._xjsl     = xjslEngine;
  }

  /** Load a KXML document (XML string or XKXGraph object). */
  async loadGraph(graphOrXML) {
    const graph = typeof graphOrXML === 'string'
      ? this._parseXML(graphOrXML)
      : graphOrXML;

    // Compile XJSL shaders for all nodes
    for (const node of graph.nodes.values()) {
      if (node.xjslShader) this._xjsl.compileShader(node.xjslShader);
    }

    this._graphs.set(graph.name, graph);
    return graph;
  }

  _parseXML(xmlStr) {
    const doc   = new DOMParser().parseFromString(xmlStr, 'application/xml');
    const root  = doc.documentElement;
    const graph = new XKXGraph(
      this._attr(root, 'name') ?? 'graph',
      this._attr(root, 'type') ?? 'bidirectional'
    );
    graph.tinyX = this._attr(root, 'tiny.x') === 'true';

    for (const el of root.querySelectorAll(':scope > node')) {
      const node = new XKXNode(
        this._attr(el, 'id'),
        this._attr(el, 'phase') ?? 'Sek',
        this._attr(el, 'domain') ?? 'compute',
        this._attr(el, 'fold')   ?? FOLD.COMPUTE,
        this._attr(el, 'device') ?? 'auto'
      );
      // XJSL shader
      const shaderEl = el.querySelector('shader');
      if (shaderEl) node.xjslShader = this._parseShader(shaderEl, node.id);
      // MathML
      const mathEl = el.querySelector('math');
      if (mathEl) node.mathml = mathEl;
      graph.addNode(node);
    }

    for (const el of root.querySelectorAll(':scope > edge')) {
      const edge = new XKXEdge(
        this._attr(el, 'id') ?? `e_${this._attr(el,'from')}_${this._attr(el,'to')}`,
        this._attr(el, 'from'),
        this._attr(el, 'to')
      );
      const fwdEl = el.querySelector('forward');
      if (fwdEl) {
        const d = fwdEl.querySelector('data');
        edge.forward = { source: this._attr(d,'source'), target: this._attr(d,'target'), transform: this._attr(d,'transform')||'none' };
      }
      const bwdEl = el.querySelector('backward');
      if (bwdEl) {
        const d = bwdEl.querySelector('data');
        const s = bwdEl.querySelector('scale');
        edge.backward = { source: this._attr(d,'source'), target: this._attr(d,'target'), scale: parseFloat(this._attr(s,'factor')||'0.001') };
      }
      const gateEl = el.querySelector('phase_gate');
      if (gateEl) {
        const fw = gateEl.querySelector('forward_requires');
        const bw = gateEl.querySelector('backward_requires');
        edge.phaseGate = {
          forward:  fw ? { fromPhase: this._attr(fw,'from_phase'), toPhase: this._attr(fw,'to_phase') } : null,
          backward: bw ? { fromPhase: this._attr(bw,'from_phase'), toPhase: this._attr(bw,'to_phase') } : null,
        };
      }
      graph.addEdge(edge);
    }

    return graph;
  }

  _parseShader(el, nodeId) {
    const inputs = {}, outputs = {}, uniforms = {};
    for (const buf of el.querySelectorAll('buffer')) {
      const obj = { type: this._attr(buf,'type'), binding: parseInt(this._attr(buf,'binding')||'0') };
      if (this._attr(buf,'access') === 'write') outputs[this._attr(buf,'name')] = obj;
      else inputs[this._attr(buf,'name')] = obj;
    }
    for (const f of el.querySelectorAll('field')) uniforms[this._attr(f,'name')] = this._attr(f,'type');
    const kernelEl = el.querySelector('kernel');
    return { name: nodeId, type: this._attr(el,'type')||'compute', inputs, outputs, uniforms, kernel: kernelEl?.textContent ?? '' };
  }

  _attr(el, name) { return el?.getAttribute?.(name) ?? el?.[name] ?? null; }

  // ── Execute ──

  async execute(graphName, inputs = {}) {
    const graph = this._graphs.get(graphName);
    if (!graph) throw new Error(`XKX: graph not found: ${graphName}`);
    return this._runCycle(graph, inputs);
  }

  async _runCycle(graph, inputs) {
    const completed = new Map();
    let phase = PHASE_ORDER[0];

    // Seed inputs onto relevant nodes
    for (const [nodeId, data] of Object.entries(inputs)) {
      const n = graph.nodes.get(nodeId);
      if (n) Object.assign(n.inputs, data);
    }

    for (let pi = 0; pi < PHASE_ORDER.length; pi++) {
      phase = PHASE_ORDER[pi];
      const batch = [...graph.nodes.values()].filter(n => n.canDispatch(phase, completed, graph.tinyX));

      // Pop: K'uhul Pop phase — load all (parallel)
      // Sek: compute-bound — can run in parallel
      await Promise.all(batch.map(n => this._executeNode(n, graph, completed)));

      batch.forEach(n => { n.completedPhase = phase; completed.set(n.id, phase); });
    }

    // Backward pass if graph has gradients
    if (graph.hasGradients()) {
      const revBatch = [...graph.nodes.values()].reverse();
      for (const n of revBatch) {
        if (n.gradient || n.backward) await this._executeBackward(n, graph);
      }
    }

    // Return outputs of terminal nodes (no outEdges)
    const terminal = [...graph.nodes.values()].filter(n => n.outEdges.length === 0);
    return Object.fromEntries(terminal.map(n => [n.id, n.outputs]));
  }

  async _executeNode(node, graph, completed) {
    // Gather inputs from incoming forward edges
    for (const edge of node.inEdges) {
      if (!edge.forward) continue;
      const src = graph.nodes.get(edge.from);
      const data = src?.outputs?.[edge.forward.source];
      if (data) node.inputs[edge.forward.target] = this._transform(data, edge.transformations?.forward);
    }

    // Execute
    if (node.xjslShader) {
      node.outputs = await this._xjsl.executeShader(
        node.id, node.inputs, node.xjslShader.outputs,
        { x: 64, y: 64, z: 1 }, {}
      );
    } else {
      node.outputs = { ...node.inputs }; // identity fallback
    }

    // Push outputs to downstream pending inputs
    for (const edge of node.outEdges) {
      if (!edge.forward) continue;
      const dst = graph.nodes.get(edge.to);
      if (dst) dst.inputs[edge.forward.target] = this._transform(node.outputs[edge.forward.source], edge.transformations?.forward);
    }
  }

  async _executeBackward(node, graph) {
    for (const edge of node.outEdges) {
      if (!edge.backward) continue;
      const dst = graph.nodes.get(edge.to);
      if (dst && dst.outputs[edge.backward.source]) {
        const grad = dst.outputs[edge.backward.source];
        const scaled = Array.isArray(grad) ? grad.map(v => v * edge.backward.scale) : grad;
        if (!node.gradients) node.gradients = {};
        node.gradients[edge.backward.target] = scaled;
      }
    }
  }

  _transform(data, spec) {
    if (!spec || spec.type === 'none' || !data) return data;
    if (spec.type === 'flatten') return data instanceof Float32Array ? data : new Float32Array(Object.values(data));
    return data;
  }
}

// ─── XCFERuntime (flow control layer over KXMLXJSLRuntime) ───────────────────
//
// Adds: for/if/switch/parallel/try-catch/templates to the graph execution model.
// XCFE = the "how it executes" plane.

export class XCFERuntime {
  constructor(xkxEngine = new KXMLXJSLRuntime()) {
    this._engine  = xkxEngine;
    this._vars    = new Map();
    this._fns     = new Map();
    this._templates = new Map();
  }

  get engine() { return this._engine; }

  setVar(name, value) { this._vars.set(name, value); return this; }
  getVar(name)        { return this._vars.get(name); }
  resolveExpr(expr)   {
    if (typeof expr !== 'string') return expr;
    return expr.replace(/\{\{\s*([^}]+)\s*\}\}/g, (_, k) => this._vars.get(k.trim()) ?? '');
  }

  async registerFunction(name, fn)  { this._fns.set(name, fn); }
  async callFunction(name, ...args) {
    const fn = this._fns.get(name);
    if (fn) return fn(...args);
    throw new Error(`XCFERuntime: function not found: ${name}`);
  }

  // ── Flow control ──

  async forEach(items, fn) {
    const results = [];
    for (const item of items) results.push(await fn(item));
    return results;
  }

  async parallel(tasks, concurrency = 4) {
    const results = [];
    for (let i = 0; i < tasks.length; i += concurrency)
      results.push(...await Promise.all(tasks.slice(i, i + concurrency).map(t => t())));
    return results;
  }

  async tryCatch(tryFn, catchFn, finallyFn) {
    try      { return await tryFn(); }
    catch(e) { return catchFn ? catchFn(e) : null; }
    finally  { if (finallyFn) await finallyFn(); }
  }

  switchOn(value, cases, defaultFn) {
    for (const [k, fn] of Object.entries(cases))
      if (String(value) === String(k)) return fn();
    return defaultFn?.();
  }

  // ── Graph execution shortcuts ──

  async executeGraph(name, inputs) { return this._engine.execute(name, inputs); }
  async loadGraph(graph)           { return this._engine.loadGraph(graph); }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

export function createXKX(xjslEngine = new XJSLEngine()) {
  const kxml = new KXMLXJSLRuntime(xjslEngine);
  const xcfe = new XCFERuntime(kxml);
  return { xcfe, kxml, xjsl: xjslEngine };
}

// ─── XKX EBNF grammar descriptor (for tooling / LSP) ─────────────────────────

export const XKX_GRAMMAR_SUMMARY = Object.freeze({
  version: '1.0.0',
  layers: {
    XCFE: 'Execution semantics + flow control (@for @if @switch @parallel @try)',
    KXML: 'Graph topology + phase gates + bidirectional edges (@node @edge @phase_sequence @soft_landing)',
    XJSL: 'GPU compute shaders + WGSL/HLSL lowering (@xjsl:shader @kernel @inputs @outputs)',
  },
  phases: PHASE_ORDER,
  folds:  Object.values(FOLD),
  edgeChannels: ['activation (forward)', 'gradient (backward)'],
  operators:    ['⊗ product', '⊕ compose', '⊖ diff', '⊘ scale', '⊛ hadamard', '⊜ satisfies', '⊝ project', '⊞ translate'],
  fileExtensions: { kxx: '.cpp', kslx: '.hlsl', kuhul: '.kxx+.kslx', kxml: 'FLUX IR' },
});
