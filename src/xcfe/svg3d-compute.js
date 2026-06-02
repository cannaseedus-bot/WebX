// svg3d-compute.js — SVG3D Parallel Compute Engine
//
// Core axiom: SVG-3D elements ARE compute nodes, not graphics.
//   circle  → compute cluster (32-core equivalent)
//   path    → data flow between clusters
//   group   → tensor core (nested execution)
//   torus   → memory hierarchy (L1/L2/L3)
//   sphere  → warp of 32 threads in lockstep
//
// iGPU executes via existing HLSL shaders (cs_5_0, HD 4600 compatible):
//   sh_propagate.hlsl    → OP_PROPAGATE = one compute tick across all clusters
//   opcode_kernels.hlsl  → OP_INJECT = inject into one cluster node
//   spherical_map_compiler.hlsl → compile the cluster topology once
//
// The optical processor ALREADY generates this:
//   ComputeOpticalMesh() vertex[i].position = cluster position
//   ComputeOpticalMesh() vertex[i].sh[9]    = cluster compute state
//   OpticalNode.neighbors = cluster interconnects
//
// π-phase scheduling maps to K'UHUL phases:
//   phase=0     Pop    initialize cluster memory
//   phase=π/4   Wo     bind input tensors to clusters
//   phase=π/2   Sek    execute iGPU dispatch (wave propagation)
//   phase=3π/4  Ch'en  collect results, update ARC weights
//   phase=π     Xul    close fold, emit updated SVG3D tensors
//
// Vector green screen = the spherical manifold as infinite coordinate plane.
// Every point on S² is addressable by (θ,φ). The green screen IS the sphere.

const PI = Math.PI;

// ─── SVG3D Compute Axioms ────────────────────────────────────────────────────

export const AXIOMS = Object.freeze({
  SVG_AS_COMPUTE:   'SVG-3D elements are compute nodes, not graphics',
  IGPU_AS_ARRAY:    'iGPU is a 2D array of compute units — perfect for optical nodes',
  CPU_AS_CONTROL:   'CPU handles π-scheduling and cluster coordination',
  CIRCLE_IS_CLUSTER: 'circle = compute cluster (32 optical nodes)',
  PATH_IS_FLOW:     'path = data flow between clusters (geodesic distance)',
  GROUP_IS_TENSOR:  'group = tensor core (nested SH band computation)',
  TORUS_IS_MEMORY:  'torus = memory tier (L1=π MB, L2=π² MB, L3=π³ MB)',
  GREEN_IS_SPHERE:  'vector green screen = S² coordinate plane, all points addressable',
});

// ─── π-Phase schedule ─────────────────────────────────────────────────────────

export const PI_PHASES = Object.freeze({
  INIT:     0,           // Pop: allocate cluster memory
  BIND:     PI / 4,      // Wo:  bind input tensors
  EXECUTE:  PI / 2,      // Sek: iGPU dispatch
  COLLECT:  3 * PI / 4,  // Ch'en: gather results
  CLOSE:    PI,          // Xul: emit output tensors
});

// ─── SVG3D Compute Node ───────────────────────────────────────────────────────
// Maps directly to an optical node from ComputeOpticalMesh()

export class SVG3DComputeNode {
  /**
   * @param {number[]}  position  [x,y,z] on the spherical manifold (= vertex.position)
   * @param {Float32Array} sh     SH wave state [9 bands × 2] (= vertex.sh)
   * @param {number[]}  neighbors adjacent cluster indices
   * @param {number}    phase     current π-phase of this cluster
   */
  constructor(position, sh, neighbors = [], phase = 0) {
    this.position  = position;
    this.sh        = sh instanceof Float32Array ? sh : new Float32Array(sh);
    this.neighbors = neighbors;
    this.phase     = phase;
    this.energy    = this._energy();
    this._instrQueue = [];   // pending compute instructions
  }

  _energy() {
    let e = 0;
    for (let i = 0; i < 9; i++) {
      const c = this.sh[i*2], s = this.sh[i*2+1];
      e += Math.sqrt(c*c + s*s);
    }
    return e / 9;
  }

  // Issue a compute instruction to this cluster
  // Maps to: [Sek ⊗ instruction] in K'UHUL
  issue(opcode, params = {}) {
    this._instrQueue.push({ opcode, params, phase: this.phase });
    return this;
  }

  // Execute all pending instructions — one π/2 compute tick
  // Maps to: sh_propagate.hlsl dispatched on this node
  execute(allNodes) {
    for (const instr of this._instrQueue) {
      switch (instr.opcode) {
        case 'PROPAGATE': this._propagate(allNodes, instr.params); break;
        case 'INJECT':    this._inject(instr.params); break;
        case 'SCALE':     this._scale(instr.params); break;
        case 'ROTATE':    this._rotatePhase(instr.params); break;
      }
    }
    this._instrQueue = [];
    this.energy = this._energy();
    this.phase  = (this.phase + PI / 4) % (2 * PI);
    return this;
  }

  // OP_PROPAGATE: mix SH state with neighbours (mirrors sh_propagate.hlsl)
  _propagate(allNodes, { decay = 0.99, coupling = 0.1, selfWeight = 0.7 } = {}) {
    const next = new Float32Array(18);
    for (let band = 0; band < 9; band++) {
      let rc = this.sh[band*2]   * selfWeight;
      let rs = this.sh[band*2+1] * selfWeight;
      for (const nbIdx of this.neighbors) {
        const nb = allNodes[nbIdx];
        if (nb) { rc += nb.sh[band*2] * coupling; rs += nb.sh[band*2+1] * coupling; }
      }
      // Phase rotation per harmonic order (l)
      const l = Math.floor(Math.sqrt(band));
      const angle = l * 0.1;
      const c = Math.cos(angle), s = Math.sin(angle);
      const rc2 = rc*c - rs*s, rs2 = rc*s + rs*c;
      // Cross-band coupling
      const nb2 = (band+1) % 9;
      let mc = rc2 + this.sh[nb2*2]*0.02;
      let ms = rs2 + this.sh[nb2*2+1]*0.02;
      // Normalize + inject + decay
      const len = Math.sqrt(mc*mc + ms*ms) || 1e-5;
      mc = mc/len * Math.min(len, 1) + 0.002;
      ms = ms/len * Math.min(len, 1) + 0.001;
      next[band*2]   = mc * decay;
      next[band*2+1] = ms * decay;
    }
    this.sh = next;
  }

  // OP_INJECT: add energy to a specific SH band
  _inject({ band = 0, amplitude = 1.0, phase: p = 0 } = {}) {
    if (band < 9) {
      this.sh[band*2]   += amplitude * Math.cos(p);
      this.sh[band*2+1] += amplitude * Math.sin(p);
    }
  }

  _scale({ factor = 1.0 } = {}) {
    for (let i = 0; i < 18; i++) this.sh[i] *= factor;
  }

  _rotatePhase({ delta = PI / 4 } = {}) {
    this.phase = (this.phase + delta) % (2 * PI);
  }
}

// ─── SVG3D Compute Graph ──────────────────────────────────────────────────────
// The full cluster topology — one graph = one iGPU dispatch batch

export class SVG3DComputeGraph {
  constructor() {
    this._nodes = [];       // SVG3DComputeNode[]
    this._tick  = 0;
    this._phase = PI_PHASES.INIT;
    this._arcBias = null;   // optional: loaded from geodesic_attention_bridge
  }

  // Pop: load from ComputeOpticalMesh() output
  // vertices from optical-mesh.js directly become compute nodes
  static fromOpticalMesh(vertices, indices = []) {
    const g = new SVG3DComputeGraph();

    // Build adjacency from indices (triangle list)
    const adj = new Array(vertices.length).fill(null).map(() => new Set());
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i], indices[i+1], indices[i+2]];
      adj[a].add(b); adj[a].add(c);
      adj[b].add(a); adj[b].add(c);
      adj[c].add(a); adj[c].add(b);
    }

    for (let i = 0; i < vertices.length; i++) {
      const v = vertices[i];
      g._nodes.push(new SVG3DComputeNode(
        v.position,
        v.sh ?? new Float32Array(18),
        [...adj[i]],
        (i / vertices.length) * 2 * PI   // staggered π-phase
      ));
    }

    console.log(`[SVG3D] Graph loaded: ${g._nodes.length} compute nodes from optical mesh`);
    return g;
  }

  // Parse SVG3D XML string into compute graph
  // <circle> → cluster node, <path> → data flow, <group> → tensor core
  static fromSVG3D(xmlString) {
    const g = new SVG3DComputeGraph();
    if (typeof DOMParser === 'undefined') return g;  // Node.js: skip DOM parse

    const doc = new DOMParser().parseFromString(xmlString, 'application/xml');

    // Extract circle elements → compute nodes
    for (const el of doc.querySelectorAll('circle, sphere')) {
      const cx = parseFloat(el.getAttribute('cx') ?? 0);
      const cy = parseFloat(el.getAttribute('cy') ?? 0);
      const cz = parseFloat(el.getAttribute('cz') ?? 0);
      const ph = parseFloat(el.getAttribute('data-phase') ?? 0);
      const cores = parseInt(el.getAttribute('data-cores') ?? 32);
      const sh = new Float32Array(18).fill(0);
      // Seed SH band 0 with core count signal
      sh[0] = cores / 32.0;
      g._nodes.push(new SVG3DComputeNode([cx, cy, cz], sh, [], ph));
    }

    console.log(`[SVG3D] Parsed: ${g._nodes.length} cluster nodes from SVG3D`);
    return g;
  }

  get nodes()  { return this._nodes; }
  get size()   { return this._nodes.length; }
  get phase()  { return this._phase; }
  get tick()   { return this._tick; }

  // Wo: inject energy into a cluster (OP_INJECT)
  inject(nodeIdx, band, amplitude, phase = 0) {
    const n = this._nodes[nodeIdx];
    if (n) n.issue('INJECT', { band, amplitude, phase });
    return this;
  }

  // Sek: dispatch one compute tick across all clusters (OP_PROPAGATE)
  // Mirrors iGPU dispatch of sh_propagate.hlsl over all nodes
  dispatch(steps = 1) {
    for (let s = 0; s < steps; s++) {
      // Phase π/4 → π/2: issue PROPAGATE to all nodes
      for (const node of this._nodes) node.issue('PROPAGATE', {});
      // Execute all (mirrors parallel GPU dispatch)
      for (const node of this._nodes) node.execute(this._nodes);
      this._tick++;
    }
    this._phase = PI_PHASES.COLLECT;
    return this;
  }

  // Ch'en: collect compute results as updated mesh tensors
  collect() {
    this._phase = PI_PHASES.CLOSE;
    return this._nodes.map((n, i) => ({
      index:    i,
      position: n.position,
      sh:       n.sh,
      energy:   n.energy,
      phase:    n.phase,
      // SVG3D semantic: energy maps to circle radius
      radius:   Math.max(0.01, n.energy * PI),
      // π-phase maps to hue for visual debugging
      color:    `hsl(${(n.phase / (2*PI)) * 360}, 80%, 50%)`,
    }));
  }

  // Xul: emit updated SVG3D tensor (compute result as SVG3D XML)
  emitSVG3D() {
    const tensors = this.collect();
    const circles = tensors.map(t =>
      `  <circle cx="${t.position[0].toFixed(4)}" `+
      `cy="${t.position[1].toFixed(4)}" `+
      `cz="${t.position[2].toFixed(4)}" `+
      `r="${t.radius.toFixed(4)}" `+
      `data-energy="${t.energy.toFixed(6)}" `+
      `data-phase="${t.phase.toFixed(4)}" `+
      `data-sh-band0="${t.sh[0].toFixed(6)}" />`
    ).join('\n');

    return `<?xml version="1.0" encoding="UTF-8"?>
<svg-3d-compute xmlns="https://schema.kuhul.os/svg-compute" tick="${this._tick}">
  <!-- K'UHUL Compute Output: ${this.size} optical nodes after ${this._tick} ticks -->
  <!-- Each circle IS a compute cluster, not a visual element -->
  <!-- energy = SH coherence, phase = π-phase, r = processing capacity -->
<cluster-nodes>
${circles}
</cluster-nodes>
</svg-3d-compute>`;
  }

  // Global coherence across all clusters
  coherence() {
    if (!this._nodes.length) return 0;
    return this._nodes.reduce((s, n) => s + n.energy, 0) / this._nodes.length;
  }

  summary() {
    return {
      nodes:    this.size,
      tick:     this._tick,
      phase:    this._phase,
      coherence: this.coherence(),
      axioms:   Object.values(AXIOMS),
    };
  }
}

// ─── SVG3D Compiler ───────────────────────────────────────────────────────────
// Maps SVG3D graph → K'UHUL program → iGPU dispatch sequence
// Mirrors Fold2DCompiler.h but for the optical compute path

export class SVG3DCompiler {
  constructor(graph) {
    this._graph = graph;
    this._program = [];   // compiled K'UHUL instruction sequence
  }

  // Pop: initialize — allocate cluster memory (π-phase 0)
  init() {
    this._program.push({ phase: PI_PHASES.INIT, op: 'ALLOC',
      clusters: this._graph.size, memory_tier: 'L1' });
    return this;
  }

  // Wo: bind input tensors (π-phase π/4)
  bind(inputMap = {}) {
    for (const [nodeIdx, data] of Object.entries(inputMap)) {
      this._program.push({ phase: PI_PHASES.BIND, op: 'INJECT',
        node: parseInt(nodeIdx), ...data });
    }
    return this;
  }

  // Sek: compile dispatch sequence (π-phase π/2)
  // Maps each OP_PROPAGATE step to iGPU thread dispatch
  compileDispatch(steps = 5) {
    for (let i = 0; i < steps; i++) {
      this._program.push({
        phase: PI_PHASES.EXECUTE + (i / steps) * (PI_PHASES.COLLECT - PI_PHASES.EXECUTE),
        op: 'PROPAGATE',
        step: i,
        // iGPU mapping: numthreads matches sh_propagate.hlsl [numthreads(256,1,1)]
        igpu: { numthreads: [256, 1, 1], dispatch: [Math.ceil(this._graph.size / 256), 1, 1] },
      });
    }
    return this;
  }

  // Ch'en + Xul: collect and close (π-phase 3π/4 → π)
  finalize() {
    this._program.push({ phase: PI_PHASES.COLLECT, op: 'COLLECT' });
    this._program.push({ phase: PI_PHASES.CLOSE,   op: 'EMIT_SVG3D' });
    return this;
  }

  // Execute the compiled program
  run() {
    console.log(`[SVG3D] Executing ${this._program.length} instructions`);
    for (const instr of this._program) {
      switch (instr.op) {
        case 'ALLOC':     /* memory already allocated in graph */   break;
        case 'INJECT':    this._graph.inject(instr.node, instr.band, instr.amplitude, instr.phase); break;
        case 'PROPAGATE': this._graph.dispatch(1); break;
        case 'COLLECT':   /* fall through */ break;
        case 'EMIT_SVG3D':/* emit handled separately */  break;
      }
    }
    return this._graph.emitSVG3D();
  }

  get program() { return [...this._program]; }
}

// ─── π-Phase scheduler ────────────────────────────────────────────────────────
// CPU coordination — parcels work to iGPU by π-phase order
// Mirrors cpu-coordination section of the SVG3D compute spec

export class PiPhaseScheduler {
  constructor(graph) {
    this._graph    = graph;
    this._barriers = new Map();  // phase → barrier count
  }

  // Schedule cluster at specific π-phase
  schedule(nodeIdx, phase, opcode, params = {}) {
    const node = this._graph.nodes[nodeIdx];
    if (!node) return this;
    // Cluster executes when its natural phase aligns with scheduled phase
    if (Math.abs(node.phase - phase) < PI / 8) {
      node.issue(opcode, params);
    }
    return this;
  }

  // π-barrier: wait for all clusters at phase multiples of π
  barrier(phase) {
    const key = Math.round(phase / (PI/4));
    this._barriers.set(key, (this._barriers.get(key) ?? 0) + 1);
    // Execute pending instructions for all nodes at this phase
    for (const node of this._graph.nodes) {
      if (node._instrQueue.length > 0) node.execute(this._graph.nodes);
    }
    return this;
  }

  // Full π-cycle: Pop → Wo → Sek → Ch'en → Xul
  cycle(injectMap = {}) {
    // Pop (phase 0): initialize
    this.barrier(PI_PHASES.INIT);

    // Wo (phase π/4): inject inputs
    for (const [idx, data] of Object.entries(injectMap)) {
      this.schedule(parseInt(idx), PI_PHASES.BIND, 'INJECT', data);
    }
    this.barrier(PI_PHASES.BIND);

    // Sek (phase π/2): execute propagation
    this._graph.dispatch(1);
    this.barrier(PI_PHASES.EXECUTE);

    // Ch'en + Xul (phase 3π/4 → π): collect and close
    this.barrier(PI_PHASES.COLLECT);
    this.barrier(PI_PHASES.CLOSE);

    return this._graph.collect();
  }

  stats() {
    return { barriers: Object.fromEntries(this._barriers),
             coherence: this._graph.coherence() };
  }
}
