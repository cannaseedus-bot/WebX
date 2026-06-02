// geo-ir.js — Geometric Tensor Intermediate Representation
//
// A geometric execution substrate with a deterministic control grammar.
// Not graphics. Not visualization. Geometry as computation.
//
// TWO PLANES, SEPARATED:
//
//   Control Plane (K'uhul grammar):
//     Pop       enter scope / begin fold
//     Xul       exit scope / end fold
//     Wo        allocate tensor in manifold M
//     Yax       read tensor from M
//     Ch'en     write tensor to M
//     Sek       apply geometric operator
//     K'ayab'   begin phase iteration
//     Kumk'u    end phase iteration
//     Muwan     invoke folded process
//
//   State Plane (SVG-3D storage):
//     <circle>  point cloud in ℝⁿ   (cx,cy → position; r → norm/density)
//     <path>    adjacency / flow     (d → geodesic topology)
//     <g>       composite fold       (transform → manifold mapping)
//     No fill/stroke/opacity in compute context.
//     viewBox = coordinate manifold bounds.
//
// Manifold M = ℝ² (or ℝ³):
//   All tensors Tᵢ embedded in M.
//   Operations: ⊗ (product) ⊕ (composition) ⊖ (difference) ⊘ (scale) ⊛ ⊜ ⊝ ⊞
//   Execution is geometric, not pixel-based.
//   Rendering is optional projection from M — separate concern.
//
// FLUX IR connection:
//   GeoIR ops are pure functions (Layer 0 — timeless)
//   K'uhul phase program is the FLUX IR reducer body
//   M is the FLUX store; Yax/Ch'en are get/set
//   Phase cycle = Pop→Wo→Sek→Ch'en as geometric steps in M

// ─── Manifold M ───────────────────────────────────────────────────────────────

export class Manifold {
  /**
   * @param {number} dim  embedding dimension (2 for ℝ², 3 for ℝ³)
   */
  constructor(dim = 2) {
    this.dim     = dim;
    this._tensors = new Map(); // id -> GeoTensor
    this._phase   = 0;         // current phase in [0, 2π)
  }

  // ── Tensor storage (Wo / Yax / Ch'en) ──

  allocate(id, coords, norm = 1.0, rank = 0, meta = {}) {
    const t = new GeoTensor(id, coords.slice(0, this.dim), norm, rank, meta);
    this._tensors.set(id, t);
    return t;
  }

  read(id)           { return this._tensors.get(id) ?? null; }         // Yax
  write(id, tensor)  { this._tensors.set(id, tensor); return this; }   // Ch'en
  has(id)            { return this._tensors.has(id); }
  all()              { return [...this._tensors.values()]; }

  // ── Phase progression ──

  advancePhase(delta = Math.PI / 4) {
    this._phase = (this._phase + delta) % (2 * Math.PI);
    return this._phase;
  }

  get phase() { return this._phase; }

  // ── Geometric operations in M (Sek) ──

  /** ⊗  Geometric product (tensor interaction in M). */
  product(a, b) {
    const ta = this.read(a), tb = this.read(b);
    if (!ta || !tb) return null;
    const coords = ta.coords.map((v, i) => v * (tb.coords[i] ?? 1));
    return new GeoTensor(`${a}⊗${b}`, coords, ta.norm * tb.norm, ta.rank + tb.rank);
  }

  /** ⊕  Manifold composition (overlay / sum). */
  compose(a, b) {
    const ta = this.read(a), tb = this.read(b);
    if (!ta || !tb) return null;
    const coords = ta.coords.map((v, i) => v + (tb.coords[i] ?? 0));
    return new GeoTensor(`${a}⊕${b}`, coords, (ta.norm + tb.norm) / 2, Math.max(ta.rank, tb.rank));
  }

  /** ⊖  Difference in M. */
  diff(a, b) {
    const ta = this.read(a), tb = this.read(b);
    if (!ta || !tb) return null;
    const coords = ta.coords.map((v, i) => v - (tb.coords[i] ?? 0));
    return new GeoTensor(`${a}⊖${b}`, coords, Math.abs(ta.norm - tb.norm), ta.rank);
  }

  /** ⊘  Scaling relative to M. */
  scale(a, scalar) {
    const ta = this.read(a);
    if (!ta) return null;
    return new GeoTensor(`${a}⊘${scalar}`, ta.coords.map(v => v * scalar), ta.norm * Math.abs(scalar), ta.rank);
  }

  /** ⊛  Hadamard-like element product. */
  hadamard(a, b) {
    const ta = this.read(a), tb = this.read(b);
    if (!ta || !tb) return null;
    const coords = ta.coords.map((v, i) => v * (tb.coords[i] ?? 1));
    return new GeoTensor(`${a}⊛${b}`, coords, ta.norm, ta.rank);
  }

  /** ⊜  Constraint check: does tensor satisfy constraint set? */
  satisfies(id, constraint) {
    const t = this.read(id);
    if (!t) return false;
    return constraint(t);
  }

  /** ⊝  Projection onto subspace (e.g. ReLU = non-negative subspace). */
  project(a, fn) {
    const ta = this.read(a);
    if (!ta) return null;
    const coords = ta.coords.map(fn);
    return new GeoTensor(`${a}⊝`, coords, Math.sqrt(coords.reduce((s,v)=>s+v*v,0)), ta.rank);
  }

  /** ⊞  Translation (bias add). */
  translate(a, delta) {
    const ta = this.read(a);
    if (!ta) return null;
    const coords = ta.coords.map((v, i) => v + (delta[i] ?? delta[0] ?? 0));
    return new GeoTensor(`${a}⊞`, coords, ta.norm, ta.rank);
  }

  // ── Geodesic distance between two tensors in M ──

  geodesicDist(a, b) {
    const ta = this.read(a), tb = this.read(b);
    if (!ta || !tb) return Infinity;
    const d2 = ta.coords.reduce((s, v, i) => s + (v - (tb.coords[i] ?? 0)) ** 2, 0);
    return Math.sqrt(d2);
  }

  // ── Schema extraction (compression invariant) ──

  schema() {
    return {
      dim:      this.dim,
      tensorIds: [...this._tensors.keys()],
      bounds:   this._bounds(),
      phase:    this._phase,
    };
  }

  /** Delta-encode tensors against a prior snapshot (natural ~90% compression). */
  deltaEncode(prior) {
    const deltas = {};
    for (const [id, t] of this._tensors) {
      const prev = prior?._tensors?.get(id);
      if (!prev) { deltas[id] = t.toJSON(); continue; }
      const dCoords = t.coords.map((v, i) => v - prev.coords[i]);
      const dNorm   = t.norm - prev.norm;
      if (dCoords.every(d => Math.abs(d) < 1e-9) && Math.abs(dNorm) < 1e-9) continue;
      deltas[id] = { coords: dCoords, norm: dNorm };
    }
    return deltas;
  }

  _bounds() {
    if (this._tensors.size === 0) return null;
    const all = this.all();
    const min = all[0].coords.map(() => Infinity);
    const max = all[0].coords.map(() => -Infinity);
    for (const t of all) t.coords.forEach((v, i) => { min[i] = Math.min(min[i], v); max[i] = Math.max(max[i], v); });
    return { min, max };
  }
}

// ─── GeoTensor ────────────────────────────────────────────────────────────────
//
// A tensor embedded in manifold M.
// SVG-3D serialisation: <circle cx cy r data-rank data-*>

export class GeoTensor {
  constructor(id, coords, norm = 1.0, rank = 0, meta = {}) {
    this.id     = id;
    this.coords = coords.slice(); // position in M
    this.norm   = norm;           // tensor norm / density
    this.rank   = rank;           // tensor rank (0=scalar, 1=vector, 2=matrix, ...)
    this.meta   = meta;           // shape, constraints, phase group, etc.
    this.phase  = meta.phase ?? 0;
  }

  /** Geodesic distance to another tensor. */
  distTo(other) {
    return Math.sqrt(this.coords.reduce((s, v, i) => s + (v - (other.coords[i] ?? 0)) ** 2, 0));
  }

  toJSON() {
    return { id: this.id, coords: this.coords, norm: this.norm, rank: this.rank,
             phase: this.phase, meta: this.meta };
  }

  /** Serialise to SVG-3D <circle> element string. */
  toSVGCircle() {
    const [cx = 0, cy = 0, cz = 0] = this.coords;
    return `<circle cx="${cx.toFixed(4)}" cy="${cy.toFixed(4)}" r="${this.norm.toFixed(4)}" `
         + `data-id="${this.id}" data-rank="${this.rank}" data-phase="${this.phase.toFixed(4)}" `
         + (cz !== 0 ? `data-cz="${cz.toFixed(4)}" ` : '')
         + `/>`;
  }

  static fromSVGElement(el) {
    const cx = parseFloat(el.getAttribute('cx') ?? '0');
    const cy = parseFloat(el.getAttribute('cy') ?? '0');
    const cz = parseFloat(el.getAttribute('data-cz') ?? '0');
    const r  = parseFloat(el.getAttribute('r') ?? '1');
    return new GeoTensor(
      el.getAttribute('data-id') ?? 'unknown',
      [cx, cy, cz].slice(0, cz !== 0 ? 3 : 2),
      r,
      parseInt(el.getAttribute('data-rank') ?? '0')
    );
  }
}

// ─── K'uhul control grammar executor ──────────────────────────────────────────
//
// Deterministic phase-aware execution of K'uhul programs against a Manifold.
// Each instruction maps to an exact operation in M.

export class KuhulExecutor {
  constructor(manifold) {
    this._m     = manifold;
    this._stack = []; // fold scope stack
    this._log   = [];
  }

  /** Execute a K'uhul instruction against the manifold. */
  exec(op, args = {}) {
    const result = this._dispatch(op, args);
    this._log.push({ op, args, result, phase: this._m.phase });
    return result;
  }

  _dispatch(op, args) {
    switch (op) {
      case 'Pop':    this._stack.push({ id: args.id, phase: this._m.phase }); return 'entered';
      case 'Xul':    return this._stack.pop() ?? null;
      case 'Wo':     return this._m.allocate(args.id, args.coords ?? [0,0], args.norm ?? 1, args.rank ?? 0, args.meta ?? {});
      case 'Yax':    return this._m.read(args.id);
      case "Ch'en":  return this._m.write(args.id, args.tensor);
      case 'Sek':    return this._applySekOp(args);
      case "K'ayab'":return { loop: args.id, phase: this._m.phase };
      case 'Kumk\'u':return this._m.advancePhase(args.delta ?? Math.PI / 4);
      case 'Muwan':  return args.fn ? args.fn(this._m) : null;
      default:       return null;
    }
  }

  _applySekOp(args) {
    const { op: geoOp, a, b, scalar, fn } = args;
    switch (geoOp) {
      case '⊗': return this._m.product(a, b);
      case '⊕': return this._m.compose(a, b);
      case '⊖': return this._m.diff(a, b);
      case '⊘': return this._m.scale(a, scalar ?? 1);
      case '⊛': return this._m.hadamard(a, b);
      case '⊜': return this._m.satisfies(a, fn ?? (() => true));
      case '⊝': return this._m.project(a, fn ?? (v => Math.max(0, v)));
      case '⊞': return this._m.translate(a, b ?? [0]);
      default:  return null;
    }
  }

  get log() { return this._log.slice(); }
  get stackDepth() { return this._stack.length; }
}

// ─── Phase cycle runner ───────────────────────────────────────────────────────
//
// Executes the K'uhul phase cycle deterministically against M.
// Binds to requestAnimationFrame for projection (optional — execution is independent).

export class PhaseCycle {
  constructor(manifold, executor) {
    this._m  = manifold;
    this._ex = executor;
    this._steps = 0;
    this._rafId = null;
    this._onStep = null;
  }

  /**
   * Run one full phase cycle (Pop→Wo→Sek→Ch'en→Xul).
   * @param {Array} program  array of {op, args} instructions
   */
  runCycle(program) {
    this._steps++;
    for (const { op, args } of program) this._ex.exec(op, args);
    return { step: this._steps, phase: this._m.phase };
  }

  /** Bind to rAF for optional projection — execution is independent. */
  startProjection(onFrame) {
    this._onStep = onFrame;
    const loop = () => {
      onFrame(this._m, this._steps);
      this._rafId = requestAnimationFrame(loop);
    };
    this._rafId = requestAnimationFrame(loop);
  }

  stopProjection() {
    if (this._rafId) cancelAnimationFrame(this._rafId);
    this._rafId = null;
  }
}

// ─── Neural network as geometry in M ─────────────────────────────────────────
//
// Dense layer: X⊗W ⊞ b ⊝relu
// No neurons. No layers. Just geometry in M.

export function denseLayerProgram(xId, wId, bId, outputId) {
  return [
    { op: 'Pop',  args: { id: 'dense_layer' } },
    // Wo: allocate output tensor
    { op: 'Wo',   args: { id: `${outputId}_xw`, coords: [0,0], rank: 2 } },
    { op: 'Wo',   args: { id: `${outputId}_xwb`, coords: [0,0], rank: 2 } },
    { op: 'Wo',   args: { id: outputId, coords: [0,0], rank: 2 } },
    // Sek: linear transform ⊗ W
    { op: 'Sek',  args: { op:'⊗', a: xId, b: wId } },
    { op: "Ch'en",args: { id: `${outputId}_xw`, tensor: null } }, // set by runtime
    // Sek: bias translation ⊞ b
    { op: 'Sek',  args: { op:'⊞', a: `${outputId}_xw`, b: [0] } },
    { op: "Ch'en",args: { id: `${outputId}_xwb`, tensor: null } },
    // Sek: ReLU = projection onto non-negative subspace ⊝
    { op: 'Sek',  args: { op:'⊝', a: `${outputId}_xwb`, fn: v => Math.max(0, v) } },
    { op: "Ch'en",args: { id: outputId, tensor: null } },
    { op: 'Xul',  args: {} },
  ];
}

// ─── SVG-3D tensor serialiser ─────────────────────────────────────────────────
//
// Canonical storage format for tensors in M.
// NO fill/stroke in compute context — only positional + data attributes.

export function manifestToSVG(manifold, viewBox = '-10 -10 20 20') {
  const tensors = manifold.all();
  const circles = tensors.map(t => t.toSVGCircle()).join('\n  ');
  const paths   = tensors.flatMap(t =>
    (t.meta.adjacency ?? []).map(adj =>
      `<path d="M ${t.coords[0]} ${t.coords[1]} L ${adj[0]} ${adj[1]}" `
      + `data-from="${t.id}" class="geo-edge"/>`
    )
  ).join('\n  ');
  return `<svg viewBox="${viewBox}" xmlns="http://www.w3.org/2000/svg">\n  ${circles}\n  ${paths}\n</svg>`;
}

export function svgToManifold(svgText, dim = 2) {
  const m   = new Manifold(dim);
  const doc = typeof DOMParser !== 'undefined'
    ? new DOMParser().parseFromString(svgText, 'image/svg+xml')
    : null;
  if (!doc) return m;
  for (const el of doc.querySelectorAll('circle[data-id]')) {
    const t = GeoTensor.fromSVGElement(el);
    m.write(t.id, t);
  }
  return m;
}
