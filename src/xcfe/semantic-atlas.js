// semantic-atlas.js — Multi-Chart Semantic Atlas (learnable KXML map)
//
// A semantic atlas is a collection of CHARTS — overlapping local maps
// that together cover the full semantic manifold.
//
// Each chart covers a different semantic domain:
//   hierarchy_chart   hyperbolic (Poincaré)         → tree structures, IS-A relations
//   similarity_chart  spherical  (stereographic)    → similarity clusters
//   temporal_chart    mixed curvature (equidistant)  → sequences, time, causation
//
// Charts overlap in transition regions connected by parallel transport.
// The atlas LEARNS: curvatures, chart centers, and transitions adapt from data.
//
// KXML spec (from user):
//   <geometric_map id="semantic_atlas" type="learnable" curvature="adaptive">
//     <charts> ... </charts>
//     <transitions> ... </transitions>
//     <learning> adapt_curvature, update_charts, optimize_transitions </learning>
//   </geometric_map>
//
// K'UHUL phase:
//   Pop   = load atlas from KXML, initialize charts
//   Wo    = bind charts to token positions, compute which chart each token lives in
//   Sek   = apply geodesic weights within each chart
//   Ch'en = update chart parameters (curvature, centers, transitions)
//   Xul   = emit updated atlas, record coverage metrics

import { SphericalManifold, GeodesicWeight, GeodesicAttention } from './geo-weights.js';

// ─── Manifold types ───────────────────────────────────────────────────────────

function makeManifold(type, curvature) {
  // All three chart types use the same SphericalManifold math
  // but different curvature signs and projection conventions.
  // Positive curvature  → spherical (similarity)
  // Negative curvature  → hyperbolic (hierarchy) — simulate via κ < 0
  // Mixed / near-zero   → flat + slight curvature (temporal)
  return new SphericalManifold(curvature);
}

// ─── Chart ────────────────────────────────────────────────────────────────────

export class Chart {
  constructor(id, opts = {}) {
    this.id         = id;
    this.center     = opts.center     ?? [0, 0, 1];
    this.radius     = opts.radius     ?? 0.8;
    this.curvature  = opts.curvature  ?? 0.1;
    this.projection = opts.projection ?? 'stereographic';
    this.manifold   = makeManifold(this.projection, this.curvature);
    this._tokens    = new Map();  // token_id → sphere point
  }

  /** True if point p falls within this chart's coverage region. */
  covers(p) {
    return this.manifold.geodesicDistance(p, this.center) <= this.radius;
  }

  /** Project a raw embedding into this chart's local coordinates. */
  embed(tokenId, rawEmbedding) {
    const projected = this.manifold.project(rawEmbedding);
    this._tokens.set(tokenId, projected);
    return projected;
  }

  tokenPosition(tokenId) { return this._tokens.get(tokenId) ?? null; }

  /** Geodesic distance between two tokens in this chart. */
  distance(idA, idB) {
    const a = this._tokens.get(idA), b = this._tokens.get(idB);
    if (!a || !b) return Infinity;
    return this.manifold.geodesicDistance(a, b);
  }

  /** Adapt curvature based on recent loss. */
  adaptCurvature(loss, lr = 0.01) {
    if (loss > 0.5)      this.curvature = Math.min(1.0, this.curvature * (1 + lr));
    else if (loss < 0.1) this.curvature = Math.max(0.01, this.curvature * (1 - lr));
    this.manifold = makeManifold(this.projection, this.curvature);
  }

  summary() {
    return { id: this.id, projection: this.projection,
             curvature: this.curvature, radius: this.radius,
             tokens: this._tokens.size };
  }
}

// ─── Transition ───────────────────────────────────────────────────────────────

export class ChartTransition {
  constructor(fromChart, toChart, opts = {}) {
    this.from        = fromChart;
    this.to          = toChart;
    this.holonomy    = opts.holonomy    ?? 0.05;
    this.overlapR    = opts.overlapR    ?? 0.2;
    this._learnHolo  = opts.learnHolonomy ?? true;
  }

  /** Check if a point is in the overlap region of both charts. */
  inOverlap(p) {
    return this.from.covers(p) && this.to.covers(p);
  }

  /**
   * Transport a vector v from fromChart's coordinates to toChart's coordinates.
   * The holonomy parameter captures the curvature-induced rotation.
   */
  transport(v, sourcePoint) {
    const targetPoint = this.to.center;
    // Parallel transport with holonomy correction
    const transported = this.from.manifold.parallelTransport(v, sourcePoint, targetPoint);
    // Apply holonomy rotation (curvature-induced phase shift)
    const angle = this.holonomy * this.from.manifold.geodesicDistance(sourcePoint, targetPoint);
    const axis  = targetPoint.map((x, i) => x - sourcePoint[i]);
    const an    = Math.sqrt(axis.reduce((s,x)=>s+x*x,0)) || 1;
    const aHat  = axis.map(x=>x/an);
    return this.from.manifold._rotateVector(transported, aHat, angle);
  }

  adaptHolonomy(gradHolonomy, lr = 0.001) {
    if (this._learnHolo) this.holonomy -= lr * gradHolonomy;
  }
}

// ─── SemanticAtlas ────────────────────────────────────────────────────────────

export class SemanticAtlas {
  constructor(opts = {}) {
    this.id         = opts.id ?? 'semantic_atlas';
    this.type       = opts.type ?? 'learnable';
    this._charts     = new Map();
    this._transitions = [];
    this._learnCurvature  = opts.adaptCurvature  ?? true;
    this._learnCharts     = opts.updateCharts     ?? true;
    this._learnTransitions = opts.optimizeTransitions ?? true;
    this._attention  = null;  // GeodesicAttention per chart

    // Build default three-chart atlas
    this._initDefaultCharts();
  }

  _initDefaultCharts() {
    this.addChart(new Chart('hierarchy_chart', {
      center:     [0.1, 0.2, 0.3], radius: 0.8,
      curvature:  -0.1,              // negative → hyperbolic-like
      projection: 'poincare',
    }));
    this.addChart(new Chart('similarity_chart', {
      center:     [0.7, -0.1, 0.4], radius: 0.6,
      curvature:   0.1,              // positive → spherical
      projection: 'stereographic',
    }));
    this.addChart(new Chart('temporal_chart', {
      center:     [-0.2, 0.5, -0.3], radius: 0.7,
      curvature:   0.02,             // near-flat → mild curvature for sequences
      projection: 'equidistant',
    }));

    // Transitions
    const h = this._charts.get('hierarchy_chart');
    const s = this._charts.get('similarity_chart');
    const t = this._charts.get('temporal_chart');
    if (h && s) this.addTransition(new ChartTransition(h, s, { holonomy:0.05, overlapR:0.2 }));
    if (s && t) this.addTransition(new ChartTransition(s, t, { holonomy:0.03, overlapR:0.15 }));
    if (t && h) this.addTransition(new ChartTransition(t, h, { holonomy:0.04, overlapR:0.18 }));
  }

  addChart(chart)      { this._charts.set(chart.id, chart); return this; }
  addTransition(tr)    { this._transitions.push(tr); return this; }
  chart(id)            { return this._charts.get(id); }
  charts()             { return [...this._charts.values()]; }

  /** Find which chart(s) cover a given point. */
  coveringCharts(point) {
    return this.charts().filter(c => c.covers(point));
  }

  /** Embed a token into the most appropriate chart. */
  embed(tokenId, rawEmbedding) {
    // Choose chart with closest center
    let best = null, bestDist = Infinity;
    for (const chart of this.charts()) {
      const d = chart.manifold.geodesicDistance(chart.center,
        chart.manifold.project(rawEmbedding));
      if (d < bestDist) { bestDist = d; best = chart; }
    }
    return best ? best.embed(tokenId, rawEmbedding) : null;
  }

  /** Geodesic distance between two tokens (may span charts via transitions). */
  distance(idA, idB) {
    // Try within same chart first
    for (const chart of this.charts()) {
      const a = chart.tokenPosition(idA);
      const b = chart.tokenPosition(idB);
      if (a && b) return chart.distance(idA, idB);
    }
    // Cross-chart: sum partial distances via transition overlap
    // (simplified: sum distances in each chart + transition cost)
    let total = 0, found = 0;
    for (const chart of this.charts()) {
      const a = chart.tokenPosition(idA);
      const b = chart.tokenPosition(idB);
      if (a) { total += chart.manifold.geodesicDistance(a, chart.center); found++; }
      if (b) { total += chart.manifold.geodesicDistance(b, chart.center); found++; }
    }
    return found > 0 ? total : Infinity;
  }

  /** Update all chart curvatures based on training loss. */
  adaptAll(loss) {
    if (!this._learnCurvature) return;
    for (const chart of this.charts()) chart.adaptCurvature(loss);
  }

  /** Load atlas from KXML string. */
  static fromKXML(xmlStr) {
    const atlas = new SemanticAtlas({ adaptCurvature:true, updateCharts:true, optimizeTransitions:true });
    if (typeof DOMParser === 'undefined') return atlas;
    const doc = new DOMParser().parseFromString(xmlStr, 'application/xml');
    const root = doc.querySelector('geometric_map');
    if (!root) return atlas;

    atlas.id   = root.getAttribute('id') ?? atlas.id;
    atlas.type = root.getAttribute('type') ?? atlas.type;

    // Override charts from XML
    atlas._charts.clear(); atlas._transitions = [];
    for (const el of root.querySelectorAll('chart')) {
      const center = JSON.parse(el.getAttribute('center') ?? '[0,0,1]');
      const radius = parseFloat(el.getAttribute('radius') ?? '0.8');
      const proj   = el.querySelector('projection')?.getAttribute('type') ?? 'stereographic';
      atlas.addChart(new Chart(el.getAttribute('id'), {
        center, radius, curvature: 0.1, projection: proj }));
    }

    // Transitions
    for (const el of root.querySelectorAll('transitions from_chart')) {
      const from = atlas.chart(el.getAttribute('from_chart'));
      const to   = atlas.chart(el.getAttribute('to_chart'));
      if (from && to) {
        const holo = parseFloat(el.querySelector('transport')?.getAttribute('holonomy') ?? '0.05');
        const ovR  = parseFloat(el.querySelector('overlap_region')?.getAttribute('radius') ?? '0.2');
        atlas.addTransition(new ChartTransition(from, to, { holonomy:holo, overlapR:ovR }));
      }
    }

    return atlas;
  }

  summary() {
    return {
      id:      this.id,
      charts:  this.charts().map(c => c.summary()),
      transitions: this._transitions.length,
      learnable: { curvature:this._learnCurvature, charts:this._learnCharts, transitions:this._learnTransitions },
    };
  }
}

// ─── Canonical KXML spec for the semantic atlas ───────────────────────────────

export const SEMANTIC_ATLAS_KXML = `<?xml version="1.0" encoding="utf-8"?>
<geometric_map id="semantic_atlas" type="learnable" curvature="adaptive">

  <charts>
    <chart id="hierarchy_chart" center="[0.1,0.2,0.3]" radius="0.8">
      <!-- Hyperbolic: tree structures, IS-A relations, word hierarchy -->
      <projection type="poincare"/>
      <weights geodesic="true"/>
      <curvature value="-0.1" learnable="true"/>
    </chart>

    <chart id="similarity_chart" center="[0.7,-0.1,0.4]" radius="0.6">
      <!-- Spherical: cosine similarity clusters, synonym groups -->
      <projection type="stereographic"/>
      <weights geodesic="true"/>
      <curvature value="0.1" learnable="true"/>
    </chart>

    <chart id="temporal_chart" center="[-0.2,0.5,-0.3]" radius="0.7">
      <!-- Near-flat: sequences, causal chains, time ordering -->
      <projection type="equidistant"/>
      <weights geodesic="true"/>
      <curvature value="0.02" learnable="true"/>
    </chart>
  </charts>

  <transitions>
    <transition from_chart="hierarchy_chart" to_chart="similarity_chart">
      <transport type="parallel" holonomy="0.05"/>
      <overlap_region radius="0.2"/>
    </transition>
    <transition from_chart="similarity_chart" to_chart="temporal_chart">
      <transport type="parallel" holonomy="0.03"/>
      <overlap_region radius="0.15"/>
    </transition>
    <transition from_chart="temporal_chart" to_chart="hierarchy_chart">
      <transport type="parallel" holonomy="0.04"/>
      <overlap_region radius="0.18"/>
    </transition>
  </transitions>

  <learning>
    <adapt_curvature>true</adapt_curvature>
    <update_charts>true</update_charts>
    <optimize_transitions>true</optimize_transitions>
  </learning>

</geometric_map>`;
