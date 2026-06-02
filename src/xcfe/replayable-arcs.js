// replayable-arcs.js — Replayable ARCs: practice for AI on the spherical map
//
// An ARC = a recorded geodesic trajectory on the semantic sphere.
// It can be REPLAYED: navigated multiple times, optimised with each replay.
// Like GPS routes the model has driven before — each replay finds a better line.
//
// What replaying does:
//   entropy along path decreases (fog clears where you've been)
//   path shortens toward geodesic ideal (less deviation each time)
//   model builds a LIBRARY of optimal routes between semantic locations
//
// K'UHUL phase mapping:
//   Pop    = record ARC from this training trajectory
//   Wo     = select ARC to replay (priority / uncertainty / novelty)
//   Sek    = replay ARC on manifold (with optional exploration noise)
//   Ch'en  = evaluate replay quality, update entropy, optimise path
//   Xul    = store improved ARC in library, update priority
//
// Grand unified theory:
//   SphericalMap        WHERE things are
//   GeodesicWeights     HOW to navigate
//   EntropicWeights     HOW SURE you are
//   ReplayableARCs      HOW TO IMPROVE (practice)

import { SphericalManifold } from './geo-weights.js';

// ─── ReplayableArc ────────────────────────────────────────────────────────────

export class ReplayableArc {
  constructor(id, start, end, opts = {}) {
    this.id            = id;
    this.start         = [...start];
    this.end           = [...end];
    this.path          = opts.path ?? [];        // recorded waypoints
    this.entropies     = opts.entropies ?? [];   // entropy at each waypoint
    this.replayCount   = 0;
    this.outcomes      = [];
    this.bestQuality   = 0;
    this.initialQuality = 0;
    this.bestPath      = null;
    this.createdAt     = Date.now();
    this.lastReplayAt  = null;
    this._manifold     = opts.manifold ?? null;
  }

  /** Replay: traverse the arc with optional exploration noise.
   *  Returns {trajectory, quality}. */
  replay(opts = {}) {
    const noise = opts.explorationNoise ?? 0;
    const mani  = this._manifold;

    this.replayCount++;
    this.lastReplayAt = Date.now();

    const base    = this.path.length > 0 ? this.path : this._interpolate(50);
    const trajectory = base.map((p, i) => {
      const ent = this.entropies[i] ?? 0.5;
      const perturb = noise * ent;
      if (perturb === 0 || !mani) return [...p];
      // Perturb in tangent direction
      const tangent = p.map(x => x + (Math.random()-0.5)*perturb);
      return mani.expMap(p, tangent.map((v,k) => v - p[k]));
    });

    const quality = this._evalQuality(trajectory);

    if (quality > this.bestQuality) {
      this.bestQuality = quality;
      this.bestPath    = trajectory;
    }
    if (this.replayCount === 1) this.initialQuality = quality;

    this.outcomes.push({ replay: this.replayCount, quality, ts: Date.now() });
    return { trajectory, quality };
  }

  _interpolate(steps) {
    if (!this._manifold) {
      return Array.from({ length: steps }, (_, i) => {
        const t = i / (steps-1);
        return this.start.map((s, k) => s + t * (this.end[k]-s));
      });
    }
    const mani = this._manifold;
    return Array.from({ length: steps }, (_, i) => {
      const t = i / (steps-1);
      const dir = mani.logMap(this.start, this.end).map(v=>v*t);
      return mani.expMap(this.start, dir);
    });
  }

  _evalQuality(trajectory) {
    if (trajectory.length < 2) return 0;
    let length = 0, smoothness = 0, meanEnt = 0;
    const mani = this._manifold;
    for (let i = 1; i < trajectory.length; i++) {
      const d = mani
        ? mani.geodesicDistance(trajectory[i-1], trajectory[i])
        : Math.sqrt(trajectory[i].reduce((s,v,k)=>s+(v-trajectory[i-1][k])**2,0));
      length += d;
    }
    for (let i = 0; i < this.entropies.length; i++) meanEnt += this.entropies[i] ?? 0.5;
    meanEnt /= Math.max(1, this.entropies.length);
    const lengthScore   = 1 / (1 + length);
    const entropyScore  = 1 - meanEnt;
    return 0.5 * lengthScore + 0.5 * entropyScore;
  }

  improvement() { return this.bestQuality - this.initialQuality; }

  summary() {
    return { id: this.id, replays: this.replayCount, bestQuality: +this.bestQuality.toFixed(4),
             improvement: +this.improvement().toFixed(4),
             meanEntropy: +(this.entropies.reduce((s,v)=>s+v,0)/Math.max(1,this.entropies.length)).toFixed(3) };
  }
}

// ─── SphericalReplayBuffer ────────────────────────────────────────────────────

export class SphericalReplayBuffer {
  constructor(capacity = 10_000) {
    this._capacity = capacity;
    this._arcs     = new Map();  // id → ReplayableArc
  }

  get size()  { return this._arcs.size; }
  get(id)     { return this._arcs.get(id); }
  all()       { return [...this._arcs.values()]; }

  record(id, start, end, opts = {}) {
    const arc = new ReplayableArc(id, start, end, opts);
    this._arcs.set(id, arc);
    if (this._arcs.size > this._capacity) {
      // Evict lowest-quality ARC
      const worst = [...this._arcs.values()].reduce((a,b) => a.bestQuality < b.bestQuality ? a : b);
      this._arcs.delete(worst.id);
    }
    return arc;
  }

  /** Select an ARC to replay based on strategy. */
  select(strategy = 'priority') {
    const arcs = this.all();
    if (!arcs.length) return null;
    switch (strategy) {
      case 'priority':
        // High quality + high entropy = replay (explore profitable uncertainty)
        return arcs.reduce((best, a) => {
          const score = a.bestQuality + 0.3 * (a.entropies.reduce((s,v)=>s+v,0)/Math.max(1,a.entropies.length));
          const bScore = best.bestQuality + 0.3 * (best.entropies.reduce((s,v)=>s+v,0)/Math.max(1,best.entropies.length));
          return score > bScore ? a : best;
        });
      case 'uncertainty':
        return arcs.reduce((best, a) => {
          const me = a.entropies.reduce((s,v)=>s+v,0)/Math.max(1,a.entropies.length);
          const bme = best.entropies.reduce((s,v)=>s+v,0)/Math.max(1,best.entropies.length);
          return me > bme ? a : best;
        });
      case 'novelty':
        return arcs.reduce((best,a) => a.replayCount < best.replayCount ? a : best);
      case 'improvement':
        return arcs.reduce((best,a) => a.improvement() > best.improvement() ? a : best);
      default:
        return arcs[Math.floor(Math.random()*arcs.length)];
    }
  }

  /** Replay a batch of n ARCs in parallel (returns array of results). */
  async replayBatch(n = 32, opts = {}) {
    const results = [];
    for (let i = 0; i < n; i++) {
      const arc = this.select(opts.strategy ?? 'priority');
      if (!arc) break;
      results.push({ arc, ...arc.replay(opts) });
    }
    return results;
  }

  /** Clear entropy along successful paths. */
  clearFogFromSuccesses(entropyField, minQuality = 0.6, reduction = 0.05) {
    for (const arc of this.all()) {
      if (arc.bestQuality >= minQuality && arc.bestPath) {
        // Find nearest indices in the entropy field (simplified: use path length)
        const n = arc.bestPath.length;
        for (let i = 0; i < n; i++) {
          const idx = Math.floor(i / n * entropyField.size);
          entropyField.values[idx] = Math.max(0, entropyField.values[idx] * (1-reduction));
        }
        // Also clear the arc's own entropies
        arc.entropies = arc.entropies.map(e => Math.max(0, e*(1-reduction)));
      }
    }
  }

  stats() {
    const arcs = this.all();
    if (!arcs.length) return { count: 0 };
    const totalReplays  = arcs.reduce((s,a)=>s+a.replayCount,0);
    const bestArc       = arcs.reduce((b,a)=>a.bestQuality>b.bestQuality?a:b);
    const mostImproved  = arcs.reduce((b,a)=>a.improvement()>b.improvement()?a:b);
    return {
      count: arcs.length, totalReplays,
      avgReplays: (totalReplays/arcs.length).toFixed(1),
      bestArc: bestArc.summary(),
      mostImproved: mostImproved.summary(),
    };
  }
}

// ─── ArcLearning ──────────────────────────────────────────────────────────────

export class ArcLearning {
  constructor(manifold, buffer, entropyField = null) {
    this._manifold     = manifold;
    this._buffer       = buffer;
    this._entropyField = entropyField;
  }

  async learnFromReplays(epochs = 5, opts = {}) {
    const history = [];
    for (let ep = 0; ep < epochs; ep++) {
      const batch   = await this._buffer.replayBatch(opts.batchSize ?? 64, opts);
      const avgQ    = batch.reduce((s,r)=>s+r.quality,0) / Math.max(1,batch.length);

      // Clear fog along successful paths
      if (this._entropyField)
        this._buffer.clearFogFromSuccesses(this._entropyField, opts.minQuality ?? 0.6);

      history.push({ epoch: ep+1, avgQuality: avgQ, bufferStats: this._buffer.stats() });
    }
    return history;
  }
}

// ─── ArcLibrary ───────────────────────────────────────────────────────────────
//
// High-level: name → ARC lookup, with automatic manifold assignment.

export class ArcLibrary {
  constructor(manifold, opts = {}) {
    this._manifold = manifold;
    this._buffer   = new SphericalReplayBuffer(opts.capacity ?? 10_000);
    this._learning = null;
  }

  record(name, start, end, entropies = []) {
    return this._buffer.record(name, start, end, {
      manifold: this._manifold,
      path:     [],
      entropies,
    });
  }

  replay(name, opts = {}) {
    const arc = this._buffer.get(name);
    if (!arc) throw new Error(`ArcLibrary: ARC not found: ${name}`);
    return arc.replay(opts);
  }

  async practice(epochs = 3, opts = {}) {
    const learning = new ArcLearning(this._manifold, this._buffer, opts.entropyField);
    return learning.learnFromReplays(epochs, opts);
  }

  best()  { return this._buffer.select('improvement'); }
  foggy() { return this._buffer.select('uncertainty'); }
  novel() { return this._buffer.select('novelty'); }

  stats() { return this._buffer.stats(); }

  /**
   * Human-readable fog report — what does the model not know yet?
   * High-entropy ARCs = foggy routes the model should explore next.
   */
  fogReport() {
    return this._buffer.all()
      .map(a => ({
        id: a.id,
        meanEntropy: +(a.entropies.reduce((s,v)=>s+v,0)/Math.max(1,a.entropies.length)).toFixed(3),
        replays: a.replayCount,
        quality: +a.bestQuality.toFixed(3),
      }))
      .sort((a,b) => b.meanEntropy - a.meanEntropy)
      .slice(0, 10);
  }
}
