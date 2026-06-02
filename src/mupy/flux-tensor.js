// flux-tensor.js — Living tensor fields that evolve, not learn.
//
// The model doesn't learn. It AGES.
//
// Each forward pass = one mutation event = one "day" in the model's life.
// Each 1000 steps   = version boundary = New Year's Day.
// Snapshots         = fossil records of evolution.
//
// K'UHUL phase mapping:
//   Pop   — restore FluxTensor from snapshot (load fossil)
//   Wo    — declare intent, project with evolving Q/K/V
//   Sek   — attention + carry field injection
//   Ch'en — mutate all tensors (the aging step)
//   Xul   — emit logits, record version snapshot
//
// Attractor types (FLUXTensor chaotic evolution):
//   lorenz   sigma=10 rho=28 beta=8/3   sensitive to initial conditions
//   rossler  a=0.2    b=0.2  c=5.7      scroll-type chaos
//   logistic r=3.9                       discrete 1D chaos map
//
// FLUX IR connection:
//   FluxTensor.mutate()  = Ch'en phase reducer action
//   FLUXTensor.evolve()  = Ch'en phase reducer with chaotic attractor
//   MayanVersionBoundary = FLUX store snapshot (TimeTraveler record)
//   carry_field          = persistent store state across forward passes

// ─── FluxTensor ────────────────────────────────────────────────────────────────
//
// A Float32Array-backed living tensor. JS equivalent of the Python FluxTensor.
// No autograd — mutation IS the update rule.

export class FluxTensor {
  /**
   * @param {number[]} data      flat initial values
   * @param {number}   mutationRate  how fast it evolves per step
   * @param {number}   halfLife      days until 50% magnitude decay
   */
  constructor(data, mutationRate = 0.01, halfLife = 365) {
    this.data         = new Float32Array(data);
    this.mutationRate = mutationRate;
    this.halfLife     = halfLife;
    this.createdAt    = Date.now();
    this.mutationCount = 0;
  }

  get size()  { return this.data.length; }
  get ageMs() { return Date.now() - this.createdAt; }
  get ageDays() { return this.ageMs / 86_400_000; }

  /** Decay factor = 0.5^(age_days / half_life). */
  decayFactor() {
    return Math.pow(0.5, this.ageDays / Math.max(1, this.halfLife));
  }

  /**
   * Mutate the tensor.
   * @param {Float32Array|null} gradient  environmental pressure; null = thermal noise
   */
  mutate(gradient = null) {
    this.mutationCount++;
    const scale = this.mutationRate * this.decayFactor();
    for (let i = 0; i < this.data.length; i++) {
      const delta = gradient
        ? gradient[i % gradient.length] * scale
        : (Math.random() * 2 - 1) * scale;
      this.data[i] += delta;
    }
    return this;
  }

  /** Snapshot as plain object (fossil record). */
  snapshot() {
    return { data: Array.from(this.data), mutationRate: this.mutationRate,
             halfLife: this.halfLife, createdAt: this.createdAt,
             mutationCount: this.mutationCount };
  }

  /** Restore from snapshot. */
  static restore(snap) {
    const t = new FluxTensor(snap.data, snap.mutationRate, snap.halfLife);
    t.createdAt     = snap.createdAt;
    t.mutationCount = snap.mutationCount ?? 0;
    return t;
  }

  mean() { return this.data.reduce((s,v) => s+v, 0) / this.data.length; }
  norm() { return Math.sqrt(this.data.reduce((s,v) => s+v*v, 0)); }
}

// ─── FLUXTensor (chaotic attractor) ──────────────────────────────────────────

export class FLUXTensor extends FluxTensor {
  constructor(data, attractor = 'lorenz') {
    super(data, 0.001, 365);
    this.attractor = attractor;
    this._s = [0.1, 0.0, 0.0]; // attractor state [x,y,z]
  }

  /**
   * Evolve via chaotic attractor dynamics.
   * @param {number} steps
   */
  evolve(steps = 1) {
    let [x, y, z] = this._s;
    for (let i = 0; i < steps; i++) {
      let dx, dy, dz;
      if (this.attractor === 'lorenz') {
        const s=10, r=28, b=8/3;
        dx = s*(y-x); dy = x*(r-z)-y; dz = x*y-b*z;
      } else if (this.attractor === 'rossler') {
        const a=.2, b=.2, c=5.7;
        dx = -y-z; dy = x+a*y; dz = b+z*(x-c);
      } else { // logistic
        const r=3.9;
        dx = r*x*(1-x)-x; dy = r*y*(1-y)-y; dz = (dx+dy)/2;
      }
      x += dx*.01; y += dy*.01; z += dz*.01;
    }
    this._s = [x, y, z];
    // Inject attractor signal into tensor
    const signal = (x + y + z) / 3;
    for (let i = 0; i < this.data.length; i++) this.data[i] += signal * 0.001;
    this.mutationCount += steps;
    return this;
  }

  snapshot() {
    return { ...super.snapshot(), attractor: this.attractor, _s: [...this._s] };
  }

  static restore(snap) {
    const t = new FLUXTensor(snap.data, snap.attractor ?? 'lorenz');
    t.createdAt      = snap.createdAt;
    t.mutationCount  = snap.mutationCount ?? 0;
    if (snap._s) t._s = snap._s;
    return t;
  }
}

// ─── CarryField ───────────────────────────────────────────────────────────────
//
// The persistent memory across forward passes — a living field that
// accumulates the "experience" of all previous conversations.

export class CarryField extends FluxTensor {
  constructor(dim, evolutionRate = 0.001) {
    super(new Array(dim).fill(0), evolutionRate, 365 * 10);
    this.dim = dim;
  }

  /** Inject signal from current layer output, then mutate. */
  update(signal) {
    // signal: Float32Array of length dim (mean over batch/seq)
    this.mutate(signal);
    return this;
  }

  /** Inject carry field into activations (additive, scaled). */
  inject(activations, scale = 0.1) {
    for (let i = 0; i < activations.length; i++)
      activations[i] += this.data[i % this.dim] * scale;
    return activations;
  }
}

// ─── MayanVersionBoundary ────────────────────────────────────────────────────
//
// Tracks version steps. Every 1000 forward passes = New Year's Day.

export class MayanVersionBoundary {
  constructor(stepsPerVersion = 1000) {
    this.stepsPerVersion = stepsPerVersion;
    this.step    = 0;
    this.version = '1.0.0';
    this.history = []; // fossil records
  }

  tick(snapshot = null) {
    this.step++;
    if (this.step % this.stepsPerVersion === 0) {
      this._incrementVersion(snapshot);
      return true; // version boundary reached
    }
    return false;
  }

  _incrementVersion(snapshot) {
    let [maj, mn, patch] = this.version.split('.').map(Number);
    patch++;
    if (patch >= 10) { patch = 0; mn++; }
    if (mn    >= 10) { mn    = 0; maj++; }
    this.version = `${maj}.${mn}.${patch}`;
    this.history.push({ version: this.version, step: this.step,
                        ts: Date.now(), snapshot });
    return this.version;
  }

  get year()         { return Math.floor(this.step / this.stepsPerVersion); }
  get fossilCount()  { return this.history.length; }
  latestFossil()     { return this.history[this.history.length - 1] ?? null; }
}

// ─── MayanEvolutionLayer (JS) ─────────────────────────────────────────────────

export class MayanEvolutionLayer {
  constructor(dim, evolutionRate = 0.01) {
    this.dim           = dim;
    this.evolutionRate = evolutionRate;
    // Minimal placeholder weights — real matmul would use linalg helpers
    this.qWeight    = new FluxTensor(new Array(dim).fill(0).map(() => (Math.random()-.5)/dim), evolutionRate);
    this.kWeight    = new FluxTensor(new Array(dim).fill(0).map(() => (Math.random()-.5)/dim), evolutionRate);
    this.vWeight    = new FluxTensor(new Array(dim).fill(0).map(() => (Math.random()-.5)/dim), evolutionRate);
    this.carryField = new CarryField(dim, evolutionRate * 0.1);
  }

  /** Forward pass: mutate Q/K/V, update carry, return carry mean (for tracing). */
  forward(x) {
    // Simplified scalar pass for JS runtime (full matmul uses linalg.js)
    const outMean  = x.reduce((s,v) => s+v, 0) / x.length;
    const carryAdj = this.carryField.mean() * 0.1;

    // Update carry with output signal
    const signal = new Float32Array(this.dim).fill(outMean + carryAdj);
    this.carryField.update(signal);

    // Mutate weights after use (aging)
    this.qWeight.mutate();
    this.kWeight.mutate();
    this.vWeight.mutate();

    return { out: outMean + carryAdj, carry: this.carryField.mean() };
  }

  snapshot() {
    return { dim: this.dim, evolutionRate: this.evolutionRate,
             q: this.qWeight.snapshot(), k: this.kWeight.snapshot(),
             v: this.vWeight.snapshot(), carry: this.carryField.snapshot() };
  }
}

// ─── MayanMatrixMath (JS runtime) ─────────────────────────────────────────────

export class MayanMatrixMath {
  constructor(dim = 256, numLayers = 4) {
    this.dim         = dim;
    this.numLayers   = numLayers;
    this.tokenEmbed  = new FluxTensor(new Array(dim).fill(0).map(() => (Math.random()-.5)/dim));
    this.posEmbed    = new FluxTensor(new Array(dim).fill(0).map(() => (Math.random()-.5)/dim));
    this.layers      = Array.from({ length: numLayers }, (_, i) =>
      new MayanEvolutionLayer(dim, 0.01 * (1 + i / numLayers)));
    this.boundary    = new MayanVersionBoundary(1000);
  }

  /** One forward pass = one aging event = one 'day'. */
  forward(inputSignal) {
    let x = inputSignal ?? [Math.random()];

    const evolution = [];
    for (const layer of this.layers) {
      const { out, carry } = layer.forward(Array.isArray(x) ? x : [x]);
      x = [out];
      evolution.push(carry);
    }

    // Age the embeddings
    this.tokenEmbed.mutate();
    this.posEmbed.mutate();

    // Check for version boundary
    const snap = this.boundary.tick() ? this.snapshot() : null;

    return { output: x[0], evolution, version: this.boundary.version,
             versionBoundary: snap !== null, step: this.boundary.step };
  }

  /** Simulate N years of aging (steps_per_day events/year). */
  simulateYears(years, stepsPerDay = 10) {
    const total = years * 365 * stepsPerDay;
    const fossils = [];
    for (let i = 0; i < total; i++) {
      const r = this.forward([Math.random()]);
      if (r.versionBoundary) fossils.push(r);
    }
    return { years, steps: total, version: this.boundary.version,
             fossils, totalMutations: this.totalMutations() };
  }

  totalMutations() {
    return this.tokenEmbed.mutationCount + this.posEmbed.mutationCount
      + this.layers.reduce((s, l) => s + l.qWeight.mutationCount
          + l.kWeight.mutationCount + l.vWeight.mutationCount
          + l.carryField.mutationCount, 0);
  }

  snapshot() {
    return {
      version:  this.boundary.version,
      step:     this.boundary.step,
      ts:       Date.now(),
      dim:      this.dim,
      embed:    this.tokenEmbed.snapshot(),
      layers:   this.layers.map(l => l.snapshot()),
      mutations: this.totalMutations(),
    };
  }

  get version() { return this.boundary.version; }
  get step()    { return this.boundary.step; }
}

// ─── µMODEL registration spec ────────────────────────────────────────────────

export const MUPY_MAYAN_KXML = `<?xml version="1.0" encoding="utf-8"?>
<kxml domain="mayan_matrix_math" phase="Pop" gravity="Float">

  <fold id="flux_embed"   domain="LIVING_EMBEDDING"/>
  <fold id="evolution_layer" domain="MAYAN_LAYER"/>
  <fold id="carry_field"  domain="PERSISTENT_MEMORY"/>
  <fold id="version_boundary" domain="VERSION_FOSSIL"/>

  <geodesic from="flux_embed"   to="evolution_layer" cost="0.1"/>
  <geodesic from="evolution_layer" to="carry_field"  cost="0.2"/>
  <geodesic from="carry_field"  to="version_boundary" cost="0.5"/>
  <geodesic from="version_boundary" to="flux_embed"  cost="0.9"/>

  <lane id="aging" type="evolution" permission="inherit"/>

  <![CDATA[
    Pop:   restore FluxTensor from fossil snapshot (no gradient needed)
    Wo:    project with evolving Q/K/V weights (mutation_rate * decay_factor)
    Sek:   attention + carry field injection (outMean + carryAdj)
    Ch'en: mutate all tensors — the aging step
           decay_factor = 0.5^(age_days / half_life)
           noise = randn * mutation_rate * decay_factor
    Xul:   check version boundary (every 1000 steps = New Year's Day)
           save fossil snapshot if boundary reached

    The model doesn't learn. It AGES.
    SafeTensors = fossil records of evolution.
    Each chat = one day. Each 1000 steps = new version.
    Lorenz attractor drives chaotic variation in FLUXTensor mode.
  ]]>

  <![CDATA[
    mayan.flux_tensor mayan.carry_field mayan.evolution_layer
    mayan.version_boundary mayan.fossil mayan.aging mayan.decay_factor
    mayan.lorenz mayan.rossler mayan.logistic mayan.attractor
    flux_tensor.mutate flux_tensor.half_life flux_tensor.mutation_count
    glyph.NEURAL_PATH glyph.QUANTUM_GATE glyph.ROTATE_COMP
  ]]>

  <shader type="mayan_projection">
    <![CDATA[
      // field_optimizer.hlsl — mayan evolution field
      // gravity=Float: tensors drift freely (no hard attraction)
      float gravity_scale  = 0.0; // Float gravity
      float decay          = pow(0.5, age_days / half_life);
      float mutation_delta = randn() * mutation_rate * decay;
      // Carry field injection
      float carry_signal   = carry_field_mean * 0.1;
    ]]>
  </shader>

</kxml>`;

export const MUPY_MAYAN_SPEC = `
[model]
domain   = "mayan_matrix_math"
phase    = "Pop"
gravity  = "Float"

[training]
steps  = 0
batch  = 1
lr     = 0.0
block  = 256
note   = "No gradient training. The model ages via FluxTensor mutation."

[routing]
trigger  = "evolve,age,mutate,version,fossil,time,year,calendar,living,flux"
fallback = "base_gpt2"

[capabilities]
tools = "mayan_fold,fibonacci_fold,pi_field,flux_runtime,kxml_run"
`;
