// gravity.js — Gravity / Antigravity field for KXML nodes
//
// Physics analogy:
//   gravity_scale = 0.0  → antigravity (node floats outside phase machine)
//   gravity_scale = 1.0  → normal gravity (fully constrained)
//   gravity_scale > 1.0  → heavy (extra constraints, debug injection)
//   gravity_scale < 0.0  → active antigravity (regularisation repulsion)
//
// Training analogy (AdamW):
//   gravity_scale  ≡ weight_decay / lr          (constraint strength ratio)
//   antigravity    ≡ lr / weight_decay           (update freedom ratio)
//   optimal_ratio  ≡ weight_decay ÷ lr ≈ 10-1000 for stable training
//
// Node taxonomy:
//   ⟁Grav⟁     gravity_scale = 1.0   constrained, phase-gated, Lipschitz-bounded
//   ⟁AntiGrav⟁ gravity_scale = 0.0   float, bypass phase gates, no gradient clip
//   ⟁HeavyGrav⟁gravity_scale = 2.0   extra constraints (loss, output head)
//   ⟁NegGrav⟁  gravity_scale = -1.0  regularisation push (L1 sparsity node)
//
// Video-game physics → KXML mapping:
//   rigid_body.gravity_scale  → node.gravity_scale
//   collision_group           → phase_gate
//   friction                  → gradient_clip_norm
//   bounce / restitution      → residual connection
//   trigger_zone              → telemetry / debug node (antigravity)
//   kinematic body            → frozen / non-trainable node

// ─── Gravity constants ────────────────────────────────────────────────────────

export const G = Object.freeze({
  FLOAT:        0.0,    // full antigravity (debug, telemetry, non-critical)
  NORMAL:       1.0,    // standard gravity (compute nodes, attention)
  HEAVY:        2.0,    // extra constraints (loss, output head)
  NEG:         -1.0,    // repulsive (L1 regularisation, sparsity gate)
  EMBED:        0.5,    // half-gravity (embedding layer can float slightly)
  THRESHOLD:    1e-3,   // |gravity| < threshold → treat as antigravity
});

// ─── GravityField ─────────────────────────────────────────────────────────────
//
// A field assigns gravity_scale to each node and computes the graph-wide
// ratio. Stable training requires gravity_sum > antigravity_sum.

export class GravityField {
  constructor(opts = {}) {
    this._fields    = new Map();   // node_id → scale
    this._defaults  = {
      compute:     G.NORMAL,
      attention:   G.NORMAL,
      loss:        G.HEAVY,
      output:      G.HEAVY,
      embedding:   G.EMBED,
      debug:       G.FLOAT,
      telemetry:   G.FLOAT,
      gradient:    G.NORMAL,
    };
    Object.assign(this._defaults, opts.defaults ?? {});
  }

  // Set gravity for a specific node
  set(nodeId, scale) { this._fields.set(nodeId, scale); return this; }

  // Assign gravity based on node domain
  setByDomain(nodeId, domain) {
    const scale = this._defaults[domain] ?? G.NORMAL;
    return this.set(nodeId, scale);
  }

  // Get effective gravity (falls back to domain default → NORMAL)
  get(nodeId, domain = null) {
    if (this._fields.has(nodeId)) return this._fields.get(nodeId);
    if (domain) return this._defaults[domain] ?? G.NORMAL;
    return G.NORMAL;
  }

  isAntigravity(nodeId, domain = null) {
    return Math.abs(this.get(nodeId, domain)) < G.THRESHOLD;
  }

  // Graph-wide ratio: gravity_sum / antigravity_sum
  // Optimal range: 10 – 10000 (matches AdamW weight_decay/lr ratio)
  ratio() {
    let gravSum = 0, antiSum = 0;
    for (const scale of this._fields.values()) {
      if (scale > G.THRESHOLD) gravSum   += scale;
      else                      antiSum  += Math.abs(scale) || 1;
    }
    if (antiSum === 0) return Infinity;   // all constrained
    return gravSum / antiSum;
  }

  // Classify the current field state
  classify() {
    const r = this.ratio();
    if (r === Infinity) return 'all_constrained';
    if (r < 0.1)   return 'unstable_antigravity_dominant';
    if (r < 1.0)   return 'borderline_unstable';
    if (r < 10)    return 'lightly_constrained';
    if (r < 1000)  return 'stable';
    if (r < 1e6)   return 'heavily_constrained';
    return 'frozen';
  }

  // Recommended ratio for a given lr / weight_decay pair
  static recommendedRatio(lr, weightDecay) {
    return weightDecay / lr;
  }

  // Auto-assign gravity to all nodes from a KXMLBridge graph
  calibrateFromBridge(bridge) {
    for (const [id, mn] of bridge.micronauts) {
      this.setByDomain(id, mn.domain);
      // Override by explicit antigravity flag
      if (mn._antigravity) this.set(id, G.FLOAT);
      if (mn._heavy)       this.set(id, G.HEAVY);
    }
    return this;
  }

  summary() {
    const counts = { float: 0, normal: 0, heavy: 0, negative: 0 };
    for (const v of this._fields.values()) {
      if (Math.abs(v) < G.THRESHOLD) counts.float++;
      else if (v < 0)                 counts.negative++;
      else if (v > G.NORMAL)          counts.heavy++;
      else                            counts.normal++;
    }
    return { ...counts, ratio: this.ratio(), classification: this.classify() };
  }
}

// ─── PhysicsDispatcher ────────────────────────────────────────────────────────
//
// Wraps KXMLBridge execution with gravity-aware phase gating and
// gradient clipping. Antigravity nodes bypass phase checks and clip.

export class PhysicsDispatcher {
  constructor(bridge, field, opts = {}) {
    this._bridge   = bridge;
    this._field    = field;
    this._clipNorm = opts.clipNorm ?? 1.0;
    this._maxLogit = opts.maxLogit ?? 20.0;
    this._log      = [];
  }

  // Should this node respect phase gating?
  _respectsPhase(nodeId) {
    const scale = this._field.get(nodeId);
    return Math.abs(scale) >= G.THRESHOLD;
  }

  // Should this node's gradient be clipped?
  _clipsGradient(nodeId) {
    const scale = this._field.get(nodeId);
    return scale >= G.NORMAL;  // heavy + normal → clip; float + negative → skip
  }

  // Apply gravity-aware forward pass
  async forward(inputs = {}) {
    const activations = { ...inputs };

    for (const id of this._bridge.execOrder) {
      const mn    = this._bridge.micronauts.get(id);
      const scale = this._field.get(id, mn.domain);
      const entry = { id, scale, phase_checked: false, logit_clamped: false };

      // Antigravity nodes skip phase check
      if (this._respectsPhase(id)) {
        entry.phase_checked = true;
        // Phase gate: node must be at or before current global phase
        if (!this._bridge.canExecute?.(mn.phase)) {
          entry.skipped = true; this._log.push(entry); continue;
        }
      }

      // Gather upstream activations
      const incoming = {};
      for (const ent of this._bridge.edges.values()) {
        if (ent.to === id && ent.activation !== null) incoming[ent.from] = ent.activation;
      }

      let result = await mn.forward({ ...activations, ...incoming });

      // Heavy-gravity nodes: clamp logits
      if (scale >= G.HEAVY) {
        if (Array.isArray(result)) {
          result = result.map(v => Math.max(-this._maxLogit, Math.min(this._maxLogit, v)));
          entry.logit_clamped = true;
        }
      }

      activations[id] = result;
      for (const ent of this._bridge.edges.values()) {
        if (ent.from === id) await ent.sendForward(result);
      }

      this._log.push(entry);
    }

    return activations;
  }

  // Apply gravity-aware backward pass
  async backward(lossGrad, lr = 0.01) {
    const gradients = {};
    let grad = lossGrad;

    for (const id of [...this._bridge.execOrder].reverse()) {
      const mn    = this._bridge.micronauts.get(id);
      const scale = this._field.get(id, mn.domain);

      let result = await mn.backward(grad, lr);

      // Antigravity nodes: gradient flows without clipping
      if (this._clipsGradient(id)) {
        const norm = Array.isArray(result.grad)
          ? Math.sqrt(result.grad.reduce((s, v) => s + v**2, 0))
          : Math.abs(result.grad);
        const clip = this._clipNorm * scale;  // heavier nodes clip harder
        if (norm > clip) {
          const s = clip / norm;
          result = {
            ...result,
            grad: Array.isArray(result.grad) ? result.grad.map(v => v * s) : result.grad * s,
            clipped: true,
            clip_scale: s,
          };
        }
      }

      gradients[id] = { ...result, gravity_scale: scale };
      grad = result.grad;

      for (const ent of this._bridge.edges.values()) {
        if (ent.to === id) await ent.sendBackward(result.grad, lr);
      }
    }

    return gradients;
  }

  // Soft-landing check (extended with gravity info)
  verifySoftLandings(gradients) {
    const base = this._bridge.verifySoftLandings(gradients);
    const gravity_report = {};
    for (const [id, g] of Object.entries(gradients)) {
      gravity_report[id] = {
        gravity_scale: g.gravity_scale ?? G.NORMAL,
        clipped:       g.clipped ?? false,
        bounded:       !base.violations.some(v => v.id === id),
      };
    }
    return { ...base, gravity_report, ratio: this._field.ratio() };
  }

  get log() { return [...this._log]; }
}

// ─── ⟁AntiGrav⟁ node decorator ────────────────────────────────────────────────
//
// Tags a KXMLMicronaut as antigravity.  Returns the same node for chaining.

export function antigravity(micronaut) {
  micronaut._antigravity = true;
  micronaut._gravity_scale = G.FLOAT;
  return micronaut;
}

export function heavyGravity(micronaut) {
  micronaut._heavy = true;
  micronaut._gravity_scale = G.HEAVY;
  return micronaut;
}

export function negativeGravity(micronaut) {
  micronaut._negative = true;
  micronaut._gravity_scale = G.NEG;
  return micronaut;
}

// ─── XML attribute parser ──────────────────────────────────────────────────────
//
// Reads gravity attributes from KXML node objects:
//   { id, gravity: "0.0" }           → antigravity
//   { id, gravity: "2.0" }           → heavy
//   { id, antigravity: "true" }      → float
//   { id, domain: "debug" }          → float (by convention)

export function parseGravityAttr(nodeAttr) {
  if (nodeAttr.antigravity === 'true' || nodeAttr.antigravity === true) return G.FLOAT;
  if (nodeAttr.gravity !== undefined) return parseFloat(nodeAttr.gravity);
  if (nodeAttr.domain === 'debug' || nodeAttr.domain === 'telemetry') return G.FLOAT;
  if (nodeAttr.domain === 'loss' || nodeAttr.domain === 'output')      return G.HEAVY;
  if (nodeAttr.domain === 'embedding')                                   return G.EMBED;
  return G.NORMAL;
}

// ─── Training stability advisor ───────────────────────────────────────────────
//
// Given a GravityField and training hyperparams, advise on stability.

export class GravityAdvisor {
  constructor(field) { this._field = field; }

  advise(lr, weightDecay, loss) {
    const fieldRatio  = this._field.ratio();
    const adamwRatio  = GravityField.recommendedRatio(lr, weightDecay);
    const mismatch    = Math.abs(Math.log10(fieldRatio / adamwRatio));

    const advice = [];

    if (fieldRatio < 1) {
      advice.push({ severity: 'critical', msg: 'Antigravity dominant — training will diverge. Add gravity to compute nodes.' });
    }
    if (fieldRatio < 10 && fieldRatio >= 1) {
      advice.push({ severity: 'warning', msg: `Ratio ${fieldRatio.toFixed(1)} is low. Consider weight_decay ≥ ${(lr * 10).toExponential(1)}.` });
    }
    if (loss > 10) {
      advice.push({ severity: 'warning', msg: 'Loss > 10 suggests antigravity explosion. Check for unclamped logits.' });
    }
    if (mismatch > 2) {
      advice.push({ severity: 'info', msg: `Field ratio (${fieldRatio.toFixed(0)}) vs AdamW ratio (${adamwRatio.toFixed(0)}) differ by ${mismatch.toFixed(1)} orders of magnitude.` });
    }
    if (advice.length === 0) {
      advice.push({ severity: 'ok', msg: `Ratio ${fieldRatio.toFixed(0)} within stable range. Field matches AdamW.` });
    }

    return {
      field_ratio:  fieldRatio,
      adamw_ratio:  adamwRatio,
      classification: this._field.classify(),
      advice,
      stable: fieldRatio >= 10 && loss <= 10,
    };
  }

  // Compute the ideal gravity_scale per domain for given lr/wd
  idealScales(lr, weightDecay) {
    const base = weightDecay / lr;   // e.g. 0.1 / 1e-5 = 10000
    return {
      loss:      Math.min(base / 1000, G.HEAVY),   // cap at 2.0
      compute:   Math.min(base / 5000, G.NORMAL),  // cap at 1.0
      embedding: Math.min(base / 10000, G.EMBED),  // cap at 0.5
      debug:     G.FLOAT,                          // always float
      telemetry: G.FLOAT,                          // always float
    };
  }
}

// ─── XCFE namespace registration ──────────────────────────────────────────────

export function registerGravityNamespaces(rt) {
  const fields = new Map();

  rt._handlers.set('@gravity', (val, ctx) => {
    const name  = val['@name'] ?? 'default';
    const field = fields.get(name) ?? new GravityField(val);
    if (val.nodes) {
      for (const [id, scale] of Object.entries(val.nodes)) field.set(id, parseFloat(scale));
    }
    if (val.auto && ctx['_bridge_default']) {
      field.calibrateFromBridge(ctx['_bridge_default']);
    }
    fields.set(name, field);
    ctx[`_gravity_${name}`] = field;
    if (val['@store']) ctx[val['@store']] = field.summary();
    return field;
  });

  rt._handlers.set('@antigravity', (val, ctx) => {
    const fieldName = val.field ?? 'default';
    const field     = fields.get(fieldName) ?? new GravityField();
    const nodeIds   = val.nodes ?? [];
    for (const id of nodeIds) field.set(id, G.FLOAT);
    fields.set(fieldName, field);
    if (val['@store']) ctx[val['@store']] = { antigravity_nodes: nodeIds, field: field.summary() };
    return field;
  });

  rt._handlers.set('@physics', (val, ctx) => {
    const bridge = ctx[`_bridge_${val.bridge ?? 'default'}`];
    const field  = fields.get(val.gravity ?? 'default') ?? new GravityField();
    if (!bridge) return null;
    const disp = new PhysicsDispatcher(bridge, field, {
      clipNorm: val.clip_norm ?? 1.0,
      maxLogit: val.max_logit ?? 20.0,
    });
    ctx['_physics_dispatcher'] = disp;
    if (val['@store']) ctx[val['@store']] = disp;
    return disp;
  });
}

// ─── KuhulPhysicsSolver — automatic gravity tuning ───────────────────────────
//
// Watches loss + gradient observations and adjusts gravity_scale per node.
// Implements a discrete Poisson solver over the KXML graph:
//
//   ∇²Φ_i = ρ_gravity_i + ρ_antigravity_i
//
// Each step:
//   1. Observe: record (loss, grad_norms) from the last training step
//   2. Diagnose: compare to stability thresholds
//   3. Adjust: increase gravity on violating nodes, relax on stable ones
//   4. Emit: updated GravityField + PhysicsDispatcher
//
// K'UHUL Physics Laws (enforced automatically):
//   Law 1 (Phase Constraint):     no node executes before its phase is ready
//   Law 2 (Antigravity Permit):   antigravity nodes bypass phase check
//   Law 3 (Gradient Conservation):||∇|| ≤ gravity_scale × Lipschitz_constant

export class KuhulPhysicsSolver {
  constructor(field, opts = {}) {
    this._field        = field;
    this._history      = [];         // [{step, loss, grad_norms, field_ratio}]
    this._alpha        = opts.alpha  ?? 0.1;   // gravity adjustment rate
    this._lossTarget   = opts.loss_target ?? 5.0;
    this._gradTarget   = opts.grad_target ?? 1.0;
    this._lossMax      = opts.loss_max ?? 10.0;
    this._minGravity   = opts.min_gravity ?? G.EMBED;
    this._maxGravity   = opts.max_gravity ?? G.HEAVY;
  }

  // Observe one training step's outcomes
  observe(step, loss, gradients) {
    const grad_norms = {};
    for (const [id, g] of Object.entries(gradients)) {
      const arr  = Array.isArray(g.grad) ? g.grad : [g.grad ?? 0];
      grad_norms[id] = Math.sqrt(arr.reduce((s, v) => s + v**2, 0));
    }
    const entry = { step, loss, grad_norms, field_ratio: this._field.ratio() };
    this._history.push(entry);
    return entry;
  }

  // Diagnose instability per node
  diagnose() {
    if (!this._history.length) return [];
    const last = this._history[this._history.length - 1];
    const issues = [];

    // Loss too high → antigravity dominant
    if (last.loss > this._lossMax) {
      issues.push({ type: 'loss_explosion', loss: last.loss,
                    action: 'increase_gravity_all', severity: 'critical' });
    }
    // Loss oscillating → gravity imbalance (check std over last 10 steps)
    if (this._history.length >= 10) {
      const recent = this._history.slice(-10).map(h => h.loss);
      const mean   = recent.reduce((s,v)=>s+v,0) / 10;
      const std    = Math.sqrt(recent.reduce((s,v)=>s+(v-mean)**2,0)/10);
      if (std > mean * 0.5) {
        issues.push({ type: 'loss_oscillation', std: std.toFixed(3), mean: mean.toFixed(3),
                      action: 'increase_gravity_critical_nodes', severity: 'warning' });
      }
    }
    // Per-node gradient explosions
    for (const [id, norm] of Object.entries(last.grad_norms)) {
      const mn = this._field.get(id);
      if (norm > this._gradTarget * 10 && mn < G.HEAVY) {
        issues.push({ type: 'grad_explosion', node: id, norm: norm.toFixed(3),
                      action: 'increase_gravity_node', severity: 'warning' });
      }
    }
    // Field ratio too low
    if (last.field_ratio < 10) {
      issues.push({ type: 'low_gravity_ratio', ratio: last.field_ratio.toFixed(1),
                    action: 'add_gravity_nodes', severity: 'warning' });
    }
    return issues;
  }

  // Adjust gravity_scale for each node based on diagnosis
  // Returns {adjusted: [{id, old_scale, new_scale}], issues}
  step(gradients) {
    const issues   = this.diagnose();
    const adjusted = [];

    for (const issue of issues) {
      switch (issue.action) {
        case 'increase_gravity_all':
          for (const [id, scale] of this._field._fields) {
            if (Math.abs(scale) >= G.THRESHOLD) {
              const newScale = Math.min(this._maxGravity, scale + this._alpha);
              adjusted.push({ id, old: scale, new: newScale, reason: issue.type });
              this._field.set(id, newScale);
            }
          }
          break;

        case 'increase_gravity_node': {
          const id       = issue.node;
          const old      = this._field.get(id);
          const newScale = Math.min(this._maxGravity, old + this._alpha);
          adjusted.push({ id, old, new: newScale, reason: issue.type });
          this._field.set(id, newScale);
          break;
        }

        case 'increase_gravity_critical_nodes':
          // Increase gravity on all heavy/normal nodes
          for (const [id, scale] of this._field._fields) {
            if (scale >= G.NORMAL) {
              const newScale = Math.min(this._maxGravity, scale + this._alpha * 0.5);
              adjusted.push({ id, old: scale, new: newScale, reason: issue.type });
              this._field.set(id, newScale);
            }
          }
          break;

        case 'add_gravity_nodes':
          // Do nothing automatically — flag for user to add gravity nodes
          break;
      }
    }

    // Relaxation: if stable, gently reduce gravity toward 1.0
    if (issues.length === 0 && this._history.length >= 5) {
      const recent = this._history.slice(-5);
      const allStable = recent.every(h => h.loss < this._lossTarget);
      if (allStable) {
        for (const [id, scale] of this._field._fields) {
          if (scale > G.NORMAL) {
            const newScale = Math.max(G.NORMAL, scale - this._alpha * 0.2);
            if (newScale !== scale) {
              adjusted.push({ id, old: scale, new: newScale, reason: 'stable_relaxation' });
              this._field.set(id, newScale);
            }
          }
        }
      }
    }

    return { adjusted, issues, ratio: this._field.ratio(), classification: this._field.classify() };
  }

  // Full Poisson field solve: N steps with observation + adjustment
  async solve(bridge, dispatcher, inputData, lossGrad, steps = 10, lr = 0.01) {
    const trajectory = [];

    for (let s = 0; s < steps; s++) {
      const activations = await dispatcher.forward(inputData);
      const gradients   = await dispatcher.backward(lossGrad, lr);
      const lastLoss    = this._history[this._history.length - 1]?.loss ?? 9999;

      this.observe(s, lastLoss, gradients);
      const result = this.step(gradients);

      trajectory.push({
        step:       s,
        loss:       lastLoss,
        ratio:      result.ratio,
        class:      result.classification,
        adjusted:   result.adjusted.length,
        issues:     result.issues.map(i => i.type),
      });
    }

    return {
      trajectory,
      final_ratio:        this._field.ratio(),
      final_class:        this._field.classify(),
      total_adjustments:  trajectory.reduce((s, t) => s + t.adjusted, 0),
      converged:          this._field.classify() === 'stable',
    };
  }

  // Field equation residual: ∇²Φ = ρ_grav + ρ_anti
  // Returns how far the field is from the Poisson equilibrium
  fieldResidual() {
    let rho_grav = 0, rho_anti = 0;
    for (const scale of this._field._fields.values()) {
      if (scale > G.THRESHOLD) rho_grav  += scale;
      else                      rho_anti  += Math.abs(scale) || 1;
    }
    // Equilibrium: rho_grav ≈ 10-1000 × rho_anti
    const laplacian = rho_grav - 10 * rho_anti;   // ∇²Φ target = 0 at ratio=10
    return { rho_grav, rho_anti, laplacian, equilibrium: Math.abs(laplacian) < rho_anti };
  }

  get history() { return [...this._history]; }
}

// ─── Opcode alignment ─────────────────────────────────────────────────────────

export const GRAVITY_OPCODE_MAP = Object.freeze({
  '⟁Grav⟁':     { opcode: 0x61, kuhul: '⟁Val⟁',   description: 'constrained execution (Lipschitz + phase gate)' },
  '⟁AntiGrav⟁': { opcode: 0x00, kuhul: 'NOP',       description: 'float — bypass phase gate and gradient clip' },
  '⟁HeavyGrav⟁':{ opcode: 0x63, kuhul: '⟁Enf⟁',   description: 'extra constraints (loss/output head)' },
  '⟁NegGrav⟁':  { opcode: 0x08, kuhul: '⟁Sek⟁!',  description: 'repulsive (L1 sparsity / regularisation)' },
});
