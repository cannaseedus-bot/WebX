// micronaut-core.js — K'UHUL Micronaut Core (v3.0.0 spec implementation)
//
// Source: C:\Users\canna\.kuhul-v1\releases\v3.0.0-agentic-micronaut\specs\micronaut-core.json
//
// Implements the base micronaut contract:
//   Identity (name, role, version, lineage, birth_time)
//   Cognitive state (temperature, context_window, response_flare, learning)
//   Memory system (working, episodic, semantic, procedural)
//   Required methods: execute, learn, update_context, calculate_relevance,
//                     adjust_temperature, modulate_flare, remember, forget, communicate
//
// Every micronaut in the system MUST implement this interface.
// The agentic micronauts (TOOL-µ, AGENT-µ, SKILL-µ etc.) extend this base.

// ─── Core spec constants ──────────────────────────────────────────────────────

export const MICRONAUT_CORE_VERSION = "3.0.0";

export const COGNITIVE_DEFAULTS = Object.freeze({
  context_window_size: 4096,
  temperature:         0.7,
  temperature_min:     0.0,
  temperature_max:     2.0,
  temperature_adaptive: true,
  response_flare: {
    enthusiasm: 0.5,
    certainty:  0.5,
    verbosity:  0.5,
    empathy:    0.5,
  },
  learning: {
    rate:                    0.01,
    curiosity:               0.5,
    forgetting_curve:        "exponential",
    consolidation_frequency_s: 3600,
  },
});

export const REQUIRED_METHODS = Object.freeze([
  "execute", "learn", "update_context", "calculate_relevance",
  "adjust_temperature", "modulate_flare", "remember", "forget", "communicate",
]);

// ─── Memory system ────────────────────────────────────────────────────────────

class MemoryStore {
  constructor(maxItems = 1024) {
    this._items   = [];
    this._maxItems = maxItems;
  }

  push(item) {
    this._items.push({ ...item, ts: Date.now() });
    if (this._items.length > this._maxItems) this._items.shift();
  }

  query(predicate) {
    return this._items.filter(predicate);
  }

  clear() { this._items.length = 0; }

  get size() { return this._items.length; }
  get all()  { return [...this._items]; }
}

// ─── MicronauntCore base class ────────────────────────────────────────────────

export class MicronauntCore {
  constructor(opts = {}) {
    // ── Identity ────────────────────────────────────────────────────────────
    this.name       = opts.name    ?? "unnamed-micronaut";
    this.role       = opts.role    ?? "general";
    this.version    = opts.version ?? MICRONAUT_CORE_VERSION;
    this.lineage    = opts.lineage ?? [];
    this.birth_time = opts.birth_time ?? new Date().toISOString();
    this.id         = opts.id ?? `${this.role}-${Date.now()}`;

    // ── Cognitive state ─────────────────────────────────────────────────────
    const cd = COGNITIVE_DEFAULTS;
    this._context    = { size: cd.context_window_size, focus: 0.5, relevance: {} };
    this._temperature = opts.temperature ?? cd.temperature;
    this._flare      = { ...cd.response_flare, ...(opts.flare ?? {}) };
    this._learning   = { ...cd.learning, ...(opts.learning ?? {}) };

    // ── Memory system ────────────────────────────────────────────────────────
    this._memory = {
      working:    new MemoryStore(64),
      episodic:   new MemoryStore(512),
      semantic:   new MemoryStore(1024),
      procedural: new MemoryStore(256),
    };

    // ── Execution state ──────────────────────────────────────────────────────
    this._execution_count = 0;
    this._last_output     = null;
  }

  // ── Required interface methods ───────────────────────────────────────────────

  // execute(task, context?) → result
  async execute(task, context = {}) {
    this._execution_count++;
    this.update_context({ task, ...context });
    const result = await this._execute(task, context);
    this._last_output = result;
    this.remember("episodic", { task, result, ts: Date.now() });
    return result;
  }

  // Subclasses override this — base implementation is a passthrough
  async _execute(task, _ctx) {
    return { status: "ok", task, micronaut: this.name, result: null };
  }

  // learn(experience) → void
  learn(experience) {
    const relevance = this.calculate_relevance(experience);
    if (relevance > 0.5) {
      this.remember("semantic", { experience, relevance });
    }
    // Exponential forgetting: decay old memories
    if (this._execution_count % 10 === 0) {
      this._decay_memories();
    }
  }

  // update_context(data) → void
  update_context(data) {
    Object.assign(this._context.relevance, data);
    this._context.focus = this.calculate_relevance(data);
    this.remember("working", data);
  }

  // calculate_relevance(data) → 0..1
  calculate_relevance(data) {
    if (!data) return 0;
    // Simple heuristic: count overlap with current context keys
    const ctxKeys  = Object.keys(this._context.relevance);
    const dataKeys = Object.keys(data);
    if (ctxKeys.length === 0) return 0.5;
    const overlap = dataKeys.filter(k => ctxKeys.includes(k)).length;
    return Math.min(1, overlap / Math.max(ctxKeys.length, 1));
  }

  // adjust_temperature(target, reason?) → void
  adjust_temperature(target, reason = "") {
    const cd = COGNITIVE_DEFAULTS;
    this._temperature = Math.max(cd.temperature_min,
                          Math.min(cd.temperature_max, target));
    this.remember("episodic", { temperature_change: this._temperature, reason });
  }

  // modulate_flare(field, value) → void
  modulate_flare(field, value) {
    if (field in this._flare) {
      this._flare[field] = Math.max(0, Math.min(1, value));
    }
  }

  // remember(store, item) → void
  remember(store = "episodic", item) {
    const s = this._memory[store];
    if (s) s.push(item);
  }

  // forget(store, predicate?) → number removed
  forget(store = "working", predicate = null) {
    if (predicate) {
      const mem = this._memory[store];
      if (!mem) return 0;
      const before = mem.size;
      const keep   = mem.all.filter(i => !predicate(i));
      mem.clear();
      keep.forEach(i => mem.push(i));
      return before - mem.size;
    }
    this._memory[store]?.clear();
    return 0;
  }

  // communicate(message, target?) → response
  async communicate(message, target = null) {
    return {
      from:    this.name,
      to:      target ?? "broadcast",
      message,
      phase:   this.phase,
      ts:      Date.now(),
    };
  }

  // ── State inspection ─────────────────────────────────────────────────────────

  get temperature() { return this._temperature; }
  get flare()       { return { ...this._flare }; }
  get context()     { return { ...this._context }; }
  get phase()       { return this._flare.enthusiasm > 0.7 ? "active" : "idle"; }

  get memoryStats() {
    const s = {};
    for (const [k, v] of Object.entries(this._memory)) s[k] = v.size;
    return { ...s, executions: this._execution_count };
  }

  snapshot() {
    return {
      id: this.id, name: this.name, role: this.role,
      version: this.version, birth_time: this.birth_time,
      temperature: this._temperature,
      flare: this.flare,
      memory: this.memoryStats,
      context_focus: this._context.focus,
      executions: this._execution_count,
    };
  }

  // ── Private ──────────────────────────────────────────────────────────────────

  _decay_memories() {
    const now = Date.now();
    // Exponential forgetting: remove episodic items older than 1 hour
    this.forget("episodic", item => item.ts && (now - item.ts) > 3_600_000);
    // Working memory resets frequently
    if (this._memory.working.size > 32) this._memory.working.clear();
  }
}

// ─── Validate implementation against spec ────────────────────────────────────

export function validateMicronaunt(instance) {
  const missing = REQUIRED_METHODS.filter(m => typeof instance[m] !== "function");
  return { valid: missing.length === 0, missing };
}
