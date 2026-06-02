// micronaut-runtime.js — Micronaut Runtime Operations
//
// Pillar 2 of the complete micronaut execution stack:
//   Lifecycle     CREATED→INITIALIZING→READY→RUNNING→PAUSED→DEGRADED→RECOVERING→TERMINATING→TERMINATED
//   Scheduling    round-robin / priority / EDF / fair-share / cooperative / work-stealing
//   Resources     CPU / memory / GPU / network / storage allocation + monitoring
//   Fault Tolerance  circuit-breaker / retry-with-backoff / checkpoint / recovery
//   Observability    health-checks / metrics / counters / histograms
//
// K'UHUL physics connection:
//   RUNNING state  = Sek phase
//   PAUSED state   = Pop phase (snapshot stored)
//   DEGRADED state = KuhulPhysicsSolver near_horizon=true
//   RECOVERING     = reserve absorbing shock before live bounds tighten
//   TERMINATED     = Xul phase (resources released, metrics emitted)
//
// FLUX IR connection:
//   MicronautInstance.execute() IS FluxRuntime.dispatch(action)
//   checkpoint = TimeTraveler.record()
//   circuit_breaker = pressure reserve drain before tightening live constraints

// ─── Lifecycle states ─────────────────────────────────────────────────────────

export const STATE = Object.freeze({
  CREATED:      'CREATED',
  INITIALIZING: 'INITIALIZING',
  READY:        'READY',
  RUNNING:      'RUNNING',
  PAUSED:       'PAUSED',
  DEGRADED:     'DEGRADED',
  RECOVERING:   'RECOVERING',
  TERMINATING:  'TERMINATING',
  TERMINATED:   'TERMINATED',
});

const VALID_TRANSITIONS = Object.freeze({
  [STATE.CREATED]:      [STATE.INITIALIZING],
  [STATE.INITIALIZING]: [STATE.READY, STATE.DEGRADED],
  [STATE.READY]:        [STATE.RUNNING, STATE.TERMINATING],
  [STATE.RUNNING]:      [STATE.PAUSED, STATE.DEGRADED, STATE.TERMINATING],
  [STATE.PAUSED]:       [STATE.RUNNING, STATE.TERMINATING],
  [STATE.DEGRADED]:     [STATE.RECOVERING, STATE.TERMINATING],
  [STATE.RECOVERING]:   [STATE.RUNNING, STATE.DEGRADED, STATE.TERMINATING],
  [STATE.TERMINATING]:  [STATE.TERMINATED],
  [STATE.TERMINATED]:   [],
});

// ─── MicronautInstance ────────────────────────────────────────────────────────

export class MicronautInstance {
  constructor(id, name, opts = {}) {
    this.id          = id;
    this.name        = name;
    this.state       = STATE.CREATED;
    this.createdAt   = Date.now();
    this.startedAt   = null;
    this.checkpoint  = null;
    this._hooks      = { onCreate:[], onStart:[], onStop:[], onError:[] };
    this._resources  = {};
    this._metrics    = new MetricsCollector();
    this._health     = [];
    this.opts        = opts;
  }

  transition(next) {
    const allowed = VALID_TRANSITIONS[this.state] ?? [];
    if (!allowed.includes(next)) throw new Error(`${this.name}: invalid transition ${this.state}→${next}`);
    this.state = next;
    return this;
  }

  onHook(event, fn) { this._hooks[event]?.push(fn); return this; }

  async initialize(ctx = {}) {
    this.transition(STATE.INITIALIZING);
    for (const h of this._hooks.onCreate) await h(this, ctx);
    this.transition(STATE.READY);
    return this;
  }

  async start(ctx = {}) {
    this.transition(STATE.RUNNING);
    this.startedAt = Date.now();
    for (const h of this._hooks.onStart) await h(this, ctx);
    return this;
  }

  async pause() {
    this.transition(STATE.PAUSED);
    this.checkpoint = { state: this.state, ts: Date.now(), data: this.opts.serialize?.() };
    return this;
  }

  async resume() {
    if (this.checkpoint) this.opts.restore?.(this.checkpoint.data);
    this.transition(STATE.RUNNING);
    return this;
  }

  async stop(graceful = true) {
    this.transition(STATE.TERMINATING);
    for (const h of this._hooks.onStop) await h(this, { graceful });
    this.transition(STATE.TERMINATED);
    return this;
  }

  async recover(strategy = 'auto') {
    this.transition(STATE.RECOVERING);
    if (strategy === 'checkpoint' && this.checkpoint) {
      this.opts.restore?.(this.checkpoint.data);
    }
    this.transition(STATE.RUNNING);
    return this;
  }

  get uptime() { return this.startedAt ? Date.now() - this.startedAt : 0; }
  get metrics() { return this._metrics; }
}

// ─── SchedulerFactory ─────────────────────────────────────────────────────────

export class SchedulerFactory {
  static roundRobin(tasks, quantumMs = 100) {
    return async function* () {
      const q = [...tasks]; let i = 0;
      while (q.length) {
        const task = q[i % q.length];
        const t0 = Date.now();
        const r = await task.execute();
        if (Date.now() - t0 < quantumMs) q.splice(i % q.length, 1);
        else i++;
        yield r;
      }
    };
  }

  static priority(tasks) {
    const order = ['critical','high','normal','low','background'];
    return async function* () {
      const sorted = [...tasks].sort((a,b) => order.indexOf(a.priority??'normal') - order.indexOf(b.priority??'normal'));
      for (const t of sorted) yield await t.execute();
    };
  }

  static earliestDeadline(tasks) {
    return async function* () {
      const q = [...tasks];
      while (q.length) {
        q.sort((a,b) => (a.deadline??Infinity) - (b.deadline??Infinity));
        const t = q.shift();
        if (t.deadline && Date.now() > t.deadline) { yield { task: t.name, status: 'deadline_missed' }; continue; }
        yield await t.execute();
      }
    };
  }

  static workStealing(workers, taskQueues) {
    return Promise.all(workers.map(async (w, i) => {
      while (true) {
        let task = taskQueues[i].shift();
        if (!task) {
          for (let j = 0; j < workers.length; j++) {
            if (j !== i && taskQueues[j].length > 10) { task = taskQueues[j].shift(); if (task) break; }
          }
        }
        if (!task) break;
        await task.execute();
      }
    }));
  }

  static create(type, config = {}) {
    const map = { round_robin: this.roundRobin, priority: this.priority,
                  earliest_deadline: this.earliestDeadline };
    const fn = map[type];
    if (!fn) throw new Error(`Unknown scheduler: ${type}`);
    return { run: (tasks) => { const gen = fn.call(this, tasks, config.quantumMs); return collectGen(gen); } };
  }
}

async function collectGen(gen) {
  const out = [];
  for await (const v of gen()) out.push(v);
  return out;
}

// ─── CircuitBreaker ───────────────────────────────────────────────────────────

export class CircuitBreaker {
  constructor(opts = {}) {
    this._state     = 'CLOSED';
    this._failures  = 0;
    this._threshold = opts.failureThreshold ?? 5;
    this._timeoutMs = opts.timeoutMs ?? 60_000;
    this._nextAttempt = 0;
  }

  get state() { return this._state; }

  async call(fn, fallback = null) {
    if (this._state === 'OPEN') {
      if (Date.now() < this._nextAttempt) {
        if (fallback) return fallback();
        throw new Error('Circuit OPEN');
      }
      this._state = 'HALF_OPEN';
    }
    try {
      const r = await fn();
      if (this._state === 'HALF_OPEN') { this._failures = 0; this._state = 'CLOSED'; }
      this._failures = Math.max(0, this._failures - 0.5);
      return r;
    } catch (e) {
      this._failures++;
      if (this._failures >= this._threshold) {
        this._state = 'OPEN';
        this._nextAttempt = Date.now() + this._timeoutMs;
      }
      if (fallback) return fallback();
      throw e;
    }
  }

  reset() { this._state = 'CLOSED'; this._failures = 0; this._nextAttempt = 0; }
  metrics() { return { state: this._state, failures: this._failures, nextAttempt: this._nextAttempt }; }
}

// ─── RetryHandler ─────────────────────────────────────────────────────────────

export const RetryStrategies = Object.freeze({
  fixed:    ()  => 1000,
  linear:   (n) => n * 1000,
  expo:     (n) => Math.min(60_000, Math.pow(2, n) * 1000),
  expo_jitter: (n) => Math.min(60_000, Math.pow(2,n)*1000) + Math.random()*1000,
  fibonacci: (n) => ([0,1000,1000,2000,3000,5000,8000,13000][n] ?? 60_000),
});

export async function retry(fn, opts = {}) {
  const max = opts.maxAttempts ?? 3;
  const backoff = RetryStrategies[opts.strategy ?? 'expo'];
  const onRetry = opts.onRetry ?? (() => {});
  let last;
  for (let a = 1; a <= max; a++) {
    try { return await fn(); }
    catch (e) {
      last = e;
      if (a < max) { const d = backoff(a); onRetry(a, d, e); await sleep(d); }
    }
  }
  throw last;
}

// ─── MetricsCollector ─────────────────────────────────────────────────────────

export class MetricsCollector {
  constructor() { this._c = new Map(); this._g = new Map(); this._h = new Map(); }

  counter(name, delta = 1) { this._c.set(name, (this._c.get(name) ?? 0) + delta); }
  gauge(name, val)          { this._g.set(name, val); }
  histogram(name, val) {
    let h = this._h.get(name);
    if (!h) { h = []; this._h.set(name, h); }
    h.push(val);
  }

  timer(name, fn) {
    return async (...args) => {
      const t = Date.now();
      try { return await fn(...args); }
      finally { this.histogram(name, Date.now() - t); }
    };
  }

  pct(vals, p) {
    const s = [...vals].sort((a,b) => a-b);
    return s[Math.ceil(p/100*s.length)-1] ?? 0;
  }

  snapshot() {
    const hists = {};
    for (const [k,v] of this._h) {
      const sum = v.reduce((a,b)=>a+b,0);
      hists[k] = { count:v.length, min:Math.min(...v), max:Math.max(...v),
                   mean: sum/v.length, p50:this.pct(v,50), p90:this.pct(v,90), p99:this.pct(v,99) };
    }
    return { counters: Object.fromEntries(this._c), gauges: Object.fromEntries(this._g), histograms: hists };
  }

  reset() { this._c.clear(); this._g.clear(); this._h.clear(); }
}

// ─── HealthChecker ────────────────────────────────────────────────────────────

export class HealthChecker {
  constructor() { this._checks = new Map(); }

  register(name, checkFn, intervalMs = 30_000) {
    let status = 'unknown', lastError = null, lastCheck = null;
    const run = async () => {
      try {
        const r = await checkFn();
        status = r.healthy ? 'healthy' : 'unhealthy';
        lastError = r.error ?? null; lastCheck = Date.now();
      } catch (e) { status = 'unhealthy'; lastError = e.message; lastCheck = Date.now(); }
    };
    const iv = setInterval(run, intervalMs);
    this._checks.set(name, { run, stop: () => clearInterval(iv),
      status: () => status, lastError: () => lastError, lastCheck: () => lastCheck });
    return this;
  }

  async runAll() {
    const out = {};
    for (const [name, check] of this._checks) {
      await check.run();
      out[name] = { status: check.status(), lastError: check.lastError(), lastCheck: check.lastCheck() };
    }
    return out;
  }

  stopAll() { for (const c of this._checks.values()) c.stop(); }
}

// ─── CheckpointManager ────────────────────────────────────────────────────────

export class CheckpointManager {
  constructor() { this._checkpoints = new Map(); }

  save(instanceId, data) {
    const cp = { id: `cp_${Date.now()}`, instanceId, ts: Date.now(), data };
    const list = this._checkpoints.get(instanceId) ?? [];
    list.push(cp);
    while (list.length > 10) list.shift(); // keep last 10
    this._checkpoints.set(instanceId, list);
    return cp;
  }

  latest(instanceId) {
    const list = this._checkpoints.get(instanceId) ?? [];
    return list[list.length - 1] ?? null;
  }

  restore(instanceId) { return this.latest(instanceId)?.data ?? null; }
  history(instanceId) { return this._checkpoints.get(instanceId) ?? []; }
}

// ─── MicronautRuntime ─────────────────────────────────────────────────────────

export class MicronautRuntime {
  constructor() {
    this._instances   = new Map();
    this._checkpoints = new CheckpointManager();
    this._health      = new HealthChecker();
    this._breakers    = new Map();
    this._counter     = 0;
  }

  create(name, opts = {}) {
    const id = `µ-${++this._counter}-${name}`;
    const inst = new MicronautInstance(id, name, opts);
    this._instances.set(id, inst);
    return inst;
  }

  get(id) { return this._instances.get(id); }
  all()   { return [...this._instances.values()]; }

  breaker(name, opts) {
    if (!this._breakers.has(name)) this._breakers.set(name, new CircuitBreaker(opts));
    return this._breakers.get(name);
  }

  checkpoint(id) {
    const inst = this._instances.get(id);
    if (!inst) return null;
    return this._checkpoints.save(id, { state: inst.state, metrics: inst.metrics.snapshot() });
  }

  async recover(id, strategy = 'auto') {
    const inst = this._instances.get(id);
    if (!inst) return null;
    return inst.recover(strategy);
  }

  scheduler(type, config) { return SchedulerFactory.create(type, config); }

  summary() {
    const counts = {};
    for (const inst of this._instances.values()) counts[inst.state] = (counts[inst.state]??0)+1;
    return { total: this._instances.size, byState: counts };
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Runtime JSONL event emitters ─────────────────────────────────────────────

export function runtimeEventJSONL(event, data) {
  return JSON.stringify({ type: 'runtime_event', event, ...data, ts: Date.now() });
}
