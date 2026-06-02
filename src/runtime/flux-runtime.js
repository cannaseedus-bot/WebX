// flux-runtime.js — FLUX Runtime: deterministic state machine
//
// The bridge between timeless math functions and physical execution.
//
//   Layer 0: Math / pure functions  (no time)
//   Layer 1: FLUX IR                (time made explicit via actions)
//   Layer 2: FLUX Runtime           (THIS FILE — lives in time)
//   Layer 3: CPU / OS / Network
//
// K'UHUL phase mapping (one action = one full cycle):
//   Pop   — dequeue next action, snapshot state_before
//   Wo    — send action to all store reducers
//   Sek   — reducers compute new state (pure, no side effects)
//   Ch'en — commit new state, notify subscribers, process effects
//
// µMODEL connection:
//   A µMODEL KXML spec is a FLUX IR program.
//   The CDATA kuhul capsule is the reducer body.
//   SemanticReader.read(kxml) produces the topology that
//   FluxRuntime.registerFromDescriptor(descriptor) compiles into stores.

// ─── ActionQueue ─────────────────────────────────────────────────────────────

export class ActionQueue {
  constructor() {
    this._q = [];
    this._seq = 0;
  }
  push(action) {
    this._q.push({ ...action, _seq: this._seq++, _ts: Date.now() });
  }
  pop()  { return this._q.shift() ?? null; }
  peek() { return this._q[0] ?? null; }
  get size() { return this._q.length; }
  drain() { const all = this._q.slice(); this._q.length = 0; return all; }
}

// ─── StoreRegistry ────────────────────────────────────────────────────────────

export class StoreRegistry {
  constructor() {
    this._stores     = new Map();
    this._subscribers = new Map();
  }

  register(name, initialState, reducer) {
    this._stores.set(name, { state: initialState, reducer });
    this._subscribers.set(name, new Set());
  }

  getState(name) {
    const s = this._stores.get(name);
    return s ? structuredClone(s.state) : undefined;
  }

  snapshot() {
    const out = {};
    for (const [name, s] of this._stores) out[name] = structuredClone(s.state);
    return out;
  }

  restore(snapshot) {
    for (const [name, state] of Object.entries(snapshot))
      if (this._stores.has(name)) this._stores.get(name).state = structuredClone(state);
  }

  applyAction(action) {
    const changed = [];
    for (const [name, store] of this._stores) {
      const next = store.reducer(store.state, action);
      if (next !== store.state) { store.state = next; changed.push(name); }
    }
    return changed;
  }

  subscribe(storeName, fn) {
    this._subscribers.get(storeName)?.add(fn);
    return () => this._subscribers.get(storeName)?.delete(fn);
  }

  notify(changed) {
    for (const name of changed)
      for (const fn of (this._subscribers.get(name) ?? []))
        try { fn(this._stores.get(name).state, name); } catch (_) {}
  }
}

// ─── EffectEngine ─────────────────────────────────────────────────────────────
//
// Effects are declarative descriptions of async work.
// They return actions (back into the queue) on completion.
// They never mutate state directly.

export class EffectEngine {
  constructor(dispatch) {
    this._effects  = new Set();
    this._dispatch = dispatch;
  }

  add(effect) {
    this._effects.add(effect);
    effect.start?.();
  }

  tick() {
    for (const e of this._effects) {
      if (e.ready()) {
        this._effects.delete(e);
        const action = e.result();
        if (action) this._dispatch(action);
      }
    }
  }

  cancelAll() { this._effects.clear(); }
  get pending() { return this._effects.size; }
}

// ─── Built-in Effect types ────────────────────────────────────────────────────

export class TimerEffect {
  constructor(delayMs, resultFn) {
    this._deadline = Date.now() + delayMs;
    this._result   = resultFn;
    this._done     = false;
  }
  start() {}
  ready() { if (!this._done && Date.now() >= this._deadline) this._done = true; return this._done; }
  result() { return this._result(); }
}

export class PromiseEffect {
  constructor(promise, onSuccess, onError) {
    this._state  = 'pending';
    this._action = null;
    promise
      .then(v => { this._action = onSuccess(v); this._state = 'done'; })
      .catch(e => { this._action = onError(e);  this._state = 'done'; });
  }
  start() {}
  ready()  { return this._state === 'done'; }
  result() { return this._action; }
}

// ─── TimeTraveler ─────────────────────────────────────────────────────────────

export class TimeTraveler {
  constructor() {
    this._log = [];
    this._cursor = -1;
  }

  record(action, stateBefore, stateAfter) {
    // Truncate forward history on new action
    this._log.splice(this._cursor + 1);
    this._log.push({ action, stateBefore, stateAfter, ts: Date.now() });
    this._cursor = this._log.length - 1;
  }

  canGoBack()    { return this._cursor > 0; }
  canGoForward() { return this._cursor < this._log.length - 1; }

  back() {
    if (!this.canGoBack()) return null;
    this._cursor--;
    return this._log[this._cursor].stateAfter;
  }

  forward() {
    if (!this.canGoForward()) return null;
    this._cursor++;
    return this._log[this._cursor].stateAfter;
  }

  jumpTo(index) {
    if (index < 0 || index >= this._log.length) return null;
    this._cursor = index;
    return this._log[index].stateAfter;
  }

  get history() { return this._log.map((e, i) => ({ ...e, index: i, current: i === this._cursor })); }
  get length()   { return this._log.length; }
}

// ─── FluxRuntime ──────────────────────────────────────────────────────────────

export class FluxRuntime {
  constructor({ enableTimeTravel = true } = {}) {
    this._queue      = new ActionQueue();
    this._stores     = new StoreRegistry();
    this._effects    = new EffectEngine(a => this.dispatch(a));
    this._traveler   = enableTimeTravel ? new TimeTraveler() : null;
    this._running    = false;
    this._processing = false;
    this._globalSubs = new Set();
    this._phase      = 'idle'; // Pop | Wo | Sek | Ch'en | idle
  }

  // ── Store registration ──

  registerStore(name, initialState, reducer) {
    this._stores.register(name, initialState, reducer);
    return this;
  }

  getState(name) { return this._stores.getState(name); }
  snapshot()     { return this._stores.snapshot(); }

  // ── Dispatch ──

  dispatch(action) {
    if (!action?.type) throw new Error(`FluxRuntime: action must have a type`);
    this._queue.push(action);
    if (!this._processing) this._flush();
    return this;
  }

  // ── Subscribe ──

  subscribe(storeOrFn, fn) {
    if (typeof storeOrFn === 'function') {
      this._globalSubs.add(storeOrFn); return () => this._globalSubs.delete(storeOrFn);
    }
    return this._stores.subscribe(storeOrFn, fn);
  }

  // ── Effect registration ──

  addEffect(effect) {
    this._effects.add(effect);
    return this;
  }

  after(delayMs, actionFn) {
    this.addEffect(new TimerEffect(delayMs, actionFn));
    return this;
  }

  fromPromise(promise, onSuccess, onError = e => ({ type: 'EFFECT_ERROR', error: String(e) })) {
    this.addEffect(new PromiseEffect(promise, onSuccess, onError));
    return this;
  }

  // ── Time travel ──

  get timeTravel() { return this._traveler; }

  travelTo(index) {
    if (!this._traveler) return;
    const snap = this._traveler.jumpTo(index);
    if (snap) { this._stores.restore(snap); this._notifyAll(); }
  }

  // ── Internal flush loop ──

  _flush() {
    this._processing = true;
    while (this._queue.size > 0) {
      const action = this._queue.pop();

      // Pop — snapshot state before
      this._phase = 'Pop';
      const before = this._traveler ? this._stores.snapshot() : null;

      // Wo + Sek — dispatch to reducers
      this._phase = 'Wo';
      const changed = this._stores.applyAction(action);

      // Ch'en — notify + record
      this._phase = "Ch'en";
      const after = this._traveler ? this._stores.snapshot() : null;
      if (this._traveler) this._traveler.record(action, before, after);

      this._stores.notify(changed);
      this._notifyAll(action);
    }

    // Tick effects after queue drains
    this._effects.tick();
    this._phase = 'idle';
    this._processing = false;

    // If effects dispatched new actions, flush again
    if (this._queue.size > 0) this._flush();
  }

  _notifyAll(action) {
    for (const fn of this._globalSubs)
      try { fn(this.snapshot(), action); } catch (_) {}
  }

  // ── µMODEL integration ──
  //
  // Register a µPY model descriptor as a store.
  // The descriptor's cdata.kuhul_programs become the reducer body.

  registerFromDescriptor(descriptor) {
    const name = descriptor.domain;
    const initial = {
      domain:       name,
      phase:        descriptor.phase,
      gravity:      descriptor.gravity,
      capabilities: descriptor.capabilities,
      activation:   descriptor.activation,
      cdata:        descriptor.cdata,
      running:      false,
    };
    const reducer = (state, action) => {
      switch (action.type) {
        case `${name}/ACTIVATE`:   return { ...state, running: true,  activation: { ...state.activation, active: true } };
        case `${name}/DEACTIVATE`: return { ...state, running: false, activation: { ...state.activation, active: false } };
        case `${name}/PHASE`:      return { ...state, phase: action.phase };
        case `${name}/RESULT`:     return { ...state, lastResult: action.result };
        default: return state;
      }
    };
    this.registerStore(name, initial, reducer);
    return this;
  }

  // ── Introspection ──

  get phase()      { return this._phase; }
  get queueSize()  { return this._queue.size; }
  get effectCount(){ return this._effects.pending; }
}

// ─── Minimal standalone runtime factory ───────────────────────────────────────

export function createRuntime(stores = {}, opts = {}) {
  const rt = new FluxRuntime(opts);
  for (const [name, { state, reducer }] of Object.entries(stores))
    rt.registerStore(name, state, reducer);
  return rt;
}
