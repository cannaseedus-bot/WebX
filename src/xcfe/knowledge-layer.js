// knowledge-layer.js — @who @what @where @when @cause @effect @event
//                       @mutate @reward @evolve @cdata @in @out
//                       @save @post @read @write @search
//
// "Computational epistemology" — XCFE's complete knowledge representation layer.
//
// Aristotelian categories:  @who @what @where @when
// Causal reasoning:         @cause @effect @event
// Adaptive systems:         @mutate @reward @evolve
// I/O & persistence:        @cdata @in @out @save @post @read @write @search
//
// K'UHUL opcode alignment:
//   @who    ≡ ⟁State⟁ 0x67  (actor identity = state track)
//   @what   ≡ ⟁Ten⟁!  0x24  (mutable entity definition)
//   @where  ≡ ⟁Path⟁  0x65  (spatial path select)
//   @when   ≡ ⟁Mon⟁   0x62  (temporal monitor)
//   @cause  ≡ ⟁Dec⟁   0x64  (decision = causal trigger)
//   @effect ≡ ⟁Trans⟁ 0x68  (transition = state change)
//   @event  ≡ ⟁Check⟁ 0x69  (checkpoint = observable)
//   @mutate ≡ ⟁Sek⟁!  0x08  (force-assign = mutation)
//   @reward ≡ ⟁Loss⟁  0x28  (loss inverse = reward signal)
//   @evolve ≡ ⟁Opt⟁   0x29  (optimizer = adapt)
//   @cdata  ≡ ⟁Tok⟁   0x22  (raw token content)
//   @in     ≡ ⟁Yax⟁   0x04  (get/access = read input)
//   @out    ≡ ⟁Ch'en⟁ 0x06  (store/persist = emit output)
//   @save   ≡ ⟁Save⟁  0x2F  (save model/state)
//   @post   ≡ ⟁Wo⟁    0x05  (call = publish)
//   @read   ≡ ⟁Load⟁  0x2E  (load = retrieve)
//   @write  ≡ ⟁Sek⟁   0x03  (set = record)
//   @search ≡ ⟁Eval⟁  0x30  (evaluate = score relevance)

import * as nodeFs from 'node:fs/promises';

// ─── @who — actor identification ──────────────────────────────────────────────

export class WhoContext {
  constructor(type = 'agent') {
    this.type        = type;   // 'agent' | 'system' | 'collective' | 'observer'
    this.identity    = null;
    this.permissions = new Set();
    this.audit       = [];
  }

  identify(id, meta = {}) {
    this.identity = { id, ...meta, ts: Date.now() };
    return this;
  }

  grant(...perms)  { perms.forEach(p => this.permissions.add(p)); return this; }
  revoke(...perms) { perms.forEach(p => this.permissions.delete(p)); return this; }
  can(perm)        { return this.permissions.has(perm) || this.permissions.has('*'); }

  log(action, data) {
    this.audit.push({ who: this.identity?.id, action, data, ts: Date.now() });
    return this;
  }
}

// ─── @what — entity specification ────────────────────────────────────────────

export class WhatEntity {
  constructor(type, schema = {}) {
    this.type     = type;
    this.schema   = schema;
    this._props   = { ...schema.properties };
    this.relations = [];   // { type, target }
  }

  property(name, typeDef) { this._props[name] = typeDef; return this; }
  relate(type, target)    { this.relations.push({ type, target }); return this; }

  validate(data) {
    const errors = [];
    for (const [field, def] of Object.entries(this._props)) {
      if (def.required && data[field] == null) errors.push(`${field} required`);
      if (def.type && data[field] != null && typeof data[field] !== def.type)
        errors.push(`${field} must be ${def.type}`);
    }
    return { valid: errors.length === 0, errors };
  }

  describe() {
    return {
      type: this.type,
      properties: this._props,
      relations: this.relations,
    };
  }
}

// ─── @where — spatial context ─────────────────────────────────────────────────

export class WhereContext {
  constructor() {
    this.physical = null;
    this.virtual  = null;
    this.logical  = null;
    this.storage  = null;
  }

  physical(lat, lon, alt = 0)  { this.physical = { lat, lon, alt }; return this; }
  network(host, port, path='') { this.virtual  = { host, port, path }; return this; }
  scope(levels)                { this.logical  = { levels, current: levels[0] }; return this; }
  storage(bucket, key)         { this.storage  = { bucket, key }; return this; }

  // Resolve the most specific location
  resolve() {
    return this.logical ?? this.virtual ?? this.physical ?? this.storage ?? { scope: 'global' };
  }
}

// ─── @when — temporal context ─────────────────────────────────────────────────

export class WhenContext {
  constructor() {
    this._ts         = Date.now();
    this._interval   = null;
    this._sequence   = [];
    this._frequency  = null;
  }

  now()                          { this._ts = Date.now(); return this; }
  at(ts)                         { this._ts = ts; return this; }
  interval(start, end)           { this._interval = { start, end, duration: end - start }; return this; }
  sequence(...events)            { this._sequence = events; return this; }
  frequency(pattern)             { this._frequency = pattern; return this; }

  // Check if a timestamp falls within this context's interval
  contains(ts) {
    if (!this._interval) return true;
    return ts >= this._interval.start && ts <= this._interval.end;
  }

  // Ordering: does event A precede event B in the declared sequence?
  precedes(a, b) {
    const ia = this._sequence.indexOf(a), ib = this._sequence.indexOf(b);
    return ia !== -1 && ib !== -1 && ia < ib;
  }

  toISO() { return new Date(this._ts).toISOString(); }
  get ts() { return this._ts; }
}

// ─── @cause — trigger & reason ────────────────────────────────────────────────

export class CausalModel {
  constructor() {
    this._graph    = new Map();   // cause → [effects]
    this._triggers = new Map();   // condition_fn → effect_name
  }

  link(cause, effect)           { if (!this._graph.has(cause)) this._graph.set(cause, []); this._graph.get(cause).push(effect); return this; }
  trigger(condFn, effectName)   { this._triggers.set(condFn, effectName); return this; }

  // Find all downstream effects of a cause
  downstream(cause, visited = new Set()) {
    if (visited.has(cause)) return [];
    visited.add(cause);
    const direct = this._graph.get(cause) ?? [];
    return [...direct, ...direct.flatMap(e => this.downstream(e, visited))];
  }

  // Evaluate which triggers fire given current state
  evaluate(state) {
    const fired = [];
    for (const [cond, effect] of this._triggers) {
      try { if (cond(state)) fired.push(effect); } catch {}
    }
    return fired;
  }

  // Root-cause analysis: work backwards from an effect
  rootCauses(effect) {
    const roots = [];
    for (const [cause, effects] of this._graph) {
      if (effects.includes(effect)) {
        const upstreamCauses = this.rootCauses(cause);
        roots.push(...(upstreamCauses.length ? upstreamCauses : [cause]));
      }
    }
    return roots.length ? roots : [effect];
  }
}

// ─── @effect — outcome & consequence ─────────────────────────────────────────

export class EffectMeasure {
  constructor() { this._measurements = []; }

  record(label, before, after, unit = '') {
    const delta = typeof before === 'number' ? after - before : null;
    this._measurements.push({ label, before, after, delta, unit, ts: Date.now() });
    return this;
  }

  // Effect size (Cohen's d for numeric measurements)
  cohensD() {
    const deltas = this._measurements.filter(m => m.delta != null).map(m => m.delta);
    if (!deltas.length) return 0;
    const mean  = deltas.reduce((s, v) => s + v, 0) / deltas.length;
    const std   = Math.sqrt(deltas.reduce((s, v) => s + (v - mean)**2, 0) / deltas.length);
    return std > 0 ? mean / std : 0;
  }

  summary() {
    return this._measurements.map(({ label, delta, unit }) => ({ label, delta, unit }));
  }
}

// ─── @event — observable occurrence ──────────────────────────────────────────

export class EventBus {
  constructor() {
    this._handlers = new Map();   // type → [handler]
    this._history  = [];
    this._schemas  = new Map();
  }

  defineSchema(type, schema) { this._schemas.set(type, schema); return this; }

  on(type, handler) {
    if (!this._handlers.has(type)) this._handlers.set(type, []);
    this._handlers.get(type).push(handler);
    return () => this.off(type, handler);
  }

  off(type, handler) {
    const list = this._handlers.get(type);
    if (list) { const i = list.indexOf(handler); if (i >= 0) list.splice(i, 1); }
  }

  async emit(type, data, who = null) {
    const event = { type, data, who, ts: Date.now(), id: `${type}_${Date.now()}` };
    this._history.push(event);
    const handlers = this._handlers.get(type) ?? [];
    const wildcard = this._handlers.get('*')  ?? [];
    await Promise.all([...handlers, ...wildcard].map(h => h(event)));
    return event;
  }

  history(type = null, limit = 100) {
    const events = type ? this._history.filter(e => e.type === type) : this._history;
    return events.slice(-limit);
  }

  // Index events for search
  buildIndex() {
    const index = {};
    for (const event of this._history) {
      const key = event.type;
      index[key] = (index[key] ?? 0) + 1;
    }
    return index;
  }
}

// ─── @mutate — state transformation ──────────────────────────────────────────

export class MutationEngine {
  constructor() { this._journal = []; }

  // Apply a mutation and record it
  apply(state, mutation) {
    const before   = JSON.parse(JSON.stringify(state));
    const after    = typeof mutation === 'function' ? mutation(state) : { ...state, ...mutation };
    const journal  = { before, after, ts: Date.now(), delta: this._diff(before, after) };
    this._journal.push(journal);
    Object.assign(state, after);
    return journal;
  }

  // Atomic mutation: only commits if validator passes
  atomic(state, mutation, validator) {
    const candidate = typeof mutation === 'function'
      ? mutation({ ...state })
      : { ...state, ...mutation };
    const result = validator(candidate);
    if (!result.valid) throw new Error(`mutation rejected: ${result.errors?.join(', ')}`);
    return this.apply(state, mutation);
  }

  // Rollback to a previous journal entry
  rollback(state, steps = 1) {
    const entry = this._journal[this._journal.length - steps];
    if (!entry) return state;
    Object.assign(state, entry.before);
    return entry.before;
  }

  _diff(before, after) {
    const delta = {};
    const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
    for (const k of keys) {
      if (JSON.stringify(before[k]) !== JSON.stringify(after[k])) {
        delta[k] = { from: before[k], to: after[k] };
      }
    }
    return delta;
  }

  get journal() { return [...this._journal]; }
}

// ─── @reward — reinforcement signal ──────────────────────────────────────────

export class RewardFunction {
  constructor(name, fn, opts = {}) {
    this.name     = name;
    this._fn      = fn;
    this._history = [];
    this._scale   = opts.scale ?? 1.0;
    this._clip    = opts.clip  ?? null;
  }

  compute(state, action, nextState) {
    let r = this._fn(state, action, nextState) * this._scale;
    if (this._clip) r = Math.max(this._clip[0], Math.min(this._clip[1], r));
    this._history.push({ state, action, reward: r, ts: Date.now() });
    return r;
  }

  // Q-learning update: Q(s,a) += lr * (r + gamma*max_Q(s') - Q(s,a))
  static qUpdate(Q, state, action, reward, nextState, lr = 0.1, gamma = 0.99) {
    const key      = `${state}:${action}`;
    const current  = Q.get(key) ?? 0;
    const nextMax  = Math.max(0, ...[...Q.entries()]
      .filter(([k]) => k.startsWith(`${nextState}:`))
      .map(([, v]) => v));
    Q.set(key, current + lr * (reward + gamma * nextMax - current));
    return Q;
  }

  average() {
    const rewards = this._history.map(h => h.reward);
    return rewards.length ? rewards.reduce((s, v) => s + v, 0) / rewards.length : 0;
  }
}

// ─── @evolve — adaptation & learning ─────────────────────────────────────────

export class EvolutionEngine {
  constructor(opts = {}) {
    this._popSize    = opts.population_size  ?? 100;
    this._mutRate    = opts.mutation_rate    ?? 0.1;
    this._crossRate  = opts.crossover_rate   ?? 0.8;
    this._generation = 0;
    this._population = [];
    this._fitness    = new Map();
  }

  initialize(genFn) {
    this._population = Array.from({ length: this._popSize }, (_, i) => genFn(i));
    return this;
  }

  evaluate(fitFn) {
    for (const individual of this._population) {
      const key = JSON.stringify(individual);
      this._fitness.set(key, fitFn(individual));
    }
    return this;
  }

  select(n = 2) {
    return [...this._population]
      .sort((a, b) => (this._fitness.get(JSON.stringify(b)) ?? 0) - (this._fitness.get(JSON.stringify(a)) ?? 0))
      .slice(0, n);
  }

  // One generation: select → crossover → mutate
  step(fitFn, crossFn, mutFn) {
    this.evaluate(fitFn);
    const elite    = this.select(Math.max(2, Math.floor(this._popSize * 0.1)));
    const nextPop  = [...elite];
    while (nextPop.length < this._popSize) {
      const [a, b] = elite.sort(() => Math.random() - 0.5).slice(0, 2);
      let child = Math.random() < this._crossRate ? crossFn(a, b) : { ...a };
      if (Math.random() < this._mutRate) child = mutFn(child);
      nextPop.push(child);
    }
    this._population = nextPop;
    this._generation++;
    return this;
  }

  best()        { return this.select(1)[0]; }
  get generation() { return this._generation; }
}

// ─── @cdata — character data content ─────────────────────────────────────────

export class CData {
  constructor(content = '', encoding = 'utf8') {
    this._raw      = content;
    this._encoding = encoding;
  }

  // Wrap in XML CDATA section
  toCDATASection() {
    const safe = this._raw.replace(/]]>/g, ']]>]]<![CDATA[>');
    return `<![CDATA[${safe}]]>`;
  }

  // Split into chunks for streaming
  *chunks(size = 65536) {
    let pos = 0;
    while (pos < this._raw.length) {
      yield this._raw.slice(pos, pos + size);
      pos += size;
    }
  }

  toBase64()  { return Buffer.from(this._raw, this._encoding).toString('base64'); }
  fromBase64(b64) { this._raw = Buffer.from(b64, 'base64').toString(this._encoding); return this; }
  get raw()   { return this._raw; }
  get length(){ return this._raw.length; }
}

// ─── @in / @out — I/O boundaries ─────────────────────────────────────────────

export class IOBoundary {
  constructor(direction = 'in') {
    this.direction = direction;
    this._pipeline = [];
    this._ctx      = {};
  }

  transform(fn) { this._pipeline.push(fn); return this; }

  async process(data) {
    let current = data;
    for (const fn of this._pipeline) current = await fn(current, this._ctx);
    return current;
  }

  static input()  { return new IOBoundary('in'); }
  static output() { return new IOBoundary('out'); }
}

// ─── @save / @read / @write — persistence ────────────────────────────────────

export class PersistenceLayer {
  constructor() {
    this._store    = new Map();   // in-memory fallback
    this._versions = new Map();   // key → [{version, data, ts}]
  }

  async save(key, data, opts = {}) {
    const entry = { data, ts: Date.now(), version: (this._versions.get(key)?.length ?? 0) + 1 };
    if (!this._versions.has(key)) this._versions.set(key, []);
    this._versions.get(key).push(entry);
    this._store.set(key, data);
    if (opts.path) {
      await nodeFs.writeFile(opts.path, typeof data === 'string' ? data : JSON.stringify(data))
        .catch(() => {});
    }
    return entry;
  }

  async read(key, opts = {}) {
    if (opts.path) {
      const text = await nodeFs.readFile(opts.path, 'utf8').catch(() => null);
      if (text !== null) {
        try { return JSON.parse(text); } catch { return text; }
      }
    }
    return this._store.get(key) ?? null;
  }

  write(key, data) { this._store.set(key, data); return { key, written: true, ts: Date.now() }; }

  history(key) { return this._versions.get(key) ?? []; }
  rollback(key, version) {
    const hist = this._versions.get(key) ?? [];
    const entry = hist.find(e => e.version === version);
    if (entry) this._store.set(key, entry.data);
    return entry?.data ?? null;
  }
}

// ─── @post — publication ──────────────────────────────────────────────────────

export class Publisher {
  constructor() { this._subscribers = new Map(); }

  subscribe(topic, fn) {
    if (!this._subscribers.has(topic)) this._subscribers.set(topic, []);
    this._subscribers.get(topic).push(fn);
    return () => this.unsubscribe(topic, fn);
  }

  unsubscribe(topic, fn) {
    const subs = this._subscribers.get(topic);
    if (subs) { const i = subs.indexOf(fn); if (i >= 0) subs.splice(i, 1); }
  }

  async publish(topic, message) {
    const subs    = this._subscribers.get(topic) ?? [];
    const wildcard= this._subscribers.get('*')   ?? [];
    const payload = { topic, message, ts: Date.now() };
    await Promise.all([...subs, ...wildcard].map(fn => fn(payload)));
    return payload;
  }
}

// ─── @search — information retrieval ──────────────────────────────────────────

export class SearchEngine {
  constructor() {
    this._index = new Map();   // term → Set of doc ids
    this._docs  = new Map();   // doc id → { content, score }
  }

  // Index a document
  index(id, content, terms = null) {
    const words = terms ?? String(content).toLowerCase().match(/\w+/g) ?? [];
    this._docs.set(id, { content, words });
    for (const word of words) {
      if (!this._index.has(word)) this._index.set(word, new Set());
      this._index.get(word).add(id);
    }
    return this;
  }

  // Boolean search: AND of all terms
  search(query, limit = 10) {
    const terms = query.toLowerCase().match(/\w+/g) ?? [];
    if (!terms.length) return [];

    // Start with docs matching first term
    let matches = new Set(this._index.get(terms[0]) ?? []);
    for (const term of terms.slice(1)) {
      const termDocs = this._index.get(term) ?? new Set();
      matches = new Set([...matches].filter(id => termDocs.has(id)));
    }

    // BM25-style scoring
    const N = this._docs.size;
    const avgLen = [...this._docs.values()].reduce((s, d) => s + d.words.length, 0) / Math.max(1, N);
    const scored = [...matches].map(id => {
      const doc = this._docs.get(id);
      const len = doc.words.length;
      const score = terms.reduce((s, term) => {
        const tf  = doc.words.filter(w => w === term).length;
        const df  = this._index.get(term)?.size ?? 0;
        const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
        return s + idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * len / avgLen));
      }, 0);
      return { id, score, content: doc.content };
    });

    return scored.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  // Semantic: cosine similarity between query vector and doc vectors
  semanticSearch(queryVec, docVecs, topK = 5) {
    const norm = v => Math.sqrt(v.reduce((s, x) => s + x*x, 0));
    const dot  = (a, b) => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0);
    const qn   = norm(queryVec);
    const results = docVecs.map(({ id, vec }) => ({
      id,
      score: qn && norm(vec) ? dot(queryVec, vec) / (qn * norm(vec)) : 0,
    }));
    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }

  get size() { return this._docs.size; }
}

// ─── Register knowledge namespaces into XCFENodeRuntime ───────────────────────

export function registerKnowledgeNamespaces(rt) {
  const bus   = new EventBus();
  const store = new PersistenceLayer();
  const pub   = new Publisher();
  const search= new SearchEngine();
  const causes= new CausalModel();
  const effects= new EffectMeasure();
  const mutations= new MutationEngine();

  rt._handlers.set('@who', (val, ctx) => {
    const type = Object.keys(val).find(k => !k.startsWith('@')) ?? 'agent';
    const who  = new WhoContext(type);
    const id   = val[type]?.id ?? val.id ?? 'system';
    who.identify(id, val[type] ?? {});
    ctx['_who'] = who;
    if (val['@store']) ctx[val['@store']] = who;
    return who;
  });

  rt._handlers.set('@what', (val, ctx) => {
    const type = val.type ?? val['@name'] ?? 'entity';
    const what = new WhatEntity(type, val.schema ?? {});
    ctx['_what'] = what;
    if (val['@store']) ctx[val['@store']] = what;
    return what;
  });

  rt._handlers.set('@where', (val, ctx) => {
    const where = new WhereContext();
    if (val.coordinates) where.physical(val.coordinates.latitude, val.coordinates.longitude);
    if (val['@network.address']) where.network(val['@network.address'].host, val['@network.address'].port);
    if (val['@path.namespace']) where.scope(val['@path.namespace'].scope ?? ['global']);
    ctx['_where'] = where;
    if (val['@store']) ctx[val['@store']] = where.resolve();
    return where;
  });

  rt._handlers.set('@when', (val, ctx) => {
    const when = new WhenContext();
    if (val.timestamp) when.at(val.timestamp);
    if (val.start && val.end) when.interval(val.start, val.end);
    if (val.sequence) when.sequence(...val.sequence);
    ctx['_when'] = when;
    if (val['@store']) ctx[val['@store']] = when.toISO();
    return when;
  });

  rt._handlers.set('@cause', (val, ctx) => {
    if (val.from && val.to) causes.link(val.from, val.to);
    const state  = ctx;
    const fired  = causes.evaluate(state);
    if (val['@store']) ctx[val['@store']] = { fired, roots: causes.rootCauses(val.effect ?? '') };
    return { causes, fired };
  });

  rt._handlers.set('@effect', (val, ctx) => {
    const label  = val.metric ?? 'delta';
    const before = ctx[val.from] ?? 0;
    const after  = ctx[val.to]   ?? 0;
    effects.record(label, before, after, val.unit);
    if (val['@store']) ctx[val['@store']] = { effect_size: effects.cohensD(), measurements: effects.summary() };
    return effects;
  });

  rt._handlers.set('@event', async (val, ctx) => {
    if (val['@event.definition']) {
      const { name, schema } = val['@event.definition'];
      bus.defineSchema(name, schema);
    }
    if (val['@event.emitter'] || val.type) {
      const type = val.type ?? 'event';
      const data = ctx[val.data] ?? val.data ?? {};
      const evt  = await bus.emit(type, data, ctx['_who']?.identity);
      if (val['@store']) ctx[val['@store']] = evt;
    }
    return bus;
  });

  rt._handlers.set('@mutate', (val, ctx) => {
    const state    = ctx[val.target] ?? ctx;
    const mutation = val.set ?? val.mutation ?? {};
    const journal  = mutations.apply(state, mutation);
    if (val['@store']) ctx[val['@store']] = journal.delta;
    return journal;
  });

  rt._handlers.set('@reward', (val, ctx) => {
    const metricKey = val.metric ?? 'value';
    const rfn = new RewardFunction(val.name ?? 'reward', (s, a, ns) => {
      const v = ns[metricKey] ?? s[metricKey] ?? 0;
      const b = val.baseline ?? 0;
      return v - b;
    }, { clip: val.clip, scale: val.scale ?? 1 });
    const reward = rfn.compute(ctx, val.action ?? '', ctx);
    if (val['@store']) ctx[val['@store']] = reward;
    return reward;
  });

  rt._handlers.set('@evolve', (val, ctx) => {
    const engine = new EvolutionEngine({
      population_size: val.population_size ?? 10,
      mutation_rate:   val.mutation_rate   ?? 0.1,
      crossover_rate:  val.crossover_rate  ?? 0.8,
    });
    engine.initialize(i => ({ id: i, value: Math.random() }));
    engine.evaluate(ind => ind.value);
    const best = engine.best();
    if (val['@store']) ctx[val['@store']] = best;
    return engine;
  });

  rt._handlers.set('@cdata', (val, ctx) => {
    const src     = ctx[val.source] ?? val.content ?? val.data ?? '';
    const cdata   = new CData(typeof src === 'string' ? src : JSON.stringify(src));
    if (val['@store']) ctx[val['@store']] = cdata.raw;
    return cdata;
  });

  rt._handlers.set('@in', (val, ctx) => {
    const boundary = IOBoundary.input();
    const data     = ctx[val.source] ?? val.data;
    if (data !== undefined && val['@store']) ctx[val['@store']] = data;
    return boundary;
  });

  rt._handlers.set('@out', (val, ctx) => {
    const boundary = IOBoundary.output();
    const result   = ctx[val.source] ?? val.data ?? ctx['_result'];
    if (val['@store']) ctx[val['@store']] = result;
    return boundary;
  });

  rt._handlers.set('@save', async (val, ctx) => {
    const key  = val.key ?? val['@name'] ?? 'default';
    const data = ctx[val.data] ?? val.data ?? ctx;
    const entry= await store.save(key, data, { path: val.path });
    if (val['@store']) ctx[val['@store']] = entry;
    return entry;
  });

  rt._handlers.set('@post', async (val, ctx) => {
    const topic   = val.topic ?? val['@name'] ?? 'event';
    const message = ctx[val.message] ?? val.message ?? val.body ?? {};
    const result  = await pub.publish(topic, message);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@read', async (val, ctx) => {
    const key  = val.key ?? val['@name'] ?? 'default';
    const data = await store.read(key, { path: val.path });
    ctx[key]   = data;
    if (val['@store']) ctx[val['@store']] = data;
    return data;
  });

  rt._handlers.set('@write', (val, ctx) => {
    const key  = val.key ?? val['@name'] ?? 'default';
    const data = ctx[val.source] ?? val.data ?? val.content;
    const r    = store.write(key, data);
    if (val['@store']) ctx[val['@store']] = r;
    return r;
  });

  rt._handlers.set('@search', (val, ctx) => {
    const query = ctx[val.query] ?? val.query ?? '';
    const docs  = ctx[val.corpus] ?? val.corpus ?? [];
    if (Array.isArray(docs)) {
      docs.forEach((d, i) => search.index(d.id ?? i, d.content ?? d, d.terms));
    }
    const results = search.search(query, val.limit ?? 10);
    if (val['@store']) ctx[val['@store']] = results;
    return results;
  });
}

// ─── Knowledge @ opcode alignment ─────────────────────────────────────────────

export const KNOWLEDGE_OPCODE_MAP = Object.freeze({
  '@who':    { kuhul: '⟁State⟁ 0x67', description: 'actor identity / state track' },
  '@what':   { kuhul: '⟁Ten⟁! 0x24',  description: 'mutable entity definition' },
  '@where':  { kuhul: '⟁Path⟁ 0x65',  description: 'spatial path select' },
  '@when':   { kuhul: '⟁Mon⟁ 0x62',   description: 'temporal monitor' },
  '@cause':  { kuhul: '⟁Dec⟁ 0x64',   description: 'decision = causal trigger' },
  '@effect': { kuhul: '⟁Trans⟁ 0x68', description: 'transition = state change' },
  '@event':  { kuhul: '⟁Check⟁ 0x69', description: 'checkpoint = observable' },
  '@mutate': { kuhul: '⟁Sek⟁! 0x08',  description: 'force-assign = mutation' },
  '@reward': { kuhul: '⟁Loss⟁ 0x28',  description: 'loss inverse = reward signal' },
  '@evolve': { kuhul: '⟁Opt⟁ 0x29',   description: 'optimizer = adapt' },
  '@cdata':  { kuhul: '⟁Tok⟁ 0x22',   description: 'raw token content' },
  '@in':     { kuhul: '⟁Yax⟁ 0x04',   description: 'get / read input' },
  '@out':    { kuhul: "⟁Ch'en⟁ 0x06", description: 'store / emit output' },
  '@save':   { kuhul: '⟁Save⟁ 0x2F',  description: 'save model/state' },
  '@post':   { kuhul: '⟁Wo⟁ 0x05',    description: 'call = publish' },
  '@read':   { kuhul: '⟁Load⟁ 0x2E',  description: 'load = retrieve' },
  '@write':  { kuhul: '⟁Sek⟁ 0x03',   description: 'set = record' },
  '@search': { kuhul: '⟁Eval⟁ 0x30',  description: 'evaluate = score relevance' },
});
