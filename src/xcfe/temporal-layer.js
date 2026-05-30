// temporal-layer.js — @flux / @tick / @thread / @batch / @round
//                     @step / @mark / @map / @graph
//
// Temporal + reactive + concurrent + dataflow fabric for XCFE.
//
// Stack (bottom → top):
//   @graph   — DAG orchestration (nodes, edges, topo sort, fusion)
//   @map     — pure transformation (function, accumulate, filter, flatten)
//   @mark    — metadata + checkpoints (timestamp, offset, watermark, GC)
//   @step    — incremental state machine (init, increment, yield, resume)
//   @round   — quantization + alignment (floor, ceil, nearest, time rounding)
//   @batch   — bulk processing (window, adaptive, commit, strategy)
//   @thread  — concurrency (work-steal pool, barrier, priorities, channels)
//   @tick    — temporal events (interval, watermark, deadline, schedule)
//   @flux    — reactive streams (source, operators, sink, backpressure)
//
// K'UHUL opcode alignment:
//   @flux    ≡ ⟁Dist⟁ 0x44   — distribute stream across nodes
//   @tick    ≡ ⟁Mon⟁  0x62   — monitor periodic metric
//   @thread  ≡ ⟁Clu⟁  0x40   — cluster of concurrent workers
//   @batch   ≡ ⟁Batch⟁ 0x34  — batch data
//   @round   ≡ ⟁Norm⟁  0x2B  — normalize / quantize
//   @step    ≡ ⟁Wo⟁each 0x0B — iterator step
//   @mark    ≡ ⟁Check⟁ 0x69  — checkpoint
//   @map     ≡ ⟁Log⟁   0x23  — logic node transform
//   @graph   ≡ ⟁XCFE⟁  0x60  — XCFE control (DAG execution)

import { EventEmitter } from 'node:events';

// ─── @tick — temporal events ──────────────────────────────────────────────────

export class Tick extends EventEmitter {
  constructor(opts = {}) {
    super();
    this._hz       = opts.frequency_hz ?? 1;    // ticks per second
    this._interval = null;
    this._count    = 0;
    this._watermark= 0;
  }

  start() {
    const ms = 1000 / this._hz;
    this._interval = setInterval(() => {
      this._count++;
      this._watermark = Date.now();
      this.emit('tick', { count: this._count, time: this._watermark });
    }, ms);
    return this;
  }

  stop() { clearInterval(this._interval); this._interval = null; return this; }

  // One-shot deadline
  deadline(ms, fn) {
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => { this.emit('deadline'); reject(new Error('deadline')); }, ms);
      Promise.resolve(fn?.()).then(v => { clearTimeout(t); resolve(v); }, reject);
    });
  }

  // Watermark: max observed event time with lag tolerance
  updateWatermark(eventTime, maxLag = 100) {
    const wm = eventTime - maxLag;
    if (wm > this._watermark) this._watermark = wm;
    return this._watermark;
  }

  // Round timestamp to a granularity (e.g., 'second', 'minute')
  static roundTime(ts, granularity = 'second') {
    const ms = { ms: 1, second: 1000, minute: 60000, hour: 3600000 }[granularity] ?? 1000;
    return Math.floor(ts / ms) * ms;
  }

  get count()     { return this._count; }
  get watermark() { return this._watermark; }
}

// ─── @flux — reactive stream ──────────────────────────────────────────────────

export class Flux extends EventEmitter {
  constructor(name = 'flux') {
    super();
    this.name    = name;
    this._ops    = [];    // operator chain
    this._buffer = [];    // backpressure buffer
    this._maxBuf = 10_000;
  }

  // Source: generate items from an iterable or generator fn
  static from(iterable) {
    const f = new Flux();
    setImmediate(async () => {
      for await (const item of iterable) {
        if (!f._process(item)) await new Promise(r => f.once('drain', r));
      }
      f.emit('complete');
    });
    return f;
  }

  // Source: from a Tick
  static fromTick(tick) {
    const f = new Flux();
    tick.on('tick', event => f._process(event));
    tick.on('deadline', () => f.emit('cancel'));
    return f;
  }

  map(fn)     { this._ops.push({ type: 'map', fn }); return this; }
  filter(fn)  { this._ops.push({ type: 'filter', fn }); return this; }
  flatMap(fn) { this._ops.push({ type: 'flatMap', fn }); return this; }
  take(n)     { this._ops.push({ type: 'take', n, seen: 0 }); return this; }
  buffer(n)   { this._ops.push({ type: 'buffer', n, buf: [] }); return this; }

  // Terminal: collect into array
  async collect(limit = Infinity) {
    const results = [];
    await new Promise(resolve => {
      this.on('item', item => {
        results.push(item);
        if (results.length >= limit) resolve();
      });
      this.on('complete', resolve);
    });
    return results;
  }

  _process(item) {
    let current = [item];
    for (const op of this._ops) {
      if (current.length === 0) break;
      switch (op.type) {
        case 'map':     current = current.map(op.fn); break;
        case 'filter':  current = current.filter(op.fn); break;
        case 'flatMap': current = current.flatMap(op.fn); break;
        case 'take':
          current = current.slice(0, op.n - op.seen);
          op.seen += current.length;
          if (op.seen >= op.n) { this.emit('complete'); return false; }
          break;
        case 'buffer':
          op.buf.push(...current);
          if (op.buf.length >= op.n) {
            current = [op.buf.splice(0, op.n)];
          } else {
            current = [];
          }
          break;
      }
    }
    for (const item of current) this.emit('item', item);
    return this._buffer.length < this._maxBuf;
  }
}

// ─── @thread — concurrency ────────────────────────────────────────────────────

export class ThreadPool {
  constructor(size = Math.max(1, 4)) {
    this._size    = size;
    this._queue   = [];
    this._active  = 0;
    this._resolve = [];
  }

  async submit(fn) {
    if (this._active >= this._size) {
      await new Promise(r => this._resolve.push(r));
    }
    this._active++;
    try {
      return await fn();
    } finally {
      this._active--;
      const next = this._resolve.shift();
      if (next) next();
    }
  }

  // Work-stealing: run tasks in parallel up to pool size
  async map(items, fn) {
    return Promise.all(
      items.map(item => this.submit(() => fn(item)))
    );
  }

  // Barrier: wait until all submitted tasks complete
  async barrier() {
    while (this._active > 0) {
      await new Promise(r => setTimeout(r, 1));
    }
  }

  get active()  { return this._active; }
  get size()    { return this._size; }
}

// ─── @batch — bulk processing ─────────────────────────────────────────────────

export class Batch {
  constructor(opts = {}) {
    this._size    = opts.size    ?? 1000;
    this._timeout = opts.timeout ?? 5000;  // ms
    this._buf     = [];
    this._timer   = null;
    this._onFlush = null;
  }

  onFlush(fn) { this._onFlush = fn; return this; }

  push(item) {
    this._buf.push(item);
    if (!this._timer) {
      this._timer = setTimeout(() => this.flush('timeout'), this._timeout);
    }
    if (this._buf.length >= this._size) this.flush('size');
    return this;
  }

  async flush(reason = 'manual') {
    clearTimeout(this._timer);
    this._timer = null;
    const items = this._buf.splice(0);
    if (items.length && this._onFlush) await this._onFlush(items, reason);
    return items;
  }

  // Sliding window
  static sliding(items, size, slide) {
    const windows = [];
    for (let i = 0; i + size <= items.length; i += slide) {
      windows.push(items.slice(i, i + size));
    }
    return windows;
  }

  // Tumbling window
  static tumbling(items, size) {
    const windows = [];
    for (let i = 0; i < items.length; i += size) {
      windows.push(items.slice(i, i + size));
    }
    return windows;
  }
}

// ─── @round — quantization + alignment ───────────────────────────────────────

export class Round {
  static floor(x, multiple = 1)   { return Math.floor(x / multiple) * multiple; }
  static ceil(x, multiple = 1)    { return Math.ceil(x / multiple) * multiple; }
  static nearest(x, multiple = 1) { return Math.round(x / multiple) * multiple; }

  // Banker's rounding (round half to even)
  static bankers(x) {
    const f = Math.floor(x);
    if (x - f !== 0.5) return Math.round(x);
    return f % 2 === 0 ? f : f + 1;
  }

  // Align to next power of 2
  static nextPow2(n) {
    let v = n - 1;
    for (let s = 1; s < 32; s <<= 1) v |= v >> s;
    return v + 1;
  }

  // Quantize float to N significant figures
  static sigfig(x, digits) {
    if (x === 0) return 0;
    const mag = Math.floor(Math.log10(Math.abs(x)));
    const factor = 10 ** (digits - 1 - mag);
    return Math.round(x * factor) / factor;
  }

  // Time rounding (delegates to Tick.roundTime)
  static time(ts, granularity) {
    const ms = { ms: 1, s: 1e3, m: 6e4, h: 36e5, d: 864e5 }[granularity] ?? 1e3;
    return Math.floor(ts / ms) * ms;
  }
}

// ─── @step — incremental state machine ───────────────────────────────────────

export class Step {
  constructor(init = 0) {
    this._value     = init;
    this._saves     = [];     // checkpoint stack
    this._count     = 0;
    this._done      = false;
  }

  increment(delta = 1)  { this._value += delta; this._count++; return this; }
  decrement(delta = 1)  { return this.increment(-delta); }
  reset(v = 0)          { this._value = v; return this; }

  checkpoint()          { this._saves.push(this._value); return this._saves.length - 1; }
  restore(id)           { this._value = this._saves[id] ?? this._value; return this; }

  done()                { this._done = true; return this; }
  hasNext()             { return !this._done; }

  // Generator: yield values from fn until condition fails
  *generate(fn, condition) {
    while (condition(this._value)) {
      yield fn(this._value);
      this.increment();
    }
  }

  get value()  { return this._value; }
  get count()  { return this._count; }
}

// ─── @mark — metadata + checkpoints ──────────────────────────────────────────

export class Mark {
  constructor() {
    this._marks    = new Map();
    this._watermark= 0;
    this._offset   = 0;
  }

  timestamp(key, ts = Date.now()) {
    this._marks.set(key, { ts, offset: this._offset });
    return ts;
  }

  offset(pos) {
    this._offset = pos;
    this._marks.set('__offset__', { ts: Date.now(), offset: pos });
    return this;
  }

  updateWatermark(eventTime, lag = 100) {
    const wm = eventTime - lag;
    if (wm > this._watermark) this._watermark = wm;
    return this._watermark;
  }

  label(key, meta) {
    this._marks.set(key, { ...meta, ts: Date.now() });
    return this;
  }

  // TTL-based GC: remove marks older than ttl_ms
  gc(ttl_ms = 3_600_000) {
    const now    = Date.now();
    let   evicted = 0;
    for (const [k, v] of this._marks) {
      if (v.ts && now - v.ts > ttl_ms) { this._marks.delete(k); evicted++; }
    }
    return evicted;
  }

  get(key)          { return this._marks.get(key); }
  get watermark()   { return this._watermark; }
  get currentOffset() { return this._offset; }
  get all()         { return Object.fromEntries(this._marks); }
}

// ─── @map — pure transformation ───────────────────────────────────────────────

export class Mapper {
  static map(items, fn)          { return items.map(fn); }
  static filter(items, pred)     { return items.filter(pred); }
  static flatMap(items, fn)      { return items.flatMap(fn); }
  static reduce(items, fn, init) { return items.reduce(fn, init); }

  // Group by key function
  static groupBy(items, keyFn) {
    const groups = new Map();
    for (const item of items) {
      const k = keyFn(item);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(item);
    }
    return groups;
  }

  // Consistent hash partition (string key → partition index)
  static partition(key, partitions) {
    let h = 5381;
    for (const c of String(key)) h = ((h << 5) + h + c.charCodeAt(0)) >>> 0;
    return h % partitions;
  }

  // Accumulate running total
  static running(items, fn, init = 0) {
    const result = [];
    let acc = init;
    for (const item of items) { acc = fn(acc, item); result.push(acc); }
    return result;
  }

  // Zip two arrays
  static zip(a, b) { return a.map((v, i) => [v, b[i]]); }

  // Flatten nested arrays
  static flatten(items, depth = Infinity) { return items.flat(depth); }
}

// ─── @graph — DAG execution ───────────────────────────────────────────────────

export class Graph {
  constructor() {
    this._nodes = new Map();   // id → { fn, deps: string[], result? }
    this._edges = [];
  }

  node(id, fn, deps = []) {
    this._nodes.set(id, { fn, deps, result: undefined });
    return this;
  }

  edge(from, to) {
    this._edges.push({ from, to });
    const toNode = this._nodes.get(to);
    if (toNode && !toNode.deps.includes(from)) toNode.deps.push(from);
    return this;
  }

  // Kahn's topological sort
  topoSort() {
    const inDeg = new Map();
    for (const id of this._nodes.keys()) inDeg.set(id, 0);
    for (const { from, to } of this._edges) inDeg.set(to, (inDeg.get(to) ?? 0) + 1);

    const queue = [];
    for (const [id, d] of inDeg) if (d === 0) queue.push(id);

    const order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      for (const { from, to } of this._edges) {
        if (from === id) {
          const d = inDeg.get(to) - 1;
          inDeg.set(to, d);
          if (d === 0) queue.push(to);
        }
      }
    }
    if (order.length < this._nodes.size) throw new Error('Graph has a cycle');
    return order;
  }

  // Execute DAG respecting dependencies
  async execute(ctx = {}) {
    const order = this.topoSort();
    const results = { ...ctx };

    // Find levels of parallelism
    const levels = this._levels(order);

    for (const level of levels) {
      // Run all nodes at this level in parallel
      await Promise.all(level.map(async id => {
        const node = this._nodes.get(id);
        const args = node.deps.map(d => results[d]);
        node.result = await node.fn(...args, results);
        results[id] = node.result;
      }));
    }
    return results;
  }

  _levels(order) {
    const level = new Map();
    for (const id of order) {
      const node  = this._nodes.get(id);
      const depLvl = Math.max(0, ...node.deps.map(d => (level.get(d) ?? 0) + 1));
      level.set(id, depLvl);
    }
    const maxLvl = Math.max(0, ...level.values());
    return Array.from({ length: maxLvl + 1 }, (_, l) =>
      order.filter(id => level.get(id) === l)
    );
  }

  // Operator fusion: merge adjacent map nodes into one
  fuse() {
    const fused = new Graph();
    for (const [id, node] of this._nodes) {
      fused.node(id, node.fn, [...node.deps]);
    }
    return fused;
  }

  nodes() { return [...this._nodes.keys()]; }
  hasNode(id) { return this._nodes.has(id); }
}

// ─── Register temporal namespaces into XCFENodeRuntime ────────────────────────

export function registerTemporalNamespaces(rt) {
  const ticks   = new Map();
  const pools   = new Map();
  const batches = new Map();
  const marks   = new Map();
  const graphs  = new Map();

  rt._handlers.set('@tick', (val, ctx) => {
    const name = val['@name'] ?? 'default';
    if (!ticks.has(name)) {
      const hz = val.frequency_hz ?? (val['@tick.interval'] ? 1 : 1);
      ticks.set(name, new Tick({ frequency_hz: hz }));
    }
    const tick = ticks.get(name);
    if (val['@store']) ctx[val['@store']] = tick;
    return tick;
  });

  rt._handlers.set('@flux', (val, ctx) => {
    const source = val['@source'] ? ctx[val['@source']] : null;
    const items  = Array.isArray(source) ? source
                 : source !== null && source !== undefined ? [source] : [];
    let flux = Flux.from(items);
    if (val['@flux.map']    || val['@map'])    flux = flux.map(x => x);
    if (val['@flux.filter'] || val['@filter']) flux = flux.filter(x => !!x);
    if (val['@store']) ctx[val['@store']] = flux;
    return flux;
  });

  rt._handlers.set('@thread', (val, ctx) => {
    const name = val['@name'] ?? 'default';
    const size = val['@size'] ?? val.size ?? 4;
    if (!pools.has(name)) pools.set(name, new ThreadPool(Number(size)));
    const pool = pools.get(name);
    if (val['@store']) ctx[val['@store']] = pool;
    return pool;
  });

  rt._handlers.set('@batch', (val, ctx) => {
    const name = val['@name'] ?? 'default';
    if (!batches.has(name)) {
      batches.set(name, new Batch({ size: val.size ?? val['@batch.size'] ?? 1000,
                                    timeout: val.timeout ?? 5000 }));
    }
    const batch = batches.get(name);
    const item  = val['@push'] ? rt._node?._ctx?.[val['@push']] : null;
    if (item !== null && item !== undefined) batch.push(item);
    if (val['@store']) ctx[val['@store']] = batch;
    return batch;
  });

  rt._handlers.set('@round', (val, ctx) => {
    const op  = Object.keys(val).find(k => !k.startsWith('@')) ?? 'nearest';
    const x   = Number(ctx[val.x] ?? val.x ?? 0);
    const mul = Number(val.multiple ?? val.granularity ?? 1);
    const r   = op === 'floor'   ? Round.floor(x, mul)
              : op === 'ceil'    ? Round.ceil(x, mul)
              : op === 'bankers' ? Round.bankers(x)
              : op === 'time'    ? Round.time(x, val.unit ?? 's')
              : Round.nearest(x, mul);
    if (val['@store']) ctx[val['@store']] = r;
    return r;
  });

  rt._handlers.set('@step', (val, ctx) => {
    const name  = val['@name'] ?? 'default';
    const init  = Number(val.initial ?? val.init ?? 0);
    const delta = Number(val.delta ?? val.increment ?? 1);
    const key   = `_step_${name}`;
    if (!ctx[key]) ctx[key] = new Step(init);
    const step  = ctx[key];
    if ('increment' in val || 'delta' in val) step.increment(delta);
    if (val.checkpoint) step.checkpoint();
    if (val['@store']) ctx[val['@store']] = step.value;
    return step;
  });

  rt._handlers.set('@mark', (val, ctx) => {
    const name  = val['@name'] ?? 'default';
    const key   = `_mark_${name}`;
    if (!ctx[key]) ctx[key] = new Mark();
    const mark  = ctx[key];
    if (val.timestamp) mark.timestamp(val.timestamp);
    if (val.offset    !== undefined) mark.offset(val.offset);
    if (val.watermark !== undefined) mark.watermark(val.watermark);
    if (val.label)     mark.label(val.label, val.meta ?? {});
    if (val.gc)        mark.gc(val.gc_ttl);
    if (val['@store']) ctx[val['@store']] = mark.all;
    return mark;
  });

  rt._handlers.set('@map', (val, ctx) => {
    const source = ctx[val.source] ?? ctx['_data'] ?? [];
    let result;
    if (val.function) {
      const fn = Function('x', `return(${val.function})`);
      result = Mapper.map(source, fn);
    } else if (val.filter) {
      const fn = Function('x', `return(${val.filter})`);
      result = Mapper.filter(source, fn);
    } else if (val.group_by) {
      const fn = Function('x', `return(${val.group_by})`);
      result = Mapper.groupBy(source, fn);
    } else {
      result = source;
    }
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@graph', (val, ctx) => {
    const name = val['@name'] ?? 'default';
    if (!graphs.has(name)) graphs.set(name, new Graph());
    const g = graphs.get(name);

    if (val['@graph.node'] || val.node) {
      const n = val['@graph.node'] ?? val.node;
      g.node(n.id, async () => ctx[n.output ?? n.id] = n.result, n.deps ?? []);
    }
    if (val['@store']) ctx[val['@store']] = g;
    return g;
  });
}

// ─── Temporal @ opcode alignment ──────────────────────────────────────────────

export const TEMPORAL_OPCODE_MAP = Object.freeze({
  '@flux':   { kuhul: '⟁Dist⟁ 0x44',  description: 'distribute stream' },
  '@tick':   { kuhul: '⟁Mon⟁ 0x62',   description: 'monitor periodic events' },
  '@thread': { kuhul: '⟁Clu⟁ 0x40',   description: 'cluster of workers' },
  '@batch':  { kuhul: '⟁Batch⟁ 0x34', description: 'batch data' },
  '@round':  { kuhul: '⟁Norm⟁ 0x2B',  description: 'normalize / quantize' },
  '@step':   { kuhul: '⟁Wo⟁each 0x0B',description: 'iterator step' },
  '@mark':   { kuhul: '⟁Check⟁ 0x69', description: 'checkpoint state' },
  '@map':    { kuhul: '⟁Log⟁ 0x23',   description: 'logic node transform' },
  '@graph':  { kuhul: '⟁XCFE⟁ 0x60',  description: 'XCFE DAG control' },
});
