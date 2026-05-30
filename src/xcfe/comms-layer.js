// comms-layer.js — @protocol / @jsonl / @notation / @ngram namespace handlers
//
// 8-layer @ computational stack (XCFE "OSI model"):
//
//   @semantics       L8  Meaning encoding
//   @ngram           L7  Pattern recognition + prediction
//   @notation        L6  Representation (AST, infix, graph)
//   @jsonl           L5  Streaming data (JSON lines, batches)
//   @protocol        L4  Transport (HTTP3, gRPC, WS, allreduce)
//   @opcodes         L3  Execution instructions
//   @context         L2  Runtime environment
//   @folds / @micro  L1  Atomic + vector composition
//
// Every layer is a namespace resolver under `@`.
// Each can plug into XCFENodeRuntime._handlers.

// ─── @protocol ────────────────────────────────────────────────────────────────

export const PROTOCOL_MAP = Object.freeze({
  http3:            { scheme: 'https', port: 443, quic: true  },
  grpc:             { scheme: 'http2', port: 50051, serial: 'protobuf' },
  websocket:        { scheme: 'wss',   port: 443, compression: 'permessage-deflate' },
  allreduce:        { type: 'collective', algo: 'ring' },
  parameter_server: { type: 'ps',    push_freq: 100, pull_freq: 10 },
  sse:              { scheme: 'https', mime: 'text/event-stream' },
  pubsub:           { type: 'fanout', ack: 'at_least_once' },
});

export class ProtocolLayer {
  constructor(name, opts = {}) {
    this.name       = name;
    this.spec       = { ...PROTOCOL_MAP[name], ...opts };
    this._handlers  = new Map();
    this._connected = false;
  }

  on(event, fn)    { this._handlers.set(event, fn); return this; }
  emit(event, data){ this._handlers.get(event)?.(data); }

  // Simulate connect (real impl would use node:http2 / ws / etc.)
  async connect() {
    this._connected = true;
    this.emit('connect', { protocol: this.name, spec: this.spec });
    return this;
  }

  // @protocol.backpressure — simple token bucket
  createBackpressure(windowSize = 1000) {
    let tokens = windowSize;
    return {
      acquire(n = 1) {
        if (tokens >= n) { tokens -= n; return true; }
        return false;
      },
      release(n = 1) { tokens = Math.min(windowSize, tokens + n); },
      available()    { return tokens; },
    };
  }

  // Allreduce ring topology
  ringAllreduce(tensors, op = 'sum') {
    const n = tensors.length;
    if (n === 0) return [];
    const result = tensors[0].slice();
    for (let i = 1; i < n; i++) {
      for (let j = 0; j < result.length; j++) {
        switch (op) {
          case 'sum': result[j] += tensors[i][j]; break;
          case 'max': result[j] = Math.max(result[j], tensors[i][j]); break;
          case 'min': result[j] = Math.min(result[j], tensors[i][j]); break;
        }
      }
    }
    const avg = op === 'sum' ? result.map(v => v / n) : result;
    return avg;
  }
}

// ─── @jsonl ───────────────────────────────────────────────────────────────────

export class JsonlStream {
  constructor(opts = {}) {
    this.batchSize  = opts.batch_size  ?? 1000;
    this.bufferSize = opts.buffer_size ?? 64 * 1024;   // 64 KB
    this._queue     = [];
    this._offset    = 0;
    this._schema    = opts.schema ?? null;
    this._transforms= [];
  }

  // Parse JSONL text into records
  parse(text) {
    const records = [];
    for (const line of text.split('\n')) {
      const l = line.trim();
      if (!l) continue;
      try {
        const r = JSON.parse(l);
        if (this._validate(r)) records.push(r);
      } catch {}
    }
    return records;
  }

  // Stream records in batches
  *batches(records) {
    for (let i = 0; i < records.length; i += this.batchSize) {
      yield records.slice(i, i + this.batchSize);
    }
  }

  // Apply @jsonl.transform pipeline
  addTransform(fn) { this._transforms.push(fn); return this; }
  applyTransforms(records) {
    return this._transforms.reduce((recs, fn) => recs.map(fn).filter(Boolean), records);
  }

  // Filter records
  filter(records, condition) {
    if (typeof condition === 'function') return records.filter(condition);
    // Simple string expression: "record.value > threshold"
    return records.filter(r => {
      try { return Function('record', `return(${condition})`)(r); }
      catch { return false; }
    });
  }

  // Sliding window aggregation
  windowAgg(records, windowSize, stride, ops = ['avg', 'max', 'min']) {
    const results = [];
    for (let i = 0; i + windowSize <= records.length; i += stride) {
      const window = records.slice(i, i + windowSize);
      const agg = {};
      for (const op of ops) {
        const values = window.map(r => typeof r === 'number' ? r : r.value ?? 0);
        switch (op) {
          case 'avg':   agg[op] = values.reduce((s,v) => s+v, 0) / values.length; break;
          case 'max':   agg[op] = Math.max(...values); break;
          case 'min':   agg[op] = Math.min(...values); break;
          case 'count': agg[op] = values.length; break;
          case 'sum':   agg[op] = values.reduce((s,v) => s+v, 0); break;
        }
      }
      results.push({ window_start: i, window_end: i + windowSize, ...agg });
    }
    return results;
  }

  // Serialize back to JSONL text
  serialize(records) {
    return records.map(r => JSON.stringify(r)).join('\n') + '\n';
  }

  // Update checkpoint offset
  checkpoint(offset) { this._offset = offset; return this._offset; }

  _validate(record) {
    if (!this._schema) return true;
    const fields = this._schema.fields ?? [];
    return fields.every(f => f in record);
  }
}

// ─── @notation ────────────────────────────────────────────────────────────────

export class NotationLayer {
  // Convert infix expression to AST
  static parseInfix(expr) {
    const PREC = { '+': 2, '-': 2, '*': 3, '/': 3, '^': 4 };
    const ops  = [], out = [];
    const tokens = expr.replace(/\s+/g, '').match(/\d+\.?\d*|[a-zA-Z_]\w*|[+\-*/^()]/g) ?? [];

    const peek = () => ops[ops.length - 1];
    const apply = () => {
      const op = ops.pop(), b = out.pop(), a = out.pop();
      out.push({ op, left: a, right: b });
    };

    for (const tok of tokens) {
      if (!isNaN(tok)) { out.push({ val: parseFloat(tok) }); continue; }
      if (/^[a-zA-Z_]\w*$/.test(tok)) { out.push({ var: tok }); continue; }
      if (tok === '(') { ops.push(tok); continue; }
      if (tok === ')') { while (peek() !== '(') apply(); ops.pop(); continue; }
      while (ops.length && peek() !== '(' && PREC[peek()] >= PREC[tok]) apply();
      ops.push(tok);
    }
    while (ops.length) apply();
    return out[0];
  }

  // Convert AST to prefix (Lisp) form
  static toPrefix(ast) {
    if (!ast) return '';
    if ('val' in ast) return String(ast.val);
    if ('var' in ast) return ast.var;
    return `(${ast.op} ${NotationLayer.toPrefix(ast.left)} ${NotationLayer.toPrefix(ast.right)})`;
  }

  // Convert AST to postfix (RPN)
  static toPostfix(ast) {
    if (!ast) return [];
    if ('val' in ast) return [ast.val];
    if ('var' in ast) return [ast.var];
    return [
      ...NotationLayer.toPostfix(ast.left),
      ...NotationLayer.toPostfix(ast.right),
      ast.op,
    ];
  }

  // DAG (directed acyclic graph) for tensor computation
  static buildDAG(nodes, edges) {
    const dag = new Map(nodes.map(n => [n, { id: n, inputs: [], outputs: [] }]));
    for (const { from, to } of edges) {
      dag.get(from)?.outputs.push(to);
      dag.get(to)?.inputs.push(from);
    }
    return dag;
  }

  // Topological sort of DAG for execution order
  static topoSort(dag) {
    const visited = new Set(), order = [];
    const visit = (id) => {
      if (visited.has(id)) return;
      visited.add(id);
      for (const dep of dag.get(id)?.inputs ?? []) visit(dep);
      order.push(id);
    };
    for (const id of dag.keys()) visit(id);
    return order;
  }

  // Evaluate postfix stack
  static evalPostfix(tokens, env = {}) {
    const stack = [];
    const OPS = { '+': (a,b) => a+b, '-': (a,b) => a-b,
                  '*': (a,b) => a*b, '/': (a,b) => a/b, '^': (a,b) => a**b };
    for (const tok of tokens) {
      if (typeof tok === 'number') { stack.push(tok); continue; }
      if (tok in env) { stack.push(env[tok]); continue; }
      if (tok in OPS) { const b=stack.pop(), a=stack.pop(); stack.push(OPS[tok](a,b)); continue; }
    }
    return stack[0];
  }
}

// ─── @ngram ───────────────────────────────────────────────────────────────────

export class NgramAnalyzer {
  constructor(order = 3) {
    this.order  = order;
    this._freq  = new Map();   // ngram → count
    this._total = 0;
    this._cache = new Map();   // LRU for memoize
    this._cacheMax = 10_000;
  }

  // Build ngram frequency table from tokens
  train(tokens) {
    for (let i = 0; i <= tokens.length - this.order; i++) {
      const gram = tokens.slice(i, i + this.order).join('\0');
      this._freq.set(gram, (this._freq.get(gram) ?? 0) + 1);
      this._total++;
    }
    return this;
  }

  // Probability of a specific ngram
  prob(tokens) {
    const gram = tokens.join('\0');
    return (this._freq.get(gram) ?? 0) / Math.max(1, this._total);
  }

  // Beam search: predict next token(s) from context
  predict(context, beamWidth = 5, maxTokens = 1) {
    const candidates = [];
    const prefix = context.slice(-(this.order - 1));

    for (const [gram, count] of this._freq) {
      const parts = gram.split('\0');
      const ctx   = parts.slice(0, -1).join('\0');
      const prefixStr = prefix.join('\0');
      if (ctx === prefixStr) {
        candidates.push({ token: parts[parts.length - 1], score: count });
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, beamWidth).map(c => c.token);
  }

  // Huffman-style frequency table for compression
  buildHuffmanTable() {
    const entries = [...this._freq.entries()]
      .sort((a, b) => b[1] - a[1]);
    const table = new Map();
    // Simple prefix coding: most frequent = shortest code
    for (let i = 0; i < entries.length; i++) {
      const bits = Math.max(1, Math.ceil(Math.log2(i + 2)));
      table.set(entries[i][0], { code: i.toString(2).padStart(bits, '0'), freq: entries[i][1] });
    }
    return table;
  }

  // LRU memoize a function call
  memoize(key, fn) {
    if (this._cache.has(key)) {
      const v = this._cache.get(key);
      this._cache.delete(key);       // move to end (LRU)
      this._cache.set(key, v);
      return v;
    }
    const result = fn();
    if (this._cache.size >= this._cacheMax) {
      this._cache.delete(this._cache.keys().next().value);
    }
    this._cache.set(key, result);
    return result;
  }

  // Anomaly detection: low-probability ngram = anomaly
  isAnomaly(tokens, threshold = 0.001) {
    return this.prob(tokens) < threshold;
  }

  get size() { return this._freq.size; }
}

// ─── Register @protocol / @jsonl / @notation / @ngram into XCFENodeRuntime ───

export function registerCommsNamespaces(rt) {
  rt._handlers.set('@protocol', (val, ctx) => {
    const name  = Object.keys(val)[0] ?? 'http3';
    const layer = new ProtocolLayer(name, val[name] ?? val);
    ctx[`_protocol_${name}`] = layer;
    return layer;
  });

  rt._handlers.set('@jsonl', (val, ctx) => {
    const stream = new JsonlStream(val);
    const source = val['@source'] ? ctx[val['@source']] : null;
    if (source) {
      const text    = typeof source === 'string' ? source : JSON.stringify(source);
      const records = stream.parse(text);
      ctx['_jsonl_records'] = records;
      if (val['@store']) ctx[val['@store']] = records;
    }
    return stream;
  });

  rt._handlers.set('@notation', (val, ctx) => {
    const type = val['@type'] ?? 'ast';
    if (val['@protocol']?.math_ml || val.expression) {
      const expr = val.expression ?? val['@protocol']?.math_ml;
      if (typeof expr === 'string') {
        const ast     = NotationLayer.parseInfix(expr);
        const prefix  = NotationLayer.toPrefix(ast);
        const postfix = NotationLayer.toPostfix(ast);
        ctx['_notation'] = { ast, prefix, postfix };
        if (val['@store']) ctx[val['@store']] = ctx['_notation'];
      }
    }
    if (val['@protocol']?.tensor_computation || val['@folds']?.dag) {
      const { nodes, edges } = val['@folds']?.dag ?? val;
      if (nodes && edges) {
        const dag   = NotationLayer.buildDAG(nodes, edges);
        const order = NotationLayer.topoSort(dag);
        ctx['_dag']   = dag;
        ctx['_exec_order'] = order;
      }
    }
  });

  rt._handlers.set('@ngram', (val, ctx) => {
    const order   = val['@order'] ?? 3;
    const corpus  = val['@corpus'] ? ctx[val['@corpus']] : null;
    const analyzer = new NgramAnalyzer(order);
    if (corpus) {
      const tokens = typeof corpus === 'string'
        ? corpus.split(/\s+/)
        : Array.isArray(corpus) ? corpus : [];
      analyzer.train(tokens);
    }
    ctx['_ngram'] = analyzer;

    if (val['@ngram.predict'] || val.predict) {
      const pSpec = val['@ngram.predict'] ?? val.predict;
      const ctx_  = pSpec.context?.split(/\s+/) ?? [];
      const preds = analyzer.predict(ctx_, pSpec['@horizontal_folds']?.beam_search?.width ?? 5);
      ctx[pSpec['@store'] ?? '_predictions'] = preds;
    }

    if (val['@store']) ctx[val['@store']] = analyzer;
    return analyzer;
  });
}

// ─── Full @ stack reference ────────────────────────────────────────────────────

export const AT_STACK = Object.freeze([
  { layer: 8, ns: '@semantics',        role: 'Meaning encoding' },
  { layer: 7, ns: '@ngram',            role: 'Pattern recognition + prediction' },
  { layer: 6, ns: '@notation',         role: 'Representation (AST, infix, graph)' },
  { layer: 5, ns: '@jsonl',            role: 'Streaming data (JSON lines, batches)' },
  { layer: 4, ns: '@protocol',         role: 'Transport (HTTP3, gRPC, WS, allreduce)' },
  { layer: 3, ns: '@opcodes',          role: 'Execution instructions (0x01-0x69)' },
  { layer: 2, ns: '@context',          role: 'Runtime environment (registers, cache)' },
  { layer: 1, ns: '@folds/@micro_folds',role: 'Atomic + vector composition' },
]);
