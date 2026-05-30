// imperative-layer.js — @verbs / @endpoints / @pagination / @generate
//                        @validate / @parse / @render / @import
//                        @class / @function / @action / @program
//
// This layer transforms XCFE from a passive tensor into an active,
// self-describing multi-paradigm programming language.
//
// Stack (bottom → top):
//   @program    — complete executable (main, lifecycle, error handling)
//   @action     — side effects (db, fs, notifications, deploy)
//   @function   — reusable logic (definition, curry, compose, async)
//   @class      — type system (properties, methods, inheritance, meta)
//   @import     — modularity (library, file, url, xcfe, dynamic)
//   @render     — output generation (template, format, visual, binary)
//   @parse      — input processing (grammar, format, binary)
//   @validate   — constraint checking (schema, business, crypto, perf)
//   @generate   — creation & synthesis (id, schema, code, data, test)
//   @pagination — streaming data access (offset, cursor, keyset, stream)
//   @endpoints  — interface boundaries (http, graphql, webhook)
//   @verbs      — action ontology (CRUD, dataflow, transform, control)
//
// K'UHUL opcode alignment:
//   @verbs     ≡ ⟁Wo⟁   0x05  call/invoke
//   @endpoints ≡ ⟁Path⟁ 0x65  path select
//   @pagination≡ ⟁Batch⟁0x34  batch data
//   @generate  ≡ ⟁Aug⟁  0x37  augment/create
//   @validate  ≡ ⟁Val⟁  0x61  XCFE validate
//   @parse     ≡ ⟁Tok⟁  0x22  tokenize / ingest
//   @render    ≡ ⟁Pred⟁ 0x31  predict / emit
//   @import    ≡ ⟁Load⟁ 0x2E  load model/module
//   @class     ≡ ⟁Ten⟁  0x20  define tensor (type)
//   @function  ≡ ⟁Log⟁  0x23  logic node
//   @action    ≡ ⟁Ch'en⟁0x06  store/persist (side effect)
//   @program   ≡ ⟁Pop⟁  0x01  begin block (main entry)

import { createHash, randomUUID } from 'node:crypto';
import * as nodePath  from 'node:path';
import * as nodeFs    from 'node:fs/promises';

// ─── @verbs — action ontology ─────────────────────────────────────────────────

export const VERBS = Object.freeze({
  // CRUD
  CREATE: 'instantiate', READ: 'retrieve', UPDATE: 'modify', DELETE: 'remove',
  // Dataflow
  PUSH: 'emit_to_downstream', PULL: 'request_from_upstream',
  FETCH: 'retrieve_external', STORE: 'persist_locally',
  // Transformation
  MAP: 'element_wise', REDUCE: 'aggregate', FILTER: 'select', FLATMAP: 'expand',
  // Control
  INIT: 'initialize', EXECUTE: 'run', PAUSE: 'suspend',
  RESUME: 'continue', TERMINATE: 'shutdown',
});

export class VerbChain {
  constructor() { this._chain = []; this._ctx = {}; }

  do(verb, fn) { this._chain.push({ verb, fn }); return this; }

  async run(input) {
    let current = input;
    for (const { verb, fn } of this._chain) {
      current = await fn(current, this._ctx);
    }
    return current;
  }

  // Common patterns
  createValidateStore(createFn, validateFn, storeFn) {
    return this.do(VERBS.CREATE, createFn)
               .do('VALIDATE', validateFn)
               .do(VERBS.STORE, storeFn);
  }
}

// ─── @endpoints — interface boundaries ───────────────────────────────────────

export class EndpointRegistry {
  constructor() {
    this._routes  = new Map();  // 'METHOD /path' → handler
    this._middleware = [];
  }

  route(method, path, handler, opts = {}) {
    const key = `${method.toUpperCase()} ${path}`;
    this._routes.set(key, { handler, opts });
    return this;
  }

  get(path, h, opts)    { return this.route('GET',    path, h, opts); }
  post(path, h, opts)   { return this.route('POST',   path, h, opts); }
  put(path, h, opts)    { return this.route('PUT',    path, h, opts); }
  del(path, h, opts)    { return this.route('DELETE', path, h, opts); }
  patch(path, h, opts)  { return this.route('PATCH',  path, h, opts); }

  use(fn) { this._middleware.push(fn); return this; }

  async handle(method, path, body = {}, headers = {}) {
    // Match route (including path params like /users/{id})
    let handler = null, params = {};
    for (const [key, { handler: h, opts }] of this._routes) {
      const [m, p] = key.split(' ');
      if (m !== method.toUpperCase()) continue;
      const match = this._matchPath(p, path);
      if (match) { handler = h; params = match; break; }
    }
    if (!handler) return { status: 404, body: { error: 'Not found' } };

    let ctx = { method, path, body, headers, params };
    for (const mw of this._middleware) ctx = await mw(ctx) ?? ctx;
    try {
      const result = await handler(ctx);
      return { status: 200, body: result };
    } catch (e) {
      return { status: 500, body: { error: e.message } };
    }
  }

  _matchPath(pattern, path) {
    const pParts = pattern.split('/');
    const rParts = path.split('/');
    if (pParts.length !== rParts.length) return null;
    const params = {};
    for (let i = 0; i < pParts.length; i++) {
      if (pParts[i].startsWith('{') && pParts[i].endsWith('}')) {
        params[pParts[i].slice(1,-1)] = rParts[i];
      } else if (pParts[i] !== rParts[i]) return null;
    }
    return params;
  }

  toOpenAPI(title = 'XCFE API', version = '1.0.0') {
    const paths = {};
    for (const [key, { opts }] of this._routes) {
      const [method, path] = key.split(' ');
      const oaPath = path.replace(/\{(\w+)\}/g, '{$1}');
      paths[oaPath] = paths[oaPath] ?? {};
      paths[oaPath][method.toLowerCase()] = {
        operationId: opts.operationId ?? `${method}_${path.replace(/\W+/g,'_')}`,
        summary: opts.summary ?? '',
        parameters: opts.params ?? [],
        responses: opts.responses ?? { '200': { description: 'OK' } },
      };
    }
    return { openapi: '3.0.0', info: { title, version }, paths };
  }

  routes() { return [...this._routes.keys()]; }
}

// ─── @pagination — streaming data access ─────────────────────────────────────

export class Paginator {
  // Offset-based page
  static offset(data, limit, offset) {
    const items = data.slice(offset, offset + limit);
    return {
      items,
      limit,
      offset,
      total: data.length,
      has_next: offset + limit < data.length,
      next_offset: offset + limit,
    };
  }

  // Cursor-based (opaque base64 cursor encodes field values)
  static cursor(data, limit, cursor, keyField = 'id') {
    let startIdx = 0;
    if (cursor) {
      const { value } = JSON.parse(Buffer.from(cursor, 'base64').toString());
      startIdx = data.findIndex(item => String(item[keyField]) === String(value));
      startIdx = startIdx >= 0 ? startIdx + 1 : 0;
    }
    const items     = data.slice(startIdx, startIdx + limit);
    const last      = items[items.length - 1];
    const nextCursor = last && startIdx + limit < data.length
      ? Buffer.from(JSON.stringify({ value: last[keyField] })).toString('base64')
      : null;
    return { items, limit, next_cursor: nextCursor, has_next: !!nextCursor };
  }

  // Keyset (sort by columns, resume after last row)
  static keyset(data, limit, lastKey = null, sortCols = ['id']) {
    let items = data;
    if (lastKey) {
      const idx = data.findIndex(r =>
        sortCols.every((c, i) => String(r[c]) === String(lastKey[i]))
      );
      items = idx >= 0 ? data.slice(idx + 1) : data;
    }
    items = items.slice(0, limit);
    const nextKey = items.length === limit
      ? sortCols.map(c => items[items.length-1][c])
      : null;
    return { items, next_key: nextKey, has_next: !!nextKey };
  }

  // Stream: async generator yielding pages
  static async *stream(fetchFn, limit = 1000) {
    let offset = 0, hasNext = true;
    while (hasNext) {
      const page = await fetchFn(limit, offset);
      yield page.items;
      hasNext   = page.has_next;
      offset   += limit;
    }
  }
}

// ─── @generate — creation & synthesis ────────────────────────────────────────

export class Generator {
  static uuid()         { return randomUUID(); }
  static id(prefix='')  { return `${prefix}${Date.now().toString(36)}_${Math.random().toString(36).slice(2,7)}`; }

  // Generate a JSON Schema skeleton from a plain object shape
  static schema(shape) {
    const props = {};
    const required = [];
    for (const [k, v] of Object.entries(shape)) {
      const type = typeof v;
      props[k] = type === 'string'  ? { type: 'string' }
               : type === 'number'  ? { type: 'number' }
               : type === 'boolean' ? { type: 'boolean' }
               : Array.isArray(v)   ? { type: 'array', items: {} }
               : { type: 'object' };
      required.push(k);
    }
    return { type: 'object', properties: props, required };
  }

  // Synthetic data from normal distribution
  static normal(mean = 0, std = 1, n = 1000) {
    const data = [];
    for (let i = 0; i < n; i += 2) {
      const u1 = Math.random(), u2 = Math.random();
      const z0 = Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2);
      const z1 = Math.sqrt(-2*Math.log(u1)) * Math.sin(2*Math.PI*u2);
      data.push(mean + z0*std);
      if (i+1 < n) data.push(mean + z1*std);
    }
    return data.slice(0, n);
  }

  // Generate test cases from a spec
  static testCases(fn, cases) {
    return cases.map(({ input, expected, label }) => ({
      label: label ?? JSON.stringify(input),
      input,
      expected,
      run: () => {
        const result = fn(...(Array.isArray(input) ? input : [input]));
        const pass   = JSON.stringify(result) === JSON.stringify(expected);
        return { pass, result, expected };
      },
    }));
  }

  // Simple code emit (template substitution)
  static code(template, vars) {
    return template.replace(/\{\{(\w+)\}\}/g, (_, k) => vars[k] ?? '');
  }
}

// ─── @validate — constraint checking ─────────────────────────────────────────

export class Validator {
  constructor() { this._rules = []; }

  required(field)                 { this._rules.push(v => v[field] != null || `${field} is required`); return this; }
  type(field, t)                  { this._rules.push(v => typeof v[field] === t || `${field} must be ${t}`); return this; }
  minLength(field, n)             { this._rules.push(v => (v[field]?.length ?? 0) >= n || `${field} min length ${n}`); return this; }
  maxLength(field, n)             { this._rules.push(v => (v[field]?.length ?? 0) <= n || `${field} max length ${n}`); return this; }
  pattern(field, re)              { this._rules.push(v => re.test(v[field]) || `${field} pattern mismatch`); return this; }
  min(field, n)                   { this._rules.push(v => v[field] >= n || `${field} min ${n}`); return this; }
  max(field, n)                   { this._rules.push(v => v[field] <= n || `${field} max ${n}`); return this; }
  custom(pred, msg)               { this._rules.push(v => pred(v) || msg); return this; }

  validate(data) {
    const errors = this._rules.map(r => r(data)).filter(r => r !== true);
    return { valid: errors.length === 0, errors };
  }

  // Schema-based validator (JSON Schema subset)
  static fromSchema(schema) {
    const v = new Validator();
    for (const field of schema.required ?? []) v.required(field);
    for (const [field, def] of Object.entries(schema.properties ?? {})) {
      if (def.type)      v.type(field, def.type === 'integer' ? 'number' : def.type);
      if (def.minLength) v.minLength(field, def.minLength);
      if (def.maxLength) v.maxLength(field, def.maxLength);
      if (def.pattern)   v.pattern(field, new RegExp(def.pattern));
      if (def.minimum)   v.min(field, def.minimum);
      if (def.maximum)   v.max(field, def.maximum);
    }
    return v;
  }

  // HMAC integrity check
  static hmac(payload, secret, algorithm = 'sha256') {
    const { createHmac } = require ? require('node:crypto') : { createHmac: () => ({ update:()=>({digest:()=>''})} )};
    const sig = createHmac(algorithm, secret)
      .update(typeof payload === 'string' ? payload : JSON.stringify(payload))
      .digest('hex');
    return sig;
  }
}

// ─── @parse — input processing ────────────────────────────────────────────────

export class Parser {
  // Tokenize a string into typed tokens
  static tokenize(src, rules) {
    // rules: [{type, pattern}]
    const tokens = [];
    let pos = 0;
    outer: while (pos < src.length) {
      for (const { type, pattern } of rules) {
        const re  = new RegExp(`^(?:${pattern.source ?? pattern})`);
        const m   = re.exec(src.slice(pos));
        if (m) {
          tokens.push({ type, value: m[0], pos });
          pos += m[0].length;
          continue outer;
        }
      }
      pos++;  // skip unknown character
    }
    return tokens;
  }

  // Parse JSONL (null-byte or newline separated, or both)
  static jsonl(text) {
    const records = [];
    // Split on both separators so mixed content works
    for (const seg of text.split('\x00')) {
      for (const line of seg.split('\n')) {
        const l = line.trim();
        if (!l) continue;
        try { records.push(JSON.parse(l)); } catch {}
      }
    }
    return records;
  }

  // Parse CSV
  static csv(text, { delimiter = ',', headers = true } = {}) {
    const lines = text.split('\n').filter(l => l.trim());
    if (!lines.length) return [];
    const cols = lines[0].split(delimiter).map(c => c.trim());
    if (!headers) return lines.map(l => l.split(delimiter).map(v => v.trim()));
    return lines.slice(1).map(line => {
      const vals = line.split(delimiter).map(v => v.trim());
      return Object.fromEntries(cols.map((c, i) => [c, vals[i] ?? '']));
    });
  }

  // Parse binary according to a schema [{name, type, length?}]
  static binary(buf, schema) {
    const view = buf instanceof ArrayBuffer ? new DataView(buf) : new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    let offset = 0;
    const result = {};
    for (const field of schema) {
      switch (field.type) {
        case 'uint8':  result[field.name] = view.getUint8(offset);   offset += 1; break;
        case 'uint16': result[field.name] = view.getUint16(offset, true); offset += 2; break;
        case 'uint32': result[field.name] = view.getUint32(offset, true); offset += 4; break;
        case 'int32':  result[field.name] = view.getInt32(offset, true);  offset += 4; break;
        case 'float32':result[field.name] = view.getFloat32(offset, true);offset += 4; break;
        case 'float64':result[field.name] = view.getFloat64(offset, true);offset += 8; break;
        case 'bytes': {
          const len = field.length ?? result[field.lengthField] ?? 0;
          result[field.name] = new Uint8Array(view.buffer, view.byteOffset + offset, len);
          offset += len;
          break;
        }
        case 'string': {
          const len = field.length ?? 0;
          const bytes = new Uint8Array(view.buffer, view.byteOffset + offset, len);
          result[field.name] = new TextDecoder().decode(bytes).replace(/\0+$/, '');
          offset += len;
          break;
        }
      }
    }
    return result;
  }

  // Cursor decode (from @pagination.cursor)
  static cursor(encoded) {
    try { return JSON.parse(Buffer.from(encoded, 'base64').toString()); }
    catch { return null; }
  }
}

// ─── @render — output generation ─────────────────────────────────────────────

export class Renderer {
  // Mustache-style template substitution
  static template(tmpl, ctx) {
    return tmpl
      .replace(/\{\{\s*([^}]+)\s*\}\}/g, (_, key) => {
        const parts = key.trim().split('.');
        let v = ctx;
        for (const p of parts) v = v?.[p];
        return v !== undefined ? String(v) : '';
      });
  }

  static json(data, pretty = false) {
    return pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
  }

  static yaml(data, indent = 0) {
    const pad = '  '.repeat(indent);
    if (data === null || data === undefined) return 'null';
    if (typeof data !== 'object') return String(data);
    if (Array.isArray(data)) {
      return data.map(v => `${pad}- ${Renderer.yaml(v, 0)}`).join('\n');
    }
    return Object.entries(data)
      .map(([k, v]) => {
        const rendered = typeof v === 'object' ? '\n' + Renderer.yaml(v, indent+1) : ' ' + v;
        return `${pad}${k}:${rendered}`;
      })
      .join('\n');
  }

  static markdown(sections) {
    return sections.map(({ heading, body, code, lang }) => {
      const parts = [];
      if (heading)  parts.push(`# ${heading}`);
      if (body)     parts.push(body);
      if (code)     parts.push(`\`\`\`${lang ?? ''}\n${code}\n\`\`\``);
      return parts.join('\n');
    }).join('\n\n');
  }

  // Base64 binary encoding
  static base64(data) {
    const buf = typeof data === 'string' ? Buffer.from(data) : Buffer.from(data);
    return buf.toString('base64');
  }

  // Pack data to binary (little-endian)
  static pack(schema, values) {
    const sizes = { uint8:1, uint16:2, uint32:4, int32:4, float32:4, float64:8 };
    const totalBytes = schema.reduce((s, f) => s + (sizes[f.type] ?? f.length ?? 0), 0);
    const buf  = new ArrayBuffer(totalBytes);
    const view = new DataView(buf);
    let offset = 0;
    for (const field of schema) {
      const v = values[field.name] ?? 0;
      switch (field.type) {
        case 'uint8':   view.setUint8(offset, v);             offset += 1; break;
        case 'uint16':  view.setUint16(offset, v, true);      offset += 2; break;
        case 'uint32':  view.setUint32(offset, v, true);      offset += 4; break;
        case 'int32':   view.setInt32(offset, v, true);       offset += 4; break;
        case 'float32': view.setFloat32(offset, v, true);     offset += 4; break;
        case 'float64': view.setFloat64(offset, v, true);     offset += 8; break;
      }
    }
    return new Uint8Array(buf);
  }
}

// ─── @import — dependency management ─────────────────────────────────────────

export class ImportManager {
  constructor() { this._cache = new Map(); this._loaders = new Map(); }

  registerLoader(ext, fn) { this._loaders.set(ext, fn); return this; }

  async load(path, opts = {}) {
    const key = `${path}:${JSON.stringify(opts)}`;
    if (this._cache.has(key)) return this._cache.get(key);

    const ext    = nodePath.extname(path).slice(1).toLowerCase();
    const loader = this._loaders.get(ext);

    let result;
    if (loader) {
      result = await loader(path, opts);
    } else {
      // Default: read file as text or JSON
      const text = await nodeFs.readFile(path, 'utf8').catch(() => null);
      result = text === null ? null
             : ext === 'json' ? JSON.parse(text)
             : text;
    }

    if (opts.ttl && result !== null) {
      this._cache.set(key, result);
      setTimeout(() => this._cache.delete(key), opts.ttl * 1000);
    } else if (result !== null) {
      this._cache.set(key, result);
    }

    return result;
  }

  // Merge two XCFE block objects (recursive)
  static merge(base, overlay) {
    const result = { ...base };
    for (const [k, v] of Object.entries(overlay)) {
      result[k] = (v && typeof v === 'object' && !Array.isArray(v) && result[k])
        ? ImportManager.merge(result[k], v)
        : v;
    }
    return result;
  }

  get size() { return this._cache.size; }
}

// ─── @class — type system ─────────────────────────────────────────────────────

export class ClassDefinition {
  constructor(name, opts = {}) {
    this.name        = name;
    this._props      = { ...opts.properties };
    this._methods    = new Map();
    this._mixins     = [];
    this._parent     = opts.extends ?? null;
    this._validators = {};
  }

  property(name, type, required = true) {
    this._props[name] = { type, required };
    return this;
  }

  method(name, fn) { this._methods.set(name, fn); return this; }

  mixin(name) { this._mixins.push(name); return this; }

  // Build a validator for this class's schema
  buildValidator() {
    const v = new Validator();
    for (const [field, def] of Object.entries(this._props)) {
      if (def.required) v.required(field);
      if (def.type)     v.type(field, def.type === 'integer' ? 'number' : def.type);
    }
    return v;
  }

  // Instantiate a plain object conforming to this class
  instantiate(data) {
    const v      = this.buildValidator();
    const result = v.validate(data);
    if (!result.valid) throw new Error(`${this.name}: ${result.errors.join(', ')}`);
    const instance = { __class__: this.name, ...data };
    for (const [name, fn] of this._methods) {
      instance[name] = fn.bind(instance);
    }
    return instance;
  }

  toJsonSchema() {
    return Generator.schema(
      Object.fromEntries(
        Object.entries(this._props).map(([k, d]) => [k, d.type ?? 'string'])
      )
    );
  }
}

// ─── @function — reusable logic ───────────────────────────────────────────────

export class FunctionDef {
  constructor(name, fn, opts = {}) {
    this.name     = name;
    this._fn      = fn;
    this._types   = opts.types ?? {};
    this._async   = opts.async ?? fn.constructor.name === 'AsyncFunction';
    this._memoize = opts.memoize ?? false;
    this._cache   = new Map();
  }

  // Partial application (curry)
  partial(...boundArgs) {
    return new FunctionDef(`${this.name}_partial`, (...args) => this._fn(...boundArgs, ...args));
  }

  // Function composition: this ∘ other
  compose(other) {
    return new FunctionDef(`${this.name}_∘_${other.name}`,
      async (...args) => this.call(await other.call(...args)));
  }

  async call(...args) {
    const key = this._memoize ? JSON.stringify(args) : null;
    if (key && this._cache.has(key)) return this._cache.get(key);
    const result = await this._fn(...args);
    if (key) this._cache.set(key, result);
    return result;
  }

  // Pipeline of FunctionDefs
  static pipeline(fns) {
    return new FunctionDef(
      fns.map(f => f.name).join('→'),
      async (input) => {
        let v = input;
        for (const fn of fns) v = await fn.call(v);
        return v;
      }
    );
  }
}

// ─── @action — side effects & state changes ───────────────────────────────────

export class ActionRunner {
  constructor() {
    this._log       = [];
    this._handlers  = new Map();
    this._rollbacks = [];
  }

  register(name, fn, rollbackFn = null) {
    this._handlers.set(name, fn);
    if (rollbackFn) this._rollbacks.push({ name, fn: rollbackFn });
    return this;
  }

  async run(name, params, ctx = {}) {
    const fn = this._handlers.get(name);
    if (!fn) throw new Error(`@action: unknown action "${name}"`);
    const entry = { name, params, ts: Date.now(), status: 'running' };
    this._log.push(entry);
    try {
      const result  = await fn(params, ctx);
      entry.status  = 'done';
      entry.result  = result;
      return result;
    } catch (e) {
      entry.status  = 'failed';
      entry.error   = e.message;
      throw e;
    }
  }

  // Rollback all registered rollback handlers
  async rollback() {
    for (const { name, fn } of [...this._rollbacks].reverse()) {
      try { await fn(); }
      catch (e) { console.error(`rollback ${name} failed:`, e.message); }
    }
  }

  get log() { return [...this._log]; }
}

// ─── @program — complete executable ──────────────────────────────────────────

export class Program {
  constructor(name, opts = {}) {
    this.name      = name;
    this._steps    = [];
    this._imports  = new ImportManager();
    this._actions  = new ActionRunner();
    this._ctx      = { ...opts.env };
    this._metrics  = { start: 0, steps: 0, errors: 0 };
    this._onError  = opts.onError ?? ((e) => { throw e; });
  }

  // Register a lifecycle step
  step(name, fn) { this._steps.push({ name, fn }); return this; }

  // Import a module/file into context
  async import(key, path, opts = {}) {
    this._ctx[key] = await this._imports.load(path, opts);
    return this;
  }

  // Register an action
  action(name, fn, rollback = null) {
    this._actions.register(name, fn, rollback);
    return this;
  }

  // Run the full lifecycle
  async run(input = {}) {
    this._metrics.start = Date.now();
    Object.assign(this._ctx, input);

    for (const { name, fn } of this._steps) {
      try {
        this._ctx = await fn(this._ctx) ?? this._ctx;
        this._metrics.steps++;
      } catch (e) {
        this._metrics.errors++;
        await this._onError(e, name, this._ctx);
        break;
      }
    }

    return {
      ctx:     this._ctx,
      metrics: {
        ...this._metrics,
        elapsed_ms: Date.now() - this._metrics.start,
        action_log: this._actions.log,
      },
    };
  }

  // Build from a declarative @program spec object
  static fromSpec(spec) {
    const p = new Program(spec['@name'] ?? 'program');
    for (const step of spec['@program.lifecycle']?.steps ?? []) {
      p.step(step, async ctx => ctx);   // default: identity pass
    }
    return p;
  }
}

// ─── Register imperative namespaces into XCFENodeRuntime ──────────────────────

export function registerImperativeNamespaces(rt) {
  const endpoints = new Map();
  const programs  = new Map();
  const actions   = new ActionRunner();

  rt._handlers.set('@verbs', (val, ctx) => {
    const verb  = Object.keys(val).find(k => !k.startsWith('@')) ?? 'EXECUTE';
    const chain = new VerbChain();
    ctx[`_verb_${verb}`] = chain;
    if (val['@store']) ctx[val['@store']] = chain;
    return chain;
  });

  rt._handlers.set('@endpoints', (val, ctx) => {
    const name = val['@name'] ?? 'default';
    if (!endpoints.has(name)) endpoints.set(name, new EndpointRegistry());
    const reg = endpoints.get(name);
    // Register inline routes
    for (const [k, v] of Object.entries(val)) {
      if (!k.startsWith('@endpoint.')) continue;
      const ep = v;
      reg.route(ep.method ?? 'GET', ep.path, async (c) => ep.handler?.(c) ?? ep);
    }
    if (val['@store']) ctx[val['@store']] = reg;
    return reg;
  });

  rt._handlers.set('@pagination', (val, ctx) => {
    const data  = ctx[val.source] ?? ctx['_data'] ?? [];
    const limit = val.limit ?? 100;
    const type  = Object.keys(val).find(k => ['offset','cursor','keyset','stream'].includes(k)) ?? 'offset';
    let result;
    if (type === 'offset') result = Paginator.offset(data, limit, val.offset ?? 0);
    else if (type === 'cursor') result = Paginator.cursor(data, limit, val.cursor, val.key_field);
    else result = Paginator.keyset(data, limit, val.last_key, val.sort_cols);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@generate', (val, ctx) => {
    const type = Object.keys(val).find(k => !k.startsWith('@')) ?? 'id';
    let result;
    if (type === 'id')     result = Generator.id(val.prefix);
    else if (type === 'uuid') result = Generator.uuid();
    else if (type === 'schema') result = Generator.schema(val.shape ?? {});
    else if (type === 'data')   result = Generator.normal(val.mean, val.std, val.n);
    else if (type === 'code')   result = Generator.code(val.template ?? '', ctx);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@validate', (val, ctx) => {
    const data    = ctx[val.source] ?? val.data ?? {};
    const schema  = val.schema ?? {};
    const v       = Validator.fromSchema(schema);
    const result  = v.validate(data);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@parse', (val, ctx) => {
    const type = Object.keys(val).find(k => ['jsonl','csv','binary','cursor'].includes(k)) ?? 'jsonl';
    const src  = ctx[val.source] ?? val.source ?? '';
    let result;
    if (type === 'jsonl')  result = Parser.jsonl(src);
    else if (type === 'csv')    result = Parser.csv(src, val.csv);
    else if (type === 'cursor') result = Parser.cursor(src);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@render', (val, ctx) => {
    const type   = Object.keys(val).find(k => ['template','json','yaml','markdown','base64'].includes(k)) ?? 'json';
    const data   = ctx[val.source] ?? val.data ?? ctx;
    let result;
    if (type === 'template') result = Renderer.template(val.template ?? '', ctx);
    else if (type === 'json')     result = Renderer.json(data, val.pretty);
    else if (type === 'yaml')     result = Renderer.yaml(data);
    else if (type === 'markdown') result = Renderer.markdown(val.sections ?? []);
    else if (type === 'base64')   result = Renderer.base64(typeof data === 'string' ? data : JSON.stringify(data));
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@import', async (val, ctx) => {
    const key  = val['@name'] ?? val.key ?? 'imported';
    const path = val.file ?? val.path ?? '';
    if (path) {
      const mgr = new ImportManager();
      const data = await mgr.load(path).catch(() => null);
      ctx[key] = data;
    }
    if (val['@folds.merge'] && val.base && val.overlay) {
      ctx[key] = ImportManager.merge(ctx[val.base] ?? {}, ctx[val.overlay] ?? {});
    }
    if (val['@store']) ctx[val['@store']] = ctx[key];
    return ctx[key];
  });

  rt._handlers.set('@class', (val, ctx) => {
    const name = Object.keys(val).find(k => !k.startsWith('@')) ?? val['@name'] ?? 'Entity';
    const cls  = new ClassDefinition(name, val[name] ?? val);
    ctx[`_class_${name}`] = cls;
    if (val['@store']) ctx[val['@store']] = cls;
    return cls;
  });

  rt._handlers.set('@function', (val, ctx) => {
    const name = val.name ?? val['@name'] ?? 'fn';
    const body = val['@program.body'] ?? val.body;
    const fn   = body ? Function('ctx', `return(${body})`).bind(null, ctx) : () => null;
    const def  = new FunctionDef(name, fn, { memoize: !!val['@memoize'] });
    ctx[`_fn_${name}`] = def;
    if (val['@store']) ctx[val['@store']] = def;
    return def;
  });

  rt._handlers.set('@action', async (val, ctx) => {
    const name = Object.keys(val).find(k => !k.startsWith('@')) ?? 'action';
    const handler = val.handler ?? (async (p, c) => ({ ...p, _action: name, _ts: Date.now() }));
    actions.register(name, handler);
    const result = await actions.run(name, val.parameters ?? val, ctx);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });

  rt._handlers.set('@program', async (val, ctx) => {
    const name = val['@name'] ?? 'main';
    if (!programs.has(name)) programs.set(name, Program.fromSpec(val));
    const prog = programs.get(name);
    const result = await prog.run(ctx);
    Object.assign(ctx, result.ctx);
    if (val['@store']) ctx[val['@store']] = result;
    return result;
  });
}

// ─── Imperative @ opcode alignment ────────────────────────────────────────────

export const IMPERATIVE_OPCODE_MAP = Object.freeze({
  '@verbs':      { kuhul: '⟁Wo⟁ 0x05',     description: 'call/invoke action' },
  '@endpoints':  { kuhul: '⟁Path⟁ 0x65',   description: 'path select / routing' },
  '@pagination': { kuhul: '⟁Batch⟁ 0x34',  description: 'batch data streaming' },
  '@generate':   { kuhul: '⟁Aug⟁ 0x37',    description: 'augment / synthesize' },
  '@validate':   { kuhul: '⟁Val⟁ 0x61',    description: 'XCFE validate' },
  '@parse':      { kuhul: '⟁Tok⟁ 0x22',    description: 'tokenize / ingest' },
  '@render':     { kuhul: '⟁Pred⟁ 0x31',   description: 'predict / emit output' },
  '@import':     { kuhul: '⟁Load⟁ 0x2E',   description: 'load module' },
  '@class':      { kuhul: '⟁Ten⟁ 0x20',    description: 'define tensor type' },
  '@function':   { kuhul: '⟁Log⟁ 0x23',    description: 'logic node' },
  '@action':     { kuhul: "⟁Ch'en⟁ 0x06",  description: 'store / side effect' },
  '@program':    { kuhul: '⟁Pop⟁ 0x01',    description: 'begin block / main entry' },
});
