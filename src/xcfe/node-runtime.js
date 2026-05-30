// node-runtime.js — XCFE → Node.js IPC runtime
//
// Law: every @ in XCFE is a namespace resolver over Node built-ins.
//
//   @node.ipc.pipe   → child_process fork + IPC channel
//   @node.cluster    → cluster.worker.send / master.on('message')
//   @node.worker     → worker_threads parentPort / MessageChannel
//   @node.net        → net.Socket / net.createServer
//   @node.stdio      → process.stdin / process.stdout streams
//   @node.fs         → fs/promises read/write/watch
//   @node.http       → http.IncomingMessage / ServerResponse
//
// The @ resolver splits on '.' and walks the node: namespace map.
// Blocks under @node.* get translated to async Node.js operations.
// XCFE control flow (@if, @for, @wait_for) wraps Node event emitters
// and promises so the declarative layer stays synchronous-looking.
//
// XCFE ↔ K'UHUL opcode mapping (for KXML integration):
//   @write  ≡ ⟁Ch'en⟁  (store/persist — push to channel)
//   @read   ≡ ⟁Yax⟁    (get — pull from channel)
//   @send   ≡ ⟁Wo⟁     (call — invoke IPC send)
//   @on     ≡ ⟁Log⟁    (logic node — event binding)
//   @fork   ≡ ⟁Clu⟁    (cluster — spawn node)

import { EventEmitter }          from 'node:events';
import { fork, spawn }           from 'node:child_process';
import { Worker, isMainThread,
         parentPort, workerData,
         MessageChannel }        from 'node:worker_threads';
import * as net                  from 'node:net';
import * as fs                   from 'node:fs/promises';
import { createServer }          from 'node:http';

// ─── @ namespace → Node module map ───────────────────────────────────────────

const NODE_NS = {
  ipc:           { _factory: 'ipc_channel'  },
  cluster:       { _factory: 'cluster_chan' },
  worker:        { _factory: 'worker_chan'  },
  net:           { _factory: 'net_chan'     },
  stdio:         { _factory: 'stdio_chan'   },
  fs:            { _factory: 'fs_ops'       },
  http:          { _factory: 'http_server'  },
};

// ─── Channel abstraction (duplex send/receive) ────────────────────────────────

class Channel extends EventEmitter {
  constructor(name) {
    super();
    this.name   = name;
    this._store = new Map();
  }

  store(key, value)   { this._store.set(key, value); this.emit('stored', key, value); }
  retrieve(key)       { return this._store.get(key); }
  waitFor(key, timeout = 30_000) {
    if (this._store.has(key)) return Promise.resolve(this._store.get(key));
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error(`timeout waiting for ${key}`)), timeout);
      this.once('stored', (k, v) => { if (k === key) { clearTimeout(t); resolve(v); } });
    });
  }
}

// ─── IPC channel (child_process.fork) ────────────────────────────────────────

class IPCChannel extends Channel {
  constructor(name, script, forkOpts = {}) {
    super(name);
    this._child = null;
    this._script = script;
    this._opts   = forkOpts;
  }

  async open() {
    return new Promise((resolve, reject) => {
      this._child = fork(this._script, [], {
        stdio: ['pipe','pipe','pipe','ipc'],
        ...this._opts,
      });
      this._child.once('message', m => { if (m?.__ready) resolve(this); });
      this._child.on('message',   m => this.emit('message', m));
      this._child.on('error',     e => this.emit('error', e));
      this._child.on('exit',      c => this.emit('exit', c));
      setTimeout(resolve, 500, this);   // open even without __ready signal
    });
  }

  send(data)   { this._child?.send(data); }
  kill()       { this._child?.kill(); }
}

// ─── Worker thread channel ────────────────────────────────────────────────────

class WorkerChannel extends Channel {
  constructor(name, script, workerDataInit = {}) {
    super(name);
    this._worker = null;
    this._script = script;
    this._data   = workerDataInit;
  }

  async open() {
    this._worker = new Worker(this._script, { workerData: this._data });
    this._worker.on('message', m => this.emit('message', m));
    this._worker.on('error',   e => this.emit('error', e));
    return this;
  }

  send(data)   { this._worker?.postMessage(data); }
  kill()       { this._worker?.terminate(); }
}

// ─── Net socket channel ───────────────────────────────────────────────────────

class NetChannel extends Channel {
  constructor(name, opts = {}) {
    super(name);
    this._opts = opts;
    this._sock = null;
  }

  async open() {
    return new Promise((resolve, reject) => {
      const sock = net.createConnection(this._opts, () => resolve(this));
      sock.on('data',  d => this.emit('message', JSON.parse(d.toString())));
      sock.on('error', e => this.emit('error', e));
      this._sock = sock;
    });
  }

  send(data)   { this._sock?.write(JSON.stringify(data)); }
  close()      { this._sock?.destroy(); }
}

// ─── STDIO channel (process.stdin/stdout) ────────────────────────────────────

class StdioChannel extends Channel {
  constructor(name) {
    super(name);
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => {
      try { this.emit('message', JSON.parse(chunk)); }
      catch { this.emit('message', chunk.trim()); }
    });
  }

  send(data) { process.stdout.write(JSON.stringify(data) + '\n'); }
}

// ─── Template binding resolver ────────────────────────────────────────────────
// Resolves "{{ expr }}" against a context object.

function resolveTemplate(str, ctx) {
  if (typeof str !== 'string') return str;
  return str.replace(/\{\{\s*([^}]+)\s*\}\}/g, (_, expr) => {
    const parts = expr.trim().split('.');
    let val = ctx;
    for (const p of parts) val = val?.[p];
    return val !== undefined ? val : '';
  });
}

// ─── XCFE block executor ──────────────────────────────────────────────────────

export class XCFENodeRuntime {
  constructor() {
    this._ctx      = {};       // binding store ({{ var }} context)
    this._channels = new Map();
    this._handlers = new Map();
    this._setup();
  }

  // ─── Public: execute an XCFE block object ──────────────────────────────────

  async execute(block) {
    for (const [key, val] of Object.entries(block)) {
      await this._dispatch(key, val);
    }
  }

  // ─── Private: dispatch one @ key ───────────────────────────────────────────

  async _dispatch(key, val) {
    if (!key.startsWith('@')) return;
    const parts = key.slice(1).split('.');   // '@node.ipc.pipe' → ['node','ipc','pipe']

    // Top-level namespace routing
    switch (parts[0]) {
      case 'node':    return this._execNode(parts.slice(1), val);
      case 'set':     return this._execSet(parts.slice(1), val);
      case 'log':     return this._execLog(val);
      case 'if':      return this._execIf(val);
      case 'for':     return this._execFor(val);
      case 'wait_for':return this._execWaitFor(val);
      case 'send':    return this._execSend(val);
      case 'on':      return this._execOn(val);
      case 'store':   return this._execStore(val);
      default:        return this._execCustom(key, val);
    }
  }

  // @node.* → Node.js subsystem
  async _execNode(parts, val) {
    const subsystem = parts[0];
    switch (subsystem) {
      case 'ipc':            return this._execIPC(parts.slice(1), val);
      case 'worker':         return this._execWorker(parts.slice(1), val);
      case 'net':            return this._execNet(parts.slice(1), val);
      case 'stdio':          return this._execStdio(val);
      case 'fs':             return this._execFS(parts.slice(1), val);
      case 'http':           return this._execHTTP(val);
      case 'master':
      case 'parent_process': return this._execParent(val);
      case 'child_process':
      case 'worker_process': return this._execChild(val);
      default:
        console.warn(`[xcfe] unknown @node.${subsystem}`);
    }
  }

  // ─── @node.ipc ────────────────────────────────────────────────────────────

  async _execIPC(parts, val) {
    const name   = val['@name'] || 'default';
    const mode   = val['@mode'] || 'duplex';
    const script = val['@script'] || val.script || './worker.js';

    if (!this._channels.has(name)) {
      const ch = new IPCChannel(name, script);
      await ch.open();
      this._channels.set(name, ch);

      ch.on('message', msg => {
        const storeKey = val['@read']?.['@store'];
        if (storeKey) this._ctx[storeKey] = msg;
        this.emit?.('ipc:message', { channel: name, msg });
      });
    }

    const ch = this._channels.get(name);

    if (mode === 'duplex' || mode === 'write') {
      const writeData = this._resolve(val['@write']);
      if (writeData !== undefined) ch.send(writeData);
    }

    // Execute nested blocks with channel in scope
    for (const [k, v] of Object.entries(val)) {
      if (k.startsWith('@') && k !== '@name' && k !== '@mode'
          && k !== '@write' && k !== '@read') {
        await this._dispatch(k, v);
      }
    }
  }

  // ─── @node.worker (worker_threads) ───────────────────────────────────────

  async _execWorker(parts, val) {
    const name   = val['@name'] || 'default_worker';
    const script = val['@script'] || val.script || './worker.js';
    const data   = this._resolve(val['@data'] || val.data || {});

    if (!this._channels.has(name)) {
      const ch = new WorkerChannel(name, script, data);
      await ch.open();
      this._channels.set(name, ch);
      ch.on('message', msg => {
        const storeKey = val['@read']?.['@store'];
        if (storeKey) this._ctx[storeKey] = msg;
      });
    }

    const payload = this._resolve(val['@send'] || val['@write']);
    if (payload !== undefined) this._channels.get(name).send(payload);
  }

  // ─── @node.net ────────────────────────────────────────────────────────────

  async _execNet(parts, val) {
    const name = val['@name'] || 'net_default';
    if (!this._channels.has(name)) {
      const ch = new NetChannel(name, {
        port: val.port || 8765,
        host: val.host || '127.0.0.1',
        path: val['@named_pipe'],      // Unix socket / Windows named pipe
      });
      await ch.open();
      this._channels.set(name, ch);
      ch.on('message', msg => {
        const storeKey = val['@read']?.['@store'];
        if (storeKey) this._ctx[storeKey] = msg;
      });
    }
    const payload = this._resolve(val['@write'] || val['@send']);
    if (payload !== undefined) this._channels.get(name).send(payload);
  }

  // ─── @node.stdio ─────────────────────────────────────────────────────────

  async _execStdio(val) {
    if (!this._channels.has('stdio')) {
      this._channels.set('stdio', new StdioChannel('stdio'));
    }
    const payload = this._resolve(val['@write'] || val['@send']);
    if (payload !== undefined) this._channels.get('stdio').send(payload);
  }

  // ─── @node.fs ─────────────────────────────────────────────────────────────

  async _execFS(parts, val) {
    const op    = parts[0] || val['@op'] || 'read';
    const path  = this._resolve(val.path || val['@path']);
    switch (op) {
      case 'read': {
        const data = await fs.readFile(path, val.encoding || 'utf8');
        if (val['@store']) this._ctx[val['@store']] = data;
        return data;
      }
      case 'write': {
        const content = this._resolve(val.data || val['@data'] || '');
        await fs.writeFile(path, typeof content === 'string' ? content : JSON.stringify(content));
        return;
      }
      case 'watch': {
        const watcher = fs.watch(path);
        for await (const evt of watcher) {
          this.emit?.('fs:change', { path, ...evt });
          if (val['@once']) break;
        }
        return;
      }
    }
  }

  // ─── @node.http ───────────────────────────────────────────────────────────

  async _execHTTP(val) {
    const port    = val.port || 3000;
    const handler = this._handlers.get(val['@handler']) || ((req, res) => res.end('ok'));
    const server  = createServer(handler);
    server.listen(port, () => console.log(`[xcfe] http server on :${port}`));
    if (val['@store']) this._ctx[val['@store']] = server;
  }

  // ─── @node.parent_process / @node.master ─────────────────────────────────

  async _execParent(val) {
    for (const [k, v] of Object.entries(val)) {
      if (k === '@send') {
        const target = v['@to'] || 'child';
        const payload = this._resolve(v);
        const ch = this._channels.get(target);
        if (ch) ch.send(payload); else process.send?.(payload);
      }
      if (k === '@on') {
        const event   = v.event || 'message';
        const storeKey = v['@store'];
        const ch = this._channels.get(v['@from']);
        const emitter = ch || process;
        emitter.on(event, msg => {
          if (storeKey) this._ctx[storeKey] = msg;
        });
      }
    }
  }

  // ─── @node.child_process / @node.worker_process ──────────────────────────

  async _execChild(val) {
    const script = this._resolve(val['@script'] || val.script);
    const name   = val['@name'] || 'child';

    const ch = new IPCChannel(name, script);
    await ch.open();
    this._channels.set(name, ch);

    for (const [k, v] of Object.entries(val)) {
      if (k === '@on_ready') {
        ch.once('message', m => {
          if (m?.__ready) {
            const payload = this._resolve(v['@send']?.data);
            if (payload !== undefined) ch.send(payload);
          }
        });
      }
      if (k === '@on') {
        ch.on('message', msg => {
          if (v['@store']) this._ctx[v['@store']] = msg;
        });
      }
    }
  }

  // ─── Control flow ─────────────────────────────────────────────────────────

  async _execIf(val) {
    const cond = this._resolve(val.condition);
    const result = typeof cond === 'string'
      ? Function('ctx', `with(ctx){return(${cond})}`)(this._ctx)
      : cond;
    if (result && val['@then']) {
      for (const [k, v] of Object.entries(val['@then'])) {
        await this._dispatch(k, v);
      }
    } else if (!result && val['@else']) {
      for (const [k, v] of Object.entries(val['@else'])) {
        await this._dispatch(k, v);
      }
    }
  }

  async _execFor(val) {
    const collection = this._resolve(val.in);
    const itemKey    = val.each || 'item';
    const items      = Array.isArray(collection) ? collection
                     : typeof collection?.next === 'function'
                     ? [...collection] : [];
    for (let i = 0; i < items.length; i++) {
      this._ctx[itemKey]   = items[i];
      this._ctx['index']   = i;
      this._ctx['counter'] = i + 1;
      if (val['@do']) {
        for (const [k, v] of Object.entries(val['@do'])) {
          await this._dispatch(k, v);
        }
      }
    }
  }

  async _execWaitFor(val) {
    const key  = typeof val === 'string' ? val : val.key;
    const ch   = val.channel ? this._channels.get(val.channel) : null;
    if (ch) {
      const result = await ch.waitFor(key, val.timeout);
      this._ctx[key] = result;
    }
  }

  async _execSend(val) {
    const target  = val['@to'] || val.to;
    const payload = this._resolve(val);
    const ch = this._channels.get(target);
    if (ch) ch.send(payload);
    else if (target === 'parent_process' || target === 'master_process') {
      process.send?.(payload);
    } else if (target === 'broadcast') {
      for (const c of this._channels.values()) c.send?.(payload);
    }
  }

  async _execOn(val) {
    const event  = val.event || 'message';
    const source = val['@from'] ? this._channels.get(val['@from']) : process;
    if (!source) return;
    source.on(event, msg => {
      if (val['@store']) this._ctx[val['@store']] = msg;
      if (val['@do']) {
        for (const [k, v] of Object.entries(val['@do'])) {
          this._dispatch(k, v);
        }
      }
    });
  }

  _execSet(parts, val) {
    const key = parts.join('.');
    this._ctx[key] = this._resolve(val);
  }

  _execLog(val) {
    console.log('[xcfe]', typeof val === 'string' ? this._resolve(val) : val);
  }

  _execStore(val) {
    const key = typeof val === 'string' ? val : val.key;
    if (val.value !== undefined) this._ctx[key] = this._resolve(val.value);
  }

  _execCustom(key, val) {
    // Hook point for user-defined @ handlers
    const h = this._handlers.get(key);
    if (h) return h(val, this._ctx);
  }

  // ─── Helpers ──────────────────────────────────────────────────────────────

  _resolve(val) {
    if (typeof val === 'string') return resolveTemplate(val, this._ctx);
    if (val && typeof val === 'object') {
      const out = {};
      for (const [k, v] of Object.entries(val)) {
        if (!k.startsWith('@')) out[k] = this._resolve(v);
      }
      return out;
    }
    return val;
  }

  _setup() {
    // Register built-in @ handlers
    this._handlers.set('@transform.xcfe_to_node', (val) => {
      console.log('[xcfe] xcfe→node transform:', JSON.stringify(val, null, 2));
    });
  }

  // ─── Public helpers ───────────────────────────────────────────────────────

  set(key, value)   { this._ctx[key] = value; return this; }
  get(key)          { return this._ctx[key]; }
  channel(name)     { return this._channels.get(name); }
  register(key, fn) { this._handlers.set(key, fn); return this; }
}

// ─── XCFE block DSL builder (for programmatic use) ───────────────────────────

export function xcfe(template) {
  return async function run(ctx = {}) {
    const rt = new XCFENodeRuntime();
    for (const [k, v] of Object.entries(ctx)) rt.set(k, v);
    await rt.execute(template);
    return rt;
  };
}

// ─── @ → Node.js opcode table (reference) ────────────────────────────────────

export const AT_OPCODE_MAP = Object.freeze({
  '@write':         { node: 'channel.send() / stream.write()',     kuhul: '⟁Ch\'en⟁ 0x06' },
  '@read':          { node: 'process.on("message") / stream.data', kuhul: '⟁Yax⟁ 0x04'   },
  '@send':          { node: 'process.send() / ws.send()',          kuhul: '⟁Wo⟁ 0x05'     },
  '@on':            { node: '.on(event, handler)',                  kuhul: '⟁Log⟁ 0x23'   },
  '@fork':          { node: 'child_process.fork()',                 kuhul: '⟁Clu⟁ 0x40'   },
  '@worker':        { node: 'new Worker(script)',                   kuhul: '⟁Nod⟁ 0x41'   },
  '@for':           { node: 'for...of / Array.forEach',            kuhul: '⟁Wo⟁each 0x0B' },
  '@if':            { node: 'if/else',                              kuhul: '⟁Wo⟁if 0x0A'  },
  '@wait_for':      { node: 'new Promise + EventEmitter.once',     kuhul: '⟁Sync⟁ 0x46'  },
  '@store':         { node: 'Map.set / variable assignment',        kuhul: '⟁Sek⟁ 0x03'   },
  '@node.ipc.pipe': { node: 'child_process.fork + IPC',            kuhul: '⟁Clu⟁ 0x40'   },
  '@node.net':      { node: 'net.createConnection / Socket',       kuhul: '⟁Nod⟁ 0x41'   },
  '@node.stdio':    { node: 'process.stdin / process.stdout',      kuhul: '⟁Wo⟁ 0x05'    },
  '@node.http':     { node: 'http.createServer',                    kuhul: '⟁Nod⟁ 0x41'   },
  '@node.fs':       { node: 'fs/promises',                          kuhul: '⟁Ch\'en⟁ 0x06'},
  '@node.worker':   { node: 'worker_threads.Worker',               kuhul: '⟁Dist⟁ 0x44'  },
  '@node.cluster':  { node: 'cluster.fork / worker.send',          kuhul: '⟁Clu⟁ 0x40'   },
});
