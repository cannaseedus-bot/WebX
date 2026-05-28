// XCFE JSON Program Format
// Parses and executes the @ops[]/@state{}/@control{}/@runtime{}/@buffers[] format.
// Based on KUHUL_V1/xvm/cluster.xjson and atomic/moe-wgsl-kernels-v1.0 examples.
//
// Op types: pure | primitive | gpu | agent
// @control ops reference @state variables with $ prefix.

// ─── Built-in native ops ──────────────────────────────────────────────────────

const NATIVE_OPS = {
  'native.LOG': (op, ctx) => {
    ctx.log(`[XCFE] [${(op['@level'] || 'info').toUpperCase()}] ${op['@msg'] || ''}`);
  },
  'native.WRITE': (op, ctx) => {
    const key = op['@key'];
    const val = op['@value'];
    if (key) ctx.state[key] = val;
  },
  'native.READ': (op, ctx) => {
    const key = op['@key'];
    const dst = op['@out'];
    if (dst && key) ctx.vars[dst] = ctx.state[key];
  },
  'xvm.alloc_fibers': (op, ctx) => {
    const count = ctx._resolve(op['@in']?.count, 0);
    ctx.vars[op['@out'] || 'fiber_pool'] = { type: 'fiber_pool', count };
  },
  'xvm.alloc_threads': (op, ctx) => {
    const count   = ctx._resolve(op['@in']?.count, 0);
    const sharedKB = ctx._resolve(op['@in']?.shared_kb, 0);
    ctx.vars[op['@out'] || 'thread_pool'] = { type: 'thread_pool', count, sharedKB };
  },
  'xvm.bind': (op, ctx) => {
    const fibers  = ctx.vars[ctx._resolve(op['@in']?.fibers, '')];
    const threads = ctx.vars[ctx._resolve(op['@in']?.threads, '')];
    const warp    = ctx._resolve(op['@in']?.warp, 32);
    ctx.vars[op['@out'] || 'cluster'] = { type: 'cluster', fibers, threads, warp };
  },
};

// ─── Execution context ────────────────────────────────────────────────────────

class XCFEContext {
  constructor(program, logger) {
    this.state  = {};
    this.vars   = {};
    this.log    = logger || ((m) => console.log(m));

    // Merge @state into context
    const stateKey = Object.keys(program).find(k => k === '@state');
    if (stateKey) Object.assign(this.state, program[stateKey]);
  }

  // Resolve $var references, literal values, or nested objects
  _resolve(val, fallback) {
    if (val === undefined || val === null) return fallback !== undefined ? fallback : val;
    if (typeof val === 'string' && val.startsWith('$')) {
      const key = val.slice(1);
      return key in this.state ? this.state[key] : (key in this.vars ? this.vars[key] : fallback);
    }
    if (typeof val === 'object' && !Array.isArray(val)) {
      const result = {};
      for (const [k, v] of Object.entries(val)) result[k] = this._resolve(v, undefined);
      return result;
    }
    return val;
  }
}

// ─── Program runner ───────────────────────────────────────────────────────────

export class XCFERuntime {
  constructor(options = {}) {
    this.ops = { ...NATIVE_OPS };
    this.log = options.log || ((m) => console.log(m));
  }

  // Register a custom op handler: handler(op, context) => void
  registerOp(name, handler) {
    this.ops[name] = handler;
    return this;
  }

  // Execute a loaded XCFE program object
  run(program) {
    const ctx = new XCFEContext(program, this.log);

    // Determine the control ops array
    const controlKey = '@control';
    const control = program[controlKey];
    if (!Array.isArray(control)) {
      // Legacy format: single control object, noop
      return ctx;
    }

    for (const op of control) {
      const opName = op['@op'];
      if (!opName) continue;
      const handler = this.ops[opName];
      if (handler) {
        handler(op, ctx);
      } else {
        ctx.log(`[XCFE] [WARN] Unknown op: ${opName}`);
      }
    }

    return ctx;
  }

  // Load from a JSON string or object
  load(source) {
    const program = typeof source === 'string' ? JSON.parse(source) : source;
    return this.run(program);
  }
}

// ─── XCFE Program builder ─────────────────────────────────────────────────────

export class XCFEProgram {
  constructor(context = 'xcfe://kuhul/v1', version = '1.0.0') {
    this['@context'] = context;
    this['@version'] = version;
    this['@state']   = {};
    this['@control'] = [];
    this['@runtime'] = {};
    this['@buffers'] = [];
    this['@ops']     = [];
  }

  state(key, value) {
    this['@state'][key] = value;
    return this;
  }

  op(type, props) {
    this['@ops'].push({ type, ...props });
    return this;
  }

  control(op, props) {
    this['@control'].push({ '@op': op, ...props });
    return this;
  }

  runtime(key, value) {
    this['@runtime'][key] = value;
    return this;
  }

  buffer(name, props) {
    this['@buffers'].push({ name, ...props });
    return this;
  }

  toJSON() {
    return JSON.stringify(this, null, 2);
  }
}

// ─── Parse a micronaut .xjson file structure ──────────────────────────────────

export function parseMicronauts(xjson) {
  const experts = {};
  const skills  = {};
  const agents  = {};

  for (const [key, val] of Object.entries(xjson)) {
    if (key.startsWith('@experts')) {
      const id = key.split('.')[1] || 'all';
      if (id === 'all' && typeof val === 'object') Object.assign(experts, val);
      else experts[id] = val;
    }
    if (key.startsWith('@skills.'))  skills[key.slice('@skills.'.length)]  = val;
    if (key.startsWith('@agent.') && !key.includes('.memory') && !key.includes('.learned') && !key.includes('.state'))
      agents[key.slice('@agent.'.length)] = val;
  }

  return { experts, skills, agents,
           modelRef: xjson['@model.ref'] || null,
           moeRouter: xjson['@moe.router'] || null,
           foldLanes: xjson['@fold.lanes'] || {},
           foldEdges: xjson['@fold.edges'] || [] };
}

export default XCFERuntime;
