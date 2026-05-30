// fold-algebra.js — @semantics / @opcodes / @context / @folds / @horizontal_folds / @micro_folds
//
// The folding hierarchy is self-similar at every scale:
//
//   @semantics       — MEANING layer (dataflow model, memory model, error strategy)
//   @opcodes         — INSTRUCTION layer (what the CPU/GPU executes)
//   @context         — ENVIRONMENT layer (registers, cache, thread-local state)
//   @folds           — MACRO layer (tree reductions, window aggregations)
//   @horizontal_folds— VECTOR/PARALLEL layer (SIMD, warp, bank-parallel)
//   @micro_folds     — ATOMIC layer (byte, nibble, bit operations)
//
// The `@` operator is a monoid on these layers:
//   Identity: @(x) = x            (pass-through)
//   Assoc:    @a.(@b.@c) = (@a.@b).@c
//   Compose:  @a ∘ @b = @a applied to the output of @b
//
// Mapping to K'UHUL opcodes:
//   @semantics        → K'UHUL source lang (meaning encoding)
//   @opcodes          → K'UHUL bytecode 0x01-0x69
//   @context          → KXML graph (execution environment)
//   @folds            → MX2LM fold objects (FoldStore / FoldObject)
//   @horizontal_folds → Distributed opcodes ⟁Dist⟁ / ⟁Reduce⟁ (0x44/0x48)
//   @micro_folds      → 3-byte bytecode instructions (BytecodeAssembler)

// ─── Semantics layer ─────────────────────────────────────────────────────────

export const SEMANTIC_MODELS = Object.freeze({
  dataflow:        { execution: 'lazy',       memory: 'sequential' },
  streaming_batch: { execution: 'eager',      memory: 'ring_buffer' },
  reactive:        { execution: 'event_driven',memory: 'immutable'  },
  quantum:         { execution: 'superposed',  memory: 'entangled'  },
  imperative:      { execution: 'sequential',  memory: 'mutable'    },
});

export class SemanticLayer {
  constructor(model = 'dataflow') {
    this.model       = SEMANTIC_MODELS[model] ?? SEMANTIC_MODELS.dataflow;
    this.name        = model;
    this.error_policy = { strategy: 'retry_with_backoff', max_attempts: 3 };
    this.optimizations = [];
  }

  setErrorPolicy(policy)    { this.error_policy = policy; return this; }
  addOptimization(opt)      { this.optimizations.push(opt); return this; }

  annotate(foldNode) {
    return { ...foldNode, '@semantics': this.name, _semantic_model: this.model };
  }
}

// ─── Opcode layer ─────────────────────────────────────────────────────────────

export const MICRO_OPCODES = Object.freeze({
  // Byte-level (0xA0-0xAF)
  LOAD_BYTE:  0xA0, STORE_BYTE: 0xA1, XOR_BYTE: 0xA2, ROT_BYTE: 0xA3,
  // Bit-level (0xB0-0xBF)
  BIT_TEST:   0xB0, BIT_SET: 0xB1, BIT_CLEAR: 0xB2,
  // SIMD (0x30-0x3F — extension of core range)
  SIMD_LOAD:  0x30, SIMD_STORE: 0x31, SIMD_MADD: 0x32,
  SIMD_GATHER:0x33, SIMD_SCATTER: 0x34, SIMD_MASKED: 0x35,
  // Vector (0x10-0x1F)
  VEC_LOAD:   0x10, VEC_ADD: 0x11, VEC_MUL: 0x12, VEC_DOT: 0x13,
  // Control (mirrors core 0x01-0x15)
  OP_JUMP:    0x20, OP_CALL: 0x21, OP_RETURN: 0x22,
  // Quantum (symbolic — maps to future quantum ISA)
  HADAMARD:   0xC0, CNOT: 0xC1, MEASURE: 0xC2,
});

export class OpcodeLayer {
  constructor(category = 'arithmetic') {
    this.category = category;
    this._table   = new Map();
  }

  define(name, code, meta = {}) {
    this._table.set(name, { code, ...meta });
    return this;
  }

  encode(name) { return this._table.get(name)?.code ?? 0x00; }
  lookup(code) {
    for (const [name, v] of this._table) if (v.code === code) return name;
    return null;
  }
  all()        { return Object.fromEntries(this._table); }
}

// ─── Context layer ────────────────────────────────────────────────────────────

export class ContextLayer {
  constructor(opts = {}) {
    this.scope       = opts.scope       ?? 'thread_local';
    this.registers   = opts.registers   ?? 32;
    this.reg_bits    = opts.reg_bits    ?? 64;
    this.stack_kb    = opts.stack_kb    ?? 1024;
    this.cache       = {
      L1_kb: opts.L1_kb ?? 32,
      L2_kb: opts.L2_kb ?? 256,
      L3_kb: opts.L3_kb ?? 8192,
    };
    this._state      = new Map();
    this._flags      = 0;
  }

  set(key, val)    { this._state.set(key, val); return this; }
  get(key)         { return this._state.get(key); }
  setFlag(bit)     { this._flags |= (1 << bit); return this; }
  clearFlag(bit)   { this._flags &= ~(1 << bit); return this; }
  testFlag(bit)    { return !!(this._flags & (1 << bit)); }

  snapshot() {
    return {
      scope: this.scope, registers: this.registers,
      flags: this._flags, stack_kb: this.stack_kb,
      cache: { ...this.cache },
      state: Object.fromEntries(this._state),
    };
  }
}

// ─── Fold hierarchy ───────────────────────────────────────────────────────────

// One fold node in the macro aggregation tree
export class Fold {
  constructor(name, opts = {}) {
    this.name      = name;
    this.operation = opts.operation ?? 'identity';
    this.associative = opts.associative ?? true;
    this.children  = [];
    this._result   = undefined;
  }

  add(child)      { this.children.push(child); return this; }

  // Apply the fold operation bottom-up
  reduce(inputs) {
    if (this.children.length === 0) return inputs;
    const childResults = this.children.map(c => c.reduce(inputs));
    return this._apply(childResults);
  }

  _apply(values) {
    switch (this.operation) {
      case 'sum':   return values.reduce((a, b) => a + b, 0);
      case 'product': return values.reduce((a, b) => a * b, 1);
      case 'max':   return Math.max(...values);
      case 'min':   return Math.min(...values);
      case 'concat':return values.flat();
      case 'tree':  return { left: values[0], right: values[1] };
      default:      return values;
    }
  }

  get result() { return this._result; }
}

// ─── Horizontal fold (vector / parallel) ──────────────────────────────────────

export class HorizontalFold {
  constructor(name, opts = {}) {
    this.name        = name;
    this.width       = opts.width       ?? 8;      // lanes or workers
    this.element_type= opts.element_type ?? 'float32';
    this.strategy    = opts.strategy    ?? 'balanced';
  }

  // Dispatch a flat array across `width` lanes
  dispatch(items) {
    const laneSize = Math.ceil(items.length / this.width);
    const lanes    = [];
    for (let i = 0; i < this.width; i++) {
      lanes.push(items.slice(i * laneSize, (i + 1) * laneSize));
    }
    return lanes;
  }

  // Gather lane results back to a flat array
  gather(lanes) { return lanes.flat(); }

  // Reduce all lanes with an operation
  reduce(lanes, op = 'sum') {
    const sums = lanes.map(lane =>
      lane.reduce((a, b) => {
        switch (op) {
          case 'sum': return a + b;
          case 'max': return Math.max(a, b);
          case 'min': return Math.min(a, b);
          default:    return b;
        }
      }, op === 'max' ? -Infinity : op === 'min' ? Infinity : 0)
    );
    return sums.reduce((a, b) => a + b, 0); // cross-lane reduce
  }

  // SIMD-style fused multiply-add across 4-element chunks
  fmadd(a, b, c) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i += 4) {
      out[i]   = a[i]*b[i]+c[i];   out[i+1] = a[i+1]*b[i+1]+c[i+1];
      out[i+2] = a[i+2]*b[i+2]+c[i+2]; out[i+3] = a[i+3]*b[i+3]+c[i+3];
    }
    return out;
  }
}

// ─── Micro fold (byte / nibble / bit) ────────────────────────────────────────

export class MicroFold {
  constructor(name, granularity = 'byte') {
    this.name        = name;
    this.granularity = granularity;  // 'bit' | 'nibble' | 'byte' | 'word'
    this._ops        = new Map();
  }

  op(name, fn) { this._ops.set(name, fn); return this; }
  run(name, ...args) {
    const fn = this._ops.get(name);
    if (!fn) throw new Error(`MicroFold ${this.name}: unknown op "${name}"`);
    return fn(...args);
  }

  // Built-in byte operations
  static byte(value) {
    return {
      high:    (value >> 4) & 0x0F,
      low:      value & 0x0F,
      xor:     (v) => value ^ v,
      rotate:  (n) => ((value << n) | (value >> (8 - n))) & 0xFF,
      bitTest: (b) => !!(value & (1 << b)),
      bitSet:  (b) => value | (1 << b),
      bitClear:(b) => value & ~(1 << b),
    };
  }

  // Quantum qubit state
  static qubit(alpha = 1, beta = 0) {
    const norm = Math.sqrt(alpha**2 + beta**2) || 1;
    return {
      alpha: alpha / norm,
      beta:  beta  / norm,
      hadamard() {
        const a = (this.alpha + this.beta) / Math.SQRT2;
        const b = (this.alpha - this.beta) / Math.SQRT2;
        return MicroFold.qubit(a, b);
      },
      measure() {
        return Math.random() < this.alpha**2 ? 0 : 1;
      },
      prob0() { return this.alpha ** 2; },
      prob1() { return this.beta  ** 2; },
    };
  }
}

// ─── Full folding pipeline builder ────────────────────────────────────────────

export class FoldPipeline {
  constructor(name) {
    this.name       = name;
    this.semantic   = new SemanticLayer('dataflow');
    this.context    = new ContextLayer();
    this.stages     = [];
  }

  withSemantic(model)  { this.semantic = new SemanticLayer(model); return this; }
  withContext(opts)    { this.context  = new ContextLayer(opts); return this; }

  addFold(fold)        { this.stages.push({ type: 'fold', node: fold }); return this; }
  addHFold(hfold)      { this.stages.push({ type: 'hfold', node: hfold }); return this; }
  addMicroFold(mfold)  { this.stages.push({ type: 'micro', node: mfold }); return this; }

  // Execute the pipeline on a data array
  run(data) {
    let current = data;
    for (const { type, node } of this.stages) {
      switch (type) {
        case 'fold':  current = [node.reduce(current)]; break;
        case 'hfold': {
          const lanes = node.dispatch(current);
          current     = node.gather(lanes);
          break;
        }
        case 'micro': {
          current = current.map(v => {
            const b = MicroFold.byte(typeof v === 'number' ? v & 0xFF : 0);
            return b.xor(0x5A);  // default micro-op: XOR scramble
          });
          break;
        }
      }
    }
    return current;
  }

  // Matrix multiply using tiled horizontal folds (GPT-2 attention scale)
  static matmul(A, B, M, K, N, tileSize = 16) {
    const C = new Float32Array(M * N);
    const hfold = new HorizontalFold('matmul_tile', { width: tileSize });
    for (let i = 0; i < M; i += tileSize) {
      for (let j = 0; j < N; j += tileSize) {
        for (let k = 0; k < K; k += tileSize) {
          // Tile multiply-accumulate
          const aSlice = Array.from(A.subarray?.(i*K+k, i*K+k+tileSize) ?? []);
          const bSlice = Array.from(B.subarray?.(k*N+j, k*N+j+tileSize) ?? []);
          const acc    = hfold.fmadd(aSlice, bSlice, new Array(tileSize).fill(0));
          for (let t = 0; t < tileSize && i+t < M; t++) {
            C[(i+t)*N+j] += acc[t] ?? 0;
          }
        }
      }
    }
    return C;
  }
}

// ─── XCFE fold namespace dispatcher (plug into XCFENodeRuntime) ───────────────

export function registerFoldNamespaces(rt) {
  const pipelines = new Map();

  rt._handlers.set('@semantics', (val, ctx) => {
    const model   = Object.keys(val)[0] ?? 'dataflow';
    const layer   = new SemanticLayer(model);
    ctx['_semantic'] = layer;
    return layer;
  });

  rt._handlers.set('@folds', (val, ctx) => {
    const name   = val['@operation'] ?? 'identity';
    const fold   = new Fold(name, val);
    const inputs = ctx['_inputs'] ?? [];
    ctx['_fold_result'] = fold.reduce(inputs);
    if (val['@store']) ctx[val['@store']] = ctx['_fold_result'];
    return fold;
  });

  rt._handlers.set('@horizontal_folds', (val, ctx) => {
    const name  = Object.keys(val)[0] ?? 'vector';
    const width = val[Object.keys(val)[0]]?.width ?? val.width ?? 8;
    const hf    = new HorizontalFold(name, { width });
    const data  = ctx['_data'] ?? [];
    const lanes = hf.dispatch(data);
    ctx['_lanes'] = lanes;
    if (val['@store']) ctx[val['@store']] = hf.gather(lanes);
    return hf;
  });

  rt._handlers.set('@micro_folds', (val, ctx) => {
    const grain = Object.keys(val)[0] ?? 'byte';
    const mf    = new MicroFold(grain, grain);
    ctx['_micro'] = mf;
    return mf;
  });

  rt._handlers.set('@context', (val, ctx) => {
    const layer = new ContextLayer(val);
    ctx['_context'] = layer;
    return layer;
  });
}
