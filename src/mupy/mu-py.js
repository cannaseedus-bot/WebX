// mu-py.js — µPY: Python→KXML transpilation bridge
//
// µPY is NOT a runtime dependency — it is a TRANSPILATION layer.
// Write Python-like tensor code → compile to KXML graph nodes → execute on µBRAIN.
//
// Pipeline:
//   Python-like source → µPYTranspiler.transpile() → KXML graph (string)
//   KXML graph → µPYRuntime.execute(code, inputs) → result via µBRAIN
//   µJSONL export → persistent, trainable, browser-runnable
//
// µModel mapping (Python op → µModel type):
//   @         mu_gemm      (matrix multiply)
//   +         mu_add       lipschitz=1.0
//   -         mu_subtract  lipschitz=1.0
//   *         mu_multiply  lipschitz=dynamic
//   /         mu_divide    lipschitz=dynamic
//   relu()    mu_relu      lipschitz=1.0
//   tanh()    mu_tanh      lipschitz=1.0
//   sigmoid() mu_sigmoid   lipschitz=0.25
//   softmax() mu_softmax   lipschitz=1.0
//
// Backward edges are auto-generated: every forward edge gets a backward
// counterpart with phase_gate (forward_requires=Sek, backward_requires=Ch'en).
// This is the static bidirectional graph that makes µBRAIN provable.

// ─── Built-in µModel definitions ─────────────────────────────────────────────

export const MU_BUILTINS = Object.freeze({
  // Operators
  '@':        { mumodel: 'mu_gemm',        lipschitz: 'dynamic', phase: 'Sek', rank: 2 },
  '+':        { mumodel: 'mu_add',         lipschitz: 1.0,       phase: 'Sek', rank: 2 },
  '-':        { mumodel: 'mu_subtract',    lipschitz: 1.0,       phase: 'Sek', rank: 2 },
  '*':        { mumodel: 'mu_multiply',    lipschitz: 'dynamic', phase: 'Sek', rank: 2 },
  '/':        { mumodel: 'mu_divide',      lipschitz: 'dynamic', phase: 'Sek', rank: 2 },
  // Activations
  relu:       { mumodel: 'mu_relu',        lipschitz: 1.0,       phase: 'Sek' },
  tanh:       { mumodel: 'mu_tanh',        lipschitz: 1.0,       phase: 'Sek' },
  sigmoid:    { mumodel: 'mu_sigmoid',     lipschitz: 0.25,      phase: 'Sek' },
  softmax:    { mumodel: 'mu_softmax',     lipschitz: 1.0,       phase: 'Sek' },
  // Loss
  cross_entropy: { mumodel: 'mu_cross_entropy', lipschitz: 2.0,  phase: 'Sek' },
  mse:           { mumodel: 'mu_mse',           lipschitz: 2.0,  phase: 'Sek' },
  // Normalization
  layer_norm: { mumodel: 'mu_layer_norm',  lipschitz: 1.0,       phase: 'Sek' },
  batch_norm: { mumodel: 'mu_batch_norm',  lipschitz: 1.0,       phase: 'Sek' },
  // Reductions
  sum:        { mumodel: 'mu_sum',         lipschitz: 1.0,       phase: 'Sek' },
  mean:       { mumodel: 'mu_mean',        lipschitz: 1.0,       phase: 'Sek' },
  max:        { mumodel: 'mu_max',         lipschitz: 1.0,       phase: 'Sek' },
  // Shape ops
  reshape:    { mumodel: 'mu_reshape',     lipschitz: 1.0,       phase: 'Pop' },
  transpose:  { mumodel: 'mu_transpose',   lipschitz: 1.0,       phase: 'Pop' },
});

// ─── µPYTranspiler ────────────────────────────────────────────────────────────

export class MuPYTranspiler {
  constructor(customModels = {}) {
    this._models = { ...MU_BUILTINS, ...customModels };
    this._nodeCounter = 0;
  }

  // ── Public entry point ──

  /** Transpile a computation description to KXML graph XML string. */
  transpile(desc) {
    const ctx = { nodes: [], edges: [], inputs: [], outputs: [] };
    if (typeof desc === 'function') {
      this._fromFn(desc, ctx);
    } else if (desc && typeof desc === 'object') {
      this._fromSpec(desc, ctx);
    }
    return this._toKXML(desc.name ?? 'graph', ctx);
  }

  /** Build a KXML graph from an explicit operation spec array. */
  fromOps(name, ops) {
    const ctx = { nodes: [], edges: [], inputs: [], outputs: [] };
    let prevOut = null;
    for (const op of ops) {
      const mid = this._models[op.op] ?? { mumodel: op.op, lipschitz: 1.0, phase: 'Sek' };
      const id  = `node_${this._nodeCounter++}`;
      const out = op.output ?? `out_${id}`;
      const ins = op.inputs ?? (prevOut ? [prevOut] : []);
      ctx.nodes.push({ id, mumodel: mid.mumodel, lipschitz: mid.lipschitz, phase: mid.phase ?? 'Sek', inputs: ins, output: out, shape: op.shape });
      if (prevOut) ctx.edges.push({ from: `node_${this._nodeCounter - 2}`, to: id });
      prevOut = out;
    }
    if (prevOut) ctx.outputs.push(prevOut);
    return this._toKXML(name, ctx);
  }

  // ── Internal helpers ──

  _fromSpec(desc, ctx) {
    const layers = desc.layers ?? desc.ops ?? [];
    let prev = null;
    for (const layer of layers) {
      const mid = this._models[layer.op ?? layer.type] ?? { mumodel: layer.op ?? layer.type, lipschitz: 1.0, phase: 'Sek' };
      const id  = `node_${this._nodeCounter++}`;
      const out = `out_${id}`;
      ctx.nodes.push({ id, mumodel: mid.mumodel, lipschitz: mid.lipschitz, phase: mid.phase ?? 'Sek',
                       inputs: layer.inputs ?? (prev ? [prev] : []), output: out, shape: layer.shape });
      if (prev) {
        const prevId = `node_${this._nodeCounter - 2}`;
        ctx.edges.push({ from: prevId, to: id });
      }
      prev = out;
    }
    if (prev) ctx.outputs.push(prev);
    if (desc.inputs) ctx.inputs.push(...desc.inputs);
  }

  _fromFn(fn, ctx) {
    // Call the function with proxy args to capture the op graph
    const trace = [];
    const makeProxy = (name) => new Proxy({ _name: name }, {
      get(t, k) {
        if (k === '_name') return t._name;
        if (k === 'matmul' || k === '__matmul__') return (other) => {
          const out = `gemm_${trace.length}`;
          trace.push({ op: '@', inputs: [t._name, other._name ?? other], output: out });
          return makeProxy(out);
        };
        return undefined;
      },
    });
    try {
      const argNames = fn.toString().match(/\(([^)]*)\)/)?.[1].split(',').map(s => s.trim()) ?? [];
      const proxies  = argNames.map(a => makeProxy(a));
      fn(...proxies);
      ctx.inputs.push(...argNames);
    } catch (_) {}
    for (const op of trace) {
      const mid = this._models[op.op] ?? { mumodel: 'mu_op', lipschitz: 1.0, phase: 'Sek' };
      const id  = `node_${this._nodeCounter++}`;
      ctx.nodes.push({ id, mumodel: mid.mumodel, lipschitz: mid.lipschitz, phase: mid.phase ?? 'Sek',
                       inputs: op.inputs, output: op.output, shape: op.shape });
    }
    if (trace.length > 0) ctx.outputs.push(trace[trace.length - 1].output);
  }

  _toKXML(name, ctx) {
    const nodeXML = ctx.nodes.map(n => `  <node id="${n.id}" mumodel="${n.mumodel}" lipschitz="${n.lipschitz}" phase="${n.phase}"${n.shape ? ` shape="${JSON.stringify(n.shape)}"` : ''}>
    <inputs>${n.inputs.join(', ')}</inputs>
    <output>${n.output}</output>
  </node>`).join('\n');

    const edgeXML = ctx.edges.map(e => `  <edge from="${e.from}" to="${e.to}">
    <forward/>
    <backward scale="0.001"/>
    <phase_gate forward_requires="Sek" backward_requires="Ch'en"/>
  </edge>`).join('\n');

    const inputXML = ctx.inputs.map(i => `  <input name="${i}"/>`).join('\n');
    const outXML   = ctx.outputs.map(o => `  <output>${o}</output>`).join('\n');

    return `<?xml version="1.0" encoding="utf-8"?>
<kxml:graph id="${name}" phase="Sek" xmlns:kxml="kuhul:kxml">
${inputXML}
${nodeXML}
${edgeXML}
${outXML}
</kxml:graph>`;
  }

  // ── µJSONL export ──

  /** Export a compiled graph as µJSONL (persistent, trainable). */
  toMuJSONL(name, ctx) {
    const lines = [`# µPY export: ${name}`];
    for (const n of (ctx?.nodes ?? [])) {
      lines.push(JSON.stringify({
        id:        `${name}.${n.id}`,
        mumodel:   n.mumodel,
        inputs:    n.inputs,
        output:    n.output,
        shape:     n.shape ?? null,
        lipschitz: n.lipschitz,
        phase:     n.phase,
        weight:    1.0,
        gradient:  0.0,
      }));
    }
    return lines.join('\n');
  }
}

// ─── µPYRuntime ────────────────────────────────────────────────────────────────

export class MuPYRuntime {
  constructor(muBrain = null) {
    this._brain     = muBrain;
    this._transpiler = new MuPYTranspiler();
    this._graphs    = new Map();
    this._weights   = new Map();
    this._lr        = 0.01;
  }

  /** Transpile + execute a computation spec with provided tensor inputs. */
  execute(spec, inputs = {}) {
    const kxml    = this._transpiler.transpile(spec);
    const graphId = spec.name ?? `graph_${Date.now()}`;
    this._graphs.set(graphId, { kxml, spec });

    // Evaluate nodes in topological order
    const env   = { ...inputs };
    const nodes = this._nodesFrom(spec);
    const grad  = {};

    // Forward pass (Sek)
    for (const node of nodes) {
      const ins = node.inputs.map(i => env[i] ?? 0);
      env[node.output] = this._evalNode(node.mumodel, ins, env);
    }

    // Return result + forward env
    const outputs = nodes.length > 0 ? env[nodes[nodes.length - 1].output] : null;
    return { graphId, outputs, env, kxml };
  }

  /** Training loop: forward + backward + weight update. */
  train(spec, examples, epochs = 10, onEpoch = null) {
    const nodes = this._nodesFrom(spec);
    const history = [];

    for (let ep = 0; ep < epochs; ep++) {
      let totalLoss = 0, count = 0;
      for (const ex of examples) {
        const { env } = this.execute(spec, ex.inputs);
        const pred    = env[nodes[nodes.length - 1]?.output];
        const loss    = this._loss(pred, ex.target);
        totalLoss    += loss;
        count++;
        // Backward (Ch'en) — update node weights via stored gradient
        const err = pred - ex.target;
        for (const node of nodes) {
          const key = node.id;
          const L   = typeof node.lipschitz === 'number' ? node.lipschitz : 1.0;
          const g   = Math.max(-L, Math.min(L, err));
          const w   = this._weights.get(key) ?? 1.0;
          this._weights.set(key, Math.max(0.1, Math.min(2.0, w + this._lr * g)));
        }
      }
      const avg = count > 0 ? totalLoss / count : 0;
      history.push({ epoch: ep + 1, loss: avg });
      onEpoch?.({ epoch: ep + 1, epochs, loss: avg });
    }
    return history;
  }

  _nodesFrom(spec) {
    if (spec.layers) return spec.layers.map((l, i) => ({
      id:       `node_${i}`,
      mumodel:  MU_BUILTINS[l.op ?? l.type]?.mumodel ?? l.op,
      inputs:   l.inputs ?? [],
      output:   l.output ?? `out_${i}`,
      lipschitz: MU_BUILTINS[l.op ?? l.type]?.lipschitz ?? 1.0,
    }));
    return [];
  }

  _evalNode(mumodel, inputs, env) {
    const nums = inputs.map(v => typeof v === 'number' ? v : 0);
    switch (mumodel) {
      case 'mu_add':         return nums.reduce((a,b) => a+b, 0);
      case 'mu_subtract':    return nums[0] - (nums[1] ?? 0);
      case 'mu_multiply':    return nums.reduce((a,b) => a*b, 1);
      case 'mu_divide':      return nums[1] !== 0 ? nums[0] / nums[1] : 0;
      case 'mu_relu':        return Math.max(0, nums[0] ?? 0);
      case 'mu_sigmoid':     return 1 / (1 + Math.exp(-(nums[0] ?? 0)));
      case 'mu_tanh':        return Math.tanh(nums[0] ?? 0);
      case 'mu_softmax': {
        const max = Math.max(...nums); const e = nums.map(x => Math.exp(x-max)); const s = e.reduce((a,b)=>a+b,0);
        return e.map(x => x/s);
      }
      case 'mu_gemm':        return (nums[0] ?? 0) * (nums[1] ?? 1); // scalar proxy
      case 'mu_cross_entropy': return -Math.log(Math.max(1e-8, nums[0] ?? 0.5));
      case 'mu_mse':         return (nums[0] - (nums[1] ?? 0)) ** 2;
      case 'mu_sum':         return nums.reduce((a,b) => a+b, 0);
      case 'mu_mean':        return nums.length > 0 ? nums.reduce((a,b)=>a+b,0)/nums.length : 0;
      case 'mu_max':         return Math.max(...nums);
      default:               return nums[0] ?? 0;
    }
  }

  _loss(pred, target) {
    const p = Array.isArray(pred) ? pred[target] ?? 0.5 : pred ?? 0;
    const t = typeof target === 'number' ? target : 0;
    return (p - t) ** 2;
  }
}

// ─── Canonical MLP spec (the two-layer example from µPY docs) ─────────────────

export const TWO_LAYER_MLP = Object.freeze({
  name: 'two_layer_mlp',
  inputs: ['x', 'W1', 'b1', 'W2', 'b2'],
  layers: [
    { op: '@',       inputs: ['x', 'W1'],   output: 'z1',     shape: [null, 256] },
    { op: '+',       inputs: ['z1', 'b1'],  output: 'h_pre',  shape: [null, 256] },
    { op: 'relu',    inputs: ['h_pre'],     output: 'h',      shape: [null, 256] },
    { op: '@',       inputs: ['h', 'W2'],   output: 'z2',     shape: [null, 10]  },
    { op: '+',       inputs: ['z2', 'b2'],  output: 'logits', shape: [null, 10]  },
    { op: 'softmax', inputs: ['logits'],    output: 'y_pred', shape: [null, 10]  },
  ],
});
