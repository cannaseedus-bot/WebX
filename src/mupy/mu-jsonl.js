// mu-jsonl.js — µJSONL: Atomic JSONL Grammar Units for µModel Composition
//
// Each line is a provable linguistic µModel:
//   - one semantic operation  (µmodel type)
//   - one pattern match       (regex with capture groups)
//   - one gradient direction  (trainable weight + Lipschitz bound)
//
// Bridge between symbolic AI (grammar rules) and connectionist AI (weights):
//   symbolic:     patterns remain human-readable JSONL
//   connectionist: weights updated via gradient descent
//   formal:       Lipschitz bound per entry = soft-landing guarantee
//
// FLUX IR connection:
//   forward()  = Sek phase  (execute reducer)
//   backward() = Ch'en phase (update weights as FLUX action)
//   compile()  = Wo phase   (bind pattern intent)
//   load()     = Pop phase  (load snapshot from storage)
//
// KXML connection:
//   lipschitz field  ↔  <soft_landing lipschitz="N"/> in KXML node
//   phase field      ↔  K'UHUL phase gate (prevents invalid states)
//   device field     ↔  gpu/cpu dispatch hint for HD4600 optimizer
//
// µMODEL connection:
//   Each base µMODEL has an associated µJSONL grammar file.
//   The µJSONL weights evolve during inference — grammar self-optimizes.

// ─── Schema constants ─────────────────────────────────────────────────────────

export const MU_MODEL_TYPES = Object.freeze([
  'mu_add', 'mu_subtract', 'mu_multiply', 'mu_divide',
  'mu_chain', 'mu_pattern', 'mu_subtract_abs',
  // Extended types wired to base µMODELS
  'mu_fibonacci', 'mu_pi', 'mu_mayan', 'mu_linalg',
  'mu_geodesic',  'mu_tensor', 'mu_logic', 'mu_phase',
]);

export const PHASES = Object.freeze(['Pop', 'Wo', 'Sek', "Ch'en", 'Xul']);

// ─── Executor factory ─────────────────────────────────────────────────────────

function makeExecutor(type, compiler) {
  switch (type) {
    case 'mu_add':          return ns => ns.reduce((a,b) => a + b, 0);
    case 'mu_subtract':     return ns => ns[0] - (ns[1] ?? 0);
    case 'mu_subtract_abs': return ns => Math.abs(ns[0] - (ns[1] ?? 0));
    case 'mu_multiply':     return ns => ns.reduce((a,b) => a * b, 1);
    case 'mu_divide':       return ns => ns[1] !== 0 ? ns[0] / ns[1] : NaN;
    case 'mu_fibonacci': {
      const fib = (n) => { let [a,b]=[0,1]; for(let i=0;i<n;i++)[a,b]=[b,a+b]; return a; };
      return ns => fib(ns[0] ?? 0);
    }
    case 'mu_pi':    return ns => Math.PI * (ns[0] ?? 1);
    case 'mu_mayan': return ns => (ns[0] ?? 0) % 20;
    case 'mu_linalg': {
      const dot = (a,b) => a.reduce((s,v,i) => s + v*(b[i]??0), 0);
      return ns => dot(ns, ns); // dot self = squared norm
    }
    case 'mu_geodesic': return ns => Math.acos(Math.max(-1, Math.min(1, ns[0] ?? 0)));
    case 'mu_tensor':   return ns => ns.reduce((a,b) => a * b, 1); // outer product scalar
    case 'mu_logic':    return ns => ns.every(Boolean) ? 1 : 0;
    case 'mu_phase':    return ns => ((ns[0] ?? 0) + Math.PI * 2) % (Math.PI * 2);
    case 'mu_chain':    return (ns, steps) => compiler._execChain(ns, steps);
    default:            return ns => ns[0] ?? 0;
  }
}

// ─── MuJSONLCompiler ──────────────────────────────────────────────────────────

export class MuJSONLCompiler {
  constructor() {
    this._entries   = new Map();  // id -> entry
    this._weights   = new Map();  // id -> weight
    this._gradients = new Map();  // id -> accumulated gradient
    this._compiled  = null;       // cached compiled patterns
    this._dirty     = false;
  }

  // ── Pop phase: load ──

  load(jsonlText) {
    for (const raw of jsonlText.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      try {
        const entry = JSON.parse(line);
        if (!entry.id || !entry.pattern) continue;
        this._entries.set(entry.id, entry);
        this._weights.set(entry.id, entry.weight ?? 1.0);
        this._gradients.set(entry.id, entry.gradient ?? 0);
        this._dirty = true;
      } catch (_) {}
    }
    this._compiled = null;
    return this;
  }

  add(entry) {
    if (!entry.id || !entry.pattern) throw new Error('µJSONL entry requires id + pattern');
    this._entries.set(entry.id, entry);
    this._weights.set(entry.id, entry.weight ?? 1.0);
    this._gradients.set(entry.id, entry.gradient ?? 0);
    this._compiled = null;
    return this;
  }

  // ── Wo phase: compile ──

  compile() {
    if (this._compiled && !this._dirty) return this._compiled;
    this._compiled = [];
    for (const [id, entry] of this._entries) {
      let re;
      try { re = new RegExp(entry.pattern, 'i'); } catch (_) { continue; }
      this._compiled.push({
        id,
        regex:    re,
        mumodel:  entry.mumodel ?? entry['µmodel'] ?? 'mu_pattern',
        weight:   this._weights.get(id),
        phase:    entry.phase ?? 'Sek',
        lipschitz: typeof entry.lipschitz === 'number' ? entry.lipschitz : 1.0,
        steps:    entry.steps ?? [],
        keywords: entry.keywords ?? [],
        device:   entry.device ?? 'auto',
        execute:  makeExecutor(entry.mumodel ?? entry['µmodel'] ?? 'mu_pattern', this),
      });
    }
    this._compiled.sort((a,b) => b.weight - a.weight);
    this._dirty = false;
    return this._compiled;
  }

  // ── Sek phase: forward ──

  forward(input) {
    const patterns = this.compile();
    let bestMatch = null, bestScore = -Infinity, matchedNums = [];

    for (const p of patterns) {
      const m = input.match(p.regex);
      if (!m) continue;
      const nums = m.slice(1).map(Number).filter(n => !isNaN(n));
      if (p.weight > bestScore) {
        bestScore = p.weight;
        bestMatch = p;
        matchedNums = nums;
      }
    }

    if (!bestMatch) return { prediction: null, patternId: null, numbers: [], phase: null };
    const prediction = bestMatch.execute(matchedNums, bestMatch.steps);
    return {
      prediction,
      patternId: bestMatch.id,
      numbers:   matchedNums,
      phase:     bestMatch.phase,
      device:    bestMatch.device,
      mumodel:   bestMatch.mumodel,
    };
  }

  // ── Ch'en phase: backward ──

  backward(errorSignal, patternId, learningRate = 0.01) {
    const entry    = this._entries.get(patternId);
    if (!entry) return 0;

    const lipschitz = typeof entry.lipschitz === 'number' ? entry.lipschitz : 1.0;
    let   gradient  = Math.max(-lipschitz, Math.min(lipschitz, errorSignal));

    const prev = this._gradients.get(patternId) ?? 0;
    this._gradients.set(patternId, prev + gradient);

    const w    = this._weights.get(patternId) ?? 1.0;
    const newW = Math.max(0.1, Math.min(2.0, w + learningRate * gradient));
    this._weights.set(patternId, newW);
    this._compiled = null; // invalidate sorted cache
    return gradient;
  }

  // ── Chain execution ──

  _execChain(numbers, stepIds) {
    let result = numbers[0] ?? 0;
    let idx    = 1;
    for (const sid of stepIds) {
      const entry = this._entries.get(sid);
      if (!entry) continue;
      const exec = makeExecutor(entry.mumodel ?? entry['µmodel'] ?? 'mu_pattern', this);
      result = exec([result, numbers[idx++] ?? 0], entry.steps ?? []);
    }
    return result;
  }

  // ── Serialise ──

  checkpoint() {
    const lines = ['# µJSONL Grammar checkpoint'];
    for (const [id, entry] of this._entries) {
      lines.push(JSON.stringify({
        ...entry,
        weight:   this._weights.get(id),
        gradient: this._gradients.get(id),
      }));
    }
    return lines.join('\n');
  }

  restore(jsonlText) { this._entries.clear(); this._weights.clear(); this._gradients.clear(); this.load(jsonlText); }

  get size()    { return this._entries.size; }
  get entries() { return [...this._entries.values()]; }
  weights()     { return Object.fromEntries(this._weights); }
  gradients()   { return Object.fromEntries(this._gradients); }
}

// ─── Loss functions ───────────────────────────────────────────────────────────

export const Loss = Object.freeze({
  mse: {
    compute:  (p, t) => (p - t) ** 2,
    gradient: (p, t) => 2 * (p - t),
  },
  mae: {
    compute:  (p, t) => Math.abs(p - t),
    gradient: (p, t) => Math.sign(p - t),
  },
  huber: {
    compute:  (p, t, d=1) => Math.abs(p-t) < d ? 0.5*(p-t)**2 : d*(Math.abs(p-t)-0.5*d),
    gradient: (p, t, d=1) => Math.abs(p-t) < d ? (p-t) : d*Math.sign(p-t),
  },
});

// ─── MuJSONLTrainer ───────────────────────────────────────────────────────────

export class MuJSONLTrainer {
  constructor(compiler, loss = Loss.mse, lr = 0.01) {
    this._compiler = compiler;
    this._loss     = loss;
    this._lr       = lr;
    this.history   = [];
  }

  train(examples, epochs = 10, onEpoch = null) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0, hits = 0;

      for (const { input, target } of examples) {
        const { prediction, patternId, numbers } = this._compiler.forward(input);
        if (prediction === null) continue;

        const loss     = this._loss.compute(prediction, target);
        const gradient = this._loss.gradient(prediction, target);
        totalLoss += loss;

        const grad = this._compiler.backward(gradient, patternId, this._lr);
        hits++;

        this.history.push({ epoch, input, prediction, target, loss, gradient: grad, patternId });
      }

      const avgLoss = hits > 0 ? totalLoss / hits : 0;
      onEpoch?.({ epoch: epoch+1, epochs, avgLoss, hits, total: examples.length });
    }
    return this;
  }

  eval(examples) {
    let correct = 0, total = 0, totalLoss = 0;
    for (const { input, target } of examples) {
      const { prediction } = this._compiler.forward(input);
      if (prediction === null) continue;
      totalLoss += this._loss.compute(prediction, target);
      if (Math.round(prediction) === Math.round(target)) correct++;
      total++;
    }
    return { accuracy: total > 0 ? correct/total : 0, avgLoss: total > 0 ? totalLoss/total : 0, total };
  }
}

// ─── HD4600 / SIMD optimizer ──────────────────────────────────────────────────
//
// Groups µJSONL entries by type for optimal dispatch on Intel HD4600:
//   arithmetic → WebAssembly SIMD f32x4 batch
//   linguistic → CPU DFA regex cache
//   tensor     → GPU D3D11 UAV dispatch

export class MuJSONLOptimizer {
  optimize(entries) {
    const groups = { arithmetic: [], linguistic: [], tensor: [], extended: [] };
    for (const e of entries) {
      const t = e.mumodel ?? e['µmodel'] ?? '';
      if (['mu_add','mu_subtract','mu_subtract_abs','mu_multiply','mu_divide'].includes(t))
        groups.arithmetic.push(e);
      else if (t === 'mu_pattern')
        groups.linguistic.push(e);
      else if (['mu_tensor','mu_linalg','mu_geodesic','mu_phase'].includes(t))
        groups.tensor.push(e);
      else
        groups.extended.push(e);
    }
    return {
      groups,
      wasmBatch:   groups.arithmetic.length > 0 ? this._wasmBatch(groups.arithmetic) : null,
      regexDFA:    groups.linguistic.length  > 0 ? this._buildDFA(groups.linguistic)  : null,
      gpuDispatch: groups.tensor.length      > 0 ? this._gpuHint(groups.tensor)       : null,
    };
  }

  _wasmBatch(entries) {
    // SIMD f32x4 WAT stub for arithmetic batch operations
    return `(module
  (memory (export "memory") 1)
  (func (export "batch_${entries[0].mumodel ?? 'op'}") (param $len i32)(param $a i32)(param $b i32)(param $out i32)
    (local $i i32)
    (loop $loop
      (v128.store (i32.add (local.get $out)(local.get $i))
        (f32x4.add
          (v128.load (i32.add (local.get $a)(local.get $i)))
          (v128.load (i32.add (local.get $b)(local.get $i)))))
      (local.set $i (i32.add (local.get $i) (i32.const 16)))
      (br_if $loop (i32.lt_u (local.get $i)(local.get $len))))))`;
  }

  _buildDFA(entries) {
    // Pre-compile all linguistic patterns into a single combined regex
    const combined = entries.map(e => `(?<${e.id.replace(/\W/g,'_')}>${e.pattern})`).join('|');
    try { return new RegExp(combined, 'i'); } catch (_) { return null; }
  }

  _gpuHint(entries) {
    return {
      dispatch: 'D3D11_UAV',
      count:    entries.length,
      ids:      entries.map(e => e.id),
    };
  }
}

// ─── Canonical base µJSONL grammar ────────────────────────────────────────────
//
// Each base µMODEL gets a seed grammar file.
// Weights start at defaults and evolve during inference.

export const BASE_GRAMMAR = `# µJSONL Base Grammar v1.0
# Trainable atomic patterns for base µMODEL routing
# Format: id | µmodel | pattern | weight | phase | lipschitz | device

{"id":"add.give","µmodel":"mu_add","pattern":"gives? (\\\\d+)","weight":1.0,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"add.buy","µmodel":"mu_add","pattern":"buys? (\\\\d+)","weight":0.95,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"add.find","µmodel":"mu_add","pattern":"finds? (\\\\d+)","weight":0.95,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"add.gain","µmodel":"mu_add","pattern":"gains? (\\\\d+)","weight":0.98,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"add.receive","µmodel":"mu_add","pattern":"receives? (\\\\d+)","weight":0.97,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"sub.give_away","µmodel":"mu_subtract","pattern":"gives? away (\\\\d+)","weight":1.0,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"sub.sell","µmodel":"mu_subtract","pattern":"sells? (\\\\d+)","weight":0.95,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"sub.lose","µmodel":"mu_subtract","pattern":"loses? (\\\\d+)","weight":0.95,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"sub.spend","µmodel":"mu_subtract","pattern":"spends? (\\\\d+)","weight":0.97,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"sub.diff","µmodel":"mu_subtract_abs","pattern":"difference between (\\\\d+) and (\\\\d+)","weight":1.0,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"mul.each","µmodel":"mu_multiply","pattern":"each \\\\w+ has (\\\\d+)","weight":0.9,"phase":"Sek","lipschitz":1.5,"device":"cpu"}
{"id":"mul.times","µmodel":"mu_multiply","pattern":"(\\\\d+) times (\\\\d+)","weight":1.0,"phase":"Sek","lipschitz":1.5,"device":"cpu"}
{"id":"mul.groups","µmodel":"mu_multiply","pattern":"(\\\\d+) groups? of (\\\\d+)","weight":0.95,"phase":"Sek","lipschitz":1.5,"device":"cpu"}
{"id":"div.split","µmodel":"mu_divide","pattern":"split (\\\\d+) (?:equally )?(?:among|between) (\\\\d+)","weight":0.9,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"div.per","µmodel":"mu_divide","pattern":"(\\\\d+) per (\\\\d+)","weight":0.85,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"chain.sell_buy","µmodel":"mu_chain","pattern":"has (\\\\d+).*?sells? (\\\\d+).*?buys? (\\\\d+)","steps":["sub.sell","add.buy"],"weight":1.0,"phase":"Sek","lipschitz":2.0,"device":"cpu"}
{"id":"chain.give_buy","µmodel":"mu_chain","pattern":"has (\\\\d+).*?gives? away (\\\\d+).*?receives? (\\\\d+)","steps":["sub.give_away","add.receive"],"weight":1.0,"phase":"Sek","lipschitz":2.0,"device":"cpu"}
{"id":"fib.of","µmodel":"mu_fibonacci","pattern":"fibonacci(?:of| number)? (\\\\d+)","weight":0.9,"phase":"Sek","lipschitz":1.0,"device":"cpu"}
{"id":"fib.fold","µmodel":"mu_fibonacci","pattern":"fold by phi over (\\\\d+)","weight":0.9,"phase":"Sek","lipschitz":1.618,"device":"cpu"}
{"id":"pi.times","µmodel":"mu_pi","pattern":"(\\\\d+) pi radians?","weight":0.9,"phase":"Wo","lipschitz":3.14159,"device":"cpu"}
{"id":"mayan.encode","µmodel":"mu_mayan","pattern":"mayan(?:encode)? (\\\\d+)","weight":0.9,"phase":"Sek","lipschitz":20.0,"device":"cpu"}
{"id":"dot.product","µmodel":"mu_linalg","pattern":"dot product of (\\\\d+)","weight":0.85,"phase":"Sek","lipschitz":1.0,"device":"gpu"}
{"id":"tensor.outer","µmodel":"mu_tensor","pattern":"outer product of (\\\\d+)","weight":0.85,"phase":"Sek","lipschitz":1.0,"device":"gpu"}
{"id":"geo.dist","µmodel":"mu_geodesic","pattern":"geodesic distance of (\\\\d+\\\\.?\\\\d*)","weight":0.85,"phase":"Wo","lipschitz":1.0,"device":"gpu"}
`;

// ─── MuJSONLGrammar facade ────────────────────────────────────────────────────
//
// One-stop: load base grammar, expose forward/backward, integrate with µMODEL registry.

export class MuJSONLGrammar {
  constructor(jsonlText = BASE_GRAMMAR, opts = {}) {
    this.compiler  = new MuJSONLCompiler();
    this.trainer   = new MuJSONLTrainer(this.compiler, opts.loss ?? Loss.mse, opts.lr ?? 0.01);
    this.optimizer = new MuJSONLOptimizer();
    this.compiler.load(jsonlText);
    this._opt = this.optimizer.optimize(this.compiler.entries);
  }

  forward(input)             { return this.compiler.forward(input); }
  backward(err, id, lr)      { return this.compiler.backward(err, id, lr); }
  train(examples, epochs, cb){ return this.trainer.train(examples, epochs, cb); }
  eval(examples)             { return this.trainer.eval(examples); }
  checkpoint()               { return this.compiler.checkpoint(); }
  restore(text)              { this.compiler.restore(text); this._opt = this.optimizer.optimize(this.compiler.entries); }

  /** Describe this grammar as µMODEL CDATA semantic grams. */
  toCdataGrams() {
    return this.compiler.entries.map(e =>
      `mujsonl.${e.id.replace(/\W/g,'_')}`
    ).join(' ');
  }

  get size()   { return this.compiler.size; }
  get groups() { return this._opt.groups; }
}
