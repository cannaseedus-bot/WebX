// semantic-resolution-engine.js — Entropy-driven semantic field resolution
//
// Intelligence as semantic thermodynamics:
//   Questions create field gradients.
//   Resolution = lawful movement toward minimum entropy.
//
// The engine doesn't "compute answers" — it resolves fields toward
// stable equilibrium using the strategy appropriate to the entropy level.
//
// Entropy thresholds → resolution strategies:
//   < 0.2  DIRECT    — answer exists in local semantic field
//   < 0.5  SHARD     — query semantic memory / local shards
//   < 0.7  INFERENCE — tensor completion from latent space
//   < 0.9  WEB       — field expansion via web traversal
//   ≥ 0.9  API       — external acquisition
//
// K'UHUL phase topology (invariant across all strategies):
//   Pop → field emergence
//   Wo  → intent/context stabilization
//   Sek → iterative resolution (loops until stable)
//   Ch'en → project answer
//   Xul → seal if stable
//
// XQuery + MathML as first-class tools:
//   XQuery queries KXML graph nodes (native XML query language)
//   MathML represents symbolic expressions in the semantic field
//   Both are XML — KXML compatibility is structural, not bolted on.

// ─── Entropy thresholds ───────────────────────────────────────────────────────

export const ENTROPY = Object.freeze({
  LOW:      0.2,   // direct answer
  MEDIUM:   0.5,   // shard retrieval
  HIGH:     0.7,   // tensor inference
  CRITICAL: 0.9,   // web traversal
});

// ─── Semantic field ───────────────────────────────────────────────────────────

export function createField(question, context = {}) {
  return {
    rawQuery:      question,
    value:         null,
    confidence:    0,
    bindings:      {},
    intent:        null,
    variables:     [],
    causalHistory: [],
    annotations:   new Map(),
    // XQuery/MathML surfaces
    xquery:        null,   // XQuery expression if field maps to KXML query
    mathml:        null,   // MathML expression if field is mathematical
    ...context,
  };
}

// ─── Entropy measurement ─────────────────────────────────────────────────────

export function computeEntropy(field) {
  return Math.min(1.0, (
    (field.bindings && Object.keys(field.bindings).length ? 0 : 0.25) +
    (field.intent?.confidence ? (1 - field.intent.confidence) * 0.2 : 0.2) +
    (field.causalHistory?.length ? 0 : 0.25) +
    (field.variables?.filter(v => !v.resolved).length ?? 0) * 0.1 +
    (field.targetDistance ?? 0) * 0.2
  ));
}

export function computeConfidence(field) {
  if (field.confidence != null) return field.confidence;
  return field.causalHistory?.length ? 0.7 : 0.3;
}

export function isStable(field) {
  return computeEntropy(field) < ENTROPY.LOW && computeConfidence(field) > 0.9;
}

// ─── Strategy selection ───────────────────────────────────────────────────────

export function selectStrategy(entropy) {
  if (entropy < ENTROPY.LOW)      return { name: 'direct',    cost: 0.0 };
  if (entropy < ENTROPY.MEDIUM)   return { name: 'shard',     cost: 0.3 };
  if (entropy < ENTROPY.HIGH)     return { name: 'inference', cost: 0.5 };
  if (entropy < ENTROPY.CRITICAL) return { name: 'web',       cost: 0.8 };
  return                                 { name: 'api',       cost: 1.0 };
}

// ─── XQuery resolver ─────────────────────────────────────────────────────────
// XQuery is the natural query language for KXML graphs — both are XML.
// When a field maps to a KXML graph query, use XQuery to resolve it.

export class XQueryResolver {
  constructor(kxmlDocument = null) {
    this._doc = kxmlDocument;
  }

  // Build an XQuery expression from a semantic field
  buildQuery(field) {
    if (field.xquery) return field.xquery;
    // Auto-generate from bindings
    const { phase, domain, id } = field.bindings;
    if (id)     return `//node[@id="${id}"]`;
    if (phase)  return `//node[@phase="${phase}"]`;
    if (domain) return `//node[@domain="${domain}"]`;
    return `//node`;
  }

  // Simulate XQuery execution (browser: no native XQuery, use XPath instead)
  resolve(field) {
    if (!this._doc) {
      // Return synthetic result for demo
      const query = this.buildQuery(field);
      return {
        query,
        nodes: [`<node id="sim" phase="Sek" domain="compute"/>`],
        confidence: 0.75,
      };
    }
    // Real XPath/XQuery via browser DOMParser
    const expr = this.buildQuery(field).replace(/\/\//g, './/');;
    try {
      const result = this._doc.evaluate(expr, this._doc, null,
        XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
      const nodes = [];
      for (let i = 0; i < result.snapshotLength; i++) {
        nodes.push(new XMLSerializer().serializeToString(result.snapshotItem(i)));
      }
      return { query: expr, nodes, confidence: nodes.length > 0 ? 0.9 : 0.3 };
    } catch (e) {
      return { query: expr, nodes: [], confidence: 0.2, error: e.message };
    }
  }
}

// ─── MathML resolver ─────────────────────────────────────────────────────────
// MathML expresses the mathematical meaning of a semantic field.
// When a field contains arithmetic/symbolic structure, MathML represents it.

export class MathMLResolver {
  // Parse a math expression into MathML content
  static toMathML(expression) {
    // Simple cases — production would use a full parser
    if (/^\d+$/.test(expression.trim())) {
      return `<cn>${expression.trim()}</cn>`;
    }
    if (/^\w+$/.test(expression.trim())) {
      return `<ci>${expression.trim()}</ci>`;
    }
    const addMatch = expression.match(/^(.+)\s*\+\s*(.+)$/);
    if (addMatch) return `<apply><plus/>${this.toMathML(addMatch[1])}${this.toMathML(addMatch[2])}</apply>`;
    const subMatch = expression.match(/^(.+)\s*-\s*(.+)$/);
    if (subMatch) return `<apply><minus/>${this.toMathML(subMatch[1])}${this.toMathML(subMatch[2])}</apply>`;
    const mulMatch = expression.match(/^(.+)\s*\*\s*(.+)$/);
    if (mulMatch) return `<apply><times/>${this.toMathML(mulMatch[1])}${this.toMathML(mulMatch[2])}</apply>`;
    return `<ci>${expression.trim()}</ci>`;
  }

  // Evaluate a MathML expression
  static eval(mathml, bindings = {}) {
    // Extract <cn> values and <ci> identifiers, compute
    const nums   = [...mathml.matchAll(/<cn>([^<]+)<\/cn>/g)].map(m => parseFloat(m[1]));
    const vars   = [...mathml.matchAll(/<ci>([^<]+)<\/ci>/g)].map(m => bindings[m[1]] ?? 0);
    const all    = [...nums, ...vars];
    if (mathml.includes('<plus/>'))  return all.reduce((a, b) => a + b, 0);
    if (mathml.includes('<minus/>')) return all[0] - (all[1] ?? 0);
    if (mathml.includes('<times/>')) return all.reduce((a, b) => a * b, 1);
    if (mathml.includes('<divide/>'))return all[0] / (all[1] ?? 1);
    return all[0] ?? 0;
  }

  resolve(field) {
    if (!field.mathml && field.rawQuery) {
      // Try to extract math from the question
      const nums = [...field.rawQuery.matchAll(/\d+\.?\d*/g)].map(m => parseFloat(m[0]));
      if (nums.length >= 2) {
        const isSubtract = /subtract|left|remain|away|fewer|minus/i.test(field.rawQuery);
        const isMultiply = /each|times|multiply|product/i.test(field.rawQuery);
        if (isSubtract) field.mathml = `<apply><minus/><cn>${nums[0]}</cn><cn>${nums[1]}</cn></apply>`;
        else if (isMultiply) field.mathml = `<apply><times/><cn>${nums[0]}</cn><cn>${nums[1]}</cn></apply>`;
        else field.mathml = `<apply><plus/><cn>${nums[0]}</cn><cn>${nums[1]}</cn></apply>`;
      }
    }
    if (!field.mathml) return null;
    const value = MathMLResolver.eval(field.mathml, field.bindings);
    return { value, mathml: field.mathml, confidence: 0.95 };
  }
}

// ─── SemanticResolutionEngine ────────────────────────────────────────────────

export class SemanticResolutionEngine {
  constructor(opts = {}) {
    this._xquery  = new XQueryResolver(opts.kxmlDocument ?? null);
    this._mathml  = new MathMLResolver();
    this._log     = [];
    this.maxIterations = opts.maxIterations ?? 10;
  }

  _trace(phase, msg) { this._log.push({ phase, msg, ts: Date.now() }); }
  get log() { return [...this._log]; }

  // Classify intent from raw question
  _intent(question) {
    const q = question.toLowerCase();
    if (/how many|how much|what is \d|calculate|compute/i.test(q))
      return { type: 'arithmetic', confidence: 0.9 };
    if (/what is|who is|where is|when/i.test(q))
      return { type: 'factual',    confidence: 0.8 };
    if (/how do|how to|explain|describe/i.test(q))
      return { type: 'procedural', confidence: 0.75 };
    if (/kxml|kuhul|micronaut|supernaut|xcfe|kml/i.test(q))
      return { type: 'kuhul',      confidence: 0.95 };
    return { type: 'unknown', confidence: 0.4 };
  }

  // Extract numeric/entity bindings from question
  _bindings(question) {
    const nums    = [...question.matchAll(/\d+\.?\d*/g)].map(m => parseFloat(m[0]));
    const give    = /gives? away (\d+)/i.exec(question);
    const has     = /has (\d+)/i.exec(question);
    const result  = {};
    if (nums.length > 0) result.values = nums;
    if (has)    result.initial  = parseFloat(has[1]);
    if (give)   result.removed  = parseFloat(give[1]);
    if (result.initial != null && result.removed != null)
      result.expected = result.initial - result.removed;
    return result;
  }

  async resolve(question, context = {}) {
    this._log = [];
    let field = createField(question, context);

    // Pop — question enters field
    this._trace("Pop", `Field created: "${question.substring(0,60)}"`);

    // Wo — stabilize intent + bindings
    field.intent   = this._intent(question);
    field.bindings = { ...field.bindings, ...this._bindings(question) };
    this._trace("Wo", `Intent: ${field.intent.type} (conf=${field.intent.confidence})`);

    // Sek — iterative resolution
    let iters = 0;
    while (iters < this.maxIterations && !isStable(field)) {
      const entropy  = computeEntropy(field);
      const strategy = selectStrategy(entropy);
      this._trace("Sek", `entropy=${entropy.toFixed(2)} strategy=${strategy.name}`);

      field = await this._applyStrategy(field, strategy);
      iters++;
    }

    // Ch'en — project result
    this._trace("Ch'en", `Projecting. confidence=${computeConfidence(field).toFixed(2)}`);
    const projection = this._project(field);

    // Xul — seal if stable
    if (isStable(field)) this._trace("Xul", "Field sealed");

    return { ...projection, phases: this._log.map(l => l.phase), iterations: iters };
  }

  async _applyStrategy(field, strategy) {
    switch (strategy.name) {
      case 'direct': {
        // Bindings have what we need — compute directly
        const b = field.bindings;
        if (b.expected != null) {
          field.value      = b.expected;
          field.confidence = 0.99;
        } else if (b.values?.length >= 2) {
          const intent = field.intent.type;
          if (/arith/i.test(intent)) field.value = b.values.reduce((a,c) => a-c, b.values[0]+b.values[0]) || b.values[0]-b.values[1];
          field.confidence = 0.9;
        }
        break;
      }
      case 'shard': {
        // Try MathML resolver
        const mr = this._mathml.resolve(field);
        if (mr) { field.value = mr.value; field.confidence = mr.confidence; }
        else    { field.value = `[shard] ${field.rawQuery}`; field.confidence = 0.65; }
        break;
      }
      case 'inference': {
        // Try XQuery on KXML graph, then MathML fallback
        if (/kxml|kuhul|kml|xcfe/i.test(field.rawQuery)) {
          const xr = this._xquery.resolve(field);
          field.value = xr.nodes[0] ?? `[kxml-inference] ${field.rawQuery}`;
          field.confidence = xr.confidence;
        } else {
          const mr = this._mathml.resolve(field);
          field.value = mr?.value ?? `[inference] ${field.rawQuery}`;
          field.confidence = mr?.confidence ?? 0.55;
        }
        break;
      }
      case 'web':
        field.value = `[web-expansion] ${field.rawQuery}`;
        field.confidence = 0.6;
        break;
      case 'api':
        field.value = `[api] ${field.rawQuery}`;
        field.confidence = 0.5;
        break;
    }
    field.causalHistory.push({ strategy: strategy.name, entropy: computeEntropy(field) });
    return field;
  }

  _project(field) {
    const entropy    = computeEntropy(field);
    const confidence = computeConfidence(field);
    const strategy   = field.causalHistory?.slice(-1)[0]?.strategy ?? 'unknown';
    return {
      value:      field.value,
      confidence,
      entropy,
      strategy,
      bindings:   field.bindings,
      mathml:     field.mathml,
      xquery:     field.xquery,
      natural:    this._natural(field),
    };
  }

  _natural(field) {
    const v = field.value;
    if (v == null) return "No answer found.";
    if (typeof v === 'number') return `The answer is ${v}.`;
    if (typeof v === 'string' && v.startsWith('<')) return `KXML node: ${v.substring(0,80)}`;
    return String(v);
  }
}
