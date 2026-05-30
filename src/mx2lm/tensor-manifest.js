// tensor-manifest.js — MX2LM Tensor Manifest + Registry + Graph
//
// The SafeTensors progression (Phase 1 → 5):
//
//   Phase 1  SafeTensors                    — raw storage
//   Phase 2  SafeTensors + Fold Runtime     — tensors get semantic meaning
//   Phase 3  SafeTensors + Geodesic Runtime — tensors become graph nodes
//   Phase 4  SVG-Tensor Runtime             — tensors get spatial coordinates
//   Phase 5  SCXQ2 Native Container         — full MX2LM object
//
// This module implements Phases 1-3:
//   TensorManifest   — attach fold semantics to safetensor names
//   TensorRegistry   — discoverable, versioned tensor inventory
//   TensorGraph      — G = (V, E) where vertices are tensors
//   FoldObject       — bigram-as-fold: Fold = {source, target, weight, memory}
//
// The key shift:
//   Traditional: P(token_{n+1}) — next token prediction
//   MX2LM:       P(F_{t+1})    — next fold/object prediction
//
// Formal: Omega = (T, G, P, M, S, R)
//   T = SafeTensors, G = geodesic graph, P = phase tensor,
//   M = micronauts, S = SVG tensor space, R = runtime state

// ─── Tensor Manifest (Phase 2) ────────────────────────────────────────────────
// Attach fold semantics to raw safetensor names.

export class TensorManifest {
  constructor(id, format = "safetensors") {
    this["@id"]     = id;
    this["@format"] = format;
    this._tensors   = new Map();   // name → {fold, role, shape, ...}
  }

  // Register a tensor with fold + role metadata
  register(name, opts = {}) {
    this._tensors.set(name, {
      "@tensor": name,
      "@fold":   opts.fold  ?? "tensor",
      "@role":   opts.role  ?? "weight",
      "@shape":  opts.shape ?? null,
      "@lane":   opts.lane  ?? 5,           // default SCXQ2 TENSOR lane
      "@dtype":  opts.dtype ?? "float32",
      ...opts,
    });
    return this;
  }

  // Auto-register a safetensors weight dict with fold inference by name
  ingestWeightNames(names) {
    for (const name of names) {
      const fold = inferFold(name);
      const role = inferRole(name);
      this.register(name, { fold, role });
    }
    return this;
  }

  get(name)     { return this._tensors.get(name) ?? null; }
  has(name)     { return this._tensors.has(name); }
  get size()    { return this._tensors.size; }
  get names()   { return [...this._tensors.keys()]; }
  get entries() { return [...this._tensors.values()]; }

  // Group by fold
  byFold() {
    const groups = {};
    for (const [, entry] of this._tensors) {
      const f = entry["@fold"];
      if (!groups[f]) groups[f] = [];
      groups[f].push(entry);
    }
    return groups;
  }

  toJSON() {
    return {
      "@id": this["@id"],
      "@format": this["@format"],
      "@tensors": Object.fromEntries(this._tensors),
    };
  }
}

// Infer fold from tensor name conventions
function inferFold(name) {
  if (/embed/.test(name))         return "semantic";
  if (/attn|attention/.test(name)) return "attention";
  if (/mlp|ffn|feed/.test(name))  return "compute";
  if (/norm|ln/.test(name))       return "normalization";
  if (/head|lm_head/.test(name))  return "output";
  if (/memory/.test(name))        return "memory";
  return "tensor";
}

function inferRole(name) {
  if (/weight$|\.w$/.test(name))  return "weight";
  if (/bias$|\.b$/.test(name))    return "bias";
  if (/\.q$|_q_/.test(name))      return "query";
  if (/\.k$|_k_/.test(name))      return "key";
  if (/\.v$|_v_/.test(name))      return "value";
  if (/proj|out/.test(name))      return "projection";
  if (/gate/.test(name))          return "gate";
  if (/embed/.test(name))         return "vocabulary";
  return "parameter";
}

// ─── Tensor Registry (Phase 2 — discoverable inventory) ──────────────────────

export class TensorRegistry {
  constructor() {
    this._entries = new Map();   // id → entry
    this._by_fold = new Map();   // fold → Set<id>
    this._by_type = new Map();   // type → Set<id>
  }

  register(id, opts = {}) {
    const entry = {
      "@id":    id,
      "@type":  opts.type  ?? "tensor",
      "@shape": opts.shape ?? null,
      "@fold":  opts.fold  ?? "tensor",
      ...opts,
    };
    this._entries.set(id, entry);

    // Index by fold
    if (!this._by_fold.has(entry["@fold"]))
      this._by_fold.set(entry["@fold"], new Set());
    this._by_fold.get(entry["@fold"]).add(id);

    // Index by type
    if (!this._by_type.has(entry["@type"]))
      this._by_type.set(entry["@type"], new Set());
    this._by_type.get(entry["@type"]).add(id);

    return entry;
  }

  get(id)       { return this._entries.get(id) ?? null; }
  has(id)       { return this._entries.has(id); }
  get size()    { return this._entries.size; }

  byFold(fold)  { return [...(this._by_fold.get(fold) ?? [])].map(id => this._entries.get(id)); }
  byType(type)  { return [...(this._by_type.get(type) ?? [])].map(id => this._entries.get(id)); }

  // Ingest from TensorManifest
  ingestManifest(manifest) {
    for (const entry of manifest.entries)
      this.register(entry["@tensor"], { ...entry, type: "tensor" });
    return this;
  }

  toJSON() {
    return { "@registry": [...this._entries.values()] };
  }
}

// ─── Tensor Graph (Phase 3 — G = (V, E) over tensors) ────────────────────────
// Tensors become graph nodes. Edges are geodesic relationships.
// Attention becomes graph traversal: A = QK^T → G.neighbours(query_node)

export class TensorGraph {
  constructor() {
    this._nodes = new Map();   // id → node data
    this._edges = new Map();   // id → [{target, weight, relation}]
  }

  addNode(id, data = {}) {
    this._nodes.set(id, { "@id": id, ...data });
    if (!this._edges.has(id)) this._edges.set(id, []);
    return this;
  }

  addEdge(from, to, weight = 1.0, relation = "geodesic") {
    if (!this._edges.has(from)) this._edges.set(from, []);
    this._edges.get(from).push({ target: to, weight, relation });
    return this;
  }

  neighbours(id)  { return this._edges.get(id) ?? []; }
  node(id)        { return this._nodes.get(id) ?? null; }
  get nodeCount() { return this._nodes.size; }
  get edgeCount() { return [...this._edges.values()].reduce((s, a) => s + a.length, 0); }

  // Build from TensorRegistry: connect tensors in same fold
  buildFromRegistry(registry, intraFoldWeight = 0.5, interFoldWeight = 2.0) {
    for (const entry of registry._entries.values()) {
      this.addNode(entry["@id"], entry);
    }
    const byFold = registry._by_fold;
    for (const [, ids] of byFold) {
      const arr = [...ids];
      for (let i = 0; i < arr.length; i++)
        for (let j = i + 1; j < arr.length; j++) {
          this.addEdge(arr[i], arr[j], intraFoldWeight, "intra_fold");
          this.addEdge(arr[j], arr[i], intraFoldWeight, "intra_fold");
        }
    }
    return this;
  }

  // Geodesic attention over tensor graph
  // h_q = Σ_{k in neighbours(query)} exp(-d(q,k)) * registry[k].value
  geodesicAttention(queryId, temperature = 1.0) {
    const nbrs = this.neighbours(queryId);
    if (!nbrs.length) return null;
    const scores = nbrs.map(({ target, weight }) => ({
      id: target, score: Math.exp(-weight / temperature)
    }));
    const sum = scores.reduce((s, r) => s + r.score, 0);
    return scores.map(r => ({ ...r, attn: r.score / (sum || 1) }));
  }

  toJSON() {
    return {
      "@graph": Object.fromEntries(
        [...this._edges.entries()].map(([id, edges]) => [id, edges])
      )
    };
  }
}

// ─── FoldObject (bigram-as-fold) ──────────────────────────────────────────────
// Replaces statistical bigrams with persistent identity objects.
//
// Traditional bigram:  P(world | hello) = 0.83
// Fold object:         Fold_001 = {source:"hello", target:"world", weight:0.83, uses:1042}
//
// Then: Token → Bigram → Fold → Fold Graph → Object → Object Server
// Inference shifts from P(token_{n+1}) to P(F_{t+1}) = next fold prediction.

export class FoldObject {
  constructor(source, target, weight = 1.0) {
    this.id      = `fold_${FoldObject._nextId++}`;
    this.source  = source;   // A
    this.target  = target;   // B
    this.weight  = weight;   // W — relationship strength
    this.uses    = 0;        // usage counter
    this.memory  = [];       // episodic context list
    this.graph   = [];       // connected fold ids
    this.created = Date.now();
  }

  // Record a usage event
  use(context = null) {
    this.uses++;
    this.weight = this.weight * 0.99 + 0.01;  // slight reinforcement
    if (context) this.memory.push({ context, ts: Date.now() });
    if (this.memory.length > 64) this.memory.shift();
    return this;
  }

  // Connect to another fold (builds the fold chain: F_A → F_B)
  connect(foldId, relation = "sequence") {
    if (!this.graph.find(e => e.id === foldId))
      this.graph.push({ id: foldId, relation });
    return this;
  }

  toJSON() {
    return {
      "@id":     this.id,
      "@source": this.source,
      "@target": this.target,
      "@weight": this.weight,
      "@uses":   this.uses,
      "@memory": this.memory.slice(-8),
      "@graph":  this.graph,
    };
  }

  static _nextId = 0;
}

// ─── FoldStore — maps (source, target) pairs → FoldObjects ───────────────────

export class FoldStore {
  constructor() {
    this._folds = new Map();   // "source::target" → FoldObject
    this._by_source = new Map();
  }

  getOrCreate(source, target, weight = 1.0) {
    const key = `${source}::${target}`;
    if (!this._folds.has(key)) {
      const f = new FoldObject(source, target, weight);
      this._folds.set(key, f);
      if (!this._by_source.has(source)) this._by_source.set(source, []);
      this._by_source.get(source).push(f);
    }
    return this._folds.get(key);
  }

  // Predict next fold given source (highest weight + uses)
  predictNext(source) {
    const candidates = this._by_source.get(source) ?? [];
    if (!candidates.length) return null;
    return candidates.reduce((best, f) =>
      (f.weight * Math.log1p(f.uses)) > (best.weight * Math.log1p(best.uses)) ? f : best
    );
  }

  get size() { return this._folds.size; }
  folds()    { return [...this._folds.values()]; }
}

// ─── MX2LM Object (Phase 5 goal) ─────────────────────────────────────────────
// Omega = (T, G, P, M, S, R)
// Not just model weights — a complete cognitive object.

export class MX2LMObject {
  constructor(id) {
    this.id       = id;
    this.manifest = new TensorManifest(id);
    this.registry = new TensorRegistry();
    this.graph    = new TensorGraph();
    this.folds    = new FoldStore();
    this.metadata = {
      "@id":      id,
      "@version": "3.5.0",
      "@system":  "K'UHUL WebX-3D",
      "@format":  "mx2lm-object",
      created:    new Date().toISOString(),
    };
  }

  // Phase 1→2: load safetensors names and attach fold semantics
  ingestWeights(weightNames) {
    this.manifest.ingestWeightNames(weightNames);
    this.registry.ingestManifest(this.manifest);
    return this;
  }

  // Phase 2→3: build geodesic graph over tensor registry
  buildGraph() {
    this.graph.buildFromRegistry(this.registry);
    return this;
  }

  // Describe the full object structure
  summary() {
    return {
      id:       this.id,
      tensors:  this.manifest.size,
      nodes:    this.graph.nodeCount,
      edges:    this.graph.edgeCount,
      folds:    this.folds.size,
      foldGroups: Object.fromEntries(
        Object.entries(this.manifest.byFold()).map(([f, ts]) => [f, ts.length])
      ),
    };
  }

  toJSON() {
    return {
      ...this.metadata,
      "@manifest": this.manifest.toJSON(),
      "@registry": this.registry.toJSON(),
      "@graph":    this.graph.toJSON(),
    };
  }
}
