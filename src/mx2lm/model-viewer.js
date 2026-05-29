// K'UHUL π SVG-3D Model Viewer v1.0 — pure data builders
// Transforms tokenizer.json + checkpoint.json → 5-shell render data.
// No DOM, no IO — pure math + data transforms. Browser rendering is in examples/.
//
// Shell topology:
//   orbital_halo  → vocabulary cortex (token freq/embedding rings)
//   stack_grid    → weight matrix (layers, heads, params)
//   tunnel_rail   → inference stream (n-gram packets, PMI lanes)
//   fractal_tree  → language tree (BPE merge hierarchy)
//   hud_ring      → executive cortex (runtime health, shard load)

const PI = Math.PI;

// ─── Math helpers (K'UHUL π) ──────────────────────────────────────────────────

export function π_vecNorm(v) {
  let sumSq = 0;
  for (const x of v) sumSq += x * x;
  return Math.sqrt(sumSq);
}

export function π_softmax(xs) {
  const maxX = Math.max(...xs);
  const exps = xs.map(x => Math.exp(x - maxX));
  const sumE = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumE);
}

export function π_entropy(probs) {
  let H = 0;
  for (const p of probs) {
    if (p > 0) H -= p * Math.log2(p);
  }
  return H;
}

export function π_ngramProb(count, total) {
  if (total <= 0) return 0;
  return count / total;
}

export function π_pmi(p_xy, p_x, p_y) {
  if (p_xy <= 0 || p_x <= 0 || p_y <= 0) return 0;
  return Math.log2(p_xy / (p_x * p_y));
}

export function π_angleFromVec(v) {
  const x = v[0] ?? 0;
  const y = v[1] ?? 0;
  let θ = Math.atan2(y, x);
  if (θ < 0) θ += 2 * PI;
  return θ;
}

export function π_clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

export function π_classToColor(clazz) {
  switch (clazz) {
    case 'special': return '#ff6b6b';
    case 'punct':   return '#feca57';
    case 'number':  return '#54a0ff';
    case 'api':     return '#5f27cd';
    default:        return '#1dd1a1';
  }
}

// ─── ISO coords helper ────────────────────────────────────────────────────────

export function isoCoords(col, row, layer, height) {
  const cell = 32;
  const x2d  = (col - row) * (cell * 0.866);
  const y2d  = (col + row) * (cell * 0.5) - (layer * 22) - height;
  const points = `-10,0 0,${-height} 10,0 0,${height}`;
  return { x: x2d, y: y2d, points };
}

// ─── Shell 1: Orbital Halo — Vocabulary cortex ───────────────────────────────
// Splits vocab into 3 frequency rings. Each token → {theta, phi, size, color, prob}.

export function buildOrbitalHaloData(tok) {
  const vocab  = tok.vocab  || [];
  const total  = tok.ngrams?.unigram_total || 1;
  const embeds = tok.embeddings?.vectors   || {};

  const freqs = vocab.map(t => t.freq).sort((a, b) => b - a);
  const fInner = freqs[Math.floor(0.05 * freqs.length)] ?? 0;
  const fMid   = freqs[Math.floor(0.35 * freqs.length)] ?? 0;

  const inner = [], mid = [], outer = [];

  for (const t of vocab) {
    const freq = t.freq || 0;
    const p    = π_ngramProb(freq, total);
    const v    = embeds[String(t.id)] || [0, 0];
    const θ    = π_angleFromVec(v);
    const elev = (1 - p) * 60;
    const size = Math.log10(freq + 10) * 2;

    const glyph = {
      id:    t.id,
      token: t.token,
      theta: θ,
      phi:   elev,
      color: π_classToColor(t.class || 'word'),
      size,
      prob:  p,
    };

    if (freq >= fInner)      inner.push(glyph);
    else if (freq >= fMid)   mid.push(glyph);
    else                     outer.push(glyph);
  }

  return { halo: { inner, mid, outer } };
}

// ─── Shell 2: Stack Grid — Weight matrix / layers ────────────────────────────
// One isometric block per layer; height = log(params), tone = normalized entropy.

export function buildStackGridData(ckpt) {
  const layers = ckpt.layers || [];
  const cols   = 4;
  const rows   = 3;
  const blocks = [];

  for (let i = 0; i < layers.length; i++) {
    const L      = layers[i];
    const col    = i % cols;
    const row    = Math.floor(i / cols) % rows;
    const layerZ = Math.floor(i / (cols * rows));
    const height = Math.log10((L.params || 1) + 10) * 6;
    const tone   = π_clamp((L.entropy || 0) / 8, 0, 1);
    const iso    = isoCoords(col, row, layerZ, height);

    blocks.push({
      col, row,
      layer:   layerZ,
      height,
      tone,
      glyphId: `layer_${L.id ?? i}`,
      iso,
    });
  }

  return { blocks };
}

// ─── Shell 3: Tunnel Rail — Inference stream / PMI lanes ─────────────────────
// Bigrams sorted into 3 lanes by PMI value: top(PMI>2), left(PMI>0), right(PMI≤0).

export function buildTunnelStreamData(tok) {
  const ngrams  = tok.ngrams  || {};
  const bigram  = ngrams.bigram || {};
  const total   = ngrams.unigram_total || 1;
  const vocab   = tok.vocab   || [];

  const uniCount = {};
  for (const t of vocab) uniCount[t.token] = t.freq || 0;

  const left = [], right = [], top = [];
  let index = 0;

  for (const [key, count] of Object.entries(bigram)) {
    const parts = key.split(' ');
    if (parts.length !== 2) continue;
    const [x, y] = parts;

    const p_xy = π_ngramProb(count, total);
    const p_x  = π_ngramProb(uniCount[x] ?? 1, total);
    const p_y  = π_ngramProb(uniCount[y] ?? 1, total);
    const pmi  = π_pmi(p_xy, p_x, p_y);
    const depth = index * 4;

    const packet = { glyph: `${x}→${y}`, depth, energy: Math.abs(pmi) };

    if (pmi > 2)       top.push({   ...packet, lane:  0 });
    else if (pmi > 0)  left.push({  ...packet, lane: -1 });
    else               right.push({ ...packet, lane:  1 });

    if (++index > 256) break;
  }

  return { leftStream: left, rightStream: right, topStream: top };
}

// ─── Shell 4: Fractal Tree — BPE merge hierarchy ─────────────────────────────
// Merges become branches; root = common ancestor; depth = merge rank group.

export function buildFractalTreeData(tok) {
  const merges = tok.merges || [];
  const nodes  = {};
  const ROOT   = 'ROOT';

  nodes[ROOT] = { id: ROOT, parent: null, depth: 0, weight: 0, glyphId: 'root', children: [] };

  for (let i = 0; i < merges.length; i++) {
    const m   = merges[i];
    const id  = `m${i}`;
    const depth = Math.floor(i / 64);

    nodes[id] = {
      id,
      parent:  ROOT,
      depth,
      weight:  Math.log10((m.count || 1) + 10),
      glyphId: `${m.left}+${m.right}`,
      children: [],
    };
    nodes[ROOT].children.push(id);
  }

  return { root: ROOT, nodes };
}

// ─── Shell 5: HUD Ring — Runtime health / executive cortex ───────────────────

export function buildHudRingData(ckpt, _tok) {
  const stats = ckpt.stats || {};

  const shards = [
    { name: 'CPU',  status: 'ok',   load: 0.4,  glyphId: 'CPU'  },
    { name: 'GPU',  status: 'warn', load: 0.7,  glyphId: 'GPU'  },
    { name: 'TPU',  status: 'ok',   load: 0.5,  glyphId: 'TPU'  },
    { name: 'RLHF', status: 'ok',   load: 0.3,  glyphId: 'RLHF' },
  ];

  const runtimes = [
    { name: 'ASXR',     status: 'ok',   load: 0.5, glyphId: 'ASXR'     },
    { name: 'ASXR-GPU', status: 'ok',   load: 0.6, glyphId: 'ASXR-GPU' },
    { name: 'TPU-OS',   status: 'warn', load: 0.7, glyphId: 'TPU-OS'   },
    { name: 'BROWSER',  status: 'ok',   load: 0.2, glyphId: 'DOM'       },
  ];

  const core = [
    { name: 'XJSON',  status: 'ok', load: 0.3,  glyphId: 'XJSON'  },
    { name: "K'UHUL", status: 'ok', load: 0.4,  glyphId: 'KUHUL'  },
    { name: 'SCXQ2',  status: 'ok', load: 0.2,  glyphId: 'SCXQ2'  },
    { name: 'KLH',    status: 'ok', load: 0.35, glyphId: 'KLH'    },
  ];

  const center = {
    name:    'MX2LM',
    status:  'ok',
    load:    π_clamp((stats.avg_entropy || 0) / 8, 0, 1),
    glyphId: 'MX2LM',
  };

  return { shards, runtimes, core, center };
}

// ─── Master entrypoint ────────────────────────────────────────────────────────
// Given parsed tokenizer + checkpoint objects, returns all 5 shell data sets.

export function buildModelShells(tok, ckpt) {
  return {
    orbitalHalo: buildOrbitalHaloData(tok),
    stackGrid:   buildStackGridData(ckpt),
    tunnelRail:  buildTunnelStreamData(tok),
    fractalTree: buildFractalTreeData(tok),
    hudRing:     buildHudRingData(ckpt, tok),
  };
}

// ─── Shell 6: Memory Constellation — episodic memory ─────────────────────────
// memory node = { id, text, strength, age, links[] }
// Returns graph suitable for force-directed or radial constellation rendering.

export function buildMemoryConstellation(memories = []) {
  const nodes = memories.map(m => ({
    id:       m.id,
    text:     m.text,
    strength: π_clamp(m.strength ?? 0.5, 0, 1),
    age:      m.age ?? 0,
    size:     4 + (m.strength ?? 0.5) * 10,
    color:    `hsl(${200 + (m.age ?? 0) * 3}, 70%, 60%)`,
    links:    m.links || [],
  }));

  const edges = [];
  for (const node of nodes) {
    for (const targetId of node.links) {
      edges.push({ from: node.id, to: targetId, weight: node.strength });
    }
  }

  return { nodes, edges };
}

// ─── Shell 7: Micronaut Neural Mesh — agent nervous system ───────────────────
// micronaut = { id, goal, tools[], memory_refs[], weight }
// Returns mesh: nodes = micronauts, edges = tool/memory connections.

export function buildMicronautMesh(micronauts = []) {
  const nodes = micronauts.map(mn => ({
    id:     mn.id,
    goal:   mn.goal,
    tools:  mn.tools   || [],
    memRefs:mn.memory_refs || [],
    weight: π_clamp(mn.weight ?? 0.5, 0, 1),
    size:   6 + (mn.weight ?? 0.5) * 12,
    color:  '#6c5ce7',
  }));

  const edges = [];
  const idSet = new Set(nodes.map(n => n.id));

  for (const node of nodes) {
    for (const ref of node.memRefs) {
      if (idSet.has(ref)) edges.push({ from: node.id, to: ref, kind: 'memory' });
    }
  }

  return { nodes, edges };
}
