// K'uhul Fold Engine v0.1 — temporal DOM + execution DAG
//
// K'uhul is a DOM of folds: each fold has an opening glyph, closing glyph,
// children, payload (JSONL/MathML/XJSL/text), temporal law (Pop→Xul),
// semantic law (domain.intent), and structural law (MathML/XJSL AST).
//
// Fold Engine: given a JSONL declaration + match context, builds a concrete
// FoldDAG that XVM/XJSL can schedule.

// ─── Temporal fold constants ──────────────────────────────────────────────────

export const TEMPORAL_FOLDS = Object.freeze(['Pop', 'Wo', 'Yax', 'Sek', "Ch'en", 'Xul']);

export const TEMPORAL_FOLD_ROLES = Object.freeze({
  Pop:     'ingest — raw input capture',
  Wo:      'classify — domain + semantic_fold assignment',
  Yax:     'decompose — structural + temporal decomposition',
  Sek:     'compute — XJSL kernel execution',
  "Ch'en": 'render — result formatting / explanation',
  Xul:     'seal — codex storage + replay record',
});

// ─── Semantic subgraph templates ──────────────────────────────────────────────
// Each semantic_fold has a fixed template: Yax sub-nodes + Sek sub-nodes.
// These expand the temporal spine into domain-specific compute.

export const SEMANTIC_SUBGRAPH_TEMPLATES = Object.freeze({
  'calculus.rate_of_change': Object.freeze({
    yax: ['Yax.parse_expression', 'Yax.build_diff_mathml'],
    sek: ['Sek.symbolic_diff', 'Sek.evaluate_at_point'],
  }),
  'calculus.accumulate': Object.freeze({
    yax: ['Yax.parse_expression', 'Yax.build_int_mathml'],
    sek: ['Sek.numeric_quad', 'Sek.emit_result'],
  }),
  'algebra.isolate': Object.freeze({
    yax: ['Yax.parse_equation', 'Yax.isolate_variable'],
    sek: ['Sek.solve_linear', 'Sek.emit_solution'],
  }),
  'algebra.transform': Object.freeze({
    yax: ['Yax.parse_shapes', 'Yax.check_dims'],
    sek: ['Sek.matmul'],
  }),
  'arithmetic.combine': Object.freeze({
    yax: ['Yax.parse_operands'],
    sek: ['Sek.add'],
  }),
  'arithmetic.scale': Object.freeze({
    yax: ['Yax.parse_operands'],
    sek: ['Sek.mul'],
  }),
  'geometry.area': Object.freeze({
    yax: ['Yax.parse_dims'],
    sek: ['Sek.area'],
  }),
  'geometry.measure': Object.freeze({
    yax: ['Yax.parse_points'],
    sek: ['Sek.distance'],
  }),
  'neural.relate': Object.freeze({
    yax: ['Yax.parse_shapes', 'Yax.check_head_dims'],
    sek: ['Sek.fused_attention'],
  }),
  'neural.transform': Object.freeze({
    yax: ['Yax.parse_shapes', 'Yax.check_mlp_dims'],
    sek: ['Sek.fused_mlp'],
  }),
  'neural.stabilize': Object.freeze({
    yax: ['Yax.parse_shapes'],
    sek: ['Sek.fused_norm'],
  }),
  'neural.normalize': Object.freeze({
    yax: ['Yax.parse_shapes'],
    sek: ['Sek.softmax'],
  }),
});

// ─── FoldNode ─────────────────────────────────────────────────────────────────

export function createFoldNode(id, temporal, opts = {}) {
  return {
    id,
    temporal,
    semantic:   opts.semantic   || null,
    structural: opts.structural || null,
    payload:    opts.payload    || {},
  };
}

// ─── FoldEdge ─────────────────────────────────────────────────────────────────

export function createFoldEdge(from, to, reason = 'temporal') {
  return { from, to, reason };
}

// ─── Build temporal spine (Pop→Wo→Yax→Sek→Ch'en→Xul) ────────────────────────

export function buildTemporalSpine(contextText = '') {
  const nodes = [
    createFoldNode('n0_Pop',   'Pop',    { payload: { text: contextText } }),
    createFoldNode('n1_Wo',    'Wo',     {}),
    createFoldNode('n2_Yax',   'Yax',   {}),
    createFoldNode('n3_Sek',   'Sek',   {}),
    createFoldNode("n4_Ch'en", "Ch'en", {}),
    createFoldNode('n5_Xul',   'Xul',   {}),
  ];
  const edges = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    edges.push(createFoldEdge(nodes[i].id, nodes[i + 1].id, 'temporal'));
  }
  return { nodes, edges };
}

// ─── Expand semantic fold → subgraph nodes ────────────────────────────────────

export function expandSemanticFold(spine, semanticFold, declaration = {}) {
  const template = SEMANTIC_SUBGRAPH_TEMPLATES[semanticFold];
  const nodes = [...spine.nodes];
  const edges = [...spine.edges];

  if (!template) {
    // No template: mark Wo node with the semantic fold and return as-is
    const woNode = nodes.find(n => n.temporal === 'Wo');
    if (woNode) {
      woNode.semantic = semanticFold;
      woNode.payload.declaration = declaration;
    }
    return { nodes, edges };
  }

  // Update Wo node
  const woIdx = nodes.findIndex(n => n.temporal === 'Wo');
  if (woIdx !== -1) {
    nodes[woIdx] = { ...nodes[woIdx], semantic: semanticFold, payload: { ...nodes[woIdx].payload, declaration } };
  }

  // Remove generic Yax + Sek nodes from spine, replace with template sub-nodes
  const yaxIdx = nodes.findIndex(n => n.temporal === 'Yax');
  const sekIdx = nodes.findIndex(n => n.temporal === 'Sek');

  const yaxNodes = template.yax.map((label, i) =>
    createFoldNode(`${label}_${i}`, 'Yax', { semantic: semanticFold, payload: { label } })
  );
  const sekNodes = template.sek.map((label, i) =>
    createFoldNode(`${label}_${i}`, 'Sek', { semantic: semanticFold, payload: { label, declaration } })
  );

  // Replace spine Yax node
  nodes.splice(yaxIdx, 1, ...yaxNodes);
  // Recalculate sekIdx after splice
  const sekIdx2 = nodes.findIndex(n => n.id === `n3_Sek`);
  if (sekIdx2 !== -1) nodes.splice(sekIdx2, 1, ...sekNodes);
  else {
    // Sek wasn't found by old id after splice — remove old Sek and insert after last Yax
    const lastYaxIdx = nodes.map((n, i) => ({ n, i })).filter(({ n }) => n.temporal === 'Yax').pop()?.i ?? -1;
    if (lastYaxIdx !== -1) nodes.splice(lastYaxIdx + 1, 0, ...sekNodes);
  }

  // Re-wire all edges (simple: sequential through the final node order)
  const wireEdges = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    wireEdges.push(createFoldEdge(nodes[i].id, nodes[i + 1].id,
      nodes[i].temporal === nodes[i + 1].temporal ? 'data' : 'temporal'));
  }

  return { nodes, edges: wireEdges };
}

// ─── Full compilation: declaration + text → FoldDAG ──────────────────────────

export function compileFoldDAG(declaration, match = {}) {
  const text        = match.text || '';
  const spine       = buildTemporalSpine(text);
  const semanticFold = declaration.semantic_fold || null;

  const dag = semanticFold
    ? expandSemanticFold(spine, semanticFold, declaration)
    : spine;

  // Attach MathML/XJSL payload to Sek nodes
  for (const node of dag.nodes) {
    if (node.temporal === 'Sek') {
      node.payload.xjsl_kernel = declaration.xjsl_kernel || declaration.operation;
      node.payload.math_technique = declaration.math_technique || null;
    }
    if (node.temporal === "Ch'en") {
      node.payload.math_equivalent = declaration.math_equivalent;
    }
    if (node.temporal === 'Xul') {
      node.payload.record = {
        declaration,
        math_equivalent: declaration.math_equivalent,
        kuhul_phase: declaration.kuhul_phase || null,
      };
    }
  }

  return { nodes: dag.nodes, edges: dag.edges };
}

// ─── FoldDAG → XVM batch schedule ────────────────────────────────────────────
// Maps each Sek-phase node to an XVM batch entry.
// Returns: [{ batchId, kernel, phase, dependencies }]

export function foldDagToXvm(dag) {
  const sekNodes = dag.nodes.filter(n => n.temporal === 'Sek');
  const batches  = [];

  for (let i = 0; i < sekNodes.length; i++) {
    const node = sekNodes[i];
    batches.push({
      batchId:      node.id,
      kernel:       node.payload.xjsl_kernel || 'unknown',
      kuhul_phase:  `Sek.${node.payload.label || 'compute'}`,
      dependencies: i === 0 ? [] : [sekNodes[i - 1].id],
      semantic_fold:node.semantic || null,
    });
  }

  return batches;
}

// ─── K'uhul DOM: KNode tree builder ──────────────────────────────────────────
// K'uhul is a DOM of folds — each fold has an open glyph, close glyph, children.
// DOM is distinct from the DAG: DOM describes structure, DAG describes execution order.

export const KUHUL_GLYPHS = Object.freeze({
  Pop:     { open: '⟁Pop',    close: '⟁Xul⟁' },
  Wo:      { open: '⟁Wo⟁',    close: '' },
  Yax:     { open: '⟁Yax⟁',   close: '' },
  Sek:     { open: '⟁Sek⟁',   close: '' },
  "Ch'en": { open: "⟁Ch'en⟁", close: '' },
  Xul:     { open: '⟁Xul⟁',   close: '' },
});

export function createKNode(tag, opts = {}) {
  const glyphs = KUHUL_GLYPHS[tag] || { open: `⟁${tag}`, close: '' };
  return {
    tag,
    glyph_open:  glyphs.open,
    glyph_close: glyphs.close,
    attributes:  opts.attributes || {},
    children:    opts.children   || [],
    payload:     opts.payload    || null,
  };
}

// Build a canonical K'uhul DOM tree from a math declaration.
// Returns the root KNode (Pop wrapping Wo > Yax > Sek > Ch'en > Xul).
export function buildKuhulDom(declaration, matchText = '') {
  const sek = createKNode('Sek', {
    payload: {
      kernel:         declaration.xjsl_kernel,
      math_technique: declaration.math_technique,
    },
  });
  const yax = createKNode('Yax', {
    payload:  { structural: declaration.math_equivalent },
    children: [sek],
  });
  const wo = createKNode('Wo', {
    attributes: { domain: declaration.semantic_fold },
    payload:    { declaration },
    children:   [yax],
  });
  const chen = createKNode("Ch'en", {
    payload: { math_equivalent: declaration.math_equivalent },
  });
  const xul  = createKNode('Xul', {
    payload: { record: { declaration, timestamp: Date.now() } },
  });
  const pop  = createKNode('Pop', {
    payload:  { text: matchText },
    children: [wo, chen, xul],
  });
  return pop;
}

// Serialize a K'uhul DOM node to a glyph string (debug / trace output).
export function serializeKuhulDom(node, depth = 0) {
  const indent = '  '.repeat(depth);
  const attrs  = Object.entries(node.attributes || {})
    .map(([k, v]) => ` ${k}="${v}"`)
    .join('');
  const open   = `${indent}${node.glyph_open}${attrs}`;
  if (!node.children || node.children.length === 0) {
    return node.glyph_close ? `${open}\n${indent}${node.glyph_close}` : open;
  }
  const body = node.children.map(c => serializeKuhulDom(c, depth + 1)).join('\n');
  const close = node.glyph_close ? `\n${indent}${node.glyph_close}` : '';
  return `${open}\n${body}${close}`;
}
