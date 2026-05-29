// kxml-parser.js — KXML v7.2 XML → graph object
//
// Pure ESM, zero deps, works in browser (DOMParser) and Node 18+.
// Produces a plain-object graph ready for KXMLGraph.compile().
//
// Output shape:
//   { id, type, phase, tinyX, metadata, nodes: Map, edges: [], phaseSequence: [], softLanding: {} }

// ─── Minimal XML tokeniser ────────────────────────────────────────────────────
// Handles: open/close/self-closing tags, attributes (single+double quote),
//          text nodes, comments, PIs.  MathML/gradient children are kept as
//          raw XML strings (they are semantics, not execution).

function parseAttrs(raw) {
  const attrs = {};
  // name="value" or name='value' — handles Ch'en inside double quotes
  const re = /([a-zA-Z_][\w.:'-]*)=(?:"([^"]*)"|'([^']*)')/g;
  let m;
  while ((m = re.exec(raw)) !== null) attrs[m[1]] = m[2] ?? m[3];
  return attrs;
}

function lex(src) {
  // Strip XML/KXML PIs and comments
  const clean = src
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<\?[\s\S]*?\?>/g, '');

  const tokens = [];
  let i = 0;
  while (i < clean.length) {
    if (clean[i] !== '<') {
      const end = clean.indexOf('<', i);
      const txt = (end < 0 ? clean.slice(i) : clean.slice(i, end)).trim();
      if (txt) tokens.push({ kind: 'text', value: txt });
      if (end < 0) break;
      i = end;
      continue;
    }
    // inside a tag
    let j = i + 1;
    let inStr = false, strCh = '';
    while (j < clean.length) {
      const c = clean[j];
      if (!inStr && (c === '"' || c === "'")) { inStr = true; strCh = c; }
      else if (inStr && c === strCh) inStr = false;
      else if (!inStr && c === '>') break;
      j++;
    }
    const inner = clean.slice(i + 1, j);
    if (inner[0] === '/') {
      tokens.push({ kind: 'close', tag: inner.slice(1).trim() });
    } else {
      const selfClose = inner[inner.length - 1] === '/';
      const body = selfClose ? inner.slice(0, -1) : inner;
      const sp = body.search(/\s/);
      const tag = sp < 0 ? body.trim() : body.slice(0, sp).trim();
      const attrs = sp < 0 ? {} : parseAttrs(body.slice(sp));
      tokens.push({ kind: selfClose ? 'self' : 'open', tag, attrs });
    }
    i = j + 1;
  }
  return tokens;
}

// Build an element tree from tokens. For <mathml> and <gradient> blocks
// we keep the raw XML snippet instead of parsing children.
const RAW_TAGS = new Set(['mathml', 'gradient', 'transport_metric']);

function buildTree(tokens, src) {
  const stack = [{ tag: '__root__', attrs: {}, children: [], raw: '' }];
  let pos = 0;
  while (pos < tokens.length) {
    const tok = tokens[pos++];
    const top = stack[stack.length - 1];
    if (tok.kind === 'text') {
      top.children.push({ tag: '#text', value: tok.value });
    } else if (tok.kind === 'self') {
      top.children.push({ tag: tok.tag, attrs: tok.attrs, children: [] });
    } else if (tok.kind === 'open') {
      if (RAW_TAGS.has(tok.tag)) {
        // Consume until matching close tag as raw string
        const start = pos;
        let depth = 1;
        while (pos < tokens.length && depth > 0) {
          if (tokens[pos].kind === 'open' && tokens[pos].tag === tok.tag) depth++;
          if (tokens[pos].kind === 'close' && tokens[pos].tag === tok.tag) depth--;
          pos++;
        }
        // Re-extract raw from original source using tag markers
        const rawText = extractRaw(src, tok.tag);
        top.children.push({ tag: tok.tag, attrs: tok.attrs, children: [], raw: rawText });
      } else {
        const el = { tag: tok.tag, attrs: tok.attrs, children: [], raw: '' };
        top.children.push(el);
        stack.push(el);
      }
    } else if (tok.kind === 'close') {
      stack.pop();
    }
  }
  return stack[0].children;
}

function extractRaw(src, tag) {
  const open = new RegExp(`<${tag}[\\s>]`, 'i');
  const start = src.search(open);
  if (start < 0) return '';
  const close = src.indexOf(`</${tag}>`, start);
  return close < 0 ? '' : src.slice(start, close + tag.length + 3);
}

// ─── KXML-specific extraction ─────────────────────────────────────────────────

function extractLipschitz(rawMathml) {
  if (!rawMathml) return null;
  const m = rawMathml.match(/encoding=["']asx\/lipschitz["'][^>]*>([^<]+)</);
  return m ? parseFloat(m[1].replace('L=', '')) : null;
}

function extractGradientBound(rawMathml) {
  if (!rawMathml) return null;
  const m = rawMathml.match(/encoding=["']asx\/gradient_bound["'][^>]*>([^<]+)</);
  return m ? parseFloat(m[1]) : null;
}

function extractPhaseGate(el) {
  const a = el.attrs;
  return {
    forwardRequiresFrom: a['forward_requires_from'] ?? a['forward_requires_to'] ?? null,
    forwardRequiresTo:   a['forward_requires_to']   ?? null,
    backwardRequiresFrom: a['backward_requires_from'] ?? null,
    backwardRequiresTo:   a['backward_requires_to']   ?? null,
  };
}

function extractNode(el) {
  const a = el.attrs;
  const ops = [];
  const dependsOn = [];
  let mathmlRaw = null, gradientRaw = null;

  for (const child of el.children) {
    if (child.tag === 'ops') {
      for (const op of child.children) {
        if (op.tag === 'op') ops.push({ ...op.attrs });
      }
    } else if (child.tag === 'mathml') {
      mathmlRaw = child.raw;
    } else if (child.tag === 'gradient') {
      gradientRaw = child.raw;
    } else if (child.tag === 'depends_on') {
      dependsOn.push({
        nodeId:   child.attrs.node,
        phase:    child.attrs.phase ?? null,
        requires: child.attrs.requires ?? 'activation',
      });
    }
  }

  return {
    id:            a.id,
    phase:         a.phase         ?? 'Pop',
    domain:        a.domain        ?? 'compute',
    fold:          a.fold          ?? 'COMPUTE_FOLD',
    device:        a.device        ?? 'cpu',
    requiresPhase: a.requires_phase ?? null,
    tinyX:         a['tiny.x'] === 'true',
    ops,
    dependsOn,
    mathmlRaw,
    gradientRaw,
    lipschitz:     extractLipschitz(mathmlRaw),
    gradientBound: extractGradientBound(mathmlRaw),
  };
}

function extractEdge(el) {
  const a = el.attrs;
  let forward = null, backward = null, phaseGate = null;

  for (const child of el.children) {
    if (child.tag === 'forward') {
      forward = {
        channel:   child.attrs.channel   ?? 'activation',
        data:      child.attrs.data       ?? null,
        transport: child.attrs.transport  ?? 'identity',
      };
    } else if (child.tag === 'backward') {
      backward = {
        channel:   child.attrs.channel   ?? 'gradient',
        data:      child.attrs.data       ?? null,
        scale:     parseFloat(child.attrs.scale ?? '1'),
        transport: child.attrs.transport  ?? 'adjoint',
      };
    } else if (child.tag === 'phase_gate') {
      phaseGate = extractPhaseGate(child);
    }
  }

  return {
    from:      a.from,
    to:        a.to,
    direction: a.direction ?? 'bidirectional',
    forward,
    backward,
    phaseGate: phaseGate ?? {
      forwardRequiresFrom: null, forwardRequiresTo: null,
      backwardRequiresFrom: null, backwardRequiresTo: null,
    },
  };
}

function extractPhaseSequence(el) {
  return el.children
    .filter(c => c.tag === 'step')
    .map(s => ({
      phase:    s.attrs.phase,
      duration: s.attrs.duration ?? 'auto',
      agents:   s.children.filter(c => c.tag === 'agent').map(a => a.children?.[0]?.value ?? ''),
    }));
}

function extractSoftLanding(el) {
  const landing = { phaseSequence: [], boundedOps: [], convergence: null };
  for (const child of el.children) {
    if (child.tag === 'phase_sequence') landing.phaseSequence = extractPhaseSequence(child);
    if (child.tag === 'bounded_operations') {
      for (const op of child.children) {
        if (op.tag === 'operation') {
          landing.boundedOps.push({
            type:          op.attrs.type,
            lipschitz:     parseFloat(op.attrs.lipschitz ?? '1'),
            gradientBound: parseFloat(op.attrs.gradient_bound ?? '1'),
          });
        }
      }
    }
    if (child.tag === 'convergence') {
      landing.convergence = {
        epsilon:    parseFloat(child.attrs.epsilon ?? '0.001'),
        maxIter:    parseInt(child.attrs.max_iterations ?? '100', 10),
        tolerance:  parseFloat(child.attrs.metric_tolerance ?? '0.0001'),
      };
    }
  }
  return landing;
}

// ─── Main parse entry point ───────────────────────────────────────────────────

export function parseKXML(src) {
  const tokens = lex(src);
  const roots  = buildTree(tokens, src);

  // Find the <graph> or <kxml:graph> element
  const graphEl = roots.find(el => el.tag === 'graph' || el.tag === 'kxml:graph')
                ?? roots[0];

  if (!graphEl) throw new Error('parseKXML: no <graph> element found');

  const a = graphEl.attrs;
  const graph = {
    id:            a.id            ?? 'unnamed',
    type:          a.type          ?? 'bidirectional',
    phase:         a.phase         ?? 'Pop',
    tinyX:         a['tiny.x'] === 'true',
    manifoldDim:   parseInt(a.manifold_dim ?? '768', 10),
    curvature:     a.curvature     ?? 'fixed',
    determinism:   a.determinism   ?? 'strict',
    metadata:      {},
    nodes:         new Map(),
    edges:         [],
    phaseSequence: [],
    softLanding:   null,
  };

  for (const child of graphEl.children) {
    switch (child.tag) {
      case 'graph_metadata':
        for (const m of child.children) {
          if (m.tag !== '#text') graph.metadata[m.tag] = m.children?.[0]?.value ?? '';
        }
        break;
      case 'node': {
        const node = extractNode(child);
        graph.nodes.set(node.id, node);
        break;
      }
      case 'edge':
        graph.edges.push(extractEdge(child));
        break;
      case 'phase_sequence':
        graph.phaseSequence = extractPhaseSequence(child);
        break;
      case 'soft_landing':
        graph.softLanding = extractSoftLanding(child);
        break;
    }
  }

  return graph;
}
