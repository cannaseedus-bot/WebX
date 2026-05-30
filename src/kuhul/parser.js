// parser.js — K'UHUL complete ngram parser
//
// Tokenizes K'UHUL source into ngrams, builds a parse tree,
// and expands syntax sugar into opcodes before dispatch.
//
// Sugar expansion is purely textual — no runtime state required.
// The parse tree uses the same Phase labels as KXML (Pop/Wo/Sek/Ch'en/Xul).

import { ALL_NGRAMS, SYNTAX_SUGAR } from './ngram-opcodes.js';

export class KuhulParser {
  constructor() {
    // Sort patterns longest-first so greedy matching picks the most specific
    this._patternKeys = Object.keys(ALL_NGRAMS).sort((a, b) => b.length - a.length);
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  parse(source) {
    const tokens    = this._tokenize(source);
    const expanded  = this._expandSugar(tokens);
    const tree      = this._buildTree(expanded);
    const analysis  = this._analyze(tree);
    return {
      tokens,
      expanded,
      tree,
      analysis,
      token_count:   tokens.length,
      unique_ngrams: new Set(tokens.map(t => t.value)).size,
      max_depth:     this._depth(tree),
    };
  }

  // ─── Tokenize ────────────────────────────────────────────────────────────────

  _tokenize(src) {
    const NGRAM_RE = /⟁[^⟁]+⟁?|→|↻|↯|←|\S+|\s+/g;
    const tokens   = [];
    let m;
    while ((m = NGRAM_RE.exec(src)) !== null) {
      const val     = m[0];
      const pattern = this._matchPattern(val);
      tokens.push({ value: val, pattern, index: m.index });
    }
    return tokens;
  }

  _matchPattern(val) {
    for (const key of this._patternKeys) {
      if (val === key || val.startsWith(key)) {
        return { ...ALL_NGRAMS[key], raw: key };
      }
    }
    return null;
  }

  // ─── Sugar expansion ─────────────────────────────────────────────────────────

  _expandSugar(tokens) {
    const out = [];
    for (const tok of tokens) {
      if (tok.pattern?.sugar) {
        // Expand: emit each opcode token in the expansion string
        const parts = tok.pattern.expands.split(/\s+/).filter(Boolean);
        for (const part of parts) {
          const p = this._matchPattern(part);
          out.push({ value: part, pattern: p, index: tok.index, expandedFrom: tok.value });
        }
      } else {
        out.push(tok);
      }
    }
    return out;
  }

  // ─── Parse tree ──────────────────────────────────────────────────────────────

  _buildTree(tokens) {
    const root  = { type: 'root', children: [], tokens: [], parent: null };
    let   scope = root;
    const stack = [root];

    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i];
      if (!tok.pattern) { scope.tokens.push(tok); continue; }

      if (tok.pattern.opcode === 0x01) {          // ⟁Pop⟁ — begin block
        const node = {
          type: 'BLOCK',
          raw:  tok.value,
          name: tokens[i+1]?.pattern ? null : tokens[i+1]?.value,
          phase: this._inferPhase(tokens, i),
          children: [], tokens: [], parent: scope,
          opcode: tok.pattern.opcode,
          category: tok.pattern.category,
        };
        scope.children.push(node);
        stack.push(node);
        scope = node;
      } else if (tok.pattern.opcode === 0x02) {   // ⟁Xul — end block
        stack.pop();
        scope = stack[stack.length - 1] ?? root;
      } else {
        const op = {
          type:     tok.pattern.description ?? tok.pattern.type,
          raw:      tok.value,
          opcode:   tok.pattern.opcode,
          category: tok.pattern.category,
          args:     this._collectArgs(tokens, i, tok.pattern),
        };
        scope.tokens.push(op);
      }
    }
    return root;
  }

  _inferPhase(tokens, idx) {
    // Peek ahead for a phase tag like (Pop) (Sek) etc. within 3 tokens
    const phases = ['Pop','Wo','Sek',"Ch'en",'Xul'];
    for (let j = idx + 1; j < Math.min(idx + 4, tokens.length); j++) {
      const v = tokens[j].value;
      if (phases.some(p => v.includes(p))) return v;
    }
    return null;
  }

  _collectArgs(tokens, i, pattern) {
    const arity = pattern.arity ?? 0;
    if (arity === 0 || arity === 'variable') return [];
    const args = [];
    for (let j = 1; j <= arity && i + j < tokens.length; j++) {
      args.push(tokens[i + j].value);
    }
    return args;
  }

  // ─── Analysis ────────────────────────────────────────────────────────────────

  _analyze(tree) {
    const by_category = {};
    const by_opcode   = {};
    const traverse = node => {
      const cat = node.category ?? node.pattern?.category;
      if (cat) by_category[cat] = (by_category[cat] ?? 0) + 1;
      if (node.opcode != null) {
        const hex = `0x${node.opcode.toString(16).padStart(2,'0')}`;
        by_opcode[hex] = (by_opcode[hex] ?? 0) + 1;
      }
      node.tokens?.forEach(traverse);
      node.children?.forEach(traverse);
    };
    traverse(tree);
    return { by_category, by_opcode };
  }

  _depth(node, d = 0) {
    if (!node.children?.length) return d;
    return Math.max(...node.children.map(c => this._depth(c, d + 1)));
  }
}
