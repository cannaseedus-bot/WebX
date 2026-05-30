// builder.js — K'UHUL unified builder
//
// Phases (per the spec):
//   1. Parse            — ngram tokenize + sugar expand
//   2. ML detection     — count ⟁Ten⟁ / ⟁Fwd⟁ / ⟁Opt⟁ etc.
//   3. Code generation  — walk AST → K'UHUL source (optimized)
//   4. Artifact wrap    — add metadata, feature flags
//
// Compression and deployment stubs are included but intentionally
// lightweight — real SCXQ7 runs in the native pipeline.

import { KuhulParser }  from './parser.js';
import { STATS }        from './ngram-opcodes.js';

export class KuhulBuilder {
  constructor() {
    this.parser = new KuhulParser();
  }

  // ─── Main entry ─────────────────────────────────────────────────────────────

  async build(source, opts = {}) {
    const t0 = Date.now();

    const parsed   = this.parser.parse(source);
    const ml_ops   = this._detectML(parsed.tree);
    const code     = this._generate(parsed.tree, opts);
    const features = this._features(parsed.tree);
    const meta     = {
      build_ms:  Date.now() - t0,
      version:   'Ω.∞.Ω.8.0',
      stats:     STATS,
      features,
      ml_ops,
      token_count:  parsed.token_count,
      unique_ngrams: parsed.unique_ngrams,
    };

    return { source, parsed, code, meta };
  }

  // ─── ML operation counter ────────────────────────────────────────────────────

  _detectML(tree) {
    const counts = { tensors: 0, neural: 0, training: 0, distributed: 0 };
    const walk = node => {
      const op = node.opcode;
      if (op >= 0x20 && op <= 0x25) counts.tensors++;
      if (op >= 0x26 && op <= 0x2D) counts.neural++;
      if (op === 0x27 || op === 0x29) counts.training++;
      if (op >= 0x40 && op <= 0x4B) counts.distributed++;
      node.tokens?.forEach(walk);
      node.children?.forEach(walk);
    };
    walk(tree);
    return counts;
  }

  // ─── Code generation ─────────────────────────────────────────────────────────

  _generate(tree, opts) {
    const lines = [];
    const emit  = (node, depth = 0) => {
      const ind = '  '.repeat(depth);
      if (node.type === 'BLOCK') {
        lines.push(`${ind}⟁Pop⟁ ${node.name ?? ''}`);
        node.tokens?.forEach(t  => emit(t, depth + 1));
        node.children?.forEach(c => emit(c, depth + 1));
        lines.push(`${ind}⟁Xul`);
        return;
      }
      if (node.raw) {
        const args = node.args?.join(' ') ?? '';
        lines.push(`${ind}${node.raw}${args ? ' ' + args : ''}`);
      }
      node.tokens?.forEach(t  => emit(t, depth));
      node.children?.forEach(c => emit(c, depth));
    };
    tree.children?.forEach(c => emit(c));
    tree.tokens?.forEach(t   => emit(t));
    let code = lines.join('\n');
    if (opts.optimize !== false) code = this._optimize(code);
    return code;
  }

  _optimize(code) {
    return code
      .replace(/[ \t]+$/gm, '')
      .replace(/\n{3,}/g, '\n\n');
  }

  // ─── Feature detection ────────────────────────────────────────────────────────

  _features(tree) {
    const f = { has_ml: false, has_distributed: false, has_xcfe: false, complexity: 0 };
    const walk = node => {
      const op = node.opcode ?? 0;
      if (op >= 0x20 && op <= 0x37) f.has_ml = true;
      if (op >= 0x40 && op <= 0x4B) f.has_distributed = true;
      if (op >= 0x60 && op <= 0x69) f.has_xcfe = true;
      f.complexity++;
      node.tokens?.forEach(walk);
      node.children?.forEach(walk);
    };
    walk(tree);
    return f;
  }
}
