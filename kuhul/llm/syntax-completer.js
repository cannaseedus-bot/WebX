/**
 * @fileoverview LLM-based syntax completer for KUHUL source.
 *
 * In production this module would call an LLM API to generate
 * context-aware completions.  The implementation below provides a
 * fully functional rule-based fallback that does not require a network
 * connection, while exposing the same async interface for drop-in
 * replacement with a real LLM backend.
 *
 * @module kuhul/llm/syntax-completer
 */

import { KuhulLexer, TokenType, KEYWORDS } from '../compiler/lexer.js';

// ------------------------------------------------------------------ //
// Completion suggestion
// ------------------------------------------------------------------ //

/**
 * @typedef {{ text: string, kind: 'keyword'|'identifier'|'glyph'|'type', detail: string }} Suggestion
 */

// ------------------------------------------------------------------ //
// SyntaxCompleter
// ------------------------------------------------------------------ //

/** Provides code-completion suggestions for KUHUL source text. */
export class SyntaxCompleter {
  /**
   * Generate completion suggestions for the given source and cursor position.
   *
   * @param {string} source   - Full KUHUL source text
   * @param {number} position - Cursor byte offset into `source`
   * @returns {Promise<Suggestion[]>}
   */
  async complete(source, position) {
    const prefix = source.slice(0, position);
    const suggestions = [];

    // Extract the last token fragment the user is typing
    const m = prefix.match(/[\w']+$/);
    const fragment = m ? m[0].toLowerCase() : '';

    // Keyword suggestions
    for (const kw of KEYWORDS) {
      if (kw.toLowerCase().startsWith(fragment)) {
        suggestions.push({ text: kw, kind: 'keyword', detail: `KUHUL keyword: ${kw}` });
      }
    }

    // Glyph suggestions
    const GLYPHS_INFO = {
      '⊗': 'tensor product / matmul',
      '⊕': 'addition / union',
      '⊖': 'subtraction',
      '⊛': 'convolution',
      '⊜': 'equality / assign',
      '⊝': 'negation',
      '⊞': 'direct sum / concat',
    };
    for (const [glyph, detail] of Object.entries(GLYPHS_INFO)) {
      suggestions.push({ text: glyph, kind: 'glyph', detail });
    }

    // Type suggestions (after "tensor<")
    if (prefix.trimEnd().endsWith('tensor<') || prefix.trimEnd().endsWith('Wo ')) {
      for (const t of ['float32', 'float64', 'int32', 'int64', 'uint8', 'bool', 'complex64']) {
        if (t.startsWith(fragment)) {
          suggestions.push({ text: t, kind: 'type', detail: `Tensor element type: ${t}` });
        }
      }
    }

    // Identifier suggestions from the source text
    const lexer = new KuhulLexer();
    try {
      const tokens = lexer.lex(source);
      const seen   = new Set();
      for (const tok of tokens) {
        if (tok.type === TokenType.IDENTIFIER && tok.value.toLowerCase().startsWith(fragment) && !seen.has(tok.value)) {
          seen.add(tok.value);
          suggestions.push({ text: tok.value, kind: 'identifier', detail: 'Declared identifier' });
        }
      }
    } catch (_) {
      // Partial source may not lex cleanly; ignore errors
    }

    return suggestions;
  }
}
