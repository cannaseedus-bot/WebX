/**
 * @fileoverview KUHUL language lexer / tokeniser.
 *
 * Converts KUHUL source text into a flat stream of Token objects.
 * KUHUL uses bracket-based syntax: every statement is wrapped in `[…]`.
 *
 * @example
 * import { KuhulLexer } from './lexer.js';
 * const lexer  = new KuhulLexer();
 * const tokens = lexer.lex('[Pop]\n  [Wo X tensor<float32, [10]>]\n[Xul]');
 */

// ------------------------------------------------------------------ //
// Token type constants
// ------------------------------------------------------------------ //

/** @enum {string} */
export const TokenType = Object.freeze({
  // Brackets
  LBRACKET:    'LBRACKET',    // [
  RBRACKET:    'RBRACKET',    // ]
  LANGLE:      'LANGLE',      // <
  RANGLE:      'RANGLE',      // >

  // Punctuation
  COMMA:       'COMMA',       // ,

  // Language keywords
  KEYWORD:     'KEYWORD',

  // Identifiers (user-defined names)
  IDENTIFIER:  'IDENTIFIER',

  // KUHUL glyph operators
  GLYPH:       'GLYPH',

  // Tensor type token  e.g.  tensor<float32, [10, 20]>
  TENSOR_TYPE: 'TENSOR_TYPE',

  // Literals
  NUMBER:      'NUMBER',
  STRING:      'STRING',

  // End-of-file sentinel
  EOF:         'EOF',
});

// ------------------------------------------------------------------ //
// Language constants
// ------------------------------------------------------------------ //

/** Reserved KUHUL keywords */
export const KEYWORDS = new Set([
  'Pop', 'Xul', 'Wo', 'Yax', "Ch'en", 'Sek', "K'ayab'", "Kumk'u", 'Muwan',
]);

/** KUHUL glyph operator symbols */
export const GLYPHS = new Set(['⊗', '⊕', '⊖', '⊛', '⊜', '⊝', '⊞']);

// ------------------------------------------------------------------ //
// LexerError
// ------------------------------------------------------------------ //

/** Thrown when the lexer encounters invalid source text. */
export class LexerError extends Error {
  /**
   * @param {string} message
   * @param {number} line
   * @param {number} col
   */
  constructor(message, line = 0, col = 0) {
    super(`LexerError at ${line}:${col} — ${message}`);
    this.name = 'LexerError';
    this.line = line;
    this.col  = col;
  }
}

// ------------------------------------------------------------------ //
// Token
// ------------------------------------------------------------------ //

/**
 * @typedef {{ type: string, value: string, line: number, col: number }} Token
 */

/**
 * Create a token object.
 *
 * @param {string} type
 * @param {string} value
 * @param {number} line
 * @param {number} col
 * @returns {Token}
 */
function token(type, value, line, col) {
  return { type, value, line, col };
}

// ------------------------------------------------------------------ //
// KuhulLexer
// ------------------------------------------------------------------ //

/** Tokenises KUHUL source text. */
export class KuhulLexer {
  /**
   * Lex (tokenise) a KUHUL source string.
   *
   * @param {string} source - Raw KUHUL source text
   * @returns {Token[]} Flat token array ending with an EOF token
   */
  lex(source) {
    const tokens = [];
    let i    = 0;
    let line = 1;
    let col  = 1;

    /**
     * Advance one character, tracking line/column.
     * @returns {string}
     */
    const advance = () => {
      const ch = source[i++];
      if (ch === '\n') { line++; col = 1; } else { col++; }
      return ch;
    };

    /** Peek at the current character without consuming. */
    const peek  = (offset = 0) => source[i + offset] ?? '';

    /**
     * Skip whitespace and line comments (# …).
     */
    const skipWS = () => {
      while (i < source.length) {
        if (/\s/.test(peek())) { advance(); continue; }
        // Line comments
        if (peek() === '#') {
          while (i < source.length && peek() !== '\n') advance();
          continue;
        }
        break;
      }
    };

    /**
     * Try to lex a complete tensor type starting at position i.
     * tensor<elementType, [dim, …]>
     * Returns the full token string or null if it doesn't match.
     *
     * @returns {string|null}
     */
    const tryLexTensorType = () => {
      // Peek ahead to see if we have "tensor<…>"
      const startPos = i;
      const startLine = line;
      const startCol  = col;

      if (source.slice(i, i + 6) !== 'tensor') return null;
      if (source[i + 6] !== '<') return null;

      // Save state
      let savedI = i, savedLine = line, savedCol = col;

      let raw = '';
      // Consume 'tensor'
      for (let k = 0; k < 6; k++) raw += advance();
      // Consume '<'
      raw += advance();

      // Collect until matching '>'
      let depth = 1;
      while (i < source.length && depth > 0) {
        const ch = peek();
        if (ch === '<') depth++;
        if (ch === '>') depth--;
        raw += advance();
      }

      if (depth !== 0) {
        // Not a valid tensor type – restore
        i = savedI; line = savedLine; col = savedCol;
        return null;
      }

      return raw;
    };

    while (i < source.length) {
      skipWS();
      if (i >= source.length) break;

      const startLine = line;
      const startCol  = col;
      const ch = peek();

      // Single-character punctuation
      if (ch === '[') { tokens.push(token(TokenType.LBRACKET, '[', startLine, startCol)); advance(); continue; }
      if (ch === ']') { tokens.push(token(TokenType.RBRACKET, ']', startLine, startCol)); advance(); continue; }
      if (ch === ',') { tokens.push(token(TokenType.COMMA,    ',', startLine, startCol)); advance(); continue; }
      if (ch === '<') { tokens.push(token(TokenType.LANGLE,   '<', startLine, startCol)); advance(); continue; }
      if (ch === '>') { tokens.push(token(TokenType.RANGLE,   '>', startLine, startCol)); advance(); continue; }

      // GLYPH symbols
      if (GLYPHS.has(ch)) {
        tokens.push(token(TokenType.GLYPH, ch, startLine, startCol));
        advance();
        continue;
      }

      // Number literal (including negative)
      if (/\d/.test(ch) || (ch === '-' && /\d/.test(peek(1)))) {
        let num = '';
        if (ch === '-') num += advance();
        while (i < source.length && /\d/.test(peek())) num += advance();
        if (peek() === '.') {
          num += advance();
          while (i < source.length && /\d/.test(peek())) num += advance();
        }
        tokens.push(token(TokenType.NUMBER, num, startLine, startCol));
        continue;
      }

      // String literal
      if (ch === '"' || ch === "'") {
        const q = ch;
        let str = '';
        advance(); // consume opening quote
        while (i < source.length && peek() !== q) {
          if (peek() === '\\') { advance(); str += advance(); }
          else str += advance();
        }
        if (i >= source.length) throw new LexerError('Unterminated string literal', startLine, startCol);
        advance(); // consume closing quote
        tokens.push(token(TokenType.STRING, str, startLine, startCol));
        continue;
      }

      // Identifiers, keywords, and tensor types
      if (/[A-Za-z_]/.test(ch)) {
        // Try tensor type first
        const tensorRaw = tryLexTensorType();
        if (tensorRaw !== null) {
          tokens.push(token(TokenType.TENSOR_TYPE, tensorRaw, startLine, startCol));
          continue;
        }

        // Read identifier/keyword (may contain apostrophes like Ch'en, K'ayab')
        let word = '';
        while (i < source.length && /[\w']/.test(peek())) word += advance();

        if (KEYWORDS.has(word)) {
          tokens.push(token(TokenType.KEYWORD, word, startLine, startCol));
        } else {
          tokens.push(token(TokenType.IDENTIFIER, word, startLine, startCol));
        }
        continue;
      }

      throw new LexerError(`Unexpected character '${ch}'`, startLine, startCol);
    }

    tokens.push(token(TokenType.EOF, '', line, col));
    return tokens;
  }
}
