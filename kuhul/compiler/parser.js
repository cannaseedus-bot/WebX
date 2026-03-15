/**
 * @fileoverview KUHUL parser – converts a token stream into an AST.
 *
 * Grammar overview:
 *   program    → statement*  EOF
 *   statement  → fold | allocation | read | write | operation
 *              | phase_cycle | invocation
 *   fold       → "[" "Pop" "]"  statement*  "[" "Xul" "]"
 *   allocation → "[" "Wo"   IDENTIFIER  TENSOR_TYPE  "]"
 *   read       → "[" "Yax"  IDENTIFIER  "]"
 *   write      → "[" "Ch'en" IDENTIFIER value "]"
 *   operation  → "[" "Sek"  GLYPH  operand+  "]"
 *   phase_cycle→ "[" "K'ayab'" "]"  statement*  "[" "Kumk'u" "]"
 *   invocation → "[" "Muwan" IDENTIFIER value* "]"
 *
 * @module kuhul/compiler/parser
 */

import { TokenType, GLYPHS } from './lexer.js';

// ------------------------------------------------------------------ //
// AST node-kind constants
// ------------------------------------------------------------------ //

/** @enum {string} */
export const NodeKind = Object.freeze({
  Program:     'Program',
  Fold:        'Fold',
  Allocation:  'Allocation',
  Read:        'Read',
  Write:       'Write',
  Operation:   'Operation',
  PhaseCycle:  'PhaseCycle',
  Invocation:  'Invocation',
  Identifier:  'Identifier',
  NumberLit:   'NumberLit',
  StringLit:   'StringLit',
  TensorType:  'TensorType',
});

// ------------------------------------------------------------------ //
// ParseError
// ------------------------------------------------------------------ //

/** Thrown when the parser encounters a syntax error. */
export class ParseError extends Error {
  /**
   * @param {string} message
   * @param {{ line: number, col: number }|null} tok
   */
  constructor(message, tok = null) {
    const loc = tok ? `${tok.line}:${tok.col}` : '?:?';
    super(`ParseError at ${loc} — ${message}`);
    this.name  = 'ParseError';
    this.token = tok;
  }
}

// ------------------------------------------------------------------ //
// AST node factories
// ------------------------------------------------------------------ //

const n = (kind, extra = {}) => ({ kind, ...extra });

// ------------------------------------------------------------------ //
// KuhulParser
// ------------------------------------------------------------------ //

/** Parses a KUHUL token stream into an AST. */
export class KuhulParser {
  /**
   * Parse the given token array (as produced by KuhulLexer) into an AST.
   *
   * @param {import('./lexer.js').Token[]} tokens
   * @returns {{ kind: 'Program', body: object[] }}
   */
  parse(tokens) {
    this._tokens = tokens;
    this._pos    = 0;

    const body = [];
    while (!this._isEOF()) {
      body.push(this._parseStatement());
    }
    return n(NodeKind.Program, { body });
  }

  // ---------------------------------------------------------------- //
  // Internal helpers
  // ---------------------------------------------------------------- //

  _peek(offset = 0) {
    return this._tokens[this._pos + offset] ?? { type: TokenType.EOF, value: '', line: 0, col: 0 };
  }

  _isEOF() {
    return this._peek().type === TokenType.EOF;
  }

  _advance() {
    const t = this._tokens[this._pos];
    if (t.type !== TokenType.EOF) this._pos++;
    return t;
  }

  /**
   * Consume the next token, asserting its type (and optionally its value).
   *
   * @param {string} type
   * @param {string|null} [value]
   * @returns {import('./lexer.js').Token}
   */
  _expect(type, value = null) {
    const t = this._peek();
    if (t.type !== type) {
      throw new ParseError(`Expected ${type}${value ? ` "${value}"` : ''} but got ${t.type} "${t.value}"`, t);
    }
    if (value !== null && t.value !== value) {
      throw new ParseError(`Expected "${value}" but got "${t.value}"`, t);
    }
    return this._advance();
  }

  /** Consume LBRACKET */
  _openBracket() { return this._expect(TokenType.LBRACKET); }
  /** Consume RBRACKET */
  _closeBracket() { return this._expect(TokenType.RBRACKET); }

  /**
   * Consume a KEYWORD token with the given value.
   * @param {string} kw
   */
  _expectKW(kw) {
    const t = this._peek();
    if (t.type !== TokenType.KEYWORD || t.value !== kw) {
      throw new ParseError(`Expected keyword "${kw}" but got ${t.type} "${t.value}"`, t);
    }
    return this._advance();
  }

  // ---------------------------------------------------------------- //
  // Statement dispatch
  // ---------------------------------------------------------------- //

  _parseStatement() {
    // All statements start with "["
    const lb = this._peek();
    if (lb.type !== TokenType.LBRACKET) {
      throw new ParseError(`Expected "[" to start a statement but got ${lb.type} "${lb.value}"`, lb);
    }

    // Peek at the keyword after "["
    const kw = this._peek(1);
    if (kw.type !== TokenType.KEYWORD) {
      throw new ParseError(`Expected a KUHUL keyword after "[" but got ${kw.type} "${kw.value}"`, kw);
    }

    switch (kw.value) {
      case 'Pop':      return this._parseFold();
      case 'Wo':       return this._parseAllocation();
      case 'Yax':      return this._parseRead();
      case "Ch'en":    return this._parseWrite();
      case 'Sek':      return this._parseOperation();
      case "K'ayab'":  return this._parsePhaseCycle();
      case 'Muwan':    return this._parseInvocation();
      default:
        throw new ParseError(`Unknown keyword "${kw.value}"`, kw);
    }
  }

  // ---------------------------------------------------------------- //
  // Individual statement parsers
  // ---------------------------------------------------------------- //

  /** [Pop] statement* [Xul] */
  _parseFold() {
    const start = this._peek();
    this._openBracket();
    this._expectKW('Pop');
    this._closeBracket();

    const body = [];
    while (!(this._peek().type === TokenType.LBRACKET && this._peek(1).value === 'Xul')) {
      if (this._isEOF()) throw new ParseError('Unterminated fold – missing [Xul]', start);
      body.push(this._parseStatement());
    }

    this._openBracket();
    this._expectKW('Xul');
    this._closeBracket();

    return n(NodeKind.Fold, { body, line: start.line, col: start.col });
  }

  /** [Wo IDENTIFIER TENSOR_TYPE] */
  _parseAllocation() {
    const start = this._peek();
    this._openBracket();
    this._expectKW('Wo');

    const identTok = this._expect(TokenType.IDENTIFIER);
    const typeTok  = this._expect(TokenType.TENSOR_TYPE);
    this._closeBracket();

    return n(NodeKind.Allocation, {
      identifier: n(NodeKind.Identifier, { name: identTok.value }),
      tensorType: this._parseTensorTypeString(typeTok.value),
      line: start.line, col: start.col,
    });
  }

  /**
   * Parse the raw tensor type string into a structured node.
   * e.g. "tensor<float32, [10, 20]>"
   *
   * @param {string} raw
   * @returns {{ kind: 'TensorType', elementType: string, shape: (number|'?')[] }}
   */
  _parseTensorTypeString(raw) {
    // tensor<elementType, [dim, ...]>
    const m = raw.match(/^tensor<\s*([\w]+)\s*,\s*\[([^\]]*)\]\s*>$/);
    if (!m) return n(NodeKind.TensorType, { elementType: 'unknown', shape: [] });

    const elementType = m[1];
    const shape = m[2]
      .split(',')
      .map(s => s.trim())
      .filter(Boolean)
      .map(s => s === '?' ? '?' : Number(s));

    return n(NodeKind.TensorType, { elementType, shape });
  }

  /** [Yax IDENTIFIER] */
  _parseRead() {
    const start = this._peek();
    this._openBracket();
    this._expectKW('Yax');
    const identTok = this._expect(TokenType.IDENTIFIER);
    this._closeBracket();

    return n(NodeKind.Read, {
      identifier: n(NodeKind.Identifier, { name: identTok.value }),
      line: start.line, col: start.col,
    });
  }

  /** [Ch'en IDENTIFIER value] */
  _parseWrite() {
    const start = this._peek();
    this._openBracket();
    this._expectKW("Ch'en");
    const identTok = this._expect(TokenType.IDENTIFIER);
    const value    = this._parseValue();
    this._closeBracket();

    return n(NodeKind.Write, {
      identifier: n(NodeKind.Identifier, { name: identTok.value }),
      value,
      line: start.line, col: start.col,
    });
  }

  /** [Sek GLYPH operand+] */
  _parseOperation() {
    const start = this._peek();
    this._openBracket();
    this._expectKW('Sek');

    const glyphTok = this._expect(TokenType.GLYPH);
    const operands = [];

    while (this._peek().type !== TokenType.RBRACKET && !this._isEOF()) {
      operands.push(this._parseOperand());
    }
    if (operands.length === 0) {
      throw new ParseError('Operation must have at least one operand', glyphTok);
    }
    this._closeBracket();

    return n(NodeKind.Operation, {
      glyph: glyphTok.value,
      operands,
      line: start.line, col: start.col,
    });
  }

  /** [K'ayab'] statement* [Kumk'u] */
  _parsePhaseCycle() {
    const start = this._peek();
    this._openBracket();
    this._expectKW("K'ayab'");
    this._closeBracket();

    const body = [];
    while (!(this._peek().type === TokenType.LBRACKET && this._peek(1).value === "Kumk'u")) {
      if (this._isEOF()) throw new ParseError("Unterminated phase cycle – missing [Kumk'u]", start);
      body.push(this._parseStatement());
    }

    this._openBracket();
    this._expectKW("Kumk'u");
    this._closeBracket();

    return n(NodeKind.PhaseCycle, { body, line: start.line, col: start.col });
  }

  /** [Muwan IDENTIFIER value*] */
  _parseInvocation() {
    const start = this._peek();
    this._openBracket();
    this._expectKW('Muwan');

    const identTok = this._expect(TokenType.IDENTIFIER);
    const args = [];

    while (this._peek().type !== TokenType.RBRACKET && !this._isEOF()) {
      args.push(this._parseValue());
    }
    this._closeBracket();

    return n(NodeKind.Invocation, {
      identifier: n(NodeKind.Identifier, { name: identTok.value }),
      args,
      line: start.line, col: start.col,
    });
  }

  // ---------------------------------------------------------------- //
  // Value / operand parsers
  // ---------------------------------------------------------------- //

  /** An operand can be an identifier, a number, or a string. */
  _parseOperand() {
    return this._parseValue();
  }

  /**
   * Parse a value: identifier | number | string.
   * @returns {object}
   */
  _parseValue() {
    const t = this._peek();
    if (t.type === TokenType.IDENTIFIER) {
      this._advance();
      return n(NodeKind.Identifier, { name: t.value, line: t.line, col: t.col });
    }
    if (t.type === TokenType.NUMBER) {
      this._advance();
      return n(NodeKind.NumberLit, { value: Number(t.value), line: t.line, col: t.col });
    }
    if (t.type === TokenType.STRING) {
      this._advance();
      return n(NodeKind.StringLit, { value: t.value, line: t.line, col: t.col });
    }
    throw new ParseError(`Expected a value (identifier, number, or string) but got ${t.type} "${t.value}"`, t);
  }
}
