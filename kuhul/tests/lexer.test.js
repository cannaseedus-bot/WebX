/**
 * @fileoverview Jest tests for KuhulLexer.
 */

import { KuhulLexer, TokenType, KEYWORDS, LexerError } from '../compiler/lexer.js';

describe('KuhulLexer', () => {
  let lexer;
  beforeEach(() => { lexer = new KuhulLexer(); });

  // ---------------------------------------------------------------- //
  // Helpers
  // ---------------------------------------------------------------- //

  /** Return all non-EOF tokens. */
  const lex = (src) => lexer.lex(src).filter(t => t.type !== TokenType.EOF);

  // ---------------------------------------------------------------- //
  // Keywords
  // ---------------------------------------------------------------- //

  test('tokenises [Pop] as LBRACKET, KEYWORD, RBRACKET', () => {
    const tokens = lex('[Pop]');
    expect(tokens).toHaveLength(3);
    expect(tokens[0].type).toBe(TokenType.LBRACKET);
    expect(tokens[1].type).toBe(TokenType.KEYWORD);
    expect(tokens[1].value).toBe('Pop');
    expect(tokens[2].type).toBe(TokenType.RBRACKET);
  });

  test('tokenises [Xul]', () => {
    const tokens = lex('[Xul]');
    expect(tokens[1].type).toBe(TokenType.KEYWORD);
    expect(tokens[1].value).toBe('Xul');
  });

  test('tokenises [Wo X tensor<float32, [10]>]', () => {
    const tokens = lex('[Wo X tensor<float32, [10]>]');
    const types  = tokens.map(t => t.type);
    expect(types).toContain(TokenType.KEYWORD);
    expect(types).toContain(TokenType.IDENTIFIER);
    expect(types).toContain(TokenType.TENSOR_TYPE);
  });

  test('tokenises Ch\'en keyword', () => {
    const tokens = lex("[Ch'en X Y]");
    const kw = tokens.find(t => t.type === TokenType.KEYWORD);
    expect(kw.value).toBe("Ch'en");
  });

  test('tokenises K\'ayab\' keyword', () => {
    const tokens = lex("[K'ayab']");
    const kw = tokens.find(t => t.type === TokenType.KEYWORD);
    expect(kw.value).toBe("K'ayab'");
  });

  test('tokenises Kumk\'u keyword', () => {
    const tokens = lex("[Kumk'u]");
    const kw = tokens.find(t => t.type === TokenType.KEYWORD);
    expect(kw).toBeTruthy();
    expect(kw.value).toBe("Kumk'u");
  });

  // ---------------------------------------------------------------- //
  // Identifiers
  // ---------------------------------------------------------------- //

  test('tokenises identifier X', () => {
    const tokens = lex('X');
    expect(tokens[0].type).toBe(TokenType.IDENTIFIER);
    expect(tokens[0].value).toBe('X');
  });

  test('tokenises multi-char identifier myTensor', () => {
    const tokens = lex('myTensor');
    expect(tokens[0].type).toBe(TokenType.IDENTIFIER);
    expect(tokens[0].value).toBe('myTensor');
  });

  // ---------------------------------------------------------------- //
  // Glyphs
  // ---------------------------------------------------------------- //

  test('tokenises glyph ⊗', () => {
    const tokens = lex('⊗');
    expect(tokens[0].type).toBe(TokenType.GLYPH);
    expect(tokens[0].value).toBe('⊗');
  });

  test('tokenises all seven glyphs', () => {
    const glyphs = ['⊗', '⊕', '⊖', '⊛', '⊜', '⊝', '⊞'];
    for (const g of glyphs) {
      const tokens = lex(g);
      expect(tokens[0].type).toBe(TokenType.GLYPH);
    }
  });

  // ---------------------------------------------------------------- //
  // Tensor types
  // ---------------------------------------------------------------- //

  test('tokenises tensor<float32, [10]> as TENSOR_TYPE', () => {
    const tokens = lex('tensor<float32, [10]>');
    expect(tokens[0].type).toBe(TokenType.TENSOR_TYPE);
    expect(tokens[0].value).toContain('float32');
  });

  test('tokenises tensor<float32, [1024, 512]>', () => {
    const tokens = lex('tensor<float32, [1024, 512]>');
    expect(tokens[0].type).toBe(TokenType.TENSOR_TYPE);
    expect(tokens[0].value).toContain('1024');
  });

  // ---------------------------------------------------------------- //
  // Number literals
  // ---------------------------------------------------------------- //

  test('tokenises integer number 42', () => {
    const tokens = lex('42');
    expect(tokens[0].type).toBe(TokenType.NUMBER);
    expect(tokens[0].value).toBe('42');
  });

  test('tokenises float 3.14', () => {
    const tokens = lex('3.14');
    expect(tokens[0].type).toBe(TokenType.NUMBER);
  });

  // ---------------------------------------------------------------- //
  // Allocation statement
  // ---------------------------------------------------------------- //

  test('tokenises full allocation statement', () => {
    const src    = '[Wo X tensor<float32, [10]>]';
    const tokens = lex(src);
    expect(tokens[0].type).toBe(TokenType.LBRACKET);
    expect(tokens[1].value).toBe('Wo');
    expect(tokens[2].type).toBe(TokenType.IDENTIFIER);
    expect(tokens[3].type).toBe(TokenType.TENSOR_TYPE);
    expect(tokens[4].type).toBe(TokenType.RBRACKET);
  });

  // ---------------------------------------------------------------- //
  // EOF token
  // ---------------------------------------------------------------- //

  test('final token is EOF', () => {
    const tokens = lexer.lex('[Pop]');
    expect(tokens[tokens.length - 1].type).toBe(TokenType.EOF);
  });

  // ---------------------------------------------------------------- //
  // Error handling
  // ---------------------------------------------------------------- //

  test('throws LexerError on unexpected character', () => {
    expect(() => lex('@')).toThrow(LexerError);
  });
});
