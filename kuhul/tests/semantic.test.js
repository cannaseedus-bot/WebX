/**
 * @fileoverview Jest tests for SemanticAnalyzer.
 */

import { KuhulLexer }        from '../compiler/lexer.js';
import { KuhulParser }       from '../compiler/parser.js';
import { SemanticAnalyzer, SemanticError } from '../compiler/semantic-analyzer.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function analyze(src) {
  const tokens = new KuhulLexer().lex(src);
  const ast    = new KuhulParser().parse(tokens);
  return new SemanticAnalyzer().analyze(ast);
}

describe('SemanticAnalyzer', () => {
  // ---------------------------------------------------------------- //
  // Valid programs
  // ---------------------------------------------------------------- //

  test('accepts a valid allocation + read + write', () => {
    const { errors } = analyze(`
[Pop]
  [Wo X tensor<float32, [10]>]
  [Yax X]
  [Ch'en X X]
[Xul]
`);
    expect(errors).toHaveLength(0);
  });

  test('accepts a valid operation', () => {
    const { errors } = analyze(`
[Pop]
  [Wo X tensor<float32, [4]>]
  [Wo Y tensor<float32, [4]>]
  [Sek ⊕ X Y]
[Xul]
`);
    expect(errors).toHaveLength(0);
  });

  test('accepts a valid phase cycle', () => {
    const { errors } = analyze(`
[Wo X tensor<float32, [4]>]
[K'ayab']
  [Yax X]
[Kumk'u]
`);
    expect(errors).toHaveLength(0);
  });

  // ---------------------------------------------------------------- //
  // Undeclared variable
  // ---------------------------------------------------------------- //

  test('errors on read of undeclared identifier', () => {
    const { errors } = analyze('[Yax UndeclaredVar]');
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0].message).toMatch(/UndeclaredVar/);
  });

  test('errors on operation using undeclared identifier', () => {
    const { errors } = analyze('[Sek ⊕ Ghost Phantom]');
    expect(errors.length).toBeGreaterThan(0);
  });

  // ---------------------------------------------------------------- //
  // Unknown glyph
  // ---------------------------------------------------------------- //

  test('errors on unknown glyph', () => {
    const src = '[Wo X tensor<float32, [4]>]\n[Sek ★ X]';
    // ★ is not a recognised glyph – lexer will reject it as unknown char
    expect(() => new KuhulLexer().lex(src)).toThrow();
  });

  // ---------------------------------------------------------------- //
  // Invalid tensor element type
  // ---------------------------------------------------------------- //

  test('errors on invalid element type', () => {
    const tokens = new KuhulLexer().lex('[Wo X tensor<bogusType, [4]>]');
    const ast    = new KuhulParser().parse(tokens);
    const { errors } = new SemanticAnalyzer().analyze(ast);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0].message).toMatch(/bogusType/);
  });

  // ---------------------------------------------------------------- //
  // SemanticError shape
  // ---------------------------------------------------------------- //

  test('SemanticError.toString() includes "SemanticError"', () => {
    const e = new SemanticError('test error');
    expect(e.toString()).toContain('SemanticError');
  });

  // ---------------------------------------------------------------- //
  // Warnings
  // ---------------------------------------------------------------- //

  test('warns when writing to undeclared identifier', () => {
    const { warnings } = analyze("[Ch'en Ghost 1]");
    expect(warnings.length).toBeGreaterThan(0);
  });
});
