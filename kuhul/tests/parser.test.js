/**
 * @fileoverview Jest tests for KuhulParser.
 */

import { KuhulLexer }  from '../compiler/lexer.js';
import { KuhulParser, NodeKind, ParseError } from '../compiler/parser.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function parse(src) {
  const tokens = new KuhulLexer().lex(src);
  return new KuhulParser().parse(tokens);
}

describe('KuhulParser', () => {
  // ---------------------------------------------------------------- //
  // Fold
  // ---------------------------------------------------------------- //

  test('parses an empty fold', () => {
    const ast = parse('[Pop]\n[Xul]');
    expect(ast.kind).toBe(NodeKind.Program);
    expect(ast.body).toHaveLength(1);
    expect(ast.body[0].kind).toBe(NodeKind.Fold);
    expect(ast.body[0].body).toHaveLength(0);
  });

  test('parses a fold with one allocation', () => {
    const ast = parse('[Pop]\n  [Wo X tensor<float32, [10]>]\n[Xul]');
    const fold = ast.body[0];
    expect(fold.kind).toBe(NodeKind.Fold);
    expect(fold.body[0].kind).toBe(NodeKind.Allocation);
  });

  // ---------------------------------------------------------------- //
  // Allocation
  // ---------------------------------------------------------------- //

  test('parses allocation correctly', () => {
    const ast   = parse('[Wo X tensor<float32, [10]>]');
    const alloc = ast.body[0];
    expect(alloc.kind).toBe(NodeKind.Allocation);
    expect(alloc.identifier.name).toBe('X');
    expect(alloc.tensorType.elementType).toBe('float32');
    expect(alloc.tensorType.shape).toEqual([10]);
  });

  test('parses 2D tensor allocation', () => {
    const ast  = parse('[Wo W tensor<float32, [1024, 512]>]');
    const alloc = ast.body[0];
    expect(alloc.tensorType.shape).toEqual([1024, 512]);
  });

  // ---------------------------------------------------------------- //
  // Read
  // ---------------------------------------------------------------- //

  test('parses [Yax X]', () => {
    const ast  = parse('[Wo X tensor<float32, [1]>]\n[Yax X]');
    const read = ast.body[1];
    expect(read.kind).toBe(NodeKind.Read);
    expect(read.identifier.name).toBe('X');
  });

  // ---------------------------------------------------------------- //
  // Write
  // ---------------------------------------------------------------- //

  test('parses [Ch\'en X Y]', () => {
    const ast   = parse("[Wo X tensor<float32, [1]>]\n[Wo Y tensor<float32, [1]>]\n[Ch'en X Y]");
    const write = ast.body[2];
    expect(write.kind).toBe(NodeKind.Write);
    expect(write.identifier.name).toBe('X');
    expect(write.value.name).toBe('Y');
  });

  // ---------------------------------------------------------------- //
  // Operation
  // ---------------------------------------------------------------- //

  test('parses [Sek ⊗ X W]', () => {
    const ast = parse('[Wo X tensor<float32, [4]>]\n[Wo W tensor<float32, [4]>]\n[Sek ⊗ X W]');
    const op  = ast.body[2];
    expect(op.kind).toBe(NodeKind.Operation);
    expect(op.glyph).toBe('⊗');
    expect(op.operands).toHaveLength(2);
    expect(op.operands[0].name).toBe('X');
    expect(op.operands[1].name).toBe('W');
  });

  test('parses [Sek ⊕ X Y]', () => {
    const ast = parse('[Wo X tensor<float32, [4]>]\n[Wo Y tensor<float32, [4]>]\n[Sek ⊕ X Y]');
    const op  = ast.body[2];
    expect(op.glyph).toBe('⊕');
  });

  // ---------------------------------------------------------------- //
  // Phase cycle
  // ---------------------------------------------------------------- //

  test('parses a phase cycle', () => {
    const src = `[Wo X tensor<float32, [4]>]
[K'ayab']
  [Yax X]
[Kumk'u]`;
    const ast   = parse(src);
    const cycle = ast.body[1];
    expect(cycle.kind).toBe(NodeKind.PhaseCycle);
    expect(cycle.body).toHaveLength(1);
    expect(cycle.body[0].kind).toBe(NodeKind.Read);
  });

  // ---------------------------------------------------------------- //
  // Invocation
  // ---------------------------------------------------------------- //

  test('parses [Muwan myFn X]', () => {
    const ast = parse('[Wo X tensor<float32, [1]>]\n[Muwan myFn X]');
    const inv = ast.body[1];
    expect(inv.kind).toBe(NodeKind.Invocation);
    expect(inv.identifier.name).toBe('myFn');
    expect(inv.args[0].name).toBe('X');
  });

  // ---------------------------------------------------------------- //
  // ParseError
  // ---------------------------------------------------------------- //

  test('throws ParseError on unterminated fold', () => {
    expect(() => parse('[Pop]')).toThrow(ParseError);
  });

  test('throws ParseError on unknown keyword', () => {
    expect(() => parse('[Bogus]')).toThrow(ParseError);
  });
});
