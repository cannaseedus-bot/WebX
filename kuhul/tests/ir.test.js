/**
 * @fileoverview Jest tests for IR infrastructure.
 */

import { IRBuilder }   from '../ir/ir-builder.js';
import { IRVerifier }  from '../ir/ir-verifier.js';
import { IROptimizer } from '../ir/ir-optimizer.js';
import { IRPrinter }   from '../ir/ir-printer.js';
import { GeometricIR, TensorType, Instruction } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function buildSimpleIR() {
  return new IRBuilder()
    .createProgram()
    .addInstruction('ALLOC', ['X', 'float32', [10]])
    .addInstruction('READ',  ['X'])
    .addInstruction('WRITE', ['Y', 'X'])
    .build();
}

describe('IRBuilder', () => {
  test('creates a GeometricIR with instructions', () => {
    const ir = buildSimpleIR();
    expect(ir).toBeInstanceOf(GeometricIR);
    expect(ir.instructions).toHaveLength(3);
  });

  test('first instruction is ALLOC', () => {
    const ir = buildSimpleIR();
    expect(ir.instructions[0].opcode).toBe('ALLOC');
  });

  test('declareSymbol adds to symbolTable', () => {
    const tt = new TensorType('float32', [10]);
    const ir = new IRBuilder()
      .createProgram()
      .declareSymbol('X', tt)
      .addInstruction('ALLOC', ['X', 'float32', [10]])
      .build();
    expect(ir.symbolTable.has('X')).toBe(true);
  });

  test('createProgram resets state', () => {
    const builder = new IRBuilder();
    builder.addInstruction('ALLOC', ['A', 'float32', [1]]);
    builder.createProgram(); // reset
    const ir = builder.build();
    expect(ir.instructions).toHaveLength(0);
  });

  test('setMetadata attaches metadata', () => {
    const ir = new IRBuilder()
      .createProgram()
      .setMetadata({ source: 'test.kuhul' })
      .build();
    expect(ir.metadata.source).toBe('test.kuhul');
  });
});

describe('IRVerifier', () => {
  const verifier = new IRVerifier();

  test('valid simple IR passes verification', () => {
    const ir = new IRBuilder()
      .createProgram()
      .addInstruction('ALLOC', ['X', 'float32', [10]])
      .addInstruction('READ',  ['X'])
      .build();
    const { valid, errors } = verifier.verify(ir);
    expect(valid).toBe(true);
    expect(errors).toHaveLength(0);
  });

  test('READ of undeclared identifier fails', () => {
    const ir = new IRBuilder()
      .createProgram()
      .addInstruction('READ', ['Undeclared'])
      .build();
    const { valid, errors } = verifier.verify(ir);
    expect(valid).toBe(false);
    expect(errors[0]).toMatch(/Undeclared/);
  });

  test('unmatched FOLD_END fails', () => {
    const ir = new IRBuilder()
      .createProgram()
      .addInstruction('FOLD_END', [])
      .build();
    const { valid } = verifier.verify(ir);
    expect(valid).toBe(false);
  });

  test('balanced FOLD_START / FOLD_END passes', () => {
    const ir = new IRBuilder()
      .createProgram()
      .addInstruction('ALLOC',      ['X', 'float32', [4]])
      .addInstruction('FOLD_START', [])
      .addInstruction('READ',       ['X'])
      .addInstruction('FOLD_END',   [])
      .build();
    const { valid } = verifier.verify(ir);
    expect(valid).toBe(true);
  });

  test('returns error for non-GeometricIR input', () => {
    const { valid, errors } = verifier.verify({});
    expect(valid).toBe(false);
    expect(errors[0]).toMatch(/GeometricIR/);
  });
});

describe('IROptimizer', () => {
  const optimizer = new IROptimizer();

  test('returns a GeometricIR', () => {
    const ir  = buildSimpleIR();
    const opt = optimizer.optimize(ir);
    expect(opt).toBeInstanceOf(GeometricIR);
  });

  test('sets optimized metadata flag', () => {
    const ir  = buildSimpleIR();
    const opt = optimizer.optimize(ir);
    expect(opt.metadata.optimized).toBe(true);
  });

  test('constant folding: ⊕ 3 4 → CONST 7', () => {
    const ir = new IRBuilder()
      .createProgram()
      .addInstruction('OP', ['⊕', 3, 4])
      .build();
    const opt = optimizer.optimize(ir);
    expect(opt.instructions[0].opcode).toBe('CONST');
    expect(opt.instructions[0].operands[0]).toBe(7);
  });

  test('removes redundant consecutive READs', () => {
    const ir = new IRBuilder()
      .createProgram()
      .addInstruction('ALLOC', ['X', 'float32', [4]])
      .addInstruction('READ',  ['X'])
      .addInstruction('READ',  ['X']) // duplicate
      .build();
    const opt = optimizer.optimize(ir);
    const reads = opt.instructions.filter(i => i.opcode === 'READ');
    expect(reads).toHaveLength(1);
  });
});

describe('IRPrinter', () => {
  test('prints a non-empty string', () => {
    const ir      = buildSimpleIR();
    const printer = new IRPrinter();
    const output  = printer.print(ir);
    expect(typeof output).toBe('string');
    expect(output.length).toBeGreaterThan(0);
    expect(output).toContain('ALLOC');
  });

  test('returns a comment for invalid IR', () => {
    const printer = new IRPrinter();
    const output  = printer.print(null);
    expect(output).toContain('invalid');
  });
});
