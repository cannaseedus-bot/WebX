/**
 * @fileoverview Jest tests for code generators.
 */

import { IRBuilder }        from '../ir/ir-builder.js';
import { JavaScriptCodegen } from '../compiler/codegen/js-codegen.js';
import { WasmCodegen }       from '../compiler/codegen/wasm-codegen.js';
import { WebGPUCodegen }     from '../compiler/codegen/webgpu-codegen.js';
import { GeometricIR }       from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function buildIR() {
  return new IRBuilder()
    .createProgram()
    .addInstruction('ALLOC',      ['X', 'float32', [10]])
    .addInstruction('FOLD_START', [])
    .addInstruction('READ',       ['X'])
    .addInstruction('OP',         ['⊕', 'X', 'X'])
    .addInstruction('WRITE',      ['Y', 'X'])
    .addInstruction('FOLD_END',   [])
    .build();
}

describe('JavaScriptCodegen', () => {
  const codegen = new JavaScriptCodegen();

  test('generate() returns a non-empty string', () => {
    const code = codegen.generate(buildIR());
    expect(typeof code).toBe('string');
    expect(code.length).toBeGreaterThan(0);
  });

  test('generated code contains "run" function', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('async function run');
  });

  test('generated code contains ALLOC → Float32Array', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('Float32Array');
  });

  test('generated code contains mem object', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('mem');
  });

  test('generated code handles ⊕ glyph', () => {
    const code = codegen.generate(buildIR());
    // ⊕ maps to addition
    expect(code).toContain('+');
  });

  test('generates PHASE_START as for loop', () => {
    const ir = new IRBuilder()
      .createProgram()
      .addInstruction('ALLOC',       ['X', 'float32', [4]])
      .addInstruction('PHASE_START', [])
      .addInstruction('READ',        ['X'])
      .addInstruction('PHASE_END',   [])
      .build();
    const code = codegen.generate(ir);
    expect(code).toContain('for');
    expect(code).toContain('Math.PI');
  });
});

describe('WasmCodegen', () => {
  const codegen = new WasmCodegen();

  test('generate() returns WAT module', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('(module');
  });

  test('contains memory declaration', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('memory');
  });

  test('contains export "run"', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('"run"');
  });
});

describe('WebGPUCodegen', () => {
  const codegen = new WebGPUCodegen();

  test('generate() returns WGSL string', () => {
    const code = codegen.generate(buildIR());
    expect(typeof code).toBe('string');
    expect(code.length).toBeGreaterThan(0);
  });

  test('contains compute entry point', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('@compute');
    expect(code).toContain('fn main');
  });

  test('contains storage buffer binding', () => {
    const code = codegen.generate(buildIR());
    expect(code).toContain('@group(0)');
    expect(code).toContain('@binding(');
  });
});
