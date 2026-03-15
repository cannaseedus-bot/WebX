/**
 * @fileoverview End-to-end integration tests for the KUHUL pipeline.
 */

import { KuhulCompiler } from '../compiler/kuhul-compiler.js';
import { KuhulVM }       from '../runtime/kuhul-vm.js';

const HELLO_WORLD = `
[Pop]
  [Wo X tensor<float32, [10]>]
  [Yax X]
  [Ch'en X X]
[Xul]
`;

const NEURAL_LAYER = `
[Pop]
  [Wo X tensor<float32, [4]>]
  [Wo W tensor<float32, [4, 2]>]
  [Wo b tensor<float32, [2]>]

  [K'ayab']
    [Yax X]
    [Yax W]
    [Sek ⊗ X W]
    [Ch'en XW X]

    [Yax XW]
    [Yax b]
    [Sek ⊕ XW b]
    [Ch'en output XW]
  [Kumk'u]
[Xul]
`;

describe('Full pipeline integration', () => {
  const compiler = new KuhulCompiler();

  // ---------------------------------------------------------------- //
  // Compile to JS
  // ---------------------------------------------------------------- //

  test('compiles hello-world to JS without errors', async () => {
    const result = await compiler.compile(HELLO_WORLD, 'js');
    expect(result).toBeDefined();
    expect(result.code).toContain('async function run');
  });

  test('compiles neural-layer to JS without errors', async () => {
    const result = await compiler.compile(NEURAL_LAYER, 'js');
    expect(result).toBeDefined();
    expect(result.target).toBe('js');
  });

  // ---------------------------------------------------------------- //
  // Compile to Wasm
  // ---------------------------------------------------------------- //

  test('compiles hello-world to wasm', async () => {
    const result = await compiler.compile(HELLO_WORLD, 'wasm');
    expect(result.code).toContain('(module');
  });

  // ---------------------------------------------------------------- //
  // Compile to WebGPU
  // ---------------------------------------------------------------- //

  test('compiles hello-world to webgpu', async () => {
    const result = await compiler.compile(HELLO_WORLD, 'webgpu');
    expect(result.code).toContain('@compute');
  });

  // ---------------------------------------------------------------- //
  // ExecutableProgram shape
  // ---------------------------------------------------------------- //

  test('result has ir, target, and code properties', async () => {
    const result = await compiler.compile(HELLO_WORLD, 'js');
    expect(result).toHaveProperty('ir');
    expect(result).toHaveProperty('target', 'js');
    expect(result).toHaveProperty('code');
  });

  // ---------------------------------------------------------------- //
  // VM execution
  // ---------------------------------------------------------------- //

  test('KuhulVM executes hello-world IR and returns memory state', async () => {
    const result = await compiler.compile(HELLO_WORLD, 'js');
    const vm     = new KuhulVM();
    const state  = await vm.execute(result.ir);
    expect(typeof state).toBe('object');
    expect(state).toHaveProperty('X');
  });

  test('KuhulVM result for X is an array', async () => {
    const result = await compiler.compile(HELLO_WORLD, 'js');
    const vm     = new KuhulVM();
    const state  = await vm.execute(result.ir);
    expect(Array.isArray(state.X)).toBe(true);
    expect(state.X).toHaveLength(10);
  });

  // ---------------------------------------------------------------- //
  // Error handling
  // ---------------------------------------------------------------- //

  test('compile throws CompileError on semantic error', async () => {
    const src = '[Yax UndeclaredVar]';
    await expect(compiler.compile(src)).rejects.toThrow();
  });

  test('compile throws CompileError on invalid target', async () => {
    await expect(compiler.compile(HELLO_WORLD, 'invalid')).rejects.toThrow();
  });
});
