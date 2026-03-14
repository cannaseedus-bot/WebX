// Tests for K'UHUL++ v2.0 Runtime

import { KuhulRuntime, RuntimeError, run } from '../runtime/runtime.js';
import { tokenize } from '../compiler/lexer.js';
import { parse } from '../compiler/parser.js';

async function execSource(source, options = {}) {
    const tokens = tokenize(source);
    const ast    = parse(tokens);
    const rt     = new KuhulRuntime(options);
    return rt.execute(ast);
}

describe('KuhulRuntime', () => {
    // ------------------------------------------------------------------ //
    // Declarations and environment
    // ------------------------------------------------------------------ //

    test('executes Tensor declaration and stores result in env', async () => {
        const { env } = await execSource(
            'Tensor t = GeometricTensor(phase: 0.75π, symmetry: 0.8, stride: 32);'
        );
        expect(env.t).toBeDefined();
        expect(env.t.type).toBe('GeometricTensor');
    });

    test('executes Cluster declaration', async () => {
        const { env } = await execSource(
            'Cluster c = TensorCluster(tensors: [], center: (0, 0, 0));'
        );
        expect(env.c).toBeDefined();
        expect(env.c.type).toBe('TensorCluster');
    });

    test('executes Model declaration', async () => {
        const { env } = await execSource(
            'Model m = GeometricModel({type: classification});'
        );
        expect(env.m).toBeDefined();
        expect(env.m.type).toBe('GeometricModel');
    });

    // ------------------------------------------------------------------ //
    // Assignment
    // ------------------------------------------------------------------ //

    test('evaluates number literal assignment', async () => {
        const { env } = await execSource('x = 42;');
        expect(env.x).toBe(42);
    });

    test('evaluates π expression', async () => {
        const { env } = await execSource('x = 0.5π;');
        expect(env.x).toBeCloseTo(0.5 * Math.PI, 5);
    });

    test('evaluates binary arithmetic', async () => {
        const { env } = await execSource('x = 3 + 4;');
        expect(env.x).toBe(7);
    });

    test('evaluates array literal', async () => {
        const { env } = await execSource('arr = [1, 2, 3];');
        expect(env.arr).toEqual([1, 2, 3]);
    });

    test('evaluates Vector3 literal', async () => {
        const { env } = await execSource('v = (1, 2, 3);');
        expect(env.v).toEqual({ x: 1, y: 2, z: 3 });
    });

    // ------------------------------------------------------------------ //
    // Glyph operations
    // ------------------------------------------------------------------ //

    test('executes glyph operation on a tensor buffer', async () => {
        const src = `
Tensor spiral = GeometricTensor(phase: 0.5π, stride: 12);
[↻] spiral angle=90;
        `;
        const { log } = await execSource(src);
        const glyphEntries = log.filter(e => e.kind === 'glyph');
        expect(glyphEntries.length).toBeGreaterThan(0);
        expect(glyphEntries[0].detail.glyph).toBe('↻');
    });

    test('logs glyph operation result', async () => {
        const src = `
Tensor t = GeometricTensor(phase: 0π, stride: 9);
[⟲] t degrees=180;
        `;
        const { log } = await execSource(src);
        const glyphLog = log.find(e => e.kind === 'glyph');
        expect(glyphLog).toBeDefined();
    });

    // ------------------------------------------------------------------ //
    // Control flow
    // ------------------------------------------------------------------ //

    test('executes if/else correctly', async () => {
        const { env } = await execSource('if (1) { x = 10; } else { x = 20; }');
        expect(env.x).toBe(10);
    });

    test('executes else branch', async () => {
        const { env } = await execSource('if (0) { x = 10; } else { x = 20; }');
        expect(env.x).toBe(20);
    });

    test('executes for loop over array range', async () => {
        const { env } = await execSource(`
            total = 0;
            for (i in [1, 2, 3]) {
                total = total + i;
            }
        `);
        expect(env.total).toBe(6);
    });

    test('limits while loop iterations via maxLoops', async () => {
        // Should not hang even if condition stays true
        const rt = new KuhulRuntime({ maxLoops: 5 });
        const tokens = tokenize('x = 0; while (1) { x = x + 1; }');
        const ast    = parse(tokens);
        const { env } = await rt.execute(ast);
        expect(env.x).toBeLessThanOrEqual(5);
    });

    // ------------------------------------------------------------------ //
    // Native DX12 block
    // ------------------------------------------------------------------ //

    test('executes dx12 block without throwing', async () => {
        const src = `
dx12 {
    RootSignature mainRS {
        CBV(0);
        SRV(0);
    }
}
        `;
        const { log } = await execSource(src);
        const dx12Entries = log.filter(e => e.kind === 'dx12');
        expect(dx12Entries.length).toBeGreaterThan(0);
    });

    // ------------------------------------------------------------------ //
    // GPU method calls
    // ------------------------------------------------------------------ //

    test('logs GPU method calls', async () => {
        const { log } = await execSource('GPU.Present();');
        const gpuLogs = log.filter(e => e.kind === 'gpu');
        expect(gpuLogs.length).toBeGreaterThan(0);
    });

    // ------------------------------------------------------------------ //
    // run() convenience function
    // ------------------------------------------------------------------ //

    test('run() compiles and executes source returning log and env', async () => {
        const result = await run('Tensor t = GeometricTensor(phase: 0.5π);');
        expect(result.env.t).toBeDefined();
        expect(Array.isArray(result.log)).toBe(true);
        expect(Array.isArray(result.errors)).toBe(true);
        expect(Array.isArray(result.warnings)).toBe(true);
    });

    test('run() returns semantic errors for invalid π-phase', async () => {
        const result = await run('x = 3π;');
        // 3 > 2, so semantic analysis should flag it
        expect(result.errors.length).toBeGreaterThan(0);
    });

    // ------------------------------------------------------------------ //
    // RuntimeError
    // ------------------------------------------------------------------ //

    test('RuntimeError has correct name', () => {
        const err = new RuntimeError('test', null);
        expect(err.name).toBe('RuntimeError');
        expect(err.message).toContain('test');
    });

    // ------------------------------------------------------------------ //
    // Execution log
    // ------------------------------------------------------------------ //

    test('log is empty at construction', () => {
        const rt = new KuhulRuntime();
        expect(rt.log).toHaveLength(0);
    });

    test('log is populated after execution', async () => {
        const rt     = new KuhulRuntime();
        const tokens = tokenize('Tensor t = GeometricTensor(phase: 0π);');
        const ast    = parse(tokens);
        await rt.execute(ast);
        expect(rt.log.length).toBeGreaterThan(0);
    });
});
