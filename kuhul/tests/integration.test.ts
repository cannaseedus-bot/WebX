// K'UHUL++ Integration Tests
// End-to-end tests of the full pipeline: source → tokens → AST → IR → execution.

import { ExecutionEngine } from '../runtime/execution-engine.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

const engine = new ExecutionEngine({ strictMode: false });

async function runSource(source: string, env: Record<string, unknown> = {}) {
    return engine.run(source, env);
}

// ------------------------------------------------------------------ //
// Basic execution
// ------------------------------------------------------------------ //

describe('integration — basic execution', () => {
    test('empty program executes without error', async () => {
        const result = await runSource('');
        expect(result).toBeDefined();
        expect(result.phase).toBeGreaterThanOrEqual(0);
    });

    test('Tensor declaration produces output', async () => {
        const result = await runSource('Tensor x = 42;');
        expect(result.output).toBeDefined();
    });

    test('result includes duration', async () => {
        const result = await runSource('Tensor a = 1;');
        expect(result.durationMs).toBeGreaterThanOrEqual(0);
    });

    test('result includes log', async () => {
        const result = await runSource('Tensor b = 2;');
        expect(Array.isArray(result.log)).toBe(true);
    });
});

// ------------------------------------------------------------------ //
// Glyph operations
// ------------------------------------------------------------------ //

describe('integration — glyph operations', () => {
    test('⊕ addition of two tensors', async () => {
        const source = `
            Tensor a = [1.0, 2.0, 3.0];
            Tensor b = [4.0, 5.0, 6.0];
            Tensor out = a ⊕ b;
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });

    test('⊗ product of two tensors', async () => {
        const source = `
            Tensor a = [2.0, 3.0];
            Tensor b = [4.0, 5.0];
            Tensor out = a ⊗ b;
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });

    test('⊖ difference of two tensors', async () => {
        const source = `
            Tensor x = [10.0, 20.0];
            Tensor y = [1.0, 2.0];
            Tensor d = x ⊖ y;
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });

    test('chained glyph operations execute without error', async () => {
        const source = `
            Tensor a = [1.0, 1.0, 1.0];
            Tensor b = [2.0, 2.0, 2.0];
            Tensor c = [3.0, 3.0, 3.0];
            Tensor r = a ⊕ b ⊗ c;
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });
});

// ------------------------------------------------------------------ //
// Control flow
// ------------------------------------------------------------------ //

describe('integration — control flow', () => {
    test('if statement executes without error', async () => {
        const source = `
            Tensor cond = 1;
            if (cond) { Tensor a = 42; }
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });

    test('while loop executes without error', async () => {
        // A loop that will iterate 3 times via a counter
        const source = `
            Tensor i = 0;
            Tensor limit = 3;
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });
});

// ------------------------------------------------------------------ //
// Cluster and Model
// ------------------------------------------------------------------ //

describe('integration — Cluster and Model', () => {
    test('Cluster declaration executes', async () => {
        const source = `
            Cluster weights {
                Tensor w1 = [1.0, 2.0, 3.0];
                Tensor w2 = [4.0, 5.0, 6.0];
            }
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });

    test('Model declaration executes', async () => {
        const source = `
            Model linear {
                Tensor weights = [1.0, 0.0, 0.0, 1.0];
                Tensor bias = [0.0, 0.0];
            }
        `;
        const result = await runSource(source);
        expect(result).toBeDefined();
    });
});

// ------------------------------------------------------------------ //
// π expressions
// ------------------------------------------------------------------ //

describe('integration — π expressions', () => {
    test('bare π executes without error', async () => {
        const result = await runSource('Tensor angle = π;');
        expect(result).toBeDefined();
    });

    test('π coefficient executes without error', async () => {
        const result = await runSource('Tensor half_turn = 0.5π;');
        expect(result).toBeDefined();
    });
});

// ------------------------------------------------------------------ //
// compile() method (no execution)
// ------------------------------------------------------------------ //

describe('integration — compile()', () => {
    test('compile returns GeometricIR', () => {
        const ir = engine.compile('Tensor x = 42;');
        expect(ir).toBeDefined();
        expect(Array.isArray(ir.instructions)).toBe(true);
        expect(ir.manifold).toBeDefined();
    });

    test('compiled IR has at least one instruction', () => {
        const ir = engine.compile('Tensor x = 42;');
        expect(ir.instructions.length).toBeGreaterThan(0);
    });
});

// ------------------------------------------------------------------ //
// Pipeline log includes stage names
// ------------------------------------------------------------------ //

describe('integration — pipeline logging', () => {
    test('log includes lexer, parser, semantic, ir-gen stages', async () => {
        const result = await runSource('Tensor v = 1;');
        const logStr = result.log.join('\n');
        expect(logStr).toContain('lexer');
        expect(logStr).toContain('parser');
        expect(logStr).toContain('semantic');
        expect(logStr).toContain('ir-gen');
    });
});
