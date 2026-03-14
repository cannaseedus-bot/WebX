// K'UHUL++ IR Unit Tests
// Tests IRBuilder construction, IROptimizer constant folding and fusion,
// and IRVerifier correctness checks.

import { IRBuilder }   from '../ir/ir-builder.js';
import { IROptimizer } from '../ir/ir-optimizer.js';
import { IRVerifier }  from '../ir/ir-verifier.js';
import type { GeometricIR } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// IRBuilder
// ------------------------------------------------------------------ //

describe('IRBuilder', () => {
    test('builds an empty IR with default manifold', () => {
        const ir = new IRBuilder().build();
        expect(ir.instructions).toHaveLength(0);
        expect(ir.manifold.dimensions).toBe(3);
        expect(ir.manifold.metric).toBe('euclidean');
        expect(ir.manifold.phase).toBe(0);
        expect(ir.phases).toHaveLength(0);
    });

    test('addConst emits a const instruction', () => {
        const ir = new IRBuilder()
            .addConst('c0', 1.0, { kind: 'scalar', dtype: 'float32' })
            .build();
        expect(ir.instructions).toHaveLength(1);
        expect(ir.instructions[0].op).toBe('const');
        expect((ir.instructions[0] as any).value).toBe(1.0);
    });

    test('addLoad emits a load and registers symbol', () => {
        const ir = new IRBuilder()
            .addLoad('r0', 'myTensor', { kind: 'tensor', dtype: 'float32', shape: [3] })
            .build();
        expect(ir.instructions[0].op).toBe('load');
        expect(ir.symbols.has('myTensor')).toBe(true);
    });

    test('addGlyphOp emits glyph instruction', () => {
        const ir = new IRBuilder()
            .addConst('a', 1, { kind: 'scalar', dtype: 'float32' })
            .addConst('b', 2, { kind: 'scalar', dtype: 'float32' })
            .addGlyphOp('g0', '⊕', 'a', 'b', { kind: 'scalar', dtype: 'float32' })
            .build();
        const glyphInstr = ir.instructions.find(i => i.op === '⊕');
        expect(glyphInstr).toBeDefined();
        expect((glyphInstr as any).left).toBe('a');
        expect((glyphInstr as any).right).toBe('b');
    });

    test('beginPhase / endPhase creates a phase entry', () => {
        const ir = new IRBuilder()
            .beginPhase('main', 0, Math.PI * 2)
            .addConst('c0', 0, { kind: 'scalar', dtype: 'float32' })
            .endPhase()
            .build();
        expect(ir.phases).toHaveLength(1);
        expect(ir.phases[0].name).toBe('main');
        expect(ir.phases[0].instructions).toHaveLength(1);
    });

    test('phase instructions are also in the flat list', () => {
        const ir = new IRBuilder()
            .beginPhase('p', 0, Math.PI)
            .addConst('c', 42, { kind: 'scalar', dtype: 'float32' })
            .endPhase()
            .build();
        expect(ir.instructions.some(i => i.op === 'const')).toBe(true);
    });

    test('setManifold customises the manifold', () => {
        const ir = new IRBuilder()
            .setManifold(4, 'riemannian', Math.PI / 2)
            .build();
        expect(ir.manifold.dimensions).toBe(4);
        expect(ir.manifold.metric).toBe('riemannian');
        expect(ir.manifold.phase).toBeCloseTo(Math.PI / 2);
    });

    test('build() is idempotent', () => {
        const builder = new IRBuilder().addConst('x', 1, { kind: 'scalar', dtype: 'float32' });
        const ir1 = builder.build();
        const ir2 = builder.build();
        expect(ir1.instructions).toHaveLength(ir2.instructions.length);
    });
});

// ------------------------------------------------------------------ //
// IROptimizer
// ------------------------------------------------------------------ //

describe('IROptimizer', () => {
    test('no-ops on an already-empty IR', () => {
        const ir   = new IRBuilder().build();
        const opt  = new IROptimizer();
        const optIR = opt.optimize(ir);
        expect(optIR.instructions).toHaveLength(0);
    });

    test('constant folding folds ⊕ of two constants', () => {
        const ir = new IRBuilder()
            .addConst('a', 3, { kind: 'scalar', dtype: 'float32' })
            .addConst('b', 4, { kind: 'scalar', dtype: 'float32' })
            .addGlyphOp('r', '⊕', 'a', 'b', { kind: 'scalar', dtype: 'float32' })
            .addReturn('ret0', 'r')   // keep 'r' live so DCE doesn't discard it
            .build();
        const opt   = new IROptimizer();
        const optIR = opt.optimize(ir);
        const folded = optIR.instructions.find(i => i.id === 'r');
        expect(folded?.op).toBe('const');
        expect((folded as any).value).toBe(7);
    });

    test('dead-code elimination removes unused consts', () => {
        const ir = new IRBuilder()
            .addConst('unused', 999, { kind: 'scalar', dtype: 'float32' })
            .addConst('used',   1,   { kind: 'scalar', dtype: 'float32' })
            .addReturn('ret', 'used')
            .build();
        const opt   = new IROptimizer();
        const optIR = opt.optimize(ir);
        const unusedInstr = optIR.instructions.find(i => i.id === 'unused');
        expect(unusedInstr).toBeUndefined();
    });

    test('preserves the IR manifold', () => {
        const builder = new IRBuilder().setManifold(3, 'riemannian');
        const ir   = builder.build();
        const opt  = new IROptimizer();
        const optIR = opt.optimize(ir);
        expect(optIR.manifold.metric).toBe('riemannian');
    });
});

// ------------------------------------------------------------------ //
// IRVerifier
// ------------------------------------------------------------------ //

describe('IRVerifier', () => {
    test('empty IR passes verification with a warning', () => {
        const ir  = new IRBuilder().build();
        const res = new IRVerifier().verify(ir);
        expect(res.valid).toBe(true);
        expect(res.warnings.some(w => w.includes('no instructions'))).toBe(true);
    });

    test('well-formed IR passes verification', () => {
        const ir = new IRBuilder()
            .beginPhase('main', 0, 2 * Math.PI)
            .addConst('a', 1, { kind: 'scalar', dtype: 'float32' })
            .addConst('b', 2, { kind: 'scalar', dtype: 'float32' })
            .addGlyphOp('r', '⊗', 'a', 'b', { kind: 'scalar', dtype: 'float32' })
            .addReturn('ret0', 'r')
            .endPhase()
            .build();
        const res = new IRVerifier().verify(ir);
        expect(res.valid).toBe(true);
        expect(res.errors).toHaveLength(0);
    });

    test('undefined operand produces an error', () => {
        const ir = new IRBuilder()
            .addGlyphOp('r', '⊗', 'noSuchId', 'alsoMissing', { kind: 'scalar', dtype: 'float32' })
            .build();
        const res = new IRVerifier().verify(ir);
        expect(res.valid).toBe(false);
        expect(res.errors.length).toBeGreaterThan(0);
    });

    test('invalid manifold dimensions produce an error', () => {
        const builder = new IRBuilder();
        (builder as any).manifold.dimensions = 0;
        const ir  = builder.build();
        const res = new IRVerifier().verify(ir);
        expect(res.valid).toBe(false);
        expect(res.errors.some(e => e.includes('dimensions'))).toBe(true);
    });

    test('phase out of [0, 2π] produces a warning', () => {
        const builder = new IRBuilder();
        (builder as any).manifold.phase = 10;
        const ir  = builder.build();
        const res = new IRVerifier().verify(ir);
        expect(res.warnings.some(w => w.includes('phase'))).toBe(true);
    });
});
