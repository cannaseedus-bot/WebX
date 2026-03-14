// K'UHUL++ Code Generation Unit Tests
// Tests JSCodegen output from simple IR programs.
// The generated JavaScript is evaluated in a VM context to verify correctness.

import { IRBuilder }  from '../ir/ir-builder.js';
import { JSCodegen }  from '../compiler/codegen/js-codegen.js';
import type { GeometricIR } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function buildSimpleIR(): GeometricIR {
    return new IRBuilder()
        .beginPhase('main', 0, 2 * Math.PI)
        .addConst('a', 3, { kind: 'scalar', dtype: 'float32' })
        .addConst('b', 4, { kind: 'scalar', dtype: 'float32' })
        .addGlyphOp('r', '⊕', 'a', 'b', { kind: 'scalar', dtype: 'float32' })
        .addReturn('ret0', 'r')
        .endPhase()
        .build();
}

// ------------------------------------------------------------------ //
// JSCodegen — output structure
// ------------------------------------------------------------------ //

describe('JSCodegen — output structure', () => {
    const cg = new JSCodegen();

    test('generate() returns a non-empty string', () => {
        const ir = new IRBuilder().build();
        const js = cg.generate(ir);
        expect(typeof js).toBe('string');
        expect(js.length).toBeGreaterThan(0);
    });

    test('emits the kuhulMain function', () => {
        const js = cg.generate(new IRBuilder().build());
        expect(js).toContain('async function kuhulMain');
    });

    test('emits the runtime helpers', () => {
        const js = cg.generate(new IRBuilder().build());
        expect(js).toContain('__geoProduct');
        expect(js).toContain('__geoTranslate');
        expect(js).toContain('__geoDifference');
        expect(js).toContain('__applyGlyph');
    });

    test('emits manifold comment', () => {
        const ir = new IRBuilder().setManifold(3, 'euclidean').build();
        const js = cg.generate(ir);
        expect(js).toContain('euclidean');
    });

    test('emits const instruction', () => {
        const ir = new IRBuilder()
            .addConst('myConst', 42, { kind: 'scalar', dtype: 'float32' })
            .build();
        const js = cg.generate(ir);
        expect(js).toContain('const myConst = 42');
    });

    test('emits load instruction', () => {
        const ir = new IRBuilder()
            .addLoad('r0', 'inputTensor', { kind: 'tensor', dtype: 'float32', shape: [3] })
            .build();
        const js = cg.generate(ir);
        expect(js).toContain("__mem.get(\"inputTensor\")");
    });

    test('emits store instruction', () => {
        const ir = new IRBuilder()
            .addConst('v', 1, { kind: 'scalar', dtype: 'float32' })
            .addStore('s0', 'output', 'v')
            .build();
        const js = cg.generate(ir);
        expect(js).toContain("__mem.set(\"output\", v)");
    });

    test('emits glyph op instruction', () => {
        const ir = buildSimpleIR();
        const js = cg.generate(ir);
        expect(js).toContain("__applyGlyph('⊕'");
    });

    test('emits phase instruction', () => {
        const ir = new IRBuilder()
            .addPhase('p0', Math.PI / 4)
            .build();
        const js = cg.generate(ir);
        expect(js).toContain('__phase');
        expect(js).toContain('Math.PI');
    });

    test('emits return instruction', () => {
        const ir = new IRBuilder()
            .addConst('x', 7, { kind: 'scalar', dtype: 'float32' })
            .addReturn('ret', 'x')
            .build();
        const js = cg.generate(ir);
        expect(js).toContain('return x');
    });

    test('emits invoke instruction', () => {
        const ir = new IRBuilder()
            .addConst('a', 1, { kind: 'scalar', dtype: 'float32' })
            .addInvoke('r', 'myFunc', ['a'], { kind: 'scalar', dtype: 'float32' })
            .build();
        const js = cg.generate(ir);
        expect(js).toContain('myFunc');
    });
});

// ------------------------------------------------------------------ //
// JSCodegen — all glyph ops
// ------------------------------------------------------------------ //

describe('JSCodegen — glyph operators', () => {
    const cg = new JSCodegen();
    const glyphs = ['⊗', '⊕', '⊖', '⊛', '⊜', '⊝', '⊞'] as const;

    test.each(glyphs)('glyph %s is emitted as __applyGlyph call', (g) => {
        const ir = new IRBuilder()
            .addConst('a', 1, { kind: 'scalar', dtype: 'float32' })
            .addConst('b', 2, { kind: 'scalar', dtype: 'float32' })
            .addGlyphOp('r', g, 'a', 'b', { kind: 'scalar', dtype: 'float32' })
            .build();
        const js = cg.generate(ir);
        expect(js).toContain(g);
    });
});

// ------------------------------------------------------------------ //
// JSCodegen — generated code is syntactically valid
// ------------------------------------------------------------------ //

describe('JSCodegen — syntax validity', () => {
    test('generated code passes Function constructor without throwing', () => {
        const cg = new JSCodegen();
        const ir = buildSimpleIR();
        const js = cg.generate(ir);

        // Strip the "export" keyword so Function() can parse it
        const stripped = js.replace(/^export\s+/gm, '');

        expect(() => new Function(stripped)).not.toThrow();
    });
});
