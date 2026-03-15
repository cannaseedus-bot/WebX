// K'UHUL++ Semantic Analyzer Unit Tests
// Tests symbol resolution, type checking, glyph validation,
// and π-phase range warnings.

import { tokenize } from '../compiler/lexer.js';
import { parse }    from '../compiler/parser.js';
import { analyze, SemanticError } from '../compiler/semantic-analyzer.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function analyzeSource(source: string) {
    const tokens = tokenize(source);
    const ast    = parse(tokens);
    return analyze(ast);
}

// ------------------------------------------------------------------ //
// Basic analysis
// ------------------------------------------------------------------ //

describe('analyze — basic', () => {
    test('empty program has no errors', () => {
        const { errors } = analyzeSource('');
        expect(errors).toHaveLength(0);
    });

    test('returns annotated AST', () => {
        const { ast } = analyzeSource('Tensor x = 1;');
        expect(ast).toBeDefined();
        expect(ast.kind).toBe('Program');
        expect(ast.typeAnnotations).toBeInstanceOf(Map);
    });

    test('simple Tensor declaration has no errors', () => {
        const { errors } = analyzeSource('Tensor v = 42;');
        expect(errors).toHaveLength(0);
    });
});

// ------------------------------------------------------------------ //
// Symbol resolution
// ------------------------------------------------------------------ //

describe('analyze — symbol resolution', () => {
    test('declared identifier resolves without error', () => {
        const { errors } = analyzeSource(`
            Tensor a = 1;
            Tensor b = a;
        `);
        expect(errors.every(e => !e.message.includes('"a"'))).toBe(true);
    });

    test('undeclared identifier produces error', () => {
        const { errors } = analyzeSource('Tensor b = undeclaredVar;');
        expect(errors.some(e => e.message.includes('undeclaredVar'))).toBe(true);
    });

    test('built-in GPU identifier does not produce error', () => {
        const { errors } = analyzeSource('Tensor g = GPU;');
        expect(errors.every(e => !e.message.includes('"GPU"'))).toBe(true);
    });

    test('built-in Math does not produce error', () => {
        const { errors } = analyzeSource('Math.sqrt(4);');
        // Math is a built-in, should not error on it
        expect(errors.every(e => !e.message.includes('"Math"'))).toBe(true);
    });

    test('Cluster declares its name in outer scope', () => {
        const { errors } = analyzeSource(`
            Cluster weights {}
            Tensor x = weights;
        `);
        // 'weights' should be declared by the cluster
        expect(errors.every(e => !e.message.includes('"weights"'))).toBe(true);
    });
});

// ------------------------------------------------------------------ //
// Glyph validation
// ------------------------------------------------------------------ //

describe('analyze — glyph validation', () => {
    test('recognised glyph produces no error', () => {
        const { errors } = analyzeSource(`
            Tensor a = 1;
            Tensor b = 2;
            a ⊗ b;
        `);
        const glyphErrors = errors.filter(e => e.message.includes('glyph'));
        expect(glyphErrors).toHaveLength(0);
    });

    test('multiple recognised glyphs produce no errors', () => {
        const { errors } = analyzeSource(`
            Tensor x = 1;
            Tensor y = 2;
            x ⊕ y;
            x ⊖ y;
            x ⊛ y;
        `);
        const glyphErrors = errors.filter(e => e.message.toLowerCase().includes('unknown glyph'));
        expect(glyphErrors).toHaveLength(0);
    });
});

// ------------------------------------------------------------------ //
// π-phase range warnings
// ------------------------------------------------------------------ //

describe('analyze — π-phase warnings', () => {
    test('coefficient in [0,2] produces no warning', () => {
        const { warnings } = analyzeSource('Tensor a = 1.5π;');
        expect(warnings.filter(w => w.includes('coefficient'))).toHaveLength(0);
    });

    test('coefficient outside [0,2] produces a warning', () => {
        const { warnings } = analyzeSource('Tensor a = 3π;');
        expect(warnings.some(w => w.includes('3'))).toBe(true);
    });
});

// ------------------------------------------------------------------ //
// Type annotations
// ------------------------------------------------------------------ //

describe('analyze — type annotations', () => {
    test('NumberLiteral is annotated as number', () => {
        const { ast } = analyzeSource('Tensor x = 42;');
        // Find the TensorDecl node
        const decl = ast.body[0] as any;
        const initType = ast.typeAnnotations.get(decl.init);
        expect(initType).toBe('number');
    });

    test('StringLiteral is annotated as string', () => {
        const { ast } = analyzeSource('Tensor s = "hello";');
        const decl = ast.body[0] as any;
        const initType = ast.typeAnnotations.get(decl.init);
        expect(initType).toBe('string');
    });

    test('ArrayLiteral is annotated as array', () => {
        const { ast } = analyzeSource('Tensor v = [1, 2, 3];');
        const decl = ast.body[0] as any;
        const initType = ast.typeAnnotations.get(decl.init);
        expect(initType).toBe('array');
    });
});

// ------------------------------------------------------------------ //
// SemanticError class
// ------------------------------------------------------------------ //

describe('SemanticError', () => {
    test('toString includes message', () => {
        const err = new SemanticError('test error');
        expect(err.toString()).toContain('test error');
    });

    test('toString includes line if node has line', () => {
        const fakeNode = { kind: 'Identifier' as any, line: 5, col: 3 };
        const err = new SemanticError('foo', fakeNode);
        expect(err.toString()).toContain('5');
    });
});
