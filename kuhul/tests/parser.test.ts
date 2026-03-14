// K'UHUL++ Parser Unit Tests
// Tests parsing of TensorDecl, GlyphOp, and basic program structure.

import { tokenize }           from '../compiler/lexer.js';
import { parse, NodeKind, ParseError } from '../compiler/parser.js';

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function parseSource(source: string) {
    return parse(tokenize(source));
}

// ------------------------------------------------------------------ //
// Program structure
// ------------------------------------------------------------------ //

describe('parse — program structure', () => {
    test('empty source → empty Program', () => {
        const ast = parseSource('');
        expect(ast.kind).toBe(NodeKind.Program);
        expect(ast.body).toHaveLength(0);
    });

    test('multiple declarations in body', () => {
        const ast = parseSource(`
            Tensor a = 1;
            Tensor b = 2;
            Tensor c = 3;
        `);
        expect(ast.body).toHaveLength(3);
    });
});

// ------------------------------------------------------------------ //
// TensorDecl
// ------------------------------------------------------------------ //

describe('parse — TensorDecl', () => {
    test('simple number initializer', () => {
        const ast = parseSource('Tensor x = 42;');
        expect(ast.body).toHaveLength(1);
        const decl = ast.body[0] as any;
        expect(decl.kind).toBe(NodeKind.TensorDecl);
        expect(decl.name).toBe('x');
        expect(decl.init.kind).toBe(NodeKind.NumberLiteral);
        expect(decl.init.value).toBe(42);
    });

    test('string initializer', () => {
        const ast = parseSource('Tensor msg = "hello";');
        const decl = ast.body[0] as any;
        expect(decl.init.kind).toBe(NodeKind.StringLiteral);
        expect(decl.init.value).toBe('hello');
    });

    test('π initializer', () => {
        const ast = parseSource('Tensor angle = π;');
        const decl = ast.body[0] as any;
        expect(decl.init.kind).toBe(NodeKind.PiExpr);
        expect(decl.init.coefficient).toBe(1);
    });

    test('π-coefficient initializer', () => {
        const ast = parseSource('Tensor angle = 0.5π;');
        const decl = ast.body[0] as any;
        expect(decl.init.kind).toBe(NodeKind.PiExpr);
        expect(decl.init.coefficient).toBe(0.5);
    });

    test('array initializer', () => {
        const ast = parseSource('Tensor v = [1, 2, 3];');
        const decl = ast.body[0] as any;
        expect(decl.init.kind).toBe(NodeKind.ArrayLiteral);
        expect(decl.init.elements).toHaveLength(3);
    });

    test('function call initializer', () => {
        const ast = parseSource('Tensor data = generate_spiral(100);');
        const decl = ast.body[0] as any;
        expect(decl.init.kind).toBe(NodeKind.FunctionCall);
        expect(decl.init.callee).toBe('generate_spiral');
        expect(decl.init.args).toHaveLength(1);
    });
});

// ------------------------------------------------------------------ //
// GlyphOp
// ------------------------------------------------------------------ //

describe('parse — GlyphOp', () => {
    test('simple glyph expression statement', () => {
        const ast = parseSource('a ⊗ b;');
        expect(ast.body).toHaveLength(1);
        const op = ast.body[0] as any;
        expect(op.kind).toBe(NodeKind.GlyphOp);
        expect(op.glyph).toBe('⊗');
        expect(op.left.name).toBe('a');
        expect(op.right.name).toBe('b');
    });

    test('glyph op assigned to tensor', () => {
        const ast = parseSource('Tensor out = a ⊕ b;');
        const decl = ast.body[0] as any;
        expect(decl.kind).toBe(NodeKind.TensorDecl);
        expect(decl.init.kind).toBe(NodeKind.GlyphOp);
        expect(decl.init.glyph).toBe('⊕');
    });

    test('chained glyph ops', () => {
        const ast = parseSource('a ⊗ b ⊕ c;');
        const stmt = ast.body[0] as any;
        // The last glyph wraps the previous
        expect(stmt.kind).toBe(NodeKind.GlyphOp);
    });

    test('all geometric glyph operators parse', () => {
        const glyphs = ['⊗', '⊕', '⊖', '⊛', '⊞'];
        for (const g of glyphs) {
            const ast = parseSource(`a ${g} b;`);
            const op = ast.body[0] as any;
            expect(op.kind).toBe(NodeKind.GlyphOp);
            expect(op.glyph).toBe(g);
        }
    });
});

// ------------------------------------------------------------------ //
// ClusterDecl / ModelDecl / PipelineDecl
// ------------------------------------------------------------------ //

describe('parse — structural declarations', () => {
    test('Cluster with body', () => {
        const ast = parseSource(`
            Cluster weights {
                Tensor w1 = [1, 2];
            }
        `);
        const cluster = ast.body[0] as any;
        expect(cluster.kind).toBe(NodeKind.ClusterDecl);
        expect(cluster.name).toBe('weights');
        expect(cluster.body).toHaveLength(1);
    });

    test('Model declaration', () => {
        const ast = parseSource('Model myNet {}');
        const model = ast.body[0] as any;
        expect(model.kind).toBe(NodeKind.ModelDecl);
        expect(model.name).toBe('myNet');
    });

    test('Pipeline declaration', () => {
        const ast = parseSource('Pipeline p {}');
        const pipeline = ast.body[0] as any;
        expect(pipeline.kind).toBe(NodeKind.PipelineDecl);
        expect(pipeline.name).toBe('p');
    });
});

// ------------------------------------------------------------------ //
// Control flow
// ------------------------------------------------------------------ //

describe('parse — control flow', () => {
    test('if statement with else', () => {
        const ast = parseSource(`
            if (x) { Tensor a = 1; } else { Tensor b = 2; }
        `);
        const stmt = ast.body[0] as any;
        expect(stmt.kind).toBe(NodeKind.IfStmt);
        expect(stmt.alternate).toBeTruthy();
    });

    test('while statement', () => {
        const ast = parseSource('while (running) {}');
        const stmt = ast.body[0] as any;
        expect(stmt.kind).toBe(NodeKind.WhileStmt);
    });

    test('return statement', () => {
        const ast = parseSource('return result;');
        const stmt = ast.body[0] as any;
        expect(stmt.kind).toBe(NodeKind.ReturnStmt);
    });
});

// ------------------------------------------------------------------ //
// Expressions
// ------------------------------------------------------------------ //

describe('parse — expressions', () => {
    test('binary arithmetic', () => {
        const ast = parseSource('Tensor r = a + b;');
        const decl = ast.body[0] as any;
        expect(decl.init.kind).toBe(NodeKind.BinaryExpr);
        expect(decl.init.op).toBe('+');
    });

    test('method call', () => {
        const ast = parseSource('model.forward(x);');
        const stmt = ast.body[0] as any;
        expect(stmt.kind).toBe(NodeKind.MethodCall);
        expect(stmt.method).toBe('forward');
    });

    test('member access', () => {
        const ast = parseSource('Tensor s = cluster.weights;');
        const decl = ast.body[0] as any;
        expect(decl.init.kind).toBe(NodeKind.MemberExpr);
    });
});
