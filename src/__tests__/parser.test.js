// Tests for K'UHUL++ v2.0 Parser

import { tokenize } from '../compiler/lexer.js';
import { parse, NodeKind, ParseError } from '../compiler/parser.js';

function ast(source) {
    return parse(tokenize(source));
}

describe('Parser — parse()', () => {
    // ------------------------------------------------------------------ //
    // Program root
    // ------------------------------------------------------------------ //

    test('returns a Program node', () => {
        const tree = ast('');
        expect(tree.kind).toBe(NodeKind.Program);
        expect(Array.isArray(tree.body)).toBe(true);
    });

    // ------------------------------------------------------------------ //
    // Declarations
    // ------------------------------------------------------------------ //

    test('parses Tensor declaration', () => {
        const tree = ast('Tensor t = GeometricTensor(phase: 0.75π);');
        expect(tree.body).toHaveLength(1);
        const decl = tree.body[0];
        expect(decl.kind).toBe(NodeKind.TensorDecl);
        expect(decl.name).toBe('t');
        expect(decl.init.kind).toBe(NodeKind.FunctionCall);
        expect(decl.init.callee).toBe('GeometricTensor');
    });

    test('parses Cluster declaration', () => {
        const tree = ast('Cluster c = TensorCluster(tensors: [t1, t2]);');
        const decl = tree.body[0];
        expect(decl.kind).toBe(NodeKind.ClusterDecl);
        expect(decl.name).toBe('c');
    });

    test('parses Model declaration', () => {
        const tree = ast('Model m = GeometricModel({type: classification});');
        const decl = tree.body[0];
        expect(decl.kind).toBe(NodeKind.ModelDecl);
        expect(decl.name).toBe('m');
    });

    // ------------------------------------------------------------------ //
    // Assignments
    // ------------------------------------------------------------------ //

    test('parses variable assignment', () => {
        const tree = ast('x = 42;');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.Assignment);
        expect(stmt.target.name).toBe('x');
        expect(stmt.value.value).toBe(42);
    });

    // ------------------------------------------------------------------ //
    // Glyph operations
    // ------------------------------------------------------------------ //

    test('parses glyph operation with params', () => {
        const tree = ast('[↻] spiral angle=45;');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.GlyphOp);
        expect(stmt.glyph).toBe('↻');
        expect(stmt.params.angle).toBeDefined();
        expect(stmt.params.angle.value).toBe(45);
    });

    test('parses glyph operation with target and member', () => {
        const tree = ast('[⟲] galaxy.radius=2.5 degrees=360;');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.GlyphOp);
        expect(stmt.glyph).toBe('⟲');
    });

    // ------------------------------------------------------------------ //
    // Native DX12 block
    // ------------------------------------------------------------------ //

    test('parses dx12 native block', () => {
        const src = `
dx12 {
    RootSignature mainRS {
        CBV(0);
        SRV(0);
    }
}`;
        const tree = ast(src);
        const block = tree.body[0];
        expect(block.kind).toBe(NodeKind.NativeBlock);
        expect(block.stmts).toHaveLength(1);
        // The RootSignature statement has kind 'DX12Stmt' with a 'kind' field of 'RootSignature'
        const rsStmt = block.stmts[0];
        expect(rsStmt.kind).toBe(NodeKind.DX12Stmt);
        expect(rsStmt.name).toBe('mainRS');
    });

    // ------------------------------------------------------------------ //
    // Control flow
    // ------------------------------------------------------------------ //

    test('parses if/else statement', () => {
        const tree = ast('if (x > 0) { y = 1; } else { y = -1; }');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.IfStmt);
        expect(stmt.condition).toBeDefined();
        expect(stmt.consequent.kind).toBe(NodeKind.Block);
        expect(stmt.alternate.kind).toBe(NodeKind.Block);
    });

    test('parses for loop', () => {
        const tree = ast('for (i in [0, 10]) { x = i; }');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.ForStmt);
        expect(stmt.ident).toBe('i');
        expect(stmt.body.kind).toBe(NodeKind.Block);
    });

    test('parses while loop', () => {
        const tree = ast('while (running) { x = 1; }');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.WhileStmt);
    });

    test('parses foreach glyph loop', () => {
        const tree = ast('foreach glyph in [↻, ⟲] { x = 1; }');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.ForEachGlyph);
        expect(stmt.glyphs).toContain('↻');
        expect(stmt.glyphs).toContain('⟲');
    });

    test('parses parallel for loop', () => {
        const tree = ast('parallel for (i in [0, 3]) { x = i; }');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.ParallelFor);
    });

    // ------------------------------------------------------------------ //
    // Expressions
    // ------------------------------------------------------------------ //

    test('parses binary expression', () => {
        const tree = ast('x = a + b;');
        const assign = tree.body[0];
        expect(assign.value.kind).toBe(NodeKind.BinaryExpr);
        expect(assign.value.op).toBe('+');
    });

    test('parses member expression', () => {
        const tree = ast('x = obj.property;');
        const assign = tree.body[0];
        expect(assign.value.kind).toBe(NodeKind.MemberExpr);
        expect(assign.value.member).toBe('property');
    });

    test('parses array literal', () => {
        const tree = ast('x = [1, 2, 3];');
        const assign = tree.body[0];
        expect(assign.value.kind).toBe(NodeKind.ArrayLiteral);
        expect(assign.value.elements).toHaveLength(3);
    });

    test('parses Vector3 literal', () => {
        const tree = ast('x = (1, 2, 3);');
        const assign = tree.body[0];
        expect(assign.value.kind).toBe(NodeKind.Vector3Literal);
        expect(assign.value.x.value).toBe(1);
        expect(assign.value.y.value).toBe(2);
        expect(assign.value.z.value).toBe(3);
    });

    test('parses π expression', () => {
        const tree = ast('x = 0.75π;');
        const assign = tree.body[0];
        expect(assign.value.kind).toBe(NodeKind.PiExpr);
        expect(assign.value.coefficient).toBeCloseTo(0.75, 5);
    });

    test('parses function call with named args', () => {
        const tree = ast('result = GeometricTensor(phase: 0.5π, symmetry: 0.8);');
        const assign = tree.body[0];
        expect(assign.value.kind).toBe(NodeKind.FunctionCall);
        expect(assign.value.callee).toBe('GeometricTensor');
        expect(assign.value.args[0].key).toBe('phase');
        expect(assign.value.args[1].key).toBe('symmetry');
    });

    // ------------------------------------------------------------------ //
    // Train statement
    // ------------------------------------------------------------------ //

    test('parses Train statement', () => {
        const tree = ast('Train myModel with myData { epochs: 100; }');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.TrainStmt);
        expect(stmt.model).toBe('myModel');
        expect(stmt.data).toBe('myData');
    });

    // ------------------------------------------------------------------ //
    // GPU method call
    // ------------------------------------------------------------------ //

    test('parses GPU.Dispatch call', () => {
        const tree = ast('GPU.Dispatch(16, 16, 1);');
        const stmt = tree.body[0];
        expect(stmt.kind).toBe(NodeKind.MethodCall);
        expect(stmt.method).toBe('Dispatch');
    });
});
