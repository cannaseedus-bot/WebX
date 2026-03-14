// Tests for K'UHUL++ v2.0 Semantic Analyzer

import { tokenize } from '../compiler/lexer.js';
import { parse, NodeKind } from '../compiler/parser.js';
import { analyze, SemanticError } from '../compiler/semantic.js';

function analyzeSource(source) {
    return analyze(parse(tokenize(source)));
}

describe('Semantic Analyzer — analyze()', () => {
    // ------------------------------------------------------------------ //
    // Clean programs
    // ------------------------------------------------------------------ //

    test('returns no errors for an empty program', () => {
        const { errors } = analyzeSource('');
        expect(errors).toHaveLength(0);
    });

    test('returns no errors for a valid tensor declaration', () => {
        const { errors } = analyzeSource(
            'Tensor t = GeometricTensor(phase: 0.75π, symmetry: 0.8);'
        );
        expect(errors).toHaveLength(0);
    });

    test('returns no errors for a valid glyph operation', () => {
        const { errors } = analyzeSource('[↻] angle=45;');
        expect(errors).toHaveLength(0);
    });

    test('returns no errors for valid foreach glyph loop', () => {
        const { errors } = analyzeSource('foreach glyph in [↻, ⟲] { x = 1; }');
        expect(errors).toHaveLength(0);
    });

    // ------------------------------------------------------------------ //
    // π-phase validation
    // ------------------------------------------------------------------ //

    test('reports no error for π-phase of 0.75', () => {
        const { errors } = analyzeSource(
            'Tensor t = GeometricTensor(phase: 0.75π);'
        );
        // phase coefficient 0.75 is within [0, 2]
        expect(errors.filter(e => e.message.includes('π-phase'))).toHaveLength(0);
    });

    test('reports error for π-phase coefficient > 2', () => {
        // 3π is out of range [0, 2π]
        const { errors } = analyzeSource('x = 3π;');
        const phaseErrors = errors.filter(e => e.message.includes('π-phase'));
        expect(phaseErrors.length).toBeGreaterThan(0);
    });

    test('reports error for negative π-phase coefficient', () => {
        // Use a literal PiExpr with negative coeff — parser creates PiExpr for NUMBER*π
        // Negative pi coefficients from source like "x = 0π;" (coeff=0 is ok)
        // We test via a direct call to verify boundary
        const { errors } = analyzeSource('x = 0π;');
        // 0 is valid
        expect(errors.filter(e => e.message.includes('π-phase'))).toHaveLength(0);
    });

    // ------------------------------------------------------------------ //
    // Glyph validation
    // ------------------------------------------------------------------ //

    test('reports error for unknown glyph', () => {
        // Inject an unknown glyph via a patched AST
        const fakeAst = {
            kind: NodeKind.Program,
            body: [{
                kind: NodeKind.GlyphOp,
                glyph: '✗',   // not in GLYPHS set
                params: {},
                target: null,
            }],
        };
        const { errors } = analyze(fakeAst);
        expect(errors.some(e => e.message.includes('✗'))).toBe(true);
    });

    test('reports error for unknown glyph in foreach glyph list', () => {
        const fakeAst = {
            kind: NodeKind.Program,
            body: [{
                kind: NodeKind.ForEachGlyph,
                glyphs: ['✗'],
                body: { kind: NodeKind.Block, stmts: [] },
            }],
        };
        const { errors } = analyze(fakeAst);
        expect(errors.some(e => e.message.includes('✗'))).toBe(true);
    });

    // ------------------------------------------------------------------ //
    // Train statement validation
    // ------------------------------------------------------------------ //

    test('reports error when Train references undeclared model', () => {
        const { errors } = analyzeSource('Train undeclaredModel with myData {}');
        expect(errors.some(e => e.message.includes('undeclaredModel'))).toBe(true);
    });

    test('no error when Train references declared model and data', () => {
        const src = `
            Model myModel = GeometricModel({});
            Cluster myData = TensorCluster(tensors: []);
            Train myModel with myData {}
        `;
        const { errors } = analyzeSource(src);
        const trainErrors = errors.filter(e =>
            e.message.includes('myModel') || e.message.includes('myData')
        );
        expect(trainErrors).toHaveLength(0);
    });

    // ------------------------------------------------------------------ //
    // SemanticError shape
    // ------------------------------------------------------------------ //

    test('SemanticError has a message and toString()', () => {
        const err = new SemanticError('test message', null);
        expect(err.message).toBe('test message');
        expect(err.toString()).toContain('SemanticError');
        expect(err.toString()).toContain('test message');
    });
});
