// Tests for K'UHUL++ v2.0 Lexer

import { tokenize, TokenType, GLYPHS, KEYWORDS, LexerError } from '../compiler/lexer.js';

describe('Lexer — tokenize()', () => {
    // ------------------------------------------------------------------ //
    // Basic tokens
    // ------------------------------------------------------------------ //

    test('tokenizes a simple number', () => {
        const tokens = tokenize('42');
        expect(tokens[0]).toMatchObject({ type: TokenType.NUMBER, value: 42 });
        expect(tokens[1].type).toBe(TokenType.EOF);
    });

    test('tokenizes a float', () => {
        const tokens = tokenize('3.14');
        expect(tokens[0]).toMatchObject({ type: TokenType.NUMBER, value: 3.14 });
    });

    test('tokenizes a string literal', () => {
        const tokens = tokenize('"hello world"');
        expect(tokens[0]).toMatchObject({ type: TokenType.STRING, value: 'hello world' });
    });

    test('tokenizes an identifier', () => {
        const tokens = tokenize('myTensor');
        expect(tokens[0]).toMatchObject({ type: TokenType.IDENTIFIER, value: 'myTensor' });
    });

    test('recognises keywords', () => {
        const tokens = tokenize('Tensor Cluster dx12');
        expect(tokens[0]).toMatchObject({ type: TokenType.KEYWORD, value: 'Tensor' });
        expect(tokens[1]).toMatchObject({ type: TokenType.KEYWORD, value: 'Cluster' });
        expect(tokens[2]).toMatchObject({ type: TokenType.KEYWORD, value: 'dx12' });
    });

    // ------------------------------------------------------------------ //
    // π expressions
    // ------------------------------------------------------------------ //

    test('tokenizes bare π as PI token', () => {
        const tokens = tokenize('π');
        expect(tokens[0]).toMatchObject({ type: TokenType.PI });
        expect(tokens[0].value).toBeCloseTo(Math.PI, 5);
    });

    test('tokenizes NUMBER followed by π as PI_EXPR', () => {
        const tokens = tokenize('0.75π');
        expect(tokens[0]).toMatchObject({ type: TokenType.PI_EXPR, value: 0.75 });
    });

    test('tokenizes NUMBER*π as PI_EXPR', () => {
        const tokens = tokenize('2*π');
        expect(tokens[0]).toMatchObject({ type: TokenType.PI_EXPR, value: 2 });
    });

    // ------------------------------------------------------------------ //
    // Glyph tokens
    // ------------------------------------------------------------------ //

    test('tokenizes ASC cipher glyph ⤍', () => {
        const tokens = tokenize('⤍');
        expect(tokens[0]).toMatchObject({ type: TokenType.GLYPH, value: '⤍' });
    });

    test('tokenizes SCX compression glyph ↻', () => {
        const tokens = tokenize('↻');
        expect(tokens[0]).toMatchObject({ type: TokenType.GLYPH, value: '↻' });
    });

    test('tokenizes 3D control glyph ⟲', () => {
        const tokens = tokenize('⟲');
        expect(tokens[0]).toMatchObject({ type: TokenType.GLYPH, value: '⟲' });
    });

    test('tokenizes neural glyph ⟿', () => {
        const tokens = tokenize('⟿');
        expect(tokens[0]).toMatchObject({ type: TokenType.GLYPH, value: '⟿' });
    });

    // ------------------------------------------------------------------ //
    // Punctuation
    // ------------------------------------------------------------------ //

    test('tokenizes punctuation correctly', () => {
        const source = '= ; : , . ( ) { } [ ] < > * + - / @ |';
        const types = [
            TokenType.EQUALS, TokenType.SEMICOLON, TokenType.COLON, TokenType.COMMA,
            TokenType.DOT, TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE,
            TokenType.RBRACE, TokenType.LBRACKET, TokenType.RBRACKET,
            TokenType.LANGLE, TokenType.RANGLE, TokenType.STAR, TokenType.PLUS,
            TokenType.MINUS, TokenType.SLASH, TokenType.AT, TokenType.PIPE,
        ];
        const tokens = tokenize(source);
        types.forEach((expectedType, i) => {
            expect(tokens[i].type).toBe(expectedType);
        });
    });

    // ------------------------------------------------------------------ //
    // Comments
    // ------------------------------------------------------------------ //

    test('skips single-line comments', () => {
        const tokens = tokenize('42 // this is a comment\n99');
        expect(tokens[0]).toMatchObject({ type: TokenType.NUMBER, value: 42 });
        expect(tokens[1]).toMatchObject({ type: TokenType.NUMBER, value: 99 });
    });

    test('skips block comments', () => {
        const tokens = tokenize('1 /* block comment */ 2');
        expect(tokens[0]).toMatchObject({ type: TokenType.NUMBER, value: 1 });
        expect(tokens[1]).toMatchObject({ type: TokenType.NUMBER, value: 2 });
    });

    // ------------------------------------------------------------------ //
    // Line / column tracking
    // ------------------------------------------------------------------ //

    test('tracks line numbers correctly', () => {
        const tokens = tokenize('a\nb\nc');
        expect(tokens[0].line).toBe(1);
        expect(tokens[1].line).toBe(2);
        expect(tokens[2].line).toBe(3);
    });

    // ------------------------------------------------------------------ //
    // Error cases
    // ------------------------------------------------------------------ //

    test('throws LexerError on unterminated string', () => {
        expect(() => tokenize('"unterminated')).toThrow(LexerError);
    });

    test('throws LexerError on unterminated block comment', () => {
        expect(() => tokenize('/* no end')).toThrow(LexerError);
    });

    // ------------------------------------------------------------------ //
    // Full declaration snippet
    // ------------------------------------------------------------------ //

    test('tokenizes a Tensor declaration', () => {
        const src = 'Tensor t = GeometricTensor(phase: 0.75π, symmetry: 0.8);';
        const tokens = tokenize(src);
        expect(tokens[0]).toMatchObject({ type: TokenType.KEYWORD, value: 'Tensor' });
        expect(tokens[1]).toMatchObject({ type: TokenType.IDENTIFIER, value: 't' });
        expect(tokens[2]).toMatchObject({ type: TokenType.EQUALS });
        // Find the PI_EXPR token
        const piExpr = tokens.find(t => t.type === TokenType.PI_EXPR);
        expect(piExpr).toBeDefined();
        expect(piExpr.value).toBeCloseTo(0.75, 5);
    });
});
