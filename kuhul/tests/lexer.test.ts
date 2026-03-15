// K'UHUL++ Lexer Unit Tests
// Tests tokenisation of keywords, glyphs, numbers, strings, and π expressions.
// Import from the TypeScript lexer (compiled to .js for Jest).

import { tokenize, TokenType, KEYWORDS, GLYPHS, LexerError } from '../compiler/lexer.js';

// ------------------------------------------------------------------ //
// TokenType enum / KEYWORDS / GLYPHS exports
// ------------------------------------------------------------------ //

describe('TokenType', () => {
    test('has required token kinds', () => {
        expect(TokenType.NUMBER).toBe('NUMBER');
        expect(TokenType.STRING).toBe('STRING');
        expect(TokenType.KEYWORD).toBe('KEYWORD');
        expect(TokenType.GLYPH).toBe('GLYPH');
        expect(TokenType.PI).toBe('PI');
        expect(TokenType.PI_EXPR).toBe('PI_EXPR');
        expect(TokenType.EOF).toBe('EOF');
    });
});

describe('KEYWORDS', () => {
    test('contains expected keywords', () => {
        expect(KEYWORDS.has('Tensor')).toBe(true);
        expect(KEYWORDS.has('Cluster')).toBe(true);
        expect(KEYWORDS.has('Model')).toBe(true);
        expect(KEYWORDS.has('Pipeline')).toBe(true);
        expect(KEYWORDS.has('if')).toBe(true);
        expect(KEYWORDS.has('for')).toBe(true);
        expect(KEYWORDS.has('return')).toBe(true);
    });

    test('does not contain non-keywords', () => {
        expect(KEYWORDS.has('foo')).toBe(false);
        expect(KEYWORDS.has('bar')).toBe(false);
    });
});

describe('GLYPHS', () => {
    test('contains geometric operator glyphs', () => {
        for (const g of ['⊗', '⊕', '⊖', '⊛', '⊜', '⊝', '⊞']) {
            expect(GLYPHS.has(g)).toBe(true);
        }
    });

    test('contains transform glyphs', () => {
        expect(GLYPHS.has('⤍')).toBe(true);
        expect(GLYPHS.has('↻')).toBe(true);
        expect(GLYPHS.has('⟲')).toBe(true);
        expect(GLYPHS.has('∿')).toBe(true);
        expect(GLYPHS.has('⊙')).toBe(true);
    });
});

// ------------------------------------------------------------------ //
// tokenize() — basic cases
// ------------------------------------------------------------------ //

describe('tokenize', () => {
    test('returns EOF for empty source', () => {
        const tokens = tokenize('');
        expect(tokens).toHaveLength(1);
        expect(tokens[0].type).toBe(TokenType.EOF);
    });

    test('tokenizes a number', () => {
        const tokens = tokenize('42');
        expect(tokens[0].type).toBe(TokenType.NUMBER);
        expect(tokens[0].value).toBe(42);
    });

    test('tokenizes a float', () => {
        const tokens = tokenize('3.14');
        expect(tokens[0].type).toBe(TokenType.NUMBER);
        expect(tokens[0].value).toBeCloseTo(3.14);
    });

    test('tokenizes a string', () => {
        const tokens = tokenize('"hello world"');
        expect(tokens[0].type).toBe(TokenType.STRING);
        expect(tokens[0].value).toBe('hello world');
    });

    test('tokenizes single-quoted string', () => {
        const tokens = tokenize("'kuhul'");
        expect(tokens[0].type).toBe(TokenType.STRING);
        expect(tokens[0].value).toBe('kuhul');
    });

    test('tokenizes an identifier', () => {
        const tokens = tokenize('myTensor');
        expect(tokens[0].type).toBe(TokenType.IDENTIFIER);
        expect(tokens[0].value).toBe('myTensor');
    });

    test('tokenizes a keyword', () => {
        const tokens = tokenize('Tensor');
        expect(tokens[0].type).toBe(TokenType.KEYWORD);
        expect(tokens[0].value).toBe('Tensor');
    });

    test('tokenizes a glyph', () => {
        const tokens = tokenize('⊗');
        expect(tokens[0].type).toBe(TokenType.GLYPH);
        expect(tokens[0].value).toBe('⊗');
    });

    test('tokenizes π as PI', () => {
        const tokens = tokenize('π');
        expect(tokens[0].type).toBe(TokenType.PI);
    });

    test('tokenizes 0.5π as PI_EXPR with coefficient 0.5', () => {
        const tokens = tokenize('0.5π');
        expect(tokens[0].type).toBe(TokenType.PI_EXPR);
        expect(tokens[0].value).toBe(0.5);
    });

    test('tokenizes punctuation', () => {
        const tokens = tokenize('= ; , . ( ) { } [ ]');
        const types = tokens.map(t => t.type).filter(t => t !== TokenType.EOF);
        expect(types).toContain(TokenType.EQUALS);
        expect(types).toContain(TokenType.SEMICOLON);
        expect(types).toContain(TokenType.COMMA);
    });

    test('tokenizes a Tensor declaration', () => {
        const tokens = tokenize('Tensor v = 1.0;');
        const types = tokens.map(t => t.type);
        expect(types[0]).toBe(TokenType.KEYWORD);
        expect(types[1]).toBe(TokenType.IDENTIFIER);
        expect(types[2]).toBe(TokenType.EQUALS);
        expect(types[3]).toBe(TokenType.NUMBER);
        expect(types[4]).toBe(TokenType.SEMICOLON);
    });

    test('skips line comments', () => {
        const tokens = tokenize('// this is a comment\n42');
        expect(tokens[0].type).toBe(TokenType.NUMBER);
    });

    test('skips block comments', () => {
        const tokens = tokenize('/* block */ 99');
        expect(tokens[0].type).toBe(TokenType.NUMBER);
        expect(tokens[0].value).toBe(99);
    });

    test('tracks line numbers', () => {
        const tokens = tokenize('Tensor\nv');
        expect(tokens[0].line).toBe(1);
        expect(tokens[1].line).toBe(2);
    });

    test('tracks column numbers', () => {
        const tokens = tokenize('  42');
        expect(tokens[0].col).toBe(3);
    });

    test('tokenizes all geometric glyph operators', () => {
        const glyphs = '⊗ ⊕ ⊖ ⊛ ⊜ ⊝ ⊞';
        const tokens = tokenize(glyphs);
        const glyphTokens = tokens.filter(t => t.type === TokenType.GLYPH);
        expect(glyphTokens).toHaveLength(7);
    });

    test('tokenizes a glyph operation expression', () => {
        const tokens = tokenize('a ⊗ b');
        const types = tokens.map(t => t.type).filter(t => t !== TokenType.EOF);
        expect(types).toEqual([TokenType.IDENTIFIER, TokenType.GLYPH, TokenType.IDENTIFIER]);
    });
});
