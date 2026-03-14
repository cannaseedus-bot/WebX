// K'UHUL++ v2.0 Lexer
// Converts .KPP source text into a flat token stream.

// ------------------------------------------------------------------ //
// Token types
// ------------------------------------------------------------------ //

export const TokenType = {
    // Literals
    NUMBER:      'NUMBER',
    STRING:      'STRING',
    IDENTIFIER:  'IDENTIFIER',

    // π constant and π-multiplied numbers
    PI:          'PI',           // bare π symbol
    PI_EXPR:     'PI_EXPR',      // e.g. 0.75π

    // Glyph tokens (Unicode operator symbols)
    GLYPH:       'GLYPH',

    // Punctuation / operators
    EQUALS:      'EQUALS',       // =
    SEMICOLON:   'SEMICOLON',    // ;
    COLON:       'COLON',        // :
    COMMA:       'COMMA',        // ,
    DOT:         'DOT',          // .
    LPAREN:      'LPAREN',       // (
    RPAREN:      'RPAREN',       // )
    LBRACE:      'LBRACE',       // {
    RBRACE:      'RBRACE',       // }
    LBRACKET:    'LBRACKET',     // [
    RBRACKET:    'RBRACKET',     // ]
    LANGLE:      'LANGLE',       // <
    RANGLE:      'RANGLE',       // >
    STAR:        'STAR',         // *
    PLUS:        'PLUS',         // +
    MINUS:       'MINUS',        // -
    SLASH:       'SLASH',        // /
    AT:          'AT',           // @
    PIPE:        'PIPE',         // |

    // Keywords
    KEYWORD:     'KEYWORD',

    // End-of-file sentinel
    EOF:         'EOF',
};

/** All K'UHUL++ reserved keywords */
export const KEYWORDS = new Set([
    'Tensor', 'Cluster', 'Model', 'Pipeline',
    'dx12', 'RootSignature', 'PipelineState',
    'VertexShader', 'PixelShader', 'ComputeShader', 'GeometryShader',
    'Buffer', 'Sampler', 'CBV', 'SRV', 'UAV',
    'PrimitiveTopology', 'RasterizerState', 'BlendState', 'DepthStencilState',
    'if', 'else', 'for', 'while', 'foreach', 'glyph', 'in', 'parallel',
    'with', 'return', 'GPU', 'Dispatch', 'Draw', 'Train',
    'GeometricTensor', 'TensorCluster', 'GeometricModel',
    'generate_spiral', 'load_dataset', 'split_tensors', 'merge_clusters',
]);

/** All recognised K'UHUL++ glyph symbols */
export const GLYPHS = new Set([
    // ASC Cipher glyphs
    '⤍', '⤎', '⤏', '⤐',
    // SCX Compression glyphs
    '↻', '↔', '⤒', '⤓',
    // 3D Control glyphs
    '⟲', '⤦', '⤧', '⤨',
    // Neural glyphs
    '⟿', '⤂', '⤃', '⤄',
    // Legacy glyphs already in kuhul.js
    '∿', '⊙',
]);

// ------------------------------------------------------------------ //
// Token helper
// ------------------------------------------------------------------ //

/**
 * @typedef {{ type: string, value: string|number, line: number, col: number }} Token
 */

function makeToken(type, value, line, col) {
    return { type, value, line, col };
}

// ------------------------------------------------------------------ //
// Lexer
// ------------------------------------------------------------------ //

/**
 * Tokenise K'UHUL++ source text.
 *
 * @param {string} source - Full .KPP source code
 * @returns {Token[]} Flat token array (last element is EOF)
 */
export function tokenize(source) {
    const tokens = [];
    let pos = 0;
    let line = 1;
    let col = 1;

    /** Peek at character `offset` ahead (default 0) without advancing */
    function peek(offset = 0) {
        return source[pos + offset] ?? '';
    }

    /** Advance by one character, updating line/col counters */
    function advance() {
        const ch = source[pos++];
        if (ch === '\n') { line++; col = 1; } else { col++; }
        return ch;
    }

    /** Consume characters while predicate holds */
    function readWhile(pred) {
        let s = '';
        while (pos < source.length && pred(source[pos])) {
            s += advance();
        }
        return s;
    }

    /** Skip whitespace (including newlines) */
    function skipWhitespace() {
        readWhile(ch => /\s/.test(ch));
    }

    /** Skip a single-line comment */
    function skipLineComment() {
        readWhile(ch => ch !== '\n');
    }

    /** Skip a block comment — errors on unterminated comment */
    function skipBlockComment() {
        while (pos < source.length) {
            if (peek() === '*' && peek(1) === '/') {
                advance(); advance(); // consume */
                return;
            }
            advance();
        }
        throw new LexerError('Unterminated block comment', line, col);
    }

    /** Read a double-quoted string literal */
    function readString(startLine, startCol) {
        advance(); // opening "
        let s = '';
        while (pos < source.length && peek() !== '"') {
            if (peek() === '\\') {
                advance(); // backslash
                const esc = advance();
                s += esc === 'n' ? '\n' : esc === 't' ? '\t' : esc;
            } else {
                s += advance();
            }
        }
        if (pos >= source.length) {
            throw new LexerError('Unterminated string literal', startLine, startCol);
        }
        advance(); // closing "
        return makeToken(TokenType.STRING, s, startLine, startCol);
    }

    /** Read a numeric literal, including optional π suffix */
    function readNumber(startLine, startCol) {
        let raw = readWhile(ch => /[0-9]/.test(ch));
        if (peek() === '.' && /[0-9]/.test(peek(1))) {
            raw += advance(); // '.'
            raw += readWhile(ch => /[0-9]/.test(ch));
        }
        // Scientific notation
        if (peek() === 'e' || peek() === 'E') {
            raw += advance();
            if (peek() === '+' || peek() === '-') raw += advance();
            raw += readWhile(ch => /[0-9]/.test(ch));
        }
        const num = parseFloat(raw);

        // Check for π or *π suffix  (e.g. 0.75π  or  2*π)
        if (peek() === 'π') {
            advance(); // consume π
            return makeToken(TokenType.PI_EXPR, num, startLine, startCol);
        }
        if (peek() === '*' && peek(1) === 'π') {
            advance(); advance(); // consume *π
            return makeToken(TokenType.PI_EXPR, num, startLine, startCol);
        }
        return makeToken(TokenType.NUMBER, num, startLine, startCol);
    }

    /** Read an identifier or keyword */
    function readIdent(startLine, startCol) {
        const name = readWhile(ch => /[a-zA-Z0-9_]/.test(ch));
        if (KEYWORDS.has(name)) {
            return makeToken(TokenType.KEYWORD, name, startLine, startCol);
        }
        return makeToken(TokenType.IDENTIFIER, name, startLine, startCol);
    }

    // Main scan loop
    while (pos < source.length) {
        skipWhitespace();
        if (pos >= source.length) break;

        const startLine = line;
        const startCol  = col;
        const ch = peek();

        // Comments
        if (ch === '/' && peek(1) === '/') {
            advance(); advance(); // //
            skipLineComment();
            continue;
        }
        if (ch === '/' && peek(1) === '*') {
            advance(); advance(); // /*
            skipBlockComment();
            continue;
        }

        // String literals
        if (ch === '"') {
            tokens.push(readString(startLine, startCol));
            continue;
        }

        // Numbers
        if (/[0-9]/.test(ch)) {
            tokens.push(readNumber(startLine, startCol));
            continue;
        }

        // Identifiers / keywords
        if (/[a-zA-Z_]/.test(ch)) {
            tokens.push(readIdent(startLine, startCol));
            continue;
        }

        // Bare π symbol
        if (ch === 'π') {
            advance();
            tokens.push(makeToken(TokenType.PI, Math.PI, startLine, startCol));
            continue;
        }

        // Glyph symbols
        if (GLYPHS.has(ch)) {
            advance();
            tokens.push(makeToken(TokenType.GLYPH, ch, startLine, startCol));
            continue;
        }

        // Single-character punctuation
        advance(); // consume the character
        switch (ch) {
            case '=': tokens.push(makeToken(TokenType.EQUALS,    ch, startLine, startCol)); break;
            case ';': tokens.push(makeToken(TokenType.SEMICOLON, ch, startLine, startCol)); break;
            case ':': tokens.push(makeToken(TokenType.COLON,     ch, startLine, startCol)); break;
            case ',': tokens.push(makeToken(TokenType.COMMA,     ch, startLine, startCol)); break;
            case '.': tokens.push(makeToken(TokenType.DOT,       ch, startLine, startCol)); break;
            case '(': tokens.push(makeToken(TokenType.LPAREN,    ch, startLine, startCol)); break;
            case ')': tokens.push(makeToken(TokenType.RPAREN,    ch, startLine, startCol)); break;
            case '{': tokens.push(makeToken(TokenType.LBRACE,    ch, startLine, startCol)); break;
            case '}': tokens.push(makeToken(TokenType.RBRACE,    ch, startLine, startCol)); break;
            case '[': tokens.push(makeToken(TokenType.LBRACKET,  ch, startLine, startCol)); break;
            case ']': tokens.push(makeToken(TokenType.RBRACKET,  ch, startLine, startCol)); break;
            case '<': tokens.push(makeToken(TokenType.LANGLE,    ch, startLine, startCol)); break;
            case '>': tokens.push(makeToken(TokenType.RANGLE,    ch, startLine, startCol)); break;
            case '*': tokens.push(makeToken(TokenType.STAR,      ch, startLine, startCol)); break;
            case '+': tokens.push(makeToken(TokenType.PLUS,      ch, startLine, startCol)); break;
            case '-': tokens.push(makeToken(TokenType.MINUS,     ch, startLine, startCol)); break;
            case '/': tokens.push(makeToken(TokenType.SLASH,     ch, startLine, startCol)); break;
            case '@': tokens.push(makeToken(TokenType.AT,        ch, startLine, startCol)); break;
            case '|': tokens.push(makeToken(TokenType.PIPE,      ch, startLine, startCol)); break;
            default:
                // Skip unrecognised characters (e.g. exotic Unicode outside glyph set)
                break;
        }
    }

    tokens.push(makeToken(TokenType.EOF, null, line, col));
    return tokens;
}

// ------------------------------------------------------------------ //
// LexerError
// ------------------------------------------------------------------ //

export class LexerError extends Error {
    /**
     * @param {string} message
     * @param {number} line
     * @param {number} col
     */
    constructor(message, line, col) {
        super(`LexerError at ${line}:${col} — ${message}`);
        this.name = 'LexerError';
        this.line = line;
        this.col  = col;
    }
}
