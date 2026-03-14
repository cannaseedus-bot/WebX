// K'UHUL++ v2.0 Lexer — TypeScript Edition
// Converts .kuhul / .kpp source text into a flat token stream.
// This is the TypeScript counterpart of src/compiler/lexer.js.

// ------------------------------------------------------------------ //
// Token types
// ------------------------------------------------------------------ //

export enum TokenType {
    // Literals
    NUMBER      = 'NUMBER',
    STRING      = 'STRING',
    IDENTIFIER  = 'IDENTIFIER',

    // π constant and π-multiplied numbers
    PI          = 'PI',       // bare π symbol
    PI_EXPR     = 'PI_EXPR',  // e.g. 0.75π

    // Glyph tokens (Unicode operator symbols)
    GLYPH       = 'GLYPH',

    // Punctuation / operators
    EQUALS      = 'EQUALS',     // =
    SEMICOLON   = 'SEMICOLON',  // ;
    COLON       = 'COLON',      // :
    COMMA       = 'COMMA',      // ,
    DOT         = 'DOT',        // .
    LPAREN      = 'LPAREN',     // (
    RPAREN      = 'RPAREN',     // )
    LBRACE      = 'LBRACE',     // {
    RBRACE      = 'RBRACE',     // }
    LBRACKET    = 'LBRACKET',   // [
    RBRACKET    = 'RBRACKET',   // ]
    LANGLE      = 'LANGLE',     // <
    RANGLE      = 'RANGLE',     // >
    STAR        = 'STAR',       // *
    PLUS        = 'PLUS',       // +
    MINUS       = 'MINUS',      // -
    SLASH       = 'SLASH',      // /
    AT          = 'AT',         // @
    PIPE        = 'PIPE',       // |

    // Keywords
    KEYWORD     = 'KEYWORD',

    // End-of-file sentinel
    EOF         = 'EOF',
}

// ------------------------------------------------------------------ //
// Token interface
// ------------------------------------------------------------------ //

/** A single lexed token produced by `tokenize()` */
export interface Token {
    type: TokenType;
    value: string | number;
    line: number;
    col: number;
}

// ------------------------------------------------------------------ //
// Reserved keywords
// ------------------------------------------------------------------ //

/** All K'UHUL++ reserved keywords */
export const KEYWORDS: Set<string> = new Set([
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

// ------------------------------------------------------------------ //
// Recognised glyph symbols
// ------------------------------------------------------------------ //

/** All recognised K'UHUL++ glyph symbols (Unicode operators) */
export const GLYPHS: Set<string> = new Set([
    // ASC Cipher glyphs
    '⤍', '⤎', '⤏', '⤐',
    // SCX Compression glyphs
    '↻', '↔', '⤒', '⤓',
    // 3D Control glyphs
    '⟲', '⤦', '⤧', '⤨',
    // Neural glyphs
    '⟿', '⤂', '⤃', '⤄',
    // Legacy / geometric glyphs
    '∿', '⊙',
    // Manifold M operators (⊗ ⊕ ⊖ ⊛ ⊜ ⊝ ⊞)
    '⊗', '⊕', '⊖', '⊛', '⊜', '⊝', '⊞',
    // Wave / modulation
    '≋',
]);

// ------------------------------------------------------------------ //
// LexerError
// ------------------------------------------------------------------ //

/** Thrown when the lexer encounters an unrecognised character */
export class LexerError extends Error {
    readonly line: number;
    readonly col: number;

    constructor(message: string, line: number, col: number) {
        super(`LexerError at ${line}:${col} — ${message}`);
        this.name = 'LexerError';
        this.line = line;
        this.col  = col;
    }
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function makeToken(type: TokenType, value: string | number, line: number, col: number): Token {
    return { type, value, line, col };
}

function isDigit(ch: string): boolean {
    return ch >= '0' && ch <= '9';
}

function isAlpha(ch: string): boolean {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_';
}

function isAlphaNumeric(ch: string): boolean {
    return isAlpha(ch) || isDigit(ch);
}

// ------------------------------------------------------------------ //
// Tokenizer
// ------------------------------------------------------------------ //

/**
 * Tokenise K'UHUL++ source text.
 *
 * @param source - Full .kuhul / .kpp source code
 * @returns Flat token array (last element is always EOF)
 * @throws {LexerError} On encountering an unrecognised character
 */
export function tokenize(source: string): Token[] {
    const tokens: Token[] = [];
    let pos  = 0;
    let line = 1;
    let lineStart = 0;

    function col(): number { return pos - lineStart + 1; }

    function advance(): string {
        const ch = source[pos++];
        if (ch === '\n') { line++; lineStart = pos; }
        return ch;
    }

    function peek(offset = 0): string {
        return source[pos + offset] ?? '\0';
    }

    while (pos < source.length) {
        // Skip whitespace
        if (/\s/.test(source[pos])) { advance(); continue; }

        // Line comments //
        if (source[pos] === '/' && peek(1) === '/') {
            while (pos < source.length && source[pos] !== '\n') advance();
            continue;
        }

        // Block comments /* ... */
        if (source[pos] === '/' && peek(1) === '*') {
            advance(); advance();
            while (pos < source.length && !(source[pos] === '*' && peek(1) === '/')) {
                advance();
            }
            advance(); advance(); // consume */
            continue;
        }

        const startCol = col();
        const ch = source[pos];

        // π — bare pi or coefficient×π
        if (ch === 'π') {
            // Check if the previous token was a number — already handled below.
            // Bare π token.
            advance();
            tokens.push(makeToken(TokenType.PI, 'π', line, startCol));
            continue;
        }

        // Glyph symbols
        if (GLYPHS.has(ch)) {
            advance();
            tokens.push(makeToken(TokenType.GLYPH, ch, line, startCol));
            continue;
        }

        // Number literals  (including trailing π → PI_EXPR)
        if (isDigit(ch) || (ch === '.' && isDigit(peek(1)))) {
            let num = '';
            while (pos < source.length && (isDigit(source[pos]) || source[pos] === '.')) {
                num += advance();
            }
            // Optional exponent
            if (source[pos] === 'e' || source[pos] === 'E') {
                num += advance();
                if (source[pos] === '+' || source[pos] === '-') num += advance();
                while (pos < source.length && isDigit(source[pos])) num += advance();
            }
            // Trailing π → PI_EXPR
            if (source[pos] === 'π') {
                advance();
                tokens.push(makeToken(TokenType.PI_EXPR, parseFloat(num), line, startCol));
            } else {
                tokens.push(makeToken(TokenType.NUMBER, parseFloat(num), line, startCol));
            }
            continue;
        }

        // String literals
        if (ch === '"' || ch === "'") {
            const quote = ch;
            advance();
            let str = '';
            while (pos < source.length && source[pos] !== quote) {
                if (source[pos] === '\\') { advance(); str += advance(); }
                else str += advance();
            }
            advance(); // closing quote
            tokens.push(makeToken(TokenType.STRING, str, line, startCol));
            continue;
        }

        // Identifiers and keywords
        if (isAlpha(ch)) {
            let ident = '';
            while (pos < source.length && isAlphaNumeric(source[pos])) {
                ident += advance();
            }
            if (KEYWORDS.has(ident)) {
                tokens.push(makeToken(TokenType.KEYWORD, ident, line, startCol));
            } else {
                tokens.push(makeToken(TokenType.IDENTIFIER, ident, line, startCol));
            }
            continue;
        }

        // Single-character tokens
        advance();
        switch (ch) {
            case '=': tokens.push(makeToken(TokenType.EQUALS,    ch, line, startCol)); break;
            case ';': tokens.push(makeToken(TokenType.SEMICOLON, ch, line, startCol)); break;
            case ':': tokens.push(makeToken(TokenType.COLON,     ch, line, startCol)); break;
            case ',': tokens.push(makeToken(TokenType.COMMA,     ch, line, startCol)); break;
            case '.': tokens.push(makeToken(TokenType.DOT,       ch, line, startCol)); break;
            case '(': tokens.push(makeToken(TokenType.LPAREN,    ch, line, startCol)); break;
            case ')': tokens.push(makeToken(TokenType.RPAREN,    ch, line, startCol)); break;
            case '{': tokens.push(makeToken(TokenType.LBRACE,    ch, line, startCol)); break;
            case '}': tokens.push(makeToken(TokenType.RBRACE,    ch, line, startCol)); break;
            case '[': tokens.push(makeToken(TokenType.LBRACKET,  ch, line, startCol)); break;
            case ']': tokens.push(makeToken(TokenType.RBRACKET,  ch, line, startCol)); break;
            case '<': tokens.push(makeToken(TokenType.LANGLE,    ch, line, startCol)); break;
            case '>': tokens.push(makeToken(TokenType.RANGLE,    ch, line, startCol)); break;
            case '*': tokens.push(makeToken(TokenType.STAR,      ch, line, startCol)); break;
            case '+': tokens.push(makeToken(TokenType.PLUS,      ch, line, startCol)); break;
            case '-': tokens.push(makeToken(TokenType.MINUS,     ch, line, startCol)); break;
            case '/': tokens.push(makeToken(TokenType.SLASH,     ch, line, startCol)); break;
            case '@': tokens.push(makeToken(TokenType.AT,        ch, line, startCol)); break;
            case '|': tokens.push(makeToken(TokenType.PIPE,      ch, line, startCol)); break;
            default:
                throw new LexerError(`Unrecognised character "${ch}"`, line, startCol);
        }
    }

    tokens.push(makeToken(TokenType.EOF, '', line, col()));
    return tokens;
}
