// K'UHUL++ v2.0 Parser
// Converts a flat token stream (from the Lexer) into an Abstract Syntax Tree.

import { TokenType, KEYWORDS } from './lexer.js';

// ------------------------------------------------------------------ //
// AST node constructors
// ------------------------------------------------------------------ //

/**
 * Create an AST node with a consistent shape.
 * @param {string} kind  - Node type name
 * @param {object} extra - Additional fields
 * @returns {object}
 */
function node(kind, extra = {}) {
    return { kind, ...extra };
}

export const NodeKind = {
    Program:         'Program',
    TensorDecl:      'TensorDecl',
    ClusterDecl:     'ClusterDecl',
    ModelDecl:       'ModelDecl',
    PipelineDecl:    'PipelineDecl',
    Assignment:      'Assignment',
    GlyphOp:         'GlyphOp',
    NativeBlock:     'NativeBlock',
    DX12Stmt:        'DX12Stmt',
    IfStmt:          'IfStmt',
    ForStmt:         'ForStmt',
    WhileStmt:       'WhileStmt',
    ForEachGlyph:    'ForEachGlyph',
    ParallelFor:     'ParallelFor',
    FunctionCall:    'FunctionCall',
    MethodCall:      'MethodCall',
    BinaryExpr:      'BinaryExpr',
    UnaryExpr:       'UnaryExpr',
    MemberExpr:      'MemberExpr',
    IndexExpr:       'IndexExpr',
    Identifier:      'Identifier',
    NumberLiteral:   'NumberLiteral',
    PiExpr:          'PiExpr',
    StringLiteral:   'StringLiteral',
    ArrayLiteral:    'ArrayLiteral',
    ObjectLiteral:   'ObjectLiteral',
    Vector3Literal:  'Vector3Literal',
    Block:           'Block',
    TrainStmt:       'TrainStmt',
};

// ------------------------------------------------------------------ //
// ParseError
// ------------------------------------------------------------------ //

export class ParseError extends Error {
    /**
     * @param {string} message
     * @param {{ line: number, col: number }} token
     */
    constructor(message, token) {
        const loc = token ? `${token.line}:${token.col}` : '?:?';
        super(`ParseError at ${loc} — ${message}`);
        this.name  = 'ParseError';
        this.token = token;
    }
}

// ------------------------------------------------------------------ //
// Parser
// ------------------------------------------------------------------ //

/**
 * Parse a K'UHUL++ token stream into an AST.
 *
 * @param {import('./lexer.js').Token[]} tokens - Output of tokenize()
 * @returns {{ kind: 'Program', body: object[] }}
 */
export function parse(tokens) {
    let pos = 0;

    // ---- helpers ----

    function current() { return tokens[pos]; }
    function peek(offset = 0) { return tokens[pos + offset] ?? tokens[tokens.length - 1]; }

    function advance() {
        const tok = tokens[pos];
        if (tok.type !== TokenType.EOF) pos++;
        return tok;
    }

    function expect(type, value) {
        const tok = current();
        if (tok.type !== type) {
            throw new ParseError(`Expected token type ${type} but got ${tok.type} ("${tok.value}")`, tok);
        }
        if (value !== undefined && tok.value !== value) {
            throw new ParseError(`Expected "${value}" but got "${tok.value}"`, tok);
        }
        return advance();
    }

    function check(type, value) {
        const tok = current();
        return tok.type === type && (value === undefined || tok.value === value);
    }

    function match(type, value) {
        if (check(type, value)) { advance(); return true; }
        return false;
    }

    function isEOF() { return current().type === TokenType.EOF; }

    // ---- program ----

    function parseProgram() {
        const body = [];
        while (!isEOF()) {
            body.push(parseStatement());
        }
        return node(NodeKind.Program, { body });
    }

    // ---- statements ----

    function parseStatement() {
        const tok = current();

        // Glyph operation: [ glyph ... ]
        if (tok.type === TokenType.LBRACKET) {
            return parseGlyphOp();
        }

        // dx12 { … }
        if (tok.type === TokenType.KEYWORD && tok.value === 'dx12') {
            return parseDX12Block();
        }

        // Parallel for
        if (tok.type === TokenType.KEYWORD && tok.value === 'parallel') {
            return parseParallelFor();
        }

        // if / for / while / foreach
        if (tok.type === TokenType.KEYWORD && tok.value === 'if')      return parseIf();
        if (tok.type === TokenType.KEYWORD && tok.value === 'for')     return parseFor();
        if (tok.type === TokenType.KEYWORD && tok.value === 'while')   return parseWhile();
        if (tok.type === TokenType.KEYWORD && tok.value === 'foreach') return parseForEachGlyph();

        // Declaration keywords
        if (tok.type === TokenType.KEYWORD) {
            switch (tok.value) {
                case 'Tensor':   return parseDeclaration('Tensor',   NodeKind.TensorDecl);
                case 'Cluster':  return parseDeclaration('Cluster',  NodeKind.ClusterDecl);
                case 'Model':    return parseDeclaration('Model',    NodeKind.ModelDecl);
                case 'Pipeline': return parseDeclaration('Pipeline', NodeKind.PipelineDecl);
                case 'Train':    return parseTrainStmt();
                case 'GPU':      return parseGPUStatement();
                default:         break;
            }
        }

        // Assignment or function call (identifier-led)
        if (tok.type === TokenType.IDENTIFIER) {
            return parseIdentifierLed();
        }

        // Unexpected token — skip so we don't spin forever
        advance();
        return node('Unknown', { token: tok });
    }

    // ---- declaration: Tensor | Cluster | Model | Pipeline ----

    function parseDeclaration(keyword, kind) {
        advance(); // consume keyword
        const name = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.EQUALS);
        const init = parseExpression();
        match(TokenType.SEMICOLON);
        return node(kind, { name, init });
    }

    // ---- glyph operation: [ glyph ] target? param=value … ----

    function parseGlyphOp() {
        expect(TokenType.LBRACKET);
        const glyph = expect(TokenType.GLYPH).value;
        // Params may appear INSIDE the brackets (spec grammar) …
        const params = {};
        while (!check(TokenType.RBRACKET) && !isEOF()) {
            const key = expect(TokenType.IDENTIFIER).value;
            expect(TokenType.EQUALS);
            params[key] = parseAtom();
        }
        expect(TokenType.RBRACKET);

        // … and/or OUTSIDE the brackets as:  target key=value key=value …
        // The optional target is any expression that is NOT an identifier=value pair.
        let target = null;
        while (!check(TokenType.SEMICOLON) && !check(TokenType.LBRACKET)
               && !isEOF() && current().type !== TokenType.EOF) {
            const tok = current();
            // If this is a bare IDENTIFIER followed by '=', it's a key=value param
            if (tok.type === TokenType.IDENTIFIER && peek(1).type === TokenType.EQUALS) {
                const key = advance().value; // identifier
                advance();                   // '='
                params[key] = parseAtom();
                continue;
            }
            // Anything else before any key=value pairs is the target expression
            if (target === null) {
                target = parsePostfix(parseAtom());
                // After target, handle key=value params like  target.member=val  or  key=val
                continue;
            }
            // Unrecognised token after target — stop parsing this glyph op
            break;
        }

        match(TokenType.SEMICOLON);
        return node(NodeKind.GlyphOp, { glyph, params, target });
    }

    // ---- dx12 native block ----

    function parseDX12Block() {
        advance(); // consume 'dx12'
        expect(TokenType.LBRACE);
        const stmts = [];
        while (!check(TokenType.RBRACE) && !isEOF()) {
            stmts.push(parseDX12Statement());
        }
        expect(TokenType.RBRACE);
        return node(NodeKind.NativeBlock, { stmts });
    }

    function parseDX12Statement() {
        const tok = current();

        // RootSignature Identifier { … }
        if (tok.type === TokenType.KEYWORD && tok.value === 'RootSignature') {
            return parseDX12NamedBlock('RootSignature');
        }
        // PipelineState Identifier { … }
        if (tok.type === TokenType.KEYWORD && tok.value === 'PipelineState') {
            return parseDX12NamedBlock('PipelineState');
        }
        // Buffer Identifier : BufferType<DataType>[size] @ flags ;
        if (tok.type === TokenType.KEYWORD && tok.value === 'Buffer') {
            return parseDX12Buffer();
        }
        // GPU.Method(…) ;
        if (tok.type === TokenType.KEYWORD && tok.value === 'GPU') {
            return parseGPUStatement();
        }

        // Generic keyword block or expression statement
        advance();
        return node(NodeKind.DX12Stmt, { token: tok });
    }

    function parseDX12NamedBlock(stmtKind) {
        advance(); // consume keyword
        const name = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.LBRACE);
        const body = [];
        while (!check(TokenType.RBRACE) && !isEOF()) {
            body.push(parseDX12InnerStmt());
        }
        expect(TokenType.RBRACE);
        return node(NodeKind.DX12Stmt, { stmtKind, name, body });
    }
    function parseDX12InnerStmt() {
        const tok = current();
        if (tok.type === TokenType.KEYWORD) {
            const kw = tok.value;
            advance(); // consume keyword
            if (check(TokenType.LPAREN)) {
                // CBV(0); SRV(0); etc.
                advance(); // (
                const reg = parseExpression();
                expect(TokenType.RPAREN);
                match(TokenType.SEMICOLON);
                return node(NodeKind.DX12Stmt, { stmtKind: kw, register: reg });
            }
            // key: value ;
            if (check(TokenType.COLON)) {
                advance(); // :
                const value = parseExpression();
                match(TokenType.SEMICOLON);
                return node(NodeKind.DX12Stmt, { stmtKind: kw, value });
            }
        }
        // Fall-through: parse an expression statement
        const expr = parseExpression();
        match(TokenType.SEMICOLON);
        return node(NodeKind.DX12Stmt, { stmtKind: 'expr', expr });
    }

    function parseDX12Buffer() {
        advance(); // consume 'Buffer'
        const name = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.COLON);
        const bufferType = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.LANGLE);
        const dataType = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.RANGLE);
        expect(TokenType.LBRACKET);
        const size = parseExpression();
        expect(TokenType.RBRACKET);
        let flags = null;
        if (match(TokenType.AT)) {
            flags = parseFlags();
        }
        match(TokenType.SEMICOLON);
        return node(NodeKind.DX12Stmt, { stmtKind: 'Buffer', name, bufferType, dataType, size, flags });
    }

    /** Parse pipe-separated resource flags: UAV | SRV | CBV … */
    function parseFlags() {
        const flags = [expect(TokenType.KEYWORD).value];
        while (match(TokenType.PIPE)) {
            flags.push(expect(TokenType.KEYWORD).value);
        }
        return flags;
    }

    // ---- GPU.Method(…); statement ----

    function parseGPUStatement() {
        advance(); // consume 'GPU'
        expect(TokenType.DOT);
        // GPU methods like Dispatch, Draw, Present may be parsed as keywords
        const methodTok = current();
        if (methodTok.type !== TokenType.IDENTIFIER && methodTok.type !== TokenType.KEYWORD) {
            throw new ParseError(`Expected method name after GPU. but got "${methodTok.value}"`, methodTok);
        }
        advance();
        const method = methodTok.value;
        const args = parseArgList();
        match(TokenType.SEMICOLON);
        return node(NodeKind.MethodCall, { object: 'GPU', method, args });
    }

    // ---- Train ----

    function parseTrainStmt() {
        advance(); // consume 'Train'
        const model = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.KEYWORD, 'with');
        const data = expect(TokenType.IDENTIFIER).value;
        let body = null;
        if (check(TokenType.LBRACE)) {
            body = parseBlock();
        } else {
            match(TokenType.SEMICOLON);
        }
        return node(NodeKind.TrainStmt, { model, data, body });
    }

    // ---- control flow ----

    function parseIf() {
        advance(); // consume 'if'
        expect(TokenType.LPAREN);
        const condition = parseExpression();
        expect(TokenType.RPAREN);
        const consequent = parseBlock();
        let alternate = null;
        if (check(TokenType.KEYWORD, 'else')) {
            advance();
            alternate = check(TokenType.KEYWORD, 'if') ? parseIf() : parseBlock();
        }
        return node(NodeKind.IfStmt, { condition, consequent, alternate });
    }

    function parseFor() {
        advance(); // consume 'for'
        expect(TokenType.LPAREN);
        const ident = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.KEYWORD, 'in');
        const range = parseExpression();
        expect(TokenType.RPAREN);
        const body = parseBlock();
        return node(NodeKind.ForStmt, { ident, range, body });
    }

    function parseWhile() {
        advance(); // consume 'while'
        expect(TokenType.LPAREN);
        const condition = parseExpression();
        expect(TokenType.RPAREN);
        const body = parseBlock();
        return node(NodeKind.WhileStmt, { condition, body });
    }

    function parseForEachGlyph() {
        advance(); // consume 'foreach'
        expect(TokenType.KEYWORD, 'glyph');
        expect(TokenType.KEYWORD, 'in');
        expect(TokenType.LBRACKET);
        const glyphs = [];
        while (!check(TokenType.RBRACKET) && !isEOF()) {
            glyphs.push(expect(TokenType.GLYPH).value);
            match(TokenType.COMMA);
        }
        expect(TokenType.RBRACKET);
        const body = parseBlock();
        return node(NodeKind.ForEachGlyph, { glyphs, body });
    }

    function parseParallelFor() {
        advance(); // consume 'parallel'
        expect(TokenType.KEYWORD, 'for');
        expect(TokenType.LPAREN);
        const ident = expect(TokenType.IDENTIFIER).value;
        expect(TokenType.KEYWORD, 'in');
        const range = parseExpression();
        expect(TokenType.RPAREN);
        const body = parseBlock();
        return node(NodeKind.ParallelFor, { ident, range, body });
    }

    // ---- block ----

    function parseBlock() {
        expect(TokenType.LBRACE);
        const stmts = [];
        while (!check(TokenType.RBRACE) && !isEOF()) {
            stmts.push(parseStatement());
        }
        expect(TokenType.RBRACE);
        return node(NodeKind.Block, { stmts });
    }

    // ---- identifier-led: assignment or call ----

    function parseIdentifierLed() {
        const expr = parsePostfix(parseAtom());

        // Assignment: expr = rhs ;
        if (check(TokenType.EQUALS)) {
            advance();
            const rhs = parseExpression();
            match(TokenType.SEMICOLON);
            return node(NodeKind.Assignment, { target: expr, value: rhs });
        }

        // Expression statement (e.g. a standalone function call)
        match(TokenType.SEMICOLON);
        return expr;
    }

    // ---- expressions ----

    function parseExpression() {
        return parseBinary(parseUnary(), 0);
    }

    const BINOP_PRECEDENCE = {
        '|': 1, '+': 2, '-': 2, '*': 3, '/': 3, '>': 4, '<': 4,
    };

    function getBinopPrec(tok) {
        if (tok.type === TokenType.PIPE)  return BINOP_PRECEDENCE['|'];
        if (tok.type === TokenType.PLUS)  return BINOP_PRECEDENCE['+'];
        if (tok.type === TokenType.MINUS) return BINOP_PRECEDENCE['-'];
        if (tok.type === TokenType.STAR)  return BINOP_PRECEDENCE['*'];
        if (tok.type === TokenType.SLASH) return BINOP_PRECEDENCE['/'];
        if (tok.type === TokenType.RANGLE) return BINOP_PRECEDENCE['>'];
        if (tok.type === TokenType.LANGLE) return BINOP_PRECEDENCE['<'];
        return -1;
    }

    function parseBinary(left, minPrec) {
        while (true) {
            const prec = getBinopPrec(current());
            if (prec < minPrec) break;
            const op = advance().value;
            let right = parseUnary();
            // Right-associativity: not needed for current ops, use left-assoc
            while (getBinopPrec(current()) > prec) {
                right = parseBinary(right, prec + 1);
            }
            left = node(NodeKind.BinaryExpr, { op, left, right });
        }
        return left;
    }

    function parseUnary() {
        if (check(TokenType.MINUS)) {
            advance();
            return node(NodeKind.UnaryExpr, { op: '-', operand: parseUnary() });
        }
        return parsePostfix(parseAtom());
    }

    /** Parse member/index/call chains after a primary atom */
    function parsePostfix(expr) {
        while (true) {
            if (check(TokenType.DOT)) {
                advance();
                const member = expect(TokenType.IDENTIFIER).value;
                if (check(TokenType.LPAREN)) {
                    const args = parseArgList();
                    expr = node(NodeKind.MethodCall, { object: expr, method: member, args });
                } else {
                    expr = node(NodeKind.MemberExpr, { object: expr, member });
                }
                continue;
            }
            if (check(TokenType.LBRACKET)) {
                advance();
                const index = parseExpression();
                expect(TokenType.RBRACKET);
                expr = node(NodeKind.IndexExpr, { object: expr, index });
                continue;
            }
            break;
        }
        return expr;
    }

    /** Parse the primary atom of an expression */
    function parseAtom() {
        const tok = current();

        // Number literal
        if (tok.type === TokenType.NUMBER) {
            advance();
            return node(NodeKind.NumberLiteral, { value: tok.value });
        }

        // π-multiplied expression: NUMBER followed by π already merged by lexer
        if (tok.type === TokenType.PI_EXPR) {
            advance();
            return node(NodeKind.PiExpr, { coefficient: tok.value });
        }

        // Bare π
        if (tok.type === TokenType.PI) {
            advance();
            return node(NodeKind.PiExpr, { coefficient: 1 });
        }

        // String literal
        if (tok.type === TokenType.STRING) {
            advance();
            return node(NodeKind.StringLiteral, { value: tok.value });
        }

        // Array literal or glyph block: [ … ]
        if (tok.type === TokenType.LBRACKET) {
            advance();
            // Empty array
            if (check(TokenType.RBRACKET)) { advance(); return node(NodeKind.ArrayLiteral, { elements: [] }); }
            // Check first element type
            const elements = [parseExpression()];
            while (match(TokenType.COMMA)) {
                if (check(TokenType.RBRACKET)) break;
                elements.push(parseExpression());
            }
            expect(TokenType.RBRACKET);
            return node(NodeKind.ArrayLiteral, { elements });
        }

        // Object literal: { key: value, … }
        if (tok.type === TokenType.LBRACE) {
            return parseObjectLiteral();
        }

        // Parenthesised expression or Vector3 literal: (x, y, z)
        if (tok.type === TokenType.LPAREN) {
            advance();
            const first = parseExpression();
            if (check(TokenType.COMMA)) {
                advance();
                const second = parseExpression();
                expect(TokenType.COMMA);
                const third = parseExpression();
                expect(TokenType.RPAREN);
                return node(NodeKind.Vector3Literal, { x: first, y: second, z: third });
            }
            expect(TokenType.RPAREN);
            return first;
        }

        // Keyword used as identifier-like value (e.g. GPU.count, spherical, phase_aligned …)
        if (tok.type === TokenType.KEYWORD) {
            advance();
            // Check for function call
            if (check(TokenType.LPAREN)) {
                const args = parseArgList();
                return node(NodeKind.FunctionCall, { callee: tok.value, args });
            }
            return node(NodeKind.Identifier, { name: tok.value });
        }

        // Identifier (possibly function call)
        if (tok.type === TokenType.IDENTIFIER) {
            advance();
            if (check(TokenType.LPAREN)) {
                const args = parseArgList();
                return node(NodeKind.FunctionCall, { callee: tok.value, args });
            }
            return node(NodeKind.Identifier, { name: tok.value });
        }

        // Glyph as value (unlikely but handle gracefully)
        if (tok.type === TokenType.GLYPH) {
            advance();
            return node(NodeKind.Identifier, { name: tok.value });
        }

        // Unknown — return a placeholder and advance to avoid infinite loops
        advance();
        return node('Unknown', { token: tok });
    }

    /** Parse a parenthesised argument list: ( arg, key: val, … ) */
    function parseArgList() {
        expect(TokenType.LPAREN);
        const args = [];
        while (!check(TokenType.RPAREN) && !isEOF()) {
            // Named argument: key: value
            if (current().type === TokenType.IDENTIFIER && peek(1).type === TokenType.COLON) {
                const key = advance().value;
                advance(); // :
                const value = parseExpression();
                args.push({ key, value });
            } else {
                args.push({ value: parseExpression() });
            }
            match(TokenType.COMMA);
        }
        expect(TokenType.RPAREN);
        return args;
    }

    /** Parse { key: value, … } */
    function parseObjectLiteral() {
        expect(TokenType.LBRACE);
        const properties = {};
        while (!check(TokenType.RBRACE) && !isEOF()) {
            let key;
            if (current().type === TokenType.IDENTIFIER || current().type === TokenType.KEYWORD) {
                key = advance().value;
            } else {
                key = advance().value ?? '';
            }
            expect(TokenType.COLON);
            const value = parseExpression();
            properties[key] = value;
            match(TokenType.COMMA);
        }
        expect(TokenType.RBRACE);
        return node(NodeKind.ObjectLiteral, { properties });
    }

    return parseProgram();
}
