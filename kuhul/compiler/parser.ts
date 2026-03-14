// K'UHUL++ v2.0 Parser — TypeScript Edition
// Converts a flat token stream (from the Lexer) into an Abstract Syntax Tree.
// This is the TypeScript counterpart of src/compiler/parser.js.

import { TokenType, KEYWORDS } from './lexer.js';
import type { Token } from './lexer.js';

// ------------------------------------------------------------------ //
// AST node kinds
// ------------------------------------------------------------------ //

export enum NodeKind {
    Program        = 'Program',
    TensorDecl     = 'TensorDecl',
    ClusterDecl    = 'ClusterDecl',
    ModelDecl      = 'ModelDecl',
    PipelineDecl   = 'PipelineDecl',
    Assignment     = 'Assignment',
    GlyphOp        = 'GlyphOp',
    NativeBlock    = 'NativeBlock',
    DX12Stmt       = 'DX12Stmt',
    IfStmt         = 'IfStmt',
    ForStmt        = 'ForStmt',
    WhileStmt      = 'WhileStmt',
    ForEachGlyph   = 'ForEachGlyph',
    ParallelFor    = 'ParallelFor',
    FunctionCall   = 'FunctionCall',
    MethodCall     = 'MethodCall',
    BinaryExpr     = 'BinaryExpr',
    UnaryExpr      = 'UnaryExpr',
    MemberExpr     = 'MemberExpr',
    IndexExpr      = 'IndexExpr',
    Identifier     = 'Identifier',
    NumberLiteral  = 'NumberLiteral',
    PiExpr         = 'PiExpr',
    StringLiteral  = 'StringLiteral',
    ArrayLiteral   = 'ArrayLiteral',
    ObjectLiteral  = 'ObjectLiteral',
    Vector3Literal = 'Vector3Literal',
    Block          = 'Block',
    TrainStmt      = 'TrainStmt',
    ReturnStmt     = 'ReturnStmt',
}

// ------------------------------------------------------------------ //
// AST node interfaces
// ------------------------------------------------------------------ //

/** Base interface every AST node extends */
export interface ASTNode {
    kind: NodeKind;
    line?: number;
    col?:  number;
}

export interface ProgramNode       extends ASTNode { kind: NodeKind.Program;       body: ASTNode[] }
export interface BlockNode         extends ASTNode { kind: NodeKind.Block;         body: ASTNode[] }
export interface TensorDeclNode    extends ASTNode { kind: NodeKind.TensorDecl;    name: string; typeParams?: string[]; init: ASTNode }
export interface ClusterDeclNode   extends ASTNode { kind: NodeKind.ClusterDecl;   name: string; body: ASTNode[] }
export interface ModelDeclNode     extends ASTNode { kind: NodeKind.ModelDecl;     name: string; body: ASTNode[] }
export interface PipelineDeclNode  extends ASTNode { kind: NodeKind.PipelineDecl;  name: string; body: ASTNode[] }
export interface AssignmentNode    extends ASTNode { kind: NodeKind.Assignment;    target: ASTNode; value: ASTNode }
export interface GlyphOpNode       extends ASTNode { kind: NodeKind.GlyphOp;       glyph: string; left: ASTNode; right: ASTNode }
export interface IfStmtNode        extends ASTNode { kind: NodeKind.IfStmt;        test: ASTNode; consequent: ASTNode; alternate?: ASTNode }
export interface ForStmtNode       extends ASTNode { kind: NodeKind.ForStmt;       init: ASTNode | null; test: ASTNode | null; update: ASTNode | null; body: ASTNode }
export interface WhileStmtNode     extends ASTNode { kind: NodeKind.WhileStmt;     test: ASTNode; body: ASTNode }
export interface ForEachGlyphNode  extends ASTNode { kind: NodeKind.ForEachGlyph;  glyph: string; target: string; source: ASTNode; body: ASTNode }
export interface ParallelForNode   extends ASTNode { kind: NodeKind.ParallelFor;   variable: string; iterable: ASTNode; body: ASTNode }
export interface FunctionCallNode  extends ASTNode { kind: NodeKind.FunctionCall;  callee: string; args: ASTNode[] }
export interface MethodCallNode    extends ASTNode { kind: NodeKind.MethodCall;    object: ASTNode; method: string; args: ASTNode[] }
export interface BinaryExprNode    extends ASTNode { kind: NodeKind.BinaryExpr;    op: string; left: ASTNode; right: ASTNode }
export interface UnaryExprNode     extends ASTNode { kind: NodeKind.UnaryExpr;     op: string; operand: ASTNode }
export interface MemberExprNode    extends ASTNode { kind: NodeKind.MemberExpr;    object: ASTNode; property: string }
export interface IndexExprNode     extends ASTNode { kind: NodeKind.IndexExpr;     object: ASTNode; index: ASTNode }
export interface IdentifierNode    extends ASTNode { kind: NodeKind.Identifier;    name: string }
export interface NumberLiteralNode extends ASTNode { kind: NodeKind.NumberLiteral; value: number }
export interface PiExprNode        extends ASTNode { kind: NodeKind.PiExpr;        coefficient: number }
export interface StringLiteralNode extends ASTNode { kind: NodeKind.StringLiteral; value: string }
export interface ArrayLiteralNode  extends ASTNode { kind: NodeKind.ArrayLiteral;  elements: ASTNode[] }
export interface ObjectLiteralNode extends ASTNode { kind: NodeKind.ObjectLiteral; properties: Array<{ key: string; value: ASTNode }> }
export interface TrainStmtNode     extends ASTNode { kind: NodeKind.TrainStmt;     model: ASTNode; options: ASTNode | null }
export interface ReturnStmtNode    extends ASTNode { kind: NodeKind.ReturnStmt;    value: ASTNode | null }

// ------------------------------------------------------------------ //
// ParseError
// ------------------------------------------------------------------ //

export class ParseError extends Error {
    readonly token: Token | null;

    constructor(message: string, token: Token | null = null) {
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
 * @param tokens - Output of `tokenize()`
 * @returns ProgramNode representing the full source file
 * @throws {ParseError} On unexpected tokens
 */
export function parse(tokens: Token[]): ProgramNode {
    let pos = 0;

    // ---- Helpers ----

    function current(): Token  { return tokens[pos]; }
    function peek(offset = 0): Token { return tokens[pos + offset] ?? tokens[tokens.length - 1]; }

    function advance(): Token {
        const tok = tokens[pos];
        if (tok.type !== TokenType.EOF) pos++;
        return tok;
    }

    function check(type: TokenType, value?: string): boolean {
        const tok = current();
        return tok.type === type && (value === undefined || tok.value === value);
    }

    function expect(type: TokenType, value?: string): Token {
        const tok = current();
        if (tok.type !== type) {
            throw new ParseError(`Expected ${type} but got ${tok.type} ("${tok.value}")`, tok);
        }
        if (value !== undefined && tok.value !== value) {
            throw new ParseError(`Expected "${value}" but got "${tok.value}"`, tok);
        }
        return advance();
    }

    function eat(type: TokenType, value?: string): boolean {
        if (check(type, value)) { advance(); return true; }
        return false;
    }

    function loc(): { line: number; col: number } {
        return { line: current().line, col: current().col };
    }

    // ---- AST helpers ----

    function node<T extends ASTNode>(kind: NodeKind, extra: Omit<T, 'kind'>): T {
        return { kind, ...extra } as T;
    }

    // ---- Top-level ----

    function parseProgram(): ProgramNode {
        const body: ASTNode[] = [];
        while (!check(TokenType.EOF)) {
            try {
                body.push(parseStatement());
            } catch (e) {
                if (e instanceof ParseError) {
                    // Skip to next semicolon or closing brace for error recovery
                    while (!check(TokenType.EOF) && !check(TokenType.SEMICOLON) && !check(TokenType.RBRACE)) advance();
                    if (check(TokenType.SEMICOLON)) advance();
                } else throw e;
            }
        }
        return node<ProgramNode>(NodeKind.Program, { body });
    }

    // ---- Statements ----

    function parseStatement(): ASTNode {
        const tok = current();

        if (tok.type === TokenType.KEYWORD) {
            switch (tok.value) {
                case 'Tensor':    return parseTensorDecl();
                case 'Cluster':   return parseClusterDecl();
                case 'Model':     return parseModelDecl();
                case 'Pipeline':  return parsePipelineDecl();
                case 'if':        return parseIfStmt();
                case 'for':       return parseForStmt();
                case 'while':     return parseWhileStmt();
                case 'foreach':   return parseForEachGlyph();
                case 'parallel':  return parseParallelFor();
                case 'Train':     return parseTrainStmt();
                case 'return':    return parseReturnStmt();
                default: break;
            }
        }

        return parseExpressionStatement();
    }

    function parseTensorDecl(): TensorDeclNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'Tensor');
        const name = expect(TokenType.IDENTIFIER).value as string;

        // Optional <TypeParams>
        let typeParams: string[] | undefined;
        if (check(TokenType.LANGLE)) {
            advance();
            typeParams = [];
            while (!check(TokenType.RANGLE) && !check(TokenType.EOF)) {
                typeParams.push(expect(TokenType.IDENTIFIER).value as string);
                eat(TokenType.COMMA);
            }
            expect(TokenType.RANGLE);
        }

        expect(TokenType.EQUALS);
        const init = parseExpression();
        eat(TokenType.SEMICOLON);
        return node<TensorDeclNode>(NodeKind.TensorDecl, { name, typeParams, init, line, col });
    }

    function parseClusterDecl(): ClusterDeclNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'Cluster');
        const name = expect(TokenType.IDENTIFIER).value as string;
        const body = parseBlock().body;
        return node<ClusterDeclNode>(NodeKind.ClusterDecl, { name, body, line, col });
    }

    function parseModelDecl(): ModelDeclNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'Model');
        const name = expect(TokenType.IDENTIFIER).value as string;
        const body = parseBlock().body;
        return node<ModelDeclNode>(NodeKind.ModelDecl, { name, body, line, col });
    }

    function parsePipelineDecl(): PipelineDeclNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'Pipeline');
        const name = expect(TokenType.IDENTIFIER).value as string;
        const body = parseBlock().body;
        return node<PipelineDeclNode>(NodeKind.PipelineDecl, { name, body, line, col });
    }

    function parseBlock(): BlockNode {
        const { line, col } = loc();
        expect(TokenType.LBRACE);
        const body: ASTNode[] = [];
        while (!check(TokenType.RBRACE) && !check(TokenType.EOF)) {
            body.push(parseStatement());
        }
        expect(TokenType.RBRACE);
        return node<BlockNode>(NodeKind.Block, { body, line, col });
    }

    function parseIfStmt(): IfStmtNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'if');
        expect(TokenType.LPAREN);
        const test = parseExpression();
        expect(TokenType.RPAREN);
        const consequent = parseBlock();
        let alternate: ASTNode | undefined;
        if (check(TokenType.KEYWORD, 'else')) {
            advance();
            alternate = check(TokenType.KEYWORD, 'if') ? parseIfStmt() : parseBlock();
        }
        return node<IfStmtNode>(NodeKind.IfStmt, { test, consequent, alternate, line, col });
    }

    function parseForStmt(): ForStmtNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'for');
        expect(TokenType.LPAREN);

        let init: ASTNode | null = null;
        if (!check(TokenType.SEMICOLON)) init = parseStatement();
        else advance();

        let test: ASTNode | null = null;
        if (!check(TokenType.SEMICOLON)) test = parseExpression();
        eat(TokenType.SEMICOLON);

        let update: ASTNode | null = null;
        if (!check(TokenType.RPAREN)) update = parseExpression();
        expect(TokenType.RPAREN);

        const body = parseBlock();
        return node<ForStmtNode>(NodeKind.ForStmt, { init, test, update, body, line, col });
    }

    function parseWhileStmt(): WhileStmtNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'while');
        expect(TokenType.LPAREN);
        const test = parseExpression();
        expect(TokenType.RPAREN);
        const body = parseBlock();
        return node<WhileStmtNode>(NodeKind.WhileStmt, { test, body, line, col });
    }

    function parseForEachGlyph(): ForEachGlyphNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'foreach');
        expect(TokenType.KEYWORD, 'glyph');
        const glyphTok = expect(TokenType.GLYPH);
        const glyph = glyphTok.value as string;
        expect(TokenType.KEYWORD, 'in');
        const target = expect(TokenType.IDENTIFIER).value as string;
        const source = parseExpression();
        const body = parseBlock();
        return node<ForEachGlyphNode>(NodeKind.ForEachGlyph, { glyph, target, source, body, line, col });
    }

    function parseParallelFor(): ParallelForNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'parallel');
        expect(TokenType.KEYWORD, 'for');
        expect(TokenType.LPAREN);
        const variable = expect(TokenType.IDENTIFIER).value as string;
        expect(TokenType.KEYWORD, 'in');
        const iterable = parseExpression();
        expect(TokenType.RPAREN);
        const body = parseBlock();
        return node<ParallelForNode>(NodeKind.ParallelFor, { variable, iterable, body, line, col });
    }

    function parseTrainStmt(): TrainStmtNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'Train');
        const model = parseExpression();
        let options: ASTNode | null = null;
        if (check(TokenType.KEYWORD, 'with')) {
            advance();
            options = parseExpression();
        }
        eat(TokenType.SEMICOLON);
        return node<TrainStmtNode>(NodeKind.TrainStmt, { model, options, line, col });
    }

    function parseReturnStmt(): ReturnStmtNode {
        const { line, col } = loc();
        expect(TokenType.KEYWORD, 'return');
        let value: ASTNode | null = null;
        if (!check(TokenType.SEMICOLON) && !check(TokenType.EOF)) {
            value = parseExpression();
        }
        eat(TokenType.SEMICOLON);
        return node<ReturnStmtNode>(NodeKind.ReturnStmt, { value, line, col });
    }

    function parseExpressionStatement(): ASTNode {
        const expr = parseExpression();
        eat(TokenType.SEMICOLON);
        return expr;
    }

    // ---- Expressions ----

    function parseExpression(): ASTNode {
        return parseAssignment();
    }

    function parseAssignment(): ASTNode {
        const left = parseGlyphOp();
        if (check(TokenType.EQUALS)) {
            const { line, col } = loc();
            advance();
            const value = parseAssignment();
            return node<AssignmentNode>(NodeKind.Assignment, { target: left, value, line, col });
        }
        return left;
    }

    function parseGlyphOp(): ASTNode {
        let left = parseBinaryOr();
        while (check(TokenType.GLYPH)) {
            const { line, col } = loc();
            const glyph = advance().value as string;
            const right = parseBinaryOr();
            left = node<GlyphOpNode>(NodeKind.GlyphOp, { glyph, left, right, line, col });
        }
        return left;
    }

    function parseBinaryOr(): ASTNode {
        let left = parseBinaryAnd();
        while (check(TokenType.PIPE)) {
            const { line, col } = loc();
            const op = advance().value as string;
            left = node<BinaryExprNode>(NodeKind.BinaryExpr, { op, left, right: parseBinaryAnd(), line, col });
        }
        return left;
    }

    function parseBinaryAnd(): ASTNode {
        return parseComparison(); // simplified — extend as needed
    }

    function parseComparison(): ASTNode {
        let left = parseAdditive();
        while (
            check(TokenType.LANGLE) || check(TokenType.RANGLE) ||
            (check(TokenType.EQUALS) && peek(1).type === TokenType.EQUALS)
        ) {
            const { line, col } = loc();
            const op = advance().value as string;
            left = node<BinaryExprNode>(NodeKind.BinaryExpr, { op, left, right: parseAdditive(), line, col });
        }
        return left;
    }

    function parseAdditive(): ASTNode {
        let left = parseMultiplicative();
        while (check(TokenType.PLUS) || check(TokenType.MINUS)) {
            const { line, col } = loc();
            const op = advance().value as string;
            left = node<BinaryExprNode>(NodeKind.BinaryExpr, { op, left, right: parseMultiplicative(), line, col });
        }
        return left;
    }

    function parseMultiplicative(): ASTNode {
        let left = parseUnary();
        while (check(TokenType.STAR) || check(TokenType.SLASH)) {
            const { line, col } = loc();
            const op = advance().value as string;
            left = node<BinaryExprNode>(NodeKind.BinaryExpr, { op, left, right: parseUnary(), line, col });
        }
        return left;
    }

    function parseUnary(): ASTNode {
        if (check(TokenType.MINUS)) {
            const { line, col } = loc();
            const op = advance().value as string;
            return node<UnaryExprNode>(NodeKind.UnaryExpr, { op, operand: parseUnary(), line, col });
        }
        return parsePostfix();
    }

    function parsePostfix(): ASTNode {
        let expr = parsePrimary();
        while (true) {
            const { line, col } = loc();
            if (check(TokenType.DOT)) {
                advance();
                const prop = expect(TokenType.IDENTIFIER).value as string;
                if (check(TokenType.LPAREN)) {
                    const args = parseArgList();
                    expr = node<MethodCallNode>(NodeKind.MethodCall, { object: expr, method: prop, args, line, col });
                } else {
                    expr = node<MemberExprNode>(NodeKind.MemberExpr, { object: expr, property: prop, line, col });
                }
            } else if (check(TokenType.LBRACKET)) {
                advance();
                const index = parseExpression();
                expect(TokenType.RBRACKET);
                expr = node<IndexExprNode>(NodeKind.IndexExpr, { object: expr, index, line, col });
            } else {
                break;
            }
        }
        return expr;
    }

    function parsePrimary(): ASTNode {
        const tok = current();
        const { line, col } = loc();

        if (tok.type === TokenType.NUMBER) {
            advance();
            return node<NumberLiteralNode>(NodeKind.NumberLiteral, { value: tok.value as number, line, col });
        }

        if (tok.type === TokenType.PI_EXPR) {
            advance();
            return node<PiExprNode>(NodeKind.PiExpr, { coefficient: tok.value as number, line, col });
        }

        if (tok.type === TokenType.PI) {
            advance();
            return node<PiExprNode>(NodeKind.PiExpr, { coefficient: 1, line, col });
        }

        if (tok.type === TokenType.STRING) {
            advance();
            return node<StringLiteralNode>(NodeKind.StringLiteral, { value: tok.value as string, line, col });
        }

        if (tok.type === TokenType.IDENTIFIER) {
            advance();
            // Function call?
            if (check(TokenType.LPAREN)) {
                const args = parseArgList();
                return node<FunctionCallNode>(NodeKind.FunctionCall, { callee: tok.value as string, args, line, col });
            }
            return node<IdentifierNode>(NodeKind.Identifier, { name: tok.value as string, line, col });
        }

        if (tok.type === TokenType.KEYWORD) {
            // Treat built-in keywords used as expressions (may be function calls)
            advance();
            if (check(TokenType.LPAREN)) {
                const args = parseArgList();
                return node<FunctionCallNode>(NodeKind.FunctionCall, { callee: tok.value as string, args, line, col });
            }
            return node<IdentifierNode>(NodeKind.Identifier, { name: tok.value as string, line, col });
        }

        if (tok.type === TokenType.LPAREN) {
            advance();
            const expr = parseExpression();
            expect(TokenType.RPAREN);
            return expr;
        }

        if (tok.type === TokenType.LBRACKET) {
            return parseArrayLiteral();
        }

        if (tok.type === TokenType.LBRACE) {
            return parseObjectLiteral();
        }

        throw new ParseError(`Unexpected token "${tok.value}" (${tok.type})`, tok);
    }

    function parseArgList(): ASTNode[] {
        expect(TokenType.LPAREN);
        const args: ASTNode[] = [];
        while (!check(TokenType.RPAREN) && !check(TokenType.EOF)) {
            args.push(parseExpression());
            eat(TokenType.COMMA);
        }
        expect(TokenType.RPAREN);
        return args;
    }

    function parseArrayLiteral(): ArrayLiteralNode {
        const { line, col } = loc();
        expect(TokenType.LBRACKET);
        const elements: ASTNode[] = [];
        while (!check(TokenType.RBRACKET) && !check(TokenType.EOF)) {
            elements.push(parseExpression());
            eat(TokenType.COMMA);
        }
        expect(TokenType.RBRACKET);
        return node<ArrayLiteralNode>(NodeKind.ArrayLiteral, { elements, line, col });
    }

    function parseObjectLiteral(): ObjectLiteralNode {
        const { line, col } = loc();
        expect(TokenType.LBRACE);
        const properties: Array<{ key: string; value: ASTNode }> = [];
        while (!check(TokenType.RBRACE) && !check(TokenType.EOF)) {
            const key = (check(TokenType.STRING) || check(TokenType.IDENTIFIER) || check(TokenType.KEYWORD))
                ? advance().value as string
                : expect(TokenType.IDENTIFIER).value as string;
            expect(TokenType.COLON);
            const value = parseExpression();
            properties.push({ key, value });
            eat(TokenType.COMMA);
        }
        expect(TokenType.RBRACE);
        return node<ObjectLiteralNode>(NodeKind.ObjectLiteral, { properties, line, col });
    }

    return parseProgram();
}
