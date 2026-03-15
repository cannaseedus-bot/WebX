// K'UHUL++ v2.0 Semantic Analyzer — TypeScript Edition
// Validates the AST produced by the Parser:
//   - Ensures identifiers are declared before use
//   - Validates π-phase values are in the range [0, 2]
//   - Verifies glyph symbols are recognised
//   - Infers and annotates types on AST nodes
//   - Collects errors and warnings without throwing

import { NodeKind } from './parser.js';
import { GLYPHS } from './lexer.js';
import type { ASTNode, ProgramNode } from './parser.js';

// ------------------------------------------------------------------ //
// Symbol table (Scope)
// ------------------------------------------------------------------ //

interface SymbolInfo {
    kind: 'tensor' | 'cluster' | 'model' | 'pipeline' | 'builtin' | 'param';
    type?: string;
}

class Scope {
    readonly parent: Scope | null;
    private symbols = new Map<string, SymbolInfo>();

    constructor(parent: Scope | null = null) {
        this.parent = parent;
    }

    define(name: string, info: SymbolInfo): void {
        this.symbols.set(name, info);
    }

    lookup(name: string): SymbolInfo | null {
        if (this.symbols.has(name)) return this.symbols.get(name)!;
        return this.parent ? this.parent.lookup(name) : null;
    }

    child(): Scope {
        return new Scope(this);
    }
}

// ------------------------------------------------------------------ //
// SemanticError
// ------------------------------------------------------------------ //

/** A non-fatal semantic error collected during analysis */
export class SemanticError {
    readonly message: string;
    readonly node: ASTNode | null;

    constructor(message: string, astNode: ASTNode | null = null) {
        this.message = message;
        this.node    = astNode;
    }

    toString(): string {
        const loc = this.node?.line != null ? ` at line ${this.node.line}` : '';
        return `SemanticError${loc} — ${this.message}`;
    }
}

// ------------------------------------------------------------------ //
// AnnotatedAST
// ------------------------------------------------------------------ //

/** AST decorated with type information after semantic analysis */
export interface AnnotatedAST extends ProgramNode {
    /** Populated by the analyzer */
    typeAnnotations: Map<ASTNode, string>;
}

// ------------------------------------------------------------------ //
// AnalysisResult
// ------------------------------------------------------------------ //

export interface AnalysisResult {
    errors:   SemanticError[];
    warnings: string[];
    ast:      AnnotatedAST;
}

// ------------------------------------------------------------------ //
// Analyzer
// ------------------------------------------------------------------ //

/**
 * Perform semantic analysis on a K'UHUL++ AST.
 *
 * @param ast - AST produced by `parse()`
 * @returns AnalysisResult containing errors, warnings, and annotated AST
 */
export function analyze(ast: ProgramNode): AnalysisResult {
    const errors:   SemanticError[] = [];
    const warnings: string[]        = [];
    const typeAnnotations           = new Map<ASTNode, string>();

    // Pre-populate global scope with built-in names
    const globalScope = new Scope();
    for (const name of [
        'GPU', 'Math',
        'generate_spiral', 'load_dataset', 'split_tensors', 'merge_clusters',
        'GeometricTensor', 'TensorCluster', 'GeometricModel',
        'running', 'universe', 'input_data', 'gradients',
        'true', 'false', 'null', 'undefined',
    ]) {
        globalScope.define(name, { kind: 'builtin' });
    }

    function annotate(node: ASTNode, type: string): void {
        typeAnnotations.set(node, type);
    }

    function error(msg: string, node: ASTNode | null = null): void {
        errors.push(new SemanticError(msg, node));
    }

    function warn(msg: string): void {
        warnings.push(msg);
    }

    // ---- Walk ----

    function visitNode(node: ASTNode, scope: Scope): void {
        switch (node.kind) {
            case NodeKind.Program: {
                const n = node as any;
                for (const stmt of n.body) visitNode(stmt, scope);
                break;
            }
            case NodeKind.TensorDecl: {
                const n = node as any;
                visitNode(n.init, scope);
                scope.define(n.name, { kind: 'tensor', type: inferType(n.init) });
                annotate(node, 'Tensor');
                break;
            }
            case NodeKind.ClusterDecl: {
                const n = node as any;
                const child = scope.child();
                scope.define(n.name, { kind: 'cluster' });
                for (const stmt of n.body) visitNode(stmt, child);
                break;
            }
            case NodeKind.ModelDecl: {
                const n = node as any;
                const child = scope.child();
                scope.define(n.name, { kind: 'model' });
                for (const stmt of n.body) visitNode(stmt, child);
                break;
            }
            case NodeKind.PipelineDecl: {
                const n = node as any;
                const child = scope.child();
                scope.define(n.name, { kind: 'pipeline' });
                for (const stmt of n.body) visitNode(stmt, child);
                break;
            }
            case NodeKind.GlyphOp: {
                const n = node as any;
                if (!GLYPHS.has(n.glyph)) {
                    error(`Unknown glyph "${n.glyph}"`, node);
                }
                visitNode(n.left, scope);
                visitNode(n.right, scope);
                annotate(node, 'Tensor');
                break;
            }
            case NodeKind.Assignment: {
                const n = node as any;
                visitNode(n.value, scope);
                visitNode(n.target, scope);
                break;
            }
            case NodeKind.IfStmt: {
                const n = node as any;
                visitNode(n.test, scope);
                visitNode(n.consequent, scope.child());
                if (n.alternate) visitNode(n.alternate, scope.child());
                break;
            }
            case NodeKind.ForStmt: {
                const n = node as any;
                const child = scope.child();
                if (n.init)   visitNode(n.init, child);
                if (n.test)   visitNode(n.test, child);
                if (n.update) visitNode(n.update, child);
                visitNode(n.body, child);
                break;
            }
            case NodeKind.WhileStmt: {
                const n = node as any;
                visitNode(n.test, scope);
                visitNode(n.body, scope.child());
                break;
            }
            case NodeKind.Block: {
                const n = node as any;
                const child = scope.child();
                for (const stmt of n.body) visitNode(stmt, child);
                break;
            }
            case NodeKind.Identifier: {
                const n = node as any;
                const sym = scope.lookup(n.name);
                if (!sym) {
                    error(`Identifier "${n.name}" is not declared`, node);
                } else {
                    annotate(node, sym.type ?? sym.kind);
                }
                break;
            }
            case NodeKind.PiExpr: {
                const n = node as any;
                if (n.coefficient < 0 || n.coefficient > 2) {
                    warn(`π-coefficient ${n.coefficient} is outside the expected range [0, 2]`);
                }
                annotate(node, 'float');
                break;
            }
            case NodeKind.NumberLiteral:
                annotate(node, 'number');
                break;
            case NodeKind.StringLiteral:
                annotate(node, 'string');
                break;
            case NodeKind.BinaryExpr: {
                const n = node as any;
                visitNode(n.left, scope);
                visitNode(n.right, scope);
                annotate(node, 'number');
                break;
            }
            case NodeKind.UnaryExpr: {
                const n = node as any;
                visitNode(n.operand, scope);
                annotate(node, 'number');
                break;
            }
            case NodeKind.FunctionCall: {
                const n = node as any;
                if (!scope.lookup(n.callee)) {
                    // Allow all known built-ins silently
                    warn(`Function "${n.callee}" may not be defined at this point`);
                }
                for (const arg of n.args) visitNode(arg, scope);
                annotate(node, 'any');
                break;
            }
            case NodeKind.MethodCall: {
                const n = node as any;
                visitNode(n.object, scope);
                for (const arg of n.args) visitNode(arg, scope);
                annotate(node, 'any');
                break;
            }
            case NodeKind.MemberExpr: {
                const n = node as any;
                visitNode(n.object, scope);
                annotate(node, 'any');
                break;
            }
            case NodeKind.IndexExpr: {
                const n = node as any;
                visitNode(n.object, scope);
                visitNode(n.index, scope);
                annotate(node, 'any');
                break;
            }
            case NodeKind.ArrayLiteral: {
                const n = node as any;
                for (const el of n.elements) visitNode(el, scope);
                annotate(node, 'array');
                break;
            }
            case NodeKind.ObjectLiteral: {
                const n = node as any;
                for (const prop of n.properties) visitNode(prop.value, scope);
                annotate(node, 'object');
                break;
            }
            case NodeKind.TrainStmt: {
                const n = node as any;
                visitNode(n.model, scope);
                if (n.options) visitNode(n.options, scope);
                break;
            }
            case NodeKind.ReturnStmt: {
                const n = node as any;
                if (n.value) visitNode(n.value, scope);
                break;
            }
            case NodeKind.ForEachGlyph: {
                const n = node as any;
                if (!GLYPHS.has(n.glyph)) {
                    error(`Unknown glyph "${n.glyph}" in foreach statement`, node);
                }
                visitNode(n.source, scope);
                const child = scope.child();
                child.define(n.target, { kind: 'tensor' });
                visitNode(n.body, child);
                break;
            }
            case NodeKind.ParallelFor: {
                const n = node as any;
                visitNode(n.iterable, scope);
                const child = scope.child();
                child.define(n.variable, { kind: 'tensor' });
                visitNode(n.body, child);
                break;
            }
            default:
                // Unknown node kinds pass through without error
                break;
        }
    }

    function inferType(node: ASTNode): string {
        switch (node.kind) {
            case NodeKind.NumberLiteral: return 'number';
            case NodeKind.StringLiteral: return 'string';
            case NodeKind.PiExpr:        return 'float';
            case NodeKind.ArrayLiteral:  return 'array';
            case NodeKind.GlyphOp:       return 'Tensor';
            default:                     return 'any';
        }
    }

    visitNode(ast, globalScope);

    const annotatedAST: AnnotatedAST = { ...ast, typeAnnotations };
    return { errors, warnings, ast: annotatedAST };
}
