// K'UHUL++ v2.0 Semantic Analyzer
// Validates an AST produced by the Parser:
//   - Ensures identifiers are declared before use
//   - Validates π-phase values are in the range [0, 2]
//   - Verifies glyph symbols are recognised
//   - Collects and reports semantic errors without throwing

import { NodeKind } from './parser.js';
import { GLYPHS } from './lexer.js';

// ------------------------------------------------------------------ //
// Scope (symbol table)
// ------------------------------------------------------------------ //

class Scope {
    constructor(parent = null) {
        this.parent  = parent;
        this.symbols = new Map();
    }

    /** Declare a symbol in this scope */
    define(name, info) {
        this.symbols.set(name, info);
    }

    /** Look up a symbol, walking parent scopes */
    lookup(name) {
        if (this.symbols.has(name)) return this.symbols.get(name);
        return this.parent ? this.parent.lookup(name) : null;
    }

    child() {
        return new Scope(this);
    }
}

// ------------------------------------------------------------------ //
// SemanticError
// ------------------------------------------------------------------ //

export class SemanticError {
    /**
     * @param {string} message
     * @param {object|null} node - AST node where the error occurred
     */
    constructor(message, astNode = null) {
        this.message = message;
        this.node    = astNode;
    }

    toString() {
        return `SemanticError — ${this.message}`;
    }
}

// ------------------------------------------------------------------ //
// Analyzer
// ------------------------------------------------------------------ //

/**
 * Perform semantic analysis on a K'UHUL++ AST.
 *
 * @param {{ kind: string, body: object[] }} ast - AST from parse()
 * @returns {{ errors: SemanticError[], warnings: string[] }}
 */
export function analyze(ast) {
    const errors   = [];
    const warnings = [];

    // Pre-define built-in identifiers and keywords as always in scope
    const globalScope = new Scope();
    const builtins = [
        'GPU', 'Math',
        'generate_spiral', 'load_dataset', 'split_tensors', 'merge_clusters',
        'GeometricTensor', 'TensorCluster', 'GeometricModel',
        'running', 'universe', 'input_data', 'gradients',
    ];
    for (const name of builtins) {
        globalScope.define(name, { kind: 'builtin' });
    }

    // ---- helpers ----

    function error(msg, n) {
        errors.push(new SemanticError(msg, n));
    }

    function warn(msg) {
        warnings.push(msg);
    }

    /**
     * Validate a π-phase value.
     * In K'UHUL++ PiExpr nodes the coefficient represents multiples of π.
     * Valid range is [0, 2] (i.e. 0π to 2π).
     */
    function validatePiPhase(n) {
        if (n.kind !== NodeKind.PiExpr) return;
        const coeff = n.coefficient;
        if (typeof coeff !== 'number') return;
        if (coeff < 0 || coeff > 2) {
            error(`π-phase coefficient ${coeff} is outside the valid range [0, 2] (i.e. [0, 2π])`, n);
        }
    }

    /**
     * Validate a glyph symbol.
     */
    function validateGlyph(glyph, n) {
        if (!GLYPHS.has(glyph)) {
            error(`Unknown glyph symbol "${glyph}"`, n);
        }
    }

    // ---- visitor ----

    function visitNode(n, scope) {
        if (!n || typeof n !== 'object') return;

        switch (n.kind) {
            case NodeKind.Program:
                visitProgram(n, scope);
                break;

            case NodeKind.TensorDecl:
            case NodeKind.ClusterDecl:
            case NodeKind.ModelDecl:
            case NodeKind.PipelineDecl:
                visitDecl(n, scope);
                break;

            case NodeKind.Assignment:
                visitAssignment(n, scope);
                break;

            case NodeKind.GlyphOp:
                visitGlyphOp(n, scope);
                break;

            case NodeKind.NativeBlock:
                visitNativeBlock(n, scope);
                break;

            case NodeKind.IfStmt:
                visitIf(n, scope);
                break;

            case NodeKind.ForStmt:
            case NodeKind.ParallelFor:
                visitFor(n, scope);
                break;

            case NodeKind.WhileStmt:
                visitWhile(n, scope);
                break;

            case NodeKind.ForEachGlyph:
                visitForEachGlyph(n, scope);
                break;

            case NodeKind.TrainStmt:
                visitTrain(n, scope);
                break;

            case NodeKind.FunctionCall:
            case NodeKind.MethodCall:
                visitCall(n, scope);
                break;

            case NodeKind.BinaryExpr:
                visitNode(n.left, scope);
                visitNode(n.right, scope);
                break;

            case NodeKind.UnaryExpr:
                visitNode(n.operand, scope);
                break;

            case NodeKind.MemberExpr:
                visitNode(n.object, scope);
                break;

            case NodeKind.IndexExpr:
                visitNode(n.object, scope);
                visitNode(n.index, scope);
                break;

            case NodeKind.ArrayLiteral:
                for (const el of n.elements) visitNode(el, scope);
                break;

            case NodeKind.ObjectLiteral:
                for (const val of Object.values(n.properties)) visitNode(val, scope);
                break;

            case NodeKind.Vector3Literal:
                visitNode(n.x, scope);
                visitNode(n.y, scope);
                visitNode(n.z, scope);
                break;

            case NodeKind.PiExpr:
                validatePiPhase(n);
                break;

            case NodeKind.Identifier:
                visitIdentifier(n, scope);
                break;

            case NodeKind.Block:
                visitBlock(n, scope.child());
                break;

            case NodeKind.NumberLiteral:
            case NodeKind.StringLiteral:
                // No validation needed for literals
                break;

            default:
                // Walk any child nodes generically
                for (const [, child] of Object.entries(n)) {
                    if (child && typeof child === 'object' && child.kind) {
                        visitNode(child, scope);
                    } else if (Array.isArray(child)) {
                        for (const item of child) {
                            if (item && typeof item === 'object' && item.kind) {
                                visitNode(item, scope);
                            }
                        }
                    }
                }
        }
    }

    function visitProgram(n, scope) {
        // First pass: hoist all top-level declarations into scope so
        // forward references work.
        for (const stmt of n.body) {
            if (stmt.name && (
                stmt.kind === NodeKind.TensorDecl   ||
                stmt.kind === NodeKind.ClusterDecl  ||
                stmt.kind === NodeKind.ModelDecl    ||
                stmt.kind === NodeKind.PipelineDecl
            )) {
                scope.define(stmt.name, { kind: stmt.kind });
            }
        }
        for (const stmt of n.body) {
            visitNode(stmt, scope);
        }
    }

    function visitDecl(n, scope) {
        // Already hoisted in visitProgram; just validate the initialiser
        visitNode(n.init, scope);
        // If the init has a phase field, validate π-phase
        if (n.init && n.init.kind === NodeKind.FunctionCall) {
            for (const arg of n.init.args ?? []) {
                if (arg.key === 'phase') {
                    validatePiPhase(arg.value);
                }
            }
        }
    }

    function visitAssignment(n, scope) {
        // Resolve target
        const targetName = resolveIdentifierName(n.target);
        if (targetName && !scope.lookup(targetName)) {
            // Auto-define on assignment (K'UHUL++ allows implicit variable creation)
            scope.define(targetName, { kind: 'variable' });
        }
        visitNode(n.value, scope);
    }

    function visitGlyphOp(n, scope) {
        validateGlyph(n.glyph, n);
        // Validate param values
        for (const val of Object.values(n.params)) {
            visitNode(val, scope);
        }
        if (n.target) visitNode(n.target, scope);
    }

    function visitNativeBlock(n, scope) {
        const dx12Scope = scope.child();
        for (const stmt of n.stmts ?? []) {
            if (stmt.name) dx12Scope.define(stmt.name, { kind: 'dx12' });
            visitNode(stmt, dx12Scope);
        }
    }

    function visitIf(n, scope) {
        visitNode(n.condition, scope);
        visitNode(n.consequent, scope);
        if (n.alternate) visitNode(n.alternate, scope);
    }

    function visitFor(n, scope) {
        const loopScope = scope.child();
        loopScope.define(n.ident, { kind: 'loopVar' });
        visitNode(n.range, scope);
        visitNode(n.body, loopScope);
    }

    function visitWhile(n, scope) {
        visitNode(n.condition, scope);
        visitNode(n.body, scope);
    }

    function visitForEachGlyph(n, scope) {
        for (const g of n.glyphs) {
            if (!GLYPHS.has(g)) {
                error(`Unknown glyph "${g}" in foreach glyph list`, n);
            }
        }
        visitNode(n.body, scope);
    }

    function visitTrain(n, scope) {
        if (!scope.lookup(n.model)) {
            error(`Undefined model "${n.model}" in Train statement`, n);
        }
        if (!scope.lookup(n.data)) {
            error(`Undefined data cluster "${n.data}" in Train statement`, n);
        }
        if (n.body) visitNode(n.body, scope);
    }

    function visitCall(n, scope) {
        const callee = typeof n.object === 'string' ? n.object : resolveIdentifierName(n.object);
        if (callee && !scope.lookup(callee)) {
            warn(`Unresolved callee "${callee}" — may be a runtime built-in`);
        }
        for (const arg of n.args ?? []) {
            if (arg.value) visitNode(arg.value, scope);
        }
    }

    function visitBlock(n, scope) {
        for (const stmt of n.stmts ?? []) {
            visitNode(stmt, scope);
        }
    }

    function visitIdentifier(n, scope) {
        if (n.name && !scope.lookup(n.name)) {
            // Only warn for identifiers that look like user-defined names
            // (not single-word keywords used as values like 'spherical')
            if (/^[a-z_][a-z0-9_]*$/i.test(n.name)) {
                warn(`Identifier "${n.name}" may not be declared in this scope`);
            }
        }
    }

    // ---- utilities ----

    function resolveIdentifierName(n) {
        if (!n) return null;
        if (n.kind === NodeKind.Identifier) return n.name;
        if (n.kind === NodeKind.MemberExpr) return resolveIdentifierName(n.object);
        return null;
    }

    // ---- run ----

    visitNode(ast, globalScope);

    return { errors, warnings };
}
