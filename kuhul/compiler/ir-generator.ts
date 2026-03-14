// K'UHUL++ IR Generator
// Lowers an annotated AST to Geometric IR instructions.
// The generator walks the AST and emits SSA-style IR instructions
// for each construct using the IRBuilder.

import { NodeKind } from './parser.js';
import type { AnnotatedAST, ASTNode } from './semantic-analyzer.js';
import type { GeometricIR, KuhulType, IRInstruction } from '../ir/ir-types.js';
import { IRBuilder } from '../ir/ir-builder.js';

// ------------------------------------------------------------------ //
// IRGenerationError
// ------------------------------------------------------------------ //

export class IRGenerationError extends Error {
    readonly node: ASTNode | null;

    constructor(message: string, node: ASTNode | null = null) {
        const loc = node?.line != null ? ` at line ${node.line}` : '';
        super(`IRGenerationError${loc} — ${message}`);
        this.name = 'IRGenerationError';
        this.node = node;
    }
}

// ------------------------------------------------------------------ //
// IR Generator
// ------------------------------------------------------------------ //

/**
 * Generate Geometric IR from an annotated AST.
 *
 * @param ast - Annotated AST produced by `analyze()`
 * @returns GeometricIR ready for optimisation or code generation
 * @throws {IRGenerationError} On constructs that cannot be lowered
 */
export function generateIR(ast: AnnotatedAST): GeometricIR {
    const builder = new IRBuilder();
    let ssaCounter = 0;

    /** Create a fresh SSA id */
    function freshId(prefix = 'v'): string {
        return `${prefix}${ssaCounter++}`;
    }

    /** Lower an expression node; return the SSA id holding its result */
    function lowerExpr(node: ASTNode): string {
        switch (node.kind) {
            case NodeKind.NumberLiteral: {
                const n = node as any;
                const id = freshId('c');
                builder.addConst(id, n.value, { kind: 'scalar', dtype: 'float32' });
                return id;
            }

            case NodeKind.StringLiteral: {
                const n = node as any;
                const id = freshId('s');
                builder.addConst(id, n.value, { kind: 'string' });
                return id;
            }

            case NodeKind.PiExpr: {
                const n = node as any;
                const id = freshId('pi');
                builder.addConst(id, n.coefficient * Math.PI, { kind: 'scalar', dtype: 'float64' });
                return id;
            }

            case NodeKind.Identifier: {
                const n = node as any;
                const id = freshId('r');
                const type = ast.typeAnnotations.get(node) ?? 'any';
                builder.addLoad(id, n.name, inferIRType(type));
                return id;
            }

            case NodeKind.BinaryExpr: {
                const n = node as any;
                const leftId  = lowerExpr(n.left);
                const rightId = lowerExpr(n.right);
                const id = freshId('bin');
                // Map arithmetic operators to glyph equivalents where possible
                const glyphOp = arithmeticToGlyph(n.op);
                if (glyphOp) {
                    builder.addGlyphOp(id, glyphOp, leftId, rightId, { kind: 'scalar', dtype: 'float32' });
                } else {
                    // Generic invoke for non-glyph ops
                    builder.addInvoke(id, `__op_${n.op}`, [leftId, rightId], { kind: 'scalar', dtype: 'float32' });
                }
                return id;
            }

            case NodeKind.UnaryExpr: {
                const n = node as any;
                const operandId = lowerExpr(n.operand);
                const id = freshId('un');
                builder.addInvoke(id, `__unary_${n.op}`, [operandId], { kind: 'scalar', dtype: 'float32' });
                return id;
            }

            case NodeKind.GlyphOp: {
                const n = node as any;
                const leftId  = lowerExpr(n.left);
                const rightId = lowerExpr(n.right);
                const id = freshId('g');
                builder.addGlyphOp(id, n.glyph, leftId, rightId,
                    { kind: 'tensor', dtype: 'float32', shape: [] });
                return id;
            }

            case NodeKind.FunctionCall: {
                const n = node as any;
                const argIds = n.args.map((a: ASTNode) => lowerExpr(a));
                const id = freshId('call');
                builder.addInvoke(id, n.callee, argIds, { kind: 'scalar', dtype: 'float32' });
                return id;
            }

            case NodeKind.MethodCall: {
                const n = node as any;
                const objId = lowerExpr(n.object);
                const argIds = n.args.map((a: ASTNode) => lowerExpr(a));
                const id = freshId('meth');
                builder.addInvoke(id, `${n.method}`, [objId, ...argIds], { kind: 'scalar', dtype: 'float32' });
                return id;
            }

            case NodeKind.MemberExpr: {
                const n = node as any;
                const objId = lowerExpr(n.object);
                const id = freshId('mem');
                builder.addInvoke(id, `__member_${n.property}`, [objId], { kind: 'scalar', dtype: 'float32' });
                return id;
            }

            case NodeKind.IndexExpr: {
                const n = node as any;
                const objId   = lowerExpr(n.object);
                const indexId = lowerExpr(n.index);
                const id = freshId('idx');
                builder.addInvoke(id, '__index', [objId, indexId], { kind: 'scalar', dtype: 'float32' });
                return id;
            }

            case NodeKind.ArrayLiteral: {
                const n = node as any;
                const elementIds = n.elements.map((e: ASTNode) => lowerExpr(e));
                const id = freshId('arr');
                builder.addInvoke(id, '__array', elementIds,
                    { kind: 'tensor', dtype: 'float32', shape: [n.elements.length] });
                return id;
            }

            default:
                throw new IRGenerationError(`Cannot lower expression node "${node.kind}"`, node);
        }
    }

    /** Lower a statement node; returns void */
    function lowerStmt(node: ASTNode): void {
        switch (node.kind) {
            case NodeKind.TensorDecl: {
                const n = node as any;
                const initId = lowerExpr(n.init);
                const allocId = freshId('alloc');
                builder.addAlloc(allocId, { kind: 'tensor', dtype: 'float32', shape: [] });
                builder.addStore(`_store_${n.name}`, n.name, initId);
                break;
            }

            case NodeKind.Assignment: {
                const n = node as any;
                const valId = lowerExpr(n.value);
                const targetName = n.target.name ?? freshId('tgt');
                builder.addStore(`_store_${targetName}`, targetName, valId);
                break;
            }

            case NodeKind.Block:
            case NodeKind.Program: {
                const n = node as any;
                for (const stmt of n.body) lowerStmt(stmt);
                break;
            }

            case NodeKind.IfStmt: {
                const n = node as any;
                const condId    = lowerExpr(n.test);
                const trueLabel = freshId('if_true');
                const falseLabel= freshId('if_false');
                const endLabel  = freshId('if_end');

                builder.addCondBranch(freshId('cb'), condId, trueLabel, falseLabel);
                builder.addLabel(freshId('lbl'), trueLabel);
                lowerStmt(n.consequent);
                builder.addBranch(freshId('br'), endLabel);
                builder.addLabel(freshId('lbl'), falseLabel);
                if (n.alternate) lowerStmt(n.alternate);
                builder.addLabel(freshId('lbl'), endLabel);
                break;
            }

            case NodeKind.WhileStmt: {
                const n = node as any;
                const loopLabel = freshId('while_top');
                const bodyLabel = freshId('while_body');
                const exitLabel = freshId('while_exit');

                builder.addLabel(freshId('lbl'), loopLabel);
                const condId = lowerExpr(n.test);
                builder.addCondBranch(freshId('cb'), condId, bodyLabel, exitLabel);
                builder.addLabel(freshId('lbl'), bodyLabel);
                lowerStmt(n.body);
                builder.addBranch(freshId('br'), loopLabel);
                builder.addLabel(freshId('lbl'), exitLabel);
                break;
            }

            case NodeKind.TrainStmt: {
                const n = node as any;
                const modelId = lowerExpr(n.model);
                const optIds  = n.options ? [lowerExpr(n.options)] : [];
                builder.addInvoke(freshId('train'), '__train', [modelId, ...optIds],
                    { kind: 'scalar', dtype: 'float32' });
                break;
            }

            case NodeKind.ReturnStmt: {
                const n = node as any;
                const valId = n.value ? lowerExpr(n.value) : undefined;
                builder.addReturn(freshId('ret'), valId);
                break;
            }

            case NodeKind.FunctionCall:
            case NodeKind.MethodCall:
            case NodeKind.GlyphOp:
            case NodeKind.BinaryExpr: {
                // Expression used as statement — lower and discard result
                lowerExpr(node);
                break;
            }

            default:
                // Unknown statement kinds are silently skipped
                break;
        }
    }

    // Begin a default phase covering the full 0..2π cycle
    builder.beginPhase('main', 0, 2 * Math.PI);
    lowerStmt(ast);
    builder.endPhase();

    return builder.build();
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function inferIRType(type: string): KuhulType {
    switch (type) {
        case 'number':  return { kind: 'scalar', dtype: 'float32' };
        case 'float':   return { kind: 'scalar', dtype: 'float64' };
        case 'string':  return { kind: 'string' };
        case 'Tensor':
        case 'tensor':
        case 'array':   return { kind: 'tensor', dtype: 'float32', shape: [] };
        default:        return { kind: 'scalar', dtype: 'float32' };
    }
}

/** Map basic arithmetic operators to manifold glyph ops where natural */
function arithmeticToGlyph(op: string): string | null {
    switch (op) {
        case '+': return '⊕';
        case '-': return '⊖';
        case '*': return '⊗';
        default:  return null;
    }
}
