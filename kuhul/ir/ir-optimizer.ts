// K'UHUL++ IR Optimizer
// Performs optimization passes over the Geometric IR before code generation.
// Passes: dead-code elimination, constant folding, and glyph transform fusion.

import type {
    GeometricIR, IRInstruction, IRConst, IRGlyphInstr, GlyphOp,
} from './ir-types.js';

// ------------------------------------------------------------------ //
// Pass interface
// ------------------------------------------------------------------ //

interface OptimizationPass {
    name: string;
    run(instructions: IRInstruction[]): IRInstruction[];
}

// ------------------------------------------------------------------ //
// Pass 1 — Dead code elimination
// ------------------------------------------------------------------ //

/**
 * Remove instructions whose result id is never referenced by any later
 * instruction, and which have no observable side effects (const, alloc, load).
 */
const deadCodeElimination: OptimizationPass = {
    name: 'dead-code-elimination',
    run(instructions) {
        // Collect all ids that are consumed
        const used = new Set<string>();
        for (const instr of instructions) {
            const i = instr as any;
            if (i.src)     used.add(i.src);
            if (i.left)    used.add(i.left);
            if (i.right)   used.add(i.right);
            if (i.cond)    used.add(i.cond);
            if (i.value && typeof i.value === 'string') used.add(i.value);
            if (i.args)    for (const a of i.args) used.add(a);
        }

        // Pure side-effect-free ops that can be eliminated if result is unused
        const pureOps = new Set(['const', 'load', 'alloc']);
        return instructions.filter(instr => {
            if (!pureOps.has(instr.op)) return true;
            return used.has(instr.id);
        });
    },
};

// ------------------------------------------------------------------ //
// Pass 2 — Constant folding
// ------------------------------------------------------------------ //

/**
 * Fold sequences of `const` followed by a glyph op where both operands are
 * compile-time constants.  Currently handles ⊕ (add) and ⊖ (subtract).
 */
const constantFolding: OptimizationPass = {
    name: 'constant-folding',
    run(instructions) {
        const constants = new Map<string, number>();
        const result: IRInstruction[] = [];

        for (const instr of instructions) {
            if (instr.op === 'const') {
                const c = instr as IRConst;
                if (typeof c.value === 'number') {
                    constants.set(c.id, c.value);
                }
                result.push(instr);
                continue;
            }

            // Try to fold binary glyph ops on numeric constants
            const isGlyph = (op: string): op is GlyphOp =>
                ['⊕', '⊖', '⊗', '⊞', '⊝'].includes(op);

            if (isGlyph(instr.op)) {
                const g = instr as IRGlyphInstr;
                const lv = constants.get(g.left);
                const rv = constants.get(g.right);

                if (lv !== undefined && rv !== undefined) {
                    let folded: number | null = null;
                    switch (g.op) {
                        case '⊕': folded = lv + rv; break;
                        case '⊖': folded = lv - rv; break;
                        case '⊗': folded = lv * rv; break;
                        case '⊞': folded = lv + rv; break;
                    }
                    if (folded !== null) {
                        constants.set(g.id, folded);
                        result.push({
                            op: 'const',
                            id: g.id,
                            value: folded,
                            type: g.type,
                        } satisfies IRConst);
                        continue;
                    }
                }
            }

            result.push(instr);
        }

        return result;
    },
};

// ------------------------------------------------------------------ //
// Pass 3 — Transform fusion
// ------------------------------------------------------------------ //

/**
 * Fuse consecutive glyph operations on the same tensor into a single
 * chained `invoke __fused` instruction when they can be composed.
 *
 * Currently fuses adjacent ⊗ (matrix multiply) chains.
 */
const transformFusion: OptimizationPass = {
    name: 'transform-fusion',
    run(instructions) {
        const result: IRInstruction[] = [];
        let i = 0;

        while (i < instructions.length) {
            const cur = instructions[i] as any;
            const next = instructions[i + 1] as any;

            // Fuse A ⊗ B followed immediately by (A⊗B) ⊗ C
            if (
                cur  && cur.op  === '⊗' &&
                next && next.op === '⊗' &&
                next.left === cur.id
            ) {
                result.push({
                    op: 'invoke',
                    id: next.id,
                    callee: '__fused_matmul',
                    args: [cur.left, cur.right, next.right],
                    returnType: next.type,
                });
                i += 2;
                continue;
            }

            result.push(instructions[i]);
            i++;
        }

        return result;
    },
};

// ------------------------------------------------------------------ //
// IROptimizer
// ------------------------------------------------------------------ //

/**
 * Applies a configurable sequence of optimization passes to a Geometric IR.
 *
 * @example
 * const optimizer = new IROptimizer();
 * const optimizedIR = optimizer.optimize(ir);
 */
export class IROptimizer {
    private passes: OptimizationPass[];

    /**
     * @param passes - Ordered list of passes (defaults to all built-in passes)
     */
    constructor(passes: OptimizationPass[] = [
        constantFolding,
        deadCodeElimination,
        transformFusion,
    ]) {
        this.passes = passes;
    }

    /**
     * Run all registered passes over the IR.
     *
     * @param ir - Input Geometric IR
     * @returns Optimised copy of the IR (original is unchanged)
     */
    optimize(ir: GeometricIR): GeometricIR {
        let instrs = [...ir.instructions];

        for (const pass of this.passes) {
            instrs = pass.run(instrs);
        }

        // Also optimise per-phase instruction lists
        const phases = ir.phases.map(phase => ({
            ...phase,
            instructions: this.passes.reduce(
                (acc, pass) => pass.run(acc),
                [...phase.instructions],
            ),
        }));

        return { ...ir, instructions: instrs, phases };
    }
}
