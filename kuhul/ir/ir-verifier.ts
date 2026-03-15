// K'UHUL++ IR Verifier
// Checks that a Geometric IR program is internally consistent before
// handing it off to code generation.  Verifies:
//   - All operand SSA ids are defined before use
//   - All branch targets reference defined labels
//   - Glyph ops use recognised glyph symbols
//   - Phase ranges are within [0, 2π]

import type { GeometricIR, IRInstruction, GlyphOp } from './ir-types.js';
import type { ValidationResult } from '../grammar/grammar-validator.js';

/** All recognised glyph operator characters */
const KNOWN_GLYPHS = new Set<GlyphOp | string>([
    '⊗', '⊕', '⊖', '⊛', '⊜', '⊝', '⊞',
    '⤍', '↻', '⟲', '∿', '⊙', '≋',
]);

// ------------------------------------------------------------------ //
// IRVerifier
// ------------------------------------------------------------------ //

/**
 * Verifies the structural and semantic correctness of a Geometric IR program.
 *
 * @example
 * const verifier = new IRVerifier();
 * const result = verifier.verify(ir);
 * if (!result.valid) console.error(result.errors);
 */
export class IRVerifier {
    /**
     * Verify a GeometricIR program.
     *
     * @param ir - IR to verify
     * @returns ValidationResult with errors and warnings
     */
    verify(ir: GeometricIR): ValidationResult {
        const errors: string[]   = [];
        const warnings: string[] = [];

        this.verifyInstructions(ir.instructions, errors, warnings);
        this.verifyPhases(ir, errors, warnings);
        this.verifyManifold(ir, errors, warnings);

        return { valid: errors.length === 0, errors, warnings };
    }

    // ---- Instruction-level checks ----

    private verifyInstructions(
        instructions: IRInstruction[],
        errors: string[],
        warnings: string[],
    ): void {
        const defined = new Set<string>();
        const labels  = new Set<string>();

        // First pass — collect all defined ids and labels
        for (const instr of instructions) {
            defined.add(instr.id);
            if (instr.op === 'label') labels.add((instr as any).name);
        }

        // Second pass — verify references
        for (const instr of instructions) {
            const i = instr as any;

            const checkId = (id: string | undefined, role: string) => {
                if (id && !defined.has(id)) {
                    errors.push(`Instruction "${instr.id}" (${instr.op}): ${role} id "${id}" is not defined.`);
                }
            };

            switch (instr.op) {
                case 'store':
                    checkId(i.src, 'src');
                    break;

                case 'invoke':
                    for (const arg of (i.args as string[])) checkId(arg, 'arg');
                    break;

                case 'branch':
                    if (!labels.has(i.target)) {
                        errors.push(`Branch "${instr.id}": target label "${i.target}" is not defined.`);
                    }
                    break;

                case 'condbranch':
                    checkId(i.cond, 'cond');
                    if (!labels.has(i.ifTrue)) {
                        errors.push(`CondBranch "${instr.id}": ifTrue label "${i.ifTrue}" is not defined.`);
                    }
                    if (!labels.has(i.ifFalse)) {
                        errors.push(`CondBranch "${instr.id}": ifFalse label "${i.ifFalse}" is not defined.`);
                    }
                    break;

                case 'return':
                    if (i.value) checkId(i.value, 'return value');
                    break;

                default:
                    // Glyph ops
                    if (i.left)  checkId(i.left, 'left');
                    if (i.right) checkId(i.right, 'right');
                    if (KNOWN_GLYPHS.size > 0 && !['load', 'alloc', 'const', 'phase', 'label', 'invoke', 'store', 'branch', 'condbranch', 'return'].includes(instr.op)) {
                        if (!KNOWN_GLYPHS.has(instr.op)) {
                            errors.push(`Unknown glyph op "${instr.op}" in instruction "${instr.id}".`);
                        }
                    }
                    break;
            }
        }

        // Warn about empty programs
        if (instructions.length === 0) {
            warnings.push('IR has no instructions.');
        }
    }

    // ---- Phase-level checks ----

    private verifyPhases(
        ir: GeometricIR,
        errors: string[],
        warnings: string[],
    ): void {
        const TWO_PI = 2 * Math.PI;

        for (const phase of ir.phases) {
            if (phase.start < 0 || phase.start > TWO_PI) {
                errors.push(`Phase "${phase.name}": start ${phase.start.toFixed(4)} is outside [0, 2π].`);
            }
            if (phase.end < 0 || phase.end > TWO_PI + 1e-9) {
                errors.push(`Phase "${phase.name}": end ${phase.end.toFixed(4)} is outside [0, 2π].`);
            }
            if (phase.start > phase.end) {
                errors.push(`Phase "${phase.name}": start (${phase.start.toFixed(4)}) > end (${phase.end.toFixed(4)}).`);
            }
            if (phase.instructions.length === 0) {
                warnings.push(`Phase "${phase.name}" contains no instructions.`);
            }

            // Verify phase-local instruction list
            this.verifyInstructions(phase.instructions, errors, warnings);
        }
    }

    // ---- Manifold checks ----

    private verifyManifold(
        ir: GeometricIR,
        errors: string[],
        warnings: string[],
    ): void {
        const { dimensions, phase } = ir.manifold;
        if (dimensions < 1 || dimensions > 16) {
            errors.push(`Manifold dimensions (${dimensions}) must be between 1 and 16.`);
        }
        if (phase < 0 || phase > 2 * Math.PI) {
            warnings.push(`Manifold initial phase (${phase.toFixed(4)}) is outside [0, 2π].`);
        }
    }
}
