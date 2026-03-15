// K'UHUL++ Glyph Validator
// Validates glyph expressions for tensor type compatibility.
// Checks that operand types are compatible with the requested glyph operation.

import type { ValidationResult } from '../grammar/grammar-validator.js';
import type { KuhulType, GlyphOp } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// Glyph compatibility rules
// ------------------------------------------------------------------ //

interface GlyphRule {
    /** Human-readable description of what this glyph does */
    description: string;
    /** Valid left-operand type kinds */
    leftKinds:   KuhulType['kind'][];
    /** Valid right-operand type kinds */
    rightKinds:  KuhulType['kind'][];
    /** Whether shapes must match */
    requireMatchingShapes: boolean;
}

const GLYPH_RULES: Partial<Record<GlyphOp, GlyphRule>> = {
    '⊗': {
        description: 'Geometric product (matrix multiply / outer product)',
        leftKinds:   ['tensor', 'scalar'],
        rightKinds:  ['tensor', 'scalar'],
        requireMatchingShapes: false,
    },
    '⊕': {
        description: 'Translation / bias addition',
        leftKinds:   ['tensor', 'scalar'],
        rightKinds:  ['tensor', 'scalar'],
        requireMatchingShapes: true,
    },
    '⊖': {
        description: 'Subtraction / difference in M',
        leftKinds:   ['tensor', 'scalar'],
        rightKinds:  ['tensor', 'scalar'],
        requireMatchingShapes: true,
    },
    '⊛': {
        description: 'Convolution in manifold M',
        leftKinds:   ['tensor'],
        rightKinds:  ['tensor'],
        requireMatchingShapes: false,
    },
    '⊞': {
        description: 'Element-wise union / addition',
        leftKinds:   ['tensor', 'scalar'],
        rightKinds:  ['tensor', 'scalar'],
        requireMatchingShapes: true,
    },
    '⊝': {
        description: 'Complement / negation',
        leftKinds:   ['tensor', 'scalar'],
        rightKinds:  ['tensor', 'scalar'],
        requireMatchingShapes: false,
    },
    '⤍': {
        description: 'Vector Encrypt (affine 4×4 transform)',
        leftKinds:   ['tensor'],
        rightKinds:  ['tensor'],
        requireMatchingShapes: false,
    },
    '↻': {
        description: 'Rotational Compression',
        leftKinds:   ['tensor'],
        rightKinds:  ['scalar', 'tensor'],
        requireMatchingShapes: false,
    },
    '⟲': {
        description: 'Spherical Loop transform',
        leftKinds:   ['tensor'],
        rightKinds:  ['scalar', 'tensor'],
        requireMatchingShapes: false,
    },
    '∿': {
        description: 'Torsion Field deformation',
        leftKinds:   ['tensor'],
        rightKinds:  ['scalar', 'tensor'],
        requireMatchingShapes: false,
    },
    '⊙': {
        description: 'Radial Projection',
        leftKinds:   ['tensor'],
        rightKinds:  ['scalar', 'tensor'],
        requireMatchingShapes: false,
    },
    '≋': {
        description: 'Wave Modulation',
        leftKinds:   ['tensor'],
        rightKinds:  ['scalar', 'tensor'],
        requireMatchingShapes: false,
    },
};

// ------------------------------------------------------------------ //
// GlyphValidator
// ------------------------------------------------------------------ //

/**
 * Validates glyph expressions for semantic correctness.
 *
 * @example
 * const validator = new GlyphValidator();
 * const result = validator.validate('⊗', tensorType, matrixType);
 */
export class GlyphValidator {
    /**
     * Validate a glyph expression string (e.g. "a ⊗ b").
     * Parses the glyph symbol and checks it is a known operator.
     *
     * @param glyphExpr - Glyph expression string
     * @returns ValidationResult
     */
    validate(glyphExpr: string): ValidationResult {
        const errors:   string[] = [];
        const warnings: string[] = [];

        // Extract glyph operator from expression
        const glyphMatch = glyphExpr.match(/[⊗⊕⊖⊛⊜⊝⊞⤍↻⟲∿⊙≋]/u);
        if (!glyphMatch) {
            errors.push(`No recognised glyph operator found in expression: "${glyphExpr}"`);
            return { valid: false, errors, warnings };
        }

        const glyph = glyphMatch[0] as GlyphOp;
        const rule  = GLYPH_RULES[glyph];
        if (!rule) {
            warnings.push(`Glyph "${glyph}" has no defined validation rule — assuming valid.`);
            return { valid: true, errors, warnings };
        }

        // Check that operands appear on both sides
        const parts = glyphExpr.split(glyph);
        if (parts.length < 2 || !parts[0].trim() || !parts[1].trim()) {
            errors.push(`Glyph "${glyph}" requires operands on both sides.`);
        }

        return { valid: errors.length === 0, errors, warnings };
    }

    /**
     * Validate a glyph operation with explicit type information.
     *
     * @param glyph      - Glyph operator symbol
     * @param leftType   - Type of the left operand
     * @param rightType  - Type of the right operand
     * @returns ValidationResult
     */
    validateTypes(glyph: GlyphOp, leftType: KuhulType, rightType: KuhulType): ValidationResult {
        const errors:   string[] = [];
        const warnings: string[] = [];

        const rule = GLYPH_RULES[glyph];
        if (!rule) {
            warnings.push(`No rule for glyph "${glyph}" — skipping type check.`);
            return { valid: true, errors, warnings };
        }

        if (!rule.leftKinds.includes(leftType.kind)) {
            errors.push(
                `Glyph "${glyph}" (${rule.description}): left operand kind "${leftType.kind}" ` +
                `is not allowed. Expected one of: ${rule.leftKinds.join(', ')}.`
            );
        }

        if (!rule.rightKinds.includes(rightType.kind)) {
            errors.push(
                `Glyph "${glyph}" (${rule.description}): right operand kind "${rightType.kind}" ` +
                `is not allowed. Expected one of: ${rule.rightKinds.join(', ')}.`
            );
        }

        if (rule.requireMatchingShapes && leftType.kind === 'tensor' && rightType.kind === 'tensor') {
            const ls = leftType.shape;
            const rs = rightType.shape;
            if (ls.length !== rs.length || ls.some((d, i) => d !== rs[i])) {
                errors.push(
                    `Glyph "${glyph}" requires matching shapes. ` +
                    `Got [${ls}] vs [${rs}].`
                );
            }
        }

        return { valid: errors.length === 0, errors, warnings };
    }
}
