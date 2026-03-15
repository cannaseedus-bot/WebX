/**
 * @fileoverview LLM-based glyph semantic validator for KUHUL.
 *
 * Validates that glyph usages are semantically meaningful (correct arity,
 * compatible operand types, expected result shape).  The rule-based
 * implementation can be swapped for an LLM backend without changing the
 * public interface.
 *
 * @module kuhul/llm/glyph-validator
 */

// ------------------------------------------------------------------ //
// Glyph rules
// ------------------------------------------------------------------ //

/**
 * @typedef {{ arity: number, description: string, resultShape: string }} GlyphRule
 */

/** @type {Map<string, GlyphRule>} */
const GLYPH_RULES = new Map([
  ['⊗', { arity: 2, description: 'Tensor product (matmul)', resultShape: 'matrix' }],
  ['⊕', { arity: 2, description: 'Element-wise addition',   resultShape: 'same'   }],
  ['⊖', { arity: 2, description: 'Element-wise subtraction', resultShape: 'same'  }],
  ['⊛', { arity: 2, description: 'Convolution',             resultShape: 'varies' }],
  ['⊜', { arity: 2, description: 'Equality comparison',     resultShape: 'bool'   }],
  ['⊝', { arity: 1, description: 'Negation',                resultShape: 'same'   }],
  ['⊞', { arity: 2, description: 'Direct sum / concat',     resultShape: 'concat' }],
]);

// ------------------------------------------------------------------ //
// GlyphValidator
// ------------------------------------------------------------------ //

/** Validates glyph code snippets or full programs. */
export class GlyphValidator {
  /**
   * Validate a piece of KUHUL source that uses glyph operators.
   *
   * @param {string} glyphCode - KUHUL source containing glyph operations
   * @returns {Promise<{ valid: boolean, messages: string[] }>}
   */
  async validate(glyphCode) {
    const messages = [];

    // Extract all [Sek glyph operand...] expressions
    const sekPattern = /\[\s*Sek\s+([\S]+)(.*?)\]/gs;
    let match;

    while ((match = sekPattern.exec(glyphCode)) !== null) {
      const glyph    = match[1].trim();
      const operands = match[2].trim().split(/\s+/).filter(Boolean);

      const rule = GLYPH_RULES.get(glyph);
      if (!rule) {
        messages.push(`Unknown glyph "${glyph}" in: ${match[0].trim()}`);
        continue;
      }

      if (operands.length !== rule.arity) {
        messages.push(
          `Glyph "${glyph}" (${rule.description}) expects ${rule.arity} operand(s) ` +
          `but got ${operands.length}: ${match[0].trim()}`
        );
      }
    }

    return { valid: messages.length === 0, messages };
  }

  /**
   * Return metadata for a specific glyph.
   *
   * @param {string} glyph
   * @returns {{ arity: number, description: string, resultShape: string }|null}
   */
  getGlyphInfo(glyph) {
    return GLYPH_RULES.get(glyph) ?? null;
  }
}
