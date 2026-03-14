/**
 * @fileoverview LLM-enhanced diagnostic engine for KUHUL errors.
 *
 * Takes raw compiler or runtime errors and enriches them with contextual
 * explanations, suggested fixes, and relevant documentation links.
 *
 * @module kuhul/llm/diagnostic-engine
 */

// ------------------------------------------------------------------ //
// Error patterns → diagnostic enrichments
// ------------------------------------------------------------------ //

/**
 * @typedef {{ pattern: RegExp, title: string, explanation: string, fix: string }} DiagnosticRule
 */

/** @type {DiagnosticRule[]} */
const DIAGNOSTIC_RULES = [
  {
    pattern:     /identifier "([^"]+)" is used before declaration/i,
    title:       'Undeclared Identifier',
    explanation: 'You are referencing a variable that has not been allocated with [Wo].',
    fix:         'Add `[Wo $1 tensor<float32, [1]>]` before the first use of `$1`.',
  },
  {
    pattern:     /unknown glyph operator "([^"]+)"/i,
    title:       'Unknown Glyph',
    explanation: 'The glyph is not in the KUHUL recognised set: ⊗ ⊕ ⊖ ⊛ ⊜ ⊝ ⊞.',
    fix:         'Replace "$1" with one of the supported glyph operators.',
  },
  {
    pattern:     /unknown tensor element type "([^"]+)"/i,
    title:       'Invalid Element Type',
    explanation: 'The tensor element type is not supported.',
    fix:         'Use one of: float32, float64, int32, int64, uint8, bool, complex64.',
  },
  {
    pattern:     /unterminated fold/i,
    title:       'Missing [Xul]',
    explanation: 'A [Pop] block was opened but never closed.',
    fix:         'Add a matching [Xul] at the end of the block.',
  },
  {
    pattern:     /unterminated phase cycle/i,
    title:       "Missing [Kumk'u]",
    explanation: "A [K'ayab'] phase cycle was opened but never closed.",
    fix:         "Add a matching [Kumk'u] after the phase body.",
  },
];

// ------------------------------------------------------------------ //
// DiagnosticEngine
// ------------------------------------------------------------------ //

/** Enhances compiler/runtime errors with human-readable explanations. */
export class DiagnosticEngine {
  /**
   * Enhance an error with contextual explanation and fix suggestions.
   *
   * @param {Error|string} error  - The raw error to enhance
   * @param {string}       source - The KUHUL source that produced the error
   * @returns {Promise<{ title: string, message: string, explanation: string, fix: string, sourceLine: string|null }>}
   */
  async enhance(error, source) {
    const message = error instanceof Error ? error.message : String(error);

    // Match against known patterns
    for (const rule of DIAGNOSTIC_RULES) {
      const m = message.match(rule.pattern);
      if (m) {
        const fix = rule.fix.replace(/\$(\d+)/g, (_, i) => m[Number(i)] ?? '');
        return {
          title:       rule.title,
          message,
          explanation: rule.explanation,
          fix,
          sourceLine:  this._findSourceLine(message, source),
        };
      }
    }

    // Generic fallback
    return {
      title:       'Compilation Error',
      message,
      explanation: 'An unexpected error occurred during compilation.',
      fix:         'Check the source for syntax errors and ensure all identifiers are declared.',
      sourceLine:  this._findSourceLine(message, source),
    };
  }

  // ---------------------------------------------------------------- //
  // Source line extraction
  // ---------------------------------------------------------------- //

  /**
   * Try to extract the relevant source line from an error message.
   *
   * @param {string} message
   * @param {string} source
   * @returns {string|null}
   */
  _findSourceLine(message, source) {
    const lineMatch = message.match(/\b(\d+):\d+/);
    if (!lineMatch) return null;
    const lineNum = parseInt(lineMatch[1], 10) - 1;
    const lines   = source.split('\n');
    return lines[lineNum] ?? null;
  }
}
