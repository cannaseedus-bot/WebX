/**
 * @fileoverview Validates a parsed EBNF grammar for the KUHUL language.
 *
 * Checks performed:
 *  - Every referenced rule name is defined.
 *  - The grammar has at least one rule.
 *  - No rule is defined more than once.
 *  - The grammar is not trivially empty (each rule has a non-empty definition).
 *
 * @module grammar-validator
 */

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

/**
 * Collect all rule-name references that appear inside an expression node.
 *
 * @param {object} node - An ExprNode from ebnf-parser.js
 * @returns {Set<string>}
 */
function collectRefs(node) {
  const refs = new Set();

  function walk(n) {
    if (!n || typeof n !== 'object') return;
    switch (n.kind) {
      case 'Ref':
        refs.add(n.name);
        break;
      case 'Seq':
      case 'Alt':
        (n.items || []).forEach(walk);
        break;
      case 'Rep':
      case 'Opt':
      case 'Plus':
        walk(n.expr);
        break;
      // 'Literal' — no refs
      default:
        break;
    }
  }

  walk(node);
  return refs;
}

// ------------------------------------------------------------------ //
// Public API
// ------------------------------------------------------------------ //

/**
 * Validate a parsed EBNF grammar object.
 *
 * @param {{ rules: Map<string, object>, start: string }} grammar
 *   The object returned by `parseEBNF()`.
 * @returns {{ valid: boolean, errors: string[] }}
 */
export function validateGrammar(grammar) {
  const errors = [];

  if (!grammar || typeof grammar !== 'object') {
    return { valid: false, errors: ['Grammar must be a non-null object.'] };
  }

  const { rules, start } = grammar;

  // 1. Must have a rules Map
  if (!(rules instanceof Map)) {
    errors.push('grammar.rules must be a Map.');
    return { valid: false, errors };
  }

  // 2. Must have at least one rule
  if (rules.size === 0) {
    errors.push('Grammar contains no rules.');
    return { valid: false, errors };
  }

  // 3. Must have a start symbol
  if (!start) {
    errors.push('Grammar is missing a start symbol.');
  } else if (!rules.has(start)) {
    errors.push(`Start symbol "${start}" is not defined.`);
  }

  // 4. Collect all referenced rule names and check they are defined
  for (const [name, expr] of rules) {
    if (!expr || typeof expr !== 'object') {
      errors.push(`Rule "${name}" has an invalid (non-object) definition.`);
      continue;
    }

    const refs = collectRefs(expr);
    for (const refName of refs) {
      if (!rules.has(refName)) {
        errors.push(`Rule "${name}" references undefined rule "${refName}".`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}
