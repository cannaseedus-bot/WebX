// K'UHUL++ Grammar Validator
// Validates a parsed Grammar object for completeness and internal consistency.
// Checks that all non-terminal references are defined, the start symbol exists,
// there are no trivially unreachable rules, and left-recursion is detected.

import type { Grammar, EBNFExpr, EBNFRule } from './ebnf-parser.js';

// ------------------------------------------------------------------ //
// Result type
// ------------------------------------------------------------------ //

/** Outcome of validating a Grammar */
export interface ValidationResult {
    /** Whether the grammar passed all required checks */
    valid: boolean;
    /** Hard errors that make the grammar unusable */
    errors: string[];
    /** Non-fatal issues to be aware of */
    warnings: string[];
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

/** Collect every non-terminal name referenced within an expression tree */
function collectRefs(expr: EBNFExpr): string[] {
    if (expr.kind === 'nonterminal' && expr.value) return [expr.value];
    if (expr.exprs) return expr.exprs.flatMap(collectRefs);
    return [];
}

/** Collect all non-terminal references in a rule's alternatives */
function ruleRefs(rule: EBNFRule): string[] {
    return rule.alternatives.flat().flatMap(collectRefs);
}

/** Find the set of rules reachable from `start` via BFS */
function reachable(grammar: Grammar): Set<string> {
    const visited = new Set<string>();
    const queue: string[] = [grammar.start];
    while (queue.length > 0) {
        const name = queue.shift()!;
        if (visited.has(name)) continue;
        visited.add(name);
        const rule = grammar.rules.get(name);
        if (rule) {
            for (const ref of ruleRefs(rule)) {
                if (!visited.has(ref)) queue.push(ref);
            }
        }
    }
    return visited;
}

/**
 * Detect direct left-recursion: a rule whose first symbol in at least one
 * alternative is itself.
 */
function isDirectlyLeftRecursive(rule: EBNFRule): boolean {
    return rule.alternatives.some(seq => {
        const first = seq.find(e => e.kind === 'nonterminal');
        return first?.value === rule.name;
    });
}

// ------------------------------------------------------------------ //
// Validator
// ------------------------------------------------------------------ //

/**
 * Validate a parsed KUHUL grammar for correctness and completeness.
 *
 * Checks performed:
 * - Grammar contains at least one rule
 * - Start symbol is defined
 * - All non-terminal references point to defined rules
 * - All defined rules are reachable from the start symbol
 * - No directly left-recursive rules (which would cause infinite loops)
 *
 * @param grammar - Grammar model produced by EBNFParser
 * @returns ValidationResult with errors and warnings
 */
export function validateGrammar(grammar: Grammar): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    // 1. Grammar must have rules
    if (grammar.rules.size === 0) {
        errors.push('Grammar contains no rules.');
        return { valid: false, errors, warnings };
    }

    // 2. Start symbol must be defined
    if (!grammar.start) {
        errors.push('Grammar has no start symbol.');
    } else if (!grammar.rules.has(grammar.start)) {
        errors.push(`Start symbol "${grammar.start}" is not defined.`);
    }

    // 3. All referenced non-terminals must be defined
    for (const [name, rule] of grammar.rules) {
        for (const ref of ruleRefs(rule)) {
            if (!grammar.rules.has(ref)) {
                errors.push(`Rule "${name}" references undefined non-terminal "${ref}".`);
            }
        }
    }

    // 4. Unreachable rules (warnings only — they may be intentional)
    if (grammar.start && grammar.rules.has(grammar.start)) {
        const reached = reachable(grammar);
        for (const name of grammar.rules.keys()) {
            if (!reached.has(name)) {
                warnings.push(`Rule "${name}" is unreachable from start symbol "${grammar.start}".`);
            }
        }
    }

    // 5. Direct left-recursion (warning — it needs special handling)
    for (const [name, rule] of grammar.rules) {
        if (isDirectlyLeftRecursive(rule)) {
            warnings.push(`Rule "${name}" is directly left-recursive. Ensure the parser handles this.`);
        }
    }

    return { valid: errors.length === 0, errors, warnings };
}
