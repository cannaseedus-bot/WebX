// K'UHUL++ Linter
// Checks K'UHUL++ source code for style issues, anti-patterns, and
// best-practice violations.  Returns a list of lint results with
// line, column, severity, and message.

import { tokenize, TokenType, GLYPHS, KEYWORDS } from '../compiler/lexer.js';
import { parse, NodeKind }                        from '../compiler/parser.js';
import type { Token }                             from '../compiler/lexer.js';
import type { ASTNode, ProgramNode }              from '../compiler/parser.js';

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

export type LintSeverity = 'error' | 'warning' | 'info';

export interface LintResult {
    rule:     string;
    message:  string;
    severity: LintSeverity;
    line:     number;
    col:      number;
}

// ------------------------------------------------------------------ //
// KuhulLinter
// ------------------------------------------------------------------ //

/**
 * Lint a K'UHUL++ source string for style and best-practice issues.
 *
 * @example
 * const linter = new KuhulLinter();
 * const results = linter.lint(source);
 * for (const r of results) console.log(`${r.line}:${r.col} [${r.severity}] ${r.message}`);
 */
export class KuhulLinter {
    /**
     * Lint the given source string.
     *
     * @param source - K'UHUL++ source code
     * @returns Array of lint results (may be empty if source is clean)
     */
    lint(source: string): LintResult[] {
        const results: LintResult[] = [];

        // Token-level checks
        let tokens: Token[];
        try {
            tokens = tokenize(source);
        } catch (e: any) {
            results.push({
                rule:     'parse-error',
                message:  e.message,
                severity: 'error',
                line:     e.line ?? 1,
                col:      e.col  ?? 1,
            });
            return results;
        }

        results.push(...this.checkTokens(tokens));

        // AST-level checks
        let ast: ProgramNode;
        try {
            ast = parse(tokens);
        } catch (e: any) {
            results.push({
                rule:     'parse-error',
                message:  e.message,
                severity: 'error',
                line:     e.token?.line ?? 1,
                col:      e.token?.col  ?? 1,
            });
            return results;
        }

        results.push(...this.checkAST(ast));

        return results.sort((a, b) => a.line - b.line || a.col - b.col);
    }

    // ---- Token-level rules ----

    private checkTokens(tokens: Token[]): LintResult[] {
        const results: LintResult[] = [];

        for (let i = 0; i < tokens.length; i++) {
            const tok = tokens[i];

            // Rule: no-magic-numbers (standalone numbers not assigned to a Tensor)
            if (tok.type === TokenType.NUMBER) {
                const prev  = tokens[i - 1];
                const next  = tokens[i + 1];
                const inDecl = prev?.value === '=' && tokens[i - 2]?.type === TokenType.IDENTIFIER;
                const isSmallInt = Number.isInteger(tok.value) && Math.abs(tok.value as number) <= 10;
                if (!inDecl && !isSmallInt) {
                    results.push({
                        rule:     'no-magic-numbers',
                        message:  `Magic number ${tok.value}. Consider assigning it to a named Tensor.`,
                        severity: 'info',
                        line:     tok.line,
                        col:      tok.col,
                    });
                }
            }

            // Rule: prefer-pi-expr  (detect 3.14159... literals)
            if (tok.type === TokenType.NUMBER) {
                const v = tok.value as number;
                if (Math.abs(v - Math.PI) < 0.001 || Math.abs(v - 2 * Math.PI) < 0.001) {
                    results.push({
                        rule:     'prefer-pi-expr',
                        message:  `Use the π literal instead of a numeric approximation (${v}).`,
                        severity: 'warning',
                        line:     tok.line,
                        col:      tok.col,
                    });
                }
            }

            // Rule: glyph-spacing (glyph should be surrounded by spaces)
            if (tok.type === TokenType.GLYPH) {
                const lines = tokens[0]?.line ? 1 : 0; // assume single line if first
                const src   = (tok as any).__src ?? '';
                // We can't easily check spacing without the raw source,
                // so we issue an info note for glyphs without adjacent spaces
                const prevIsSpace = !tokens[i-1] || tokens[i-1].col + String(tokens[i-1].value).length < tok.col - 1;
                if (!prevIsSpace) {
                    results.push({
                        rule:     'glyph-spacing',
                        message:  `Glyph "${tok.value}" should be separated from its operands by spaces.`,
                        severity: 'info',
                        line:     tok.line,
                        col:      tok.col,
                    });
                }
            }
        }

        return results;
    }

    // ---- AST-level rules ----

    private checkAST(ast: ProgramNode): LintResult[] {
        const results: LintResult[] = [];
        this.visitNode(ast, results, 0);
        return results;
    }

    private visitNode(node: ASTNode, results: LintResult[], depth: number): void {
        const n = node as any;

        switch (node.kind) {
            case NodeKind.TensorDecl:
                // Rule: naming-convention — Tensor names should start with lowercase
                if (n.name && /^[A-Z]/.test(n.name) && n.name.length > 1) {
                    results.push({
                        rule:     'naming-convention',
                        message:  `Tensor "${n.name}" starts with uppercase. Consider lowercase for tensor variables.`,
                        severity: 'info',
                        line:     node.line ?? 0,
                        col:      node.col  ?? 0,
                    });
                }
                this.visitNode(n.init, results, depth + 1);
                break;

            case NodeKind.WhileStmt:
                // Rule: no-infinite-while — warn on while(true)
                if (n.test?.kind === NodeKind.Identifier && n.test?.name === 'true') {
                    results.push({
                        rule:     'no-infinite-while',
                        message:  `Potentially infinite while(true) loop. Add an exit condition.`,
                        severity: 'warning',
                        line:     node.line ?? 0,
                        col:      node.col  ?? 0,
                    });
                }
                this.visitNode(n.body, results, depth + 1);
                break;

            case NodeKind.GlyphOp:
                // Rule: nested-glyph-depth — deeply nested glyph ops hurt readability
                if (depth > 5) {
                    results.push({
                        rule:     'nested-glyph-depth',
                        message:  `Glyph operation "${n.glyph}" is nested ${depth} levels deep. Consider extracting sub-expressions.`,
                        severity: 'warning',
                        line:     node.line ?? 0,
                        col:      node.col  ?? 0,
                    });
                }
                this.visitNode(n.left,  results, depth + 1);
                this.visitNode(n.right, results, depth + 1);
                break;

            case NodeKind.Program:
            case NodeKind.Block:
            case NodeKind.ClusterDecl:
            case NodeKind.ModelDecl:
            case NodeKind.PipelineDecl: {
                const body = n.body ?? [];
                for (const child of body) this.visitNode(child, results, depth + 1);
                break;
            }

            default:
                // Visit child nodes generically
                for (const key of Object.keys(n)) {
                    const child = n[key];
                    if (child && typeof child === 'object' && 'kind' in child) {
                        this.visitNode(child, results, depth + 1);
                    }
                    if (Array.isArray(child)) {
                        for (const item of child) {
                            if (item && typeof item === 'object' && 'kind' in item) {
                                this.visitNode(item, results, depth + 1);
                            }
                        }
                    }
                }
                break;
        }
    }
}
