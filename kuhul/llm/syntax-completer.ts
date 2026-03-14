// K'UHUL++ Syntax Completer
// Provides code-completion suggestions for partial K'UHUL++ source.
// Uses rule-based heuristics derived from the grammar; falls back to
// LLM-assisted completion when a provider is configured.

import type { Token, TokenType } from '../compiler/lexer.js';
import { tokenize, KEYWORDS, GLYPHS } from '../compiler/lexer.js';

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

export interface CompletionSuggestion {
    /** The completion text to insert */
    text:  string;
    /** Human-readable label shown in a completion menu */
    label: string;
    /** Category of completion */
    kind:  'keyword' | 'glyph' | 'identifier' | 'snippet';
    /** Short documentation string */
    documentation?: string;
    /** Sort priority — lower is better */
    priority: number;
}

// ------------------------------------------------------------------ //
// Built-in snippet completions
// ------------------------------------------------------------------ //

const SNIPPETS: CompletionSuggestion[] = [
    {
        text:  'Tensor ${1:name} = ${2:value};',
        label: 'Tensor declaration',
        kind:  'snippet',
        documentation: 'Declare a new tensor variable',
        priority: 10,
    },
    {
        text:  'Cluster ${1:name} {\n  ${0}\n}',
        label: 'Cluster block',
        kind:  'snippet',
        documentation: 'Declare a tensor cluster',
        priority: 20,
    },
    {
        text:  'Model ${1:name} {\n  ${0}\n}',
        label: 'Model block',
        kind:  'snippet',
        documentation: 'Declare a neural model',
        priority: 20,
    },
    {
        text:  'Pipeline ${1:name} {\n  ${0}\n}',
        label: 'Pipeline block',
        kind:  'snippet',
        documentation: 'Declare a compute pipeline',
        priority: 20,
    },
    {
        text:  'foreach glyph ⊗ in ${1:tensor} {\n  ${0}\n}',
        label: 'foreach glyph loop',
        kind:  'snippet',
        documentation: 'Loop over glyph operations on a tensor',
        priority: 30,
    },
    {
        text:  'parallel for (${1:item} in ${2:collection}) {\n  ${0}\n}',
        label: 'parallel for loop',
        kind:  'snippet',
        documentation: 'GPU-parallel for loop',
        priority: 30,
    },
];

// ------------------------------------------------------------------ //
// SyntaxCompleter
// ------------------------------------------------------------------ //

/**
 * Provides code completions for K'UHUL++ source.
 *
 * @example
 * const completer = new SyntaxCompleter();
 * const suggestions = completer.complete('Tens', 4);
 * // → [{ text: 'Tensor', label: 'Tensor', kind: 'keyword', priority: 0 }]
 */
export class SyntaxCompleter {
    private llmProvider?: (prefix: string) => Promise<string[]>;

    /**
     * @param llmProvider - Optional async function that returns LLM-generated completions
     */
    constructor(llmProvider?: (prefix: string) => Promise<string[]>) {
        this.llmProvider = llmProvider;
    }

    /**
     * Get completion suggestions for the given source at the cursor position.
     *
     * @param partialCode - Source code up to (and possibly including) the cursor
     * @param cursor      - Cursor offset in characters
     * @returns Ranked list of completion suggestions
     */
    async complete(partialCode: string, cursor: number): Promise<CompletionSuggestion[]> {
        const prefix = partialCode.slice(0, cursor);
        const word   = this.currentWord(prefix);

        const results: CompletionSuggestion[] = [];

        // Rule-based completions
        results.push(...this.keywordCompletions(word));
        results.push(...this.glyphCompletions(word));
        results.push(...this.snippetCompletions(word));

        // LLM-assisted completions (if provider available)
        if (this.llmProvider && word.length >= 2) {
            try {
                const llmResults = await this.llmProvider(prefix);
                for (const text of llmResults) {
                    results.push({
                        text,
                        label: text,
                        kind:  'identifier',
                        documentation: 'LLM suggestion',
                        priority: 50,
                    });
                }
            } catch {
                // LLM unavailable — rule-based results still returned
            }
        }

        // Sort by priority then alphabetically
        results.sort((a, b) => a.priority - b.priority || a.text.localeCompare(b.text));
        return results;
    }

    // ---- Helpers ----

    /** Extract the word the cursor is currently on */
    private currentWord(prefix: string): string {
        const match = prefix.match(/[\w⊗⊕⊖⊛⊜⊝⊞⤍↻⟲∿⊙≋]+$/u);
        return match ? match[0] : '';
    }

    private keywordCompletions(word: string): CompletionSuggestion[] {
        if (word === '') return [];
        const results: CompletionSuggestion[] = [];
        for (const kw of KEYWORDS) {
            if (kw.toLowerCase().startsWith(word.toLowerCase())) {
                results.push({
                    text: kw,
                    label: kw,
                    kind: 'keyword',
                    documentation: `K'UHUL++ keyword`,
                    priority: 0,
                });
            }
        }
        return results;
    }

    private glyphCompletions(word: string): CompletionSuggestion[] {
        const GLYPH_DOCS: Record<string, string> = {
            '⊗': 'Geometric product / tensor product',
            '⊕': 'Translation / bias addition in M',
            '⊖': 'Difference / subtraction in M',
            '⊛': 'Convolution in M',
            '⊜': 'Identity element in M',
            '⊝': 'Complement / negation in M',
            '⊞': 'Union / element-wise addition',
            '⤍': 'Vector Encrypt — affine transform',
            '↻': 'Rotational Compression',
            '⟲': 'Spherical Loop transform',
            '∿': 'Torsion Field deformation',
            '⊙': 'Radial Projection',
            '≋': 'Wave Modulation',
        };
        if (word === '') {
            return [...GLYPHS].map(g => ({
                text:  g,
                label: `${g} — ${GLYPH_DOCS[g] ?? 'glyph op'}`,
                kind:  'glyph' as const,
                documentation: GLYPH_DOCS[g],
                priority: 5,
            }));
        }
        return [];
    }

    private snippetCompletions(word: string): CompletionSuggestion[] {
        if (word.length < 2) return [];
        return SNIPPETS.filter(s => s.label.toLowerCase().includes(word.toLowerCase()));
    }
}
