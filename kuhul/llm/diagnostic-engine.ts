// K'UHUL++ Diagnostic Engine
// Enhances raw errors with contextual information, suggestions for fixes,
// and links to the relevant grammar rules or documentation sections.

// ------------------------------------------------------------------ //
// Enhanced diagnostic types
// ------------------------------------------------------------------ //

export interface FixSuggestion {
    /** Short description of the fix */
    title:       string;
    /** The replacement text to apply */
    replacement: string;
    /** Range in the source to replace */
    range?:      { start: number; end: number };
}

export interface EnhancedDiagnostic {
    /** Original error message */
    originalMessage: string;
    /** Improved, user-friendly error message */
    message:         string;
    /** Error category */
    category:        'syntax' | 'semantic' | 'type' | 'glyph' | 'ir' | 'runtime' | 'unknown';
    /** Possible fixes */
    suggestions:     FixSuggestion[];
    /** Link to relevant documentation section */
    docLink?:        string;
    /** Severity level */
    severity:        'error' | 'warning' | 'info';
}

// ------------------------------------------------------------------ //
// Error pattern registry
// ------------------------------------------------------------------ //

interface ErrorPattern {
    /** Regular expression to match against the error message */
    match:    RegExp;
    category: EnhancedDiagnostic['category'];
    message:  (m: RegExpMatchArray) => string;
    suggestions: (m: RegExpMatchArray) => FixSuggestion[];
    docLink?: string;
}

const ERROR_PATTERNS: ErrorPattern[] = [
    {
        match:    /LexerError.*Unrecognised character "(.+)"/,
        category: 'syntax',
        message:  m => `Unrecognised character "${m[1]}". K'UHUL++ only accepts standard ASCII, Unicode identifiers, and recognised glyph symbols.`,
        suggestions: m => [{
            title:       `Remove the character "${m[1]}"`,
            replacement: '',
        }],
        docLink: 'docs/KUHUL.md#lexical-structure',
    },
    {
        match:    /ParseError.*Expected (\w+) but got (\w+)/,
        category: 'syntax',
        message:  m => `Syntax error: expected ${m[1]} but found ${m[2]}.`,
        suggestions: _ => [{
            title:       'Check for missing semicolons or mismatched braces',
            replacement: '',
        }],
        docLink: 'docs/KUHUL.md#statements',
    },
    {
        match:    /Identifier "(.+)" is not declared/,
        category: 'semantic',
        message:  m => `"${m[1]}" is used before being declared. Add a \`Tensor ${m[1]} = ...\` declaration.`,
        suggestions: m => [{
            title:       `Declare "${m[1]}"`,
            replacement: `Tensor ${m[1]} = /* value */;`,
        }],
        docLink: 'docs/KUHUL.md#tensor-declarations',
    },
    {
        match:    /Unknown glyph "(.+)"/,
        category: 'glyph',
        message:  m => `"${m[1]}" is not a recognised glyph operator. See docs/KUHUL.md for the list of valid glyphs.`,
        suggestions: _ => [
            { title: 'Use ⊗ for geometric product',       replacement: '⊗' },
            { title: 'Use ⊕ for translation/bias',        replacement: '⊕' },
            { title: 'Use ⊖ for difference',              replacement: '⊖' },
            { title: 'Use ⤍ for vector encryption',       replacement: '⤍' },
        ],
        docLink: 'docs/KUHUL.md#glyph-operators',
    },
    {
        match:    /IR verification failed/i,
        category: 'ir',
        message:  _ => 'The generated IR failed verification. This is usually caused by a semantic error in the source that was not caught by the analyzer.',
        suggestions: _ => [{
            title:       'Enable strict semantic analysis to catch errors earlier',
            replacement: '',
        }],
        docLink: 'docs/IR.md#verification',
    },
    {
        match:    /RuntimeError — (.+)/,
        category: 'runtime',
        message:  m => `Runtime error during execution: ${m[1]}`,
        suggestions: _ => [{
            title:       'Add null/undefined checks before glyph operations',
            replacement: '',
        }],
        docLink: 'docs/KUHUL.md#runtime',
    },
];

// ------------------------------------------------------------------ //
// DiagnosticEngine
// ------------------------------------------------------------------ //

/**
 * Enhances raw K'UHUL++ errors with user-friendly messages and fix suggestions.
 *
 * @example
 * const engine = new DiagnosticEngine();
 * const diag = engine.enhance(new LexerError('Unrecognised character "£"', 1, 5));
 * console.log(diag.message);
 * console.log(diag.suggestions[0].title);
 */
export class DiagnosticEngine {
    private llmProvider?: (errorMsg: string) => Promise<string>;

    constructor(llmProvider?: (errorMsg: string) => Promise<string>) {
        this.llmProvider = llmProvider;
    }

    /**
     * Enhance a raw error with context and suggestions.
     *
     * @param error - Any Error (LexerError, ParseError, SemanticError, etc.)
     * @returns EnhancedDiagnostic
     */
    async enhance(error: Error): Promise<EnhancedDiagnostic> {
        const msg = error.message;

        // Try each pattern
        for (const pattern of ERROR_PATTERNS) {
            const match = msg.match(pattern.match);
            if (match) {
                const base: EnhancedDiagnostic = {
                    originalMessage: msg,
                    message:         pattern.message(match),
                    category:        pattern.category,
                    suggestions:     pattern.suggestions(match),
                    docLink:         pattern.docLink,
                    severity:        'error',
                };
                // Try LLM enhancement
                if (this.llmProvider) {
                    try {
                        const extra = await this.llmProvider(msg);
                        if (extra) base.suggestions.push({ title: 'LLM suggestion', replacement: extra });
                    } catch { /* ignore */ }
                }
                return base;
            }
        }

        // Fallback for unrecognised errors
        return {
            originalMessage: msg,
            message:         msg,
            category:        'unknown',
            suggestions:     [],
            severity:        'error',
        };
    }

    /**
     * Synchronous version — no LLM call.
     */
    enhanceSync(error: Error): EnhancedDiagnostic {
        const msg = error.message;
        for (const pattern of ERROR_PATTERNS) {
            const match = msg.match(pattern.match);
            if (match) {
                return {
                    originalMessage: msg,
                    message:         pattern.message(match),
                    category:        pattern.category,
                    suggestions:     pattern.suggestions(match),
                    docLink:         pattern.docLink,
                    severity:        'error',
                };
            }
        }
        return { originalMessage: msg, message: msg, category: 'unknown', suggestions: [], severity: 'error' };
    }
}
