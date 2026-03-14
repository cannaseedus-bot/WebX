// K'UHUL++ EBNF Parser
// Parses EBNF grammar definitions into an in-memory grammar model.
// Supports the subset of EBNF used by KUHUL-LLM.ebnf.

// ------------------------------------------------------------------ //
// Grammar model types
// ------------------------------------------------------------------ //

/** A single expression node within an EBNF rule */
export interface EBNFExpr {
    kind: 'terminal' | 'nonterminal' | 'optional' | 'repeat' | 'group';
    /** Literal string value for terminal / identifier for nonterminal */
    value?: string;
    /** Child expressions for optional / repeat / group */
    exprs?: EBNFExpr[];
}

/** One named production rule with one or more alternative sequences */
export interface EBNFRule {
    name: string;
    /** Each inner array is a sequence of expressions (one alternative) */
    alternatives: EBNFExpr[][];
}

/** A parsed grammar containing all rules and the designated start rule */
export interface Grammar {
    rules: Map<string, EBNFRule>;
    /** Name of the first rule defined in the grammar */
    start: string;
}

// ------------------------------------------------------------------ //
// Internal token types for the EBNF tokenizer
// ------------------------------------------------------------------ //

type EBNFTokenKind =
    | 'IDENT'       // non-terminal name, e.g. Expression
    | 'TERMINAL'    // quoted string, e.g. 'Tensor'
    | 'EQUALS'      // =
    | 'PIPE'        // |
    | 'SEMICOLON'   // ;
    | 'LPAREN'      // (
    | 'RPAREN'      // )
    | 'LBRACKET'    // [
    | 'RBRACKET'    // ]
    | 'LBRACE'      // {
    | 'RBRACE'      // }
    | 'EOF';

interface EBNFToken {
    kind: EBNFTokenKind;
    value: string;
    line: number;
}

// ------------------------------------------------------------------ //
// EBNF Lexer
// ------------------------------------------------------------------ //

function tokenizeEBNF(source: string): EBNFToken[] {
    const tokens: EBNFToken[] = [];
    let pos = 0;
    let line = 1;

    while (pos < source.length) {
        // Skip whitespace
        if (/\s/.test(source[pos])) {
            if (source[pos] === '\n') line++;
            pos++;
            continue;
        }

        // Skip EBNF comments (* ... *)
        if (source[pos] === '(' && source[pos + 1] === '*') {
            pos += 2;
            while (pos < source.length && !(source[pos] === '*' && source[pos + 1] === ')')) {
                if (source[pos] === '\n') line++;
                pos++;
            }
            pos += 2;
            continue;
        }

        // Skip line comments -- or //
        if ((source[pos] === '-' && source[pos + 1] === '-') ||
            (source[pos] === '/' && source[pos + 1] === '/')) {
            while (pos < source.length && source[pos] !== '\n') pos++;
            continue;
        }

        // Quoted terminal: 'foo' or "foo"
        if (source[pos] === "'" || source[pos] === '"') {
            const quote = source[pos++];
            let value = '';
            while (pos < source.length && source[pos] !== quote) {
                value += source[pos++];
            }
            pos++; // closing quote
            tokens.push({ kind: 'TERMINAL', value, line });
            continue;
        }

        // Identifiers (non-terminals)
        if (/[A-Za-z_]/.test(source[pos])) {
            let ident = '';
            while (pos < source.length && /[A-Za-z0-9_-]/.test(source[pos])) {
                ident += source[pos++];
            }
            tokens.push({ kind: 'IDENT', value: ident, line });
            continue;
        }

        // Single-char tokens
        const ch = source[pos++];
        switch (ch) {
            case '=': tokens.push({ kind: 'EQUALS',    value: ch, line }); break;
            case '|': tokens.push({ kind: 'PIPE',      value: ch, line }); break;
            case ';': tokens.push({ kind: 'SEMICOLON', value: ch, line }); break;
            case '(': tokens.push({ kind: 'LPAREN',    value: ch, line }); break;
            case ')': tokens.push({ kind: 'RPAREN',    value: ch, line }); break;
            case '[': tokens.push({ kind: 'LBRACKET',  value: ch, line }); break;
            case ']': tokens.push({ kind: 'RBRACKET',  value: ch, line }); break;
            case '{': tokens.push({ kind: 'LBRACE',    value: ch, line }); break;
            case '}': tokens.push({ kind: 'RBRACE',    value: ch, line }); break;
            // Ignore unknown characters (e.g. Unicode glyphs, punctuation)
        }
    }

    tokens.push({ kind: 'EOF', value: '', line });
    return tokens;
}

// ------------------------------------------------------------------ //
// EBNF Recursive-Descent Parser
// ------------------------------------------------------------------ //

class EBNFParserImpl {
    private tokens: EBNFToken[];
    private pos = 0;

    constructor(tokens: EBNFToken[]) {
        this.tokens = tokens;
    }

    private current(): EBNFToken {
        return this.tokens[this.pos];
    }

    private advance(): EBNFToken {
        const tok = this.tokens[this.pos];
        if (tok.kind !== 'EOF') this.pos++;
        return tok;
    }

    private expect(kind: EBNFTokenKind): EBNFToken {
        const tok = this.current();
        if (tok.kind !== kind) {
            throw new Error(`EBNF parse error at line ${tok.line}: expected ${kind} but got ${tok.kind} ("${tok.value}")`);
        }
        return this.advance();
    }

    /** Parse all rules until EOF */
    parseGrammar(): Grammar {
        const rules = new Map<string, EBNFRule>();
        let start = '';

        while (this.current().kind !== 'EOF') {
            // Skip stray semicolons
            if (this.current().kind === 'SEMICOLON') { this.advance(); continue; }

            if (this.current().kind !== 'IDENT') { this.advance(); continue; }

            const rule = this.parseRule();
            if (!rules.has(rule.name)) {
                if (start === '') start = rule.name;
                rules.set(rule.name, rule);
            }
        }

        return { rules, start };
    }

    /** Parse:  RuleName = Alternatives ; */
    private parseRule(): EBNFRule {
        const name = this.expect('IDENT').value;
        this.expect('EQUALS');
        const alternatives = this.parseAlternatives();
        if (this.current().kind === 'SEMICOLON') this.advance();
        return { name, alternatives };
    }

    /** Parse pipe-separated sequences */
    private parseAlternatives(): EBNFExpr[][] {
        const alts: EBNFExpr[][] = [this.parseSequence()];
        while (this.current().kind === 'PIPE') {
            this.advance();
            alts.push(this.parseSequence());
        }
        return alts;
    }

    /** Parse a sequence of expressions until |, ), ], }, ; or EOF */
    private parseSequence(): EBNFExpr[] {
        const seq: EBNFExpr[] = [];
        while (true) {
            const k = this.current().kind;
            if (k === 'PIPE' || k === 'RPAREN' || k === 'RBRACKET' ||
                k === 'RBRACE' || k === 'SEMICOLON' || k === 'EOF') break;
            const expr = this.parseAtom();
            if (expr) seq.push(expr);
        }
        return seq;
    }

    /** Parse a single atomic expression */
    private parseAtom(): EBNFExpr | null {
        const tok = this.current();

        switch (tok.kind) {
            case 'TERMINAL':
                this.advance();
                return { kind: 'terminal', value: tok.value };

            case 'IDENT':
                this.advance();
                return { kind: 'nonterminal', value: tok.value };

            case 'LBRACKET': {
                // [ ... ] — optional
                this.advance();
                const exprs = this.parseAlternatives().flat();
                this.expect('RBRACKET');
                return { kind: 'optional', exprs };
            }

            case 'LBRACE': {
                // { ... } — zero or more
                this.advance();
                const exprs = this.parseAlternatives().flat();
                this.expect('RBRACE');
                return { kind: 'repeat', exprs };
            }

            case 'LPAREN': {
                // ( ... ) — group
                this.advance();
                const exprs = this.parseAlternatives().flat();
                this.expect('RPAREN');
                return { kind: 'group', exprs };
            }

            default:
                // Skip unrecognised token
                this.advance();
                return null;
        }
    }
}

// ------------------------------------------------------------------ //
// Public API
// ------------------------------------------------------------------ //

/**
 * Parses an EBNF grammar source string into a Grammar model.
 *
 * @param source - Raw EBNF text (e.g. contents of kuhul.ebnf)
 * @returns Parsed Grammar with all rules and start symbol
 */
export class EBNFParser {
    parse(source: string): Grammar {
        const tokens = tokenizeEBNF(source);
        const parser = new EBNFParserImpl(tokens);
        return parser.parseGrammar();
    }
}

/**
 * Convenience wrapper — parse an EBNF grammar string.
 *
 * @param source - EBNF grammar text
 * @returns Grammar model
 */
export function parseGrammar(source: string): Grammar {
    return new EBNFParser().parse(source);
}
