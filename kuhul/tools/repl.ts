// K'UHUL++ Interactive REPL
// Provides an interactive Read-Eval-Print Loop for K'UHUL++ programs.
// Can be run from the command line with: node --loader ts-node/esm tools/repl.ts

import * as readline from 'readline';
import { ExecutionEngine } from '../runtime/execution-engine.js';

// ------------------------------------------------------------------ //
// KuhulREPL
// ------------------------------------------------------------------ //

/**
 * Interactive K'UHUL++ REPL shell.
 * Supports multi-line input (lines ending with `{` open a block),
 * special meta-commands prefixed with `.`, and pretty-printed results.
 *
 * @example
 * const repl = new KuhulREPL();
 * await repl.start();
 */
export class KuhulREPL {
    private readonly engine: ExecutionEngine;
    private context: Record<string, unknown> = {};
    private history: string[] = [];
    private rl?: readline.Interface;

    constructor() {
        this.engine = new ExecutionEngine({ strictMode: false });
    }

    // ---- Lifecycle ----

    /**
     * Start the interactive REPL session.
     * Reads from stdin and writes to stdout.
     */
    async start(): Promise<void> {
        this.rl = readline.createInterface({
            input:  process.stdin,
            output: process.stdout,
            prompt: 'kuhul> ',
        });

        this.printBanner();
        this.rl.prompt();

        let buffer = '';
        let depth  = 0;

        this.rl.on('line', async (line) => {
            // Meta-commands
            if (buffer === '' && line.trim().startsWith('.')) {
                await this.handleMetaCommand(line.trim());
                this.rl!.prompt();
                return;
            }

            buffer += line + '\n';
            depth  += (line.match(/\{/g) ?? []).length;
            depth  -= (line.match(/\}/g) ?? []).length;

            if (depth <= 0) {
                const code = buffer.trim();
                buffer = '';
                depth  = 0;
                if (code) {
                    await this.evaluate(code);
                }
            } else {
                process.stdout.write('  ..  ');
            }

            this.rl!.prompt();
        });

        this.rl.on('close', () => {
            console.log('\nGoodbye.');
            process.exit(0);
        });
    }

    /**
     * Evaluate a K'UHUL++ code snippet and print the result.
     *
     * @param line - Source code to evaluate
     */
    async evaluate(line: string): Promise<void> {
        this.history.push(line);
        const start = Date.now();

        try {
            const result = await this.engine.run(line, this.context);

            // Merge outputs back into the persistent context
            for (const [k, v] of result.output) {
                this.context[k] = v;
            }

            console.log(`\x1b[32m=> phase=${result.phase.toFixed(4)} rad  (${result.durationMs}ms)\x1b[0m`);

            if (result.output.size > 0) {
                for (const [k, v] of result.output) {
                    console.log(`   \x1b[36m${k}\x1b[0m = ${formatValue(v)}`);
                }
            }
        } catch (err: any) {
            console.error(`\x1b[31m✗ ${err.message}\x1b[0m`);
        }
    }

    // ---- Meta-commands ----

    private async handleMetaCommand(cmd: string): Promise<void> {
        switch (cmd) {
            case '.help':
                console.log([
                    '.help    — Show this help',
                    '.clear   — Clear the execution context',
                    '.history — Show command history',
                    '.exit    — Exit the REPL',
                ].join('\n'));
                break;
            case '.clear':
                this.context = {};
                console.log('Context cleared.');
                break;
            case '.history':
                this.history.forEach((h, i) => console.log(`${i + 1}  ${h}`));
                break;
            case '.exit':
                this.rl?.close();
                break;
            default:
                console.log(`Unknown command "${cmd}". Type .help for a list.`);
        }
    }

    private printBanner(): void {
        console.log([
            `\x1b[35m╔══════════════════════════════════════╗`,
            `║  K'UHUL++ Interactive REPL  v2.0     ║`,
            `║  Type .help for commands              ║`,
            `╚══════════════════════════════════════╝\x1b[0m`,
        ].join('\n'));
    }
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function formatValue(v: unknown): string {
    if (v instanceof Float32Array) {
        const preview = Array.from(v.slice(0, 8)).map(x => x.toFixed(3)).join(', ');
        return `Float32Array(${v.length})[${preview}${v.length > 8 ? '...' : ''}]`;
    }
    return JSON.stringify(v);
}

// ------------------------------------------------------------------ //
// CLI entry point
// ------------------------------------------------------------------ //

if (typeof process !== 'undefined' && process.argv[1]?.endsWith('repl.ts')) {
    const repl = new KuhulREPL();
    repl.start().catch(console.error);
}
