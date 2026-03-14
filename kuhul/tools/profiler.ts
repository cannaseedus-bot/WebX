// K'UHUL++ Performance Profiler
// Measures wall-clock time for each stage of the K'UHUL++ compilation
// and execution pipeline, then produces a structured report.

import { tokenize }     from '../compiler/lexer.js';
import { parse }        from '../compiler/parser.js';
import { analyze }      from '../compiler/semantic-analyzer.js';
import { generateIR }   from '../compiler/ir-generator.js';
import { IROptimizer }  from '../ir/ir-optimizer.js';
import { IRVerifier }   from '../ir/ir-verifier.js';
import { KuhulVM }      from '../runtime/kuhul-vm.js';
import { ExecutionEngine } from '../runtime/execution-engine.js';

// ------------------------------------------------------------------ //
// Report types
// ------------------------------------------------------------------ //

export interface StageTimings {
    lex:      number;
    parse:    number;
    semantic: number;
    irGen:    number;
    opt:      number;
    verify:   number;
    execute:  number;
    total:    number;
}

export interface ProfileReport {
    source:   string;
    timings:  StageTimings;
    /** Token count from the lexer */
    tokens:   number;
    /** Top-level AST node count */
    astNodes: number;
    /** IR instruction count after optimisation */
    irInstrs: number;
    /** Whether all stages completed without error */
    success:  boolean;
    errors:   string[];
}

// ------------------------------------------------------------------ //
// KuhulProfiler
// ------------------------------------------------------------------ //

/**
 * Profiles the complete K'UHUL++ pipeline stage-by-stage.
 *
 * @example
 * const profiler = new KuhulProfiler();
 * profiler.start();
 * const report = await profiler.profile(source);
 * console.log(profiler.report());
 */
export class KuhulProfiler {
    private sessions: ProfileReport[] = [];
    private running   = false;
    private startTime = 0;

    /** Begin a profiling session */
    start(): void {
        this.running   = true;
        this.startTime = Date.now();
        this.sessions  = [];
    }

    /** Stop the profiling session */
    stop(): void {
        this.running = false;
    }

    /**
     * Profile a complete compilation + execution run.
     *
     * @param source   - K'UHUL++ source to profile
     * @param env      - Optional execution environment
     * @returns ProfileReport with per-stage timings
     */
    async profile(source: string, env: Record<string, unknown> = {}): Promise<ProfileReport> {
        const errors: string[] = [];
        const timings: Partial<StageTimings> = {};
        let tokens = 0, astNodes = 0, irInstrs = 0;
        let success = true;

        const t = (label: keyof StageTimings, fn: () => unknown) => {
            const s = Date.now();
            const result = fn();
            timings[label] = Date.now() - s;
            return result;
        };

        try {
            // Lex
            const tokenArr = t('lex', () => tokenize(source)) as ReturnType<typeof tokenize>;
            tokens = tokenArr.length;

            // Parse
            const ast = t('parse', () => parse(tokenArr)) as ReturnType<typeof parse>;
            astNodes = ast.body.length;

            // Semantic
            const { ast: annotated, errors: semErrors } =
                t('semantic', () => analyze(ast)) as ReturnType<typeof analyze>;
            if (semErrors.length > 0) errors.push(...semErrors.map(e => e.toString()));

            // IR generation
            const rawIR = t('irGen', () => generateIR(annotated)) as ReturnType<typeof generateIR>;

            // Optimisation
            const optimizer = new IROptimizer();
            const optIR = t('opt', () => optimizer.optimize(rawIR)) as ReturnType<typeof IROptimizer.prototype.optimize>;
            irInstrs = optIR.instructions.length;

            // Verification
            const verifier = new IRVerifier();
            t('verify', () => verifier.verify(optIR));

            // Execution
            const vm = new KuhulVM();
            const execStart = Date.now();
            await vm.execute(optIR, env);
            timings.execute = Date.now() - execStart;

        } catch (err: any) {
            errors.push(err.message);
            success = false;
        }

        timings.total = Object.values(timings).reduce((a, b) => a + (b ?? 0), 0) as number;

        const report: ProfileReport = {
            source: source.slice(0, 60) + (source.length > 60 ? '...' : ''),
            timings: timings as StageTimings,
            tokens,
            astNodes,
            irInstrs,
            success,
            errors,
        };

        this.sessions.push(report);
        return report;
    }

    /**
     * Generate a human-readable text report for all profiled sessions.
     */
    report(): string {
        if (this.sessions.length === 0) return 'No profiling sessions recorded.';

        const lines: string[] = ['K\'UHUL++ Profiler Report', '═'.repeat(50)];

        this.sessions.forEach((s, i) => {
            lines.push(`\nSession ${i + 1}: ${s.source}`);
            lines.push(`  Status:      ${s.success ? '✓ OK' : '✗ FAILED'}`);
            if (s.errors.length > 0) lines.push(`  Errors:      ${s.errors.join('; ')}`);
            lines.push(`  Tokens:      ${s.tokens}`);
            lines.push(`  AST nodes:   ${s.astNodes}`);
            lines.push(`  IR instrs:   ${s.irInstrs}`);
            lines.push(`  Timings:`);
            lines.push(`    lex:       ${s.timings.lex}ms`);
            lines.push(`    parse:     ${s.timings.parse}ms`);
            lines.push(`    semantic:  ${s.timings.semantic}ms`);
            lines.push(`    ir-gen:    ${s.timings.irGen}ms`);
            lines.push(`    opt:       ${s.timings.opt}ms`);
            lines.push(`    verify:    ${s.timings.verify}ms`);
            lines.push(`    execute:   ${s.timings.execute}ms`);
            lines.push(`    total:     ${s.timings.total}ms`);
        });

        return lines.join('\n');
    }
}
