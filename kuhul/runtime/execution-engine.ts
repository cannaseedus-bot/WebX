// K'UHUL++ Execution Engine
// Top-level pipeline: source text → tokens → AST → IR → execution.
// Combines the lexer, parser, semantic analyzer, IR generator,
// optimizer, verifier, and VM into a single `run()` call.

import { tokenize }     from '../compiler/lexer.js';
import { parse }        from '../compiler/parser.js';
import { analyze }      from '../compiler/semantic-analyzer.js';
import { generateIR }   from '../compiler/ir-generator.js';
import { IROptimizer }  from '../ir/ir-optimizer.js';
import { IRVerifier }   from '../ir/ir-verifier.js';
import { KuhulVM }      from './kuhul-vm.js';
import { PhaseManager } from './phase-manager.js';
import { MemoryManager }from './memory-manager.js';

// ------------------------------------------------------------------ //
// ExecutionResult
// ------------------------------------------------------------------ //

/** Outcome of a full K'UHUL++ program execution */
export interface ExecutionResult {
    /** Named output values written to manifold M memory */
    output:     Map<string, unknown>;
    /** Final phase angle in radians */
    phase:      number;
    /** Execution log messages */
    log:        string[];
    /** Wall-clock duration in milliseconds */
    durationMs: number;
}

// ------------------------------------------------------------------ //
// ExecutionEngine
// ------------------------------------------------------------------ //

/**
 * End-to-end K'UHUL++ execution engine.
 *
 * @example
 * const engine = new ExecutionEngine();
 * const result = await engine.run(`
 *   Tensor v = [1.0, 2.0, 3.0];
 *   Tensor w = [4.0, 5.0, 6.0];
 *   Tensor out = v ⊗ w;
 * `);
 */
export class ExecutionEngine {
    readonly vm:      KuhulVM;
    readonly phase:   PhaseManager;
    readonly memory:  MemoryManager;

    private readonly optimizer: IROptimizer;
    private readonly verifier:  IRVerifier;

    /** If true, verification errors abort execution */
    readonly strictMode: boolean;

    constructor(options: { strictMode?: boolean; useSharedMemory?: boolean } = {}) {
        this.strictMode = options.strictMode ?? false;
        this.vm         = new KuhulVM();
        this.phase      = new PhaseManager();
        this.memory     = new MemoryManager(options.useSharedMemory);
        this.optimizer  = new IROptimizer();
        this.verifier   = new IRVerifier();
    }

    // ---- Main entry point ----

    /**
     * Compile and execute a K'UHUL++ source string end-to-end.
     *
     * @param source - K'UHUL++ source code
     * @param env    - Host environment bindings made available to the program
     * @returns Execution result
     * @throws On lexer / parser / semantic errors (unless strict mode is off)
     */
    async run(source: string, env: Record<string, unknown> = {}): Promise<ExecutionResult> {
        const pipeline: string[] = [];
        const startTotal = Date.now();

        // 1. Tokenize
        const t0 = Date.now();
        const tokens = tokenize(source);
        pipeline.push(`[lexer]    ${tokens.length} tokens  (${Date.now() - t0}ms)`);

        // 2. Parse
        const t1 = Date.now();
        const ast = parse(tokens);
        pipeline.push(`[parser]   ${ast.body.length} top-level nodes  (${Date.now() - t1}ms)`);

        // 3. Semantic analysis
        const t2 = Date.now();
        const { errors: semErrors, warnings, ast: annotatedAST } = analyze(ast);
        pipeline.push(`[semantic] ${semErrors.length} errors, ${warnings.length} warnings  (${Date.now() - t2}ms)`);
        if (this.strictMode && semErrors.length > 0) {
            throw new Error(`Semantic errors:\n${semErrors.map(e => e.toString()).join('\n')}`);
        }

        // 4. IR generation
        const t3 = Date.now();
        const rawIR = generateIR(annotatedAST);
        pipeline.push(`[ir-gen]   ${rawIR.instructions.length} instructions  (${Date.now() - t3}ms)`);

        // 5. IR optimisation
        const t4 = Date.now();
        const optimIR = this.optimizer.optimize(rawIR);
        pipeline.push(`[opt]      ${optimIR.instructions.length} instructions  (${Date.now() - t4}ms)`);

        // 6. IR verification
        const t5 = Date.now();
        const verifyResult = this.verifier.verify(optimIR);
        pipeline.push(`[verify]   valid=${verifyResult.valid}  (${Date.now() - t5}ms)`);
        if (this.strictMode && !verifyResult.valid) {
            throw new Error(`IR verification failed:\n${verifyResult.errors.join('\n')}`);
        }

        // 7. Sync phase manager with IR manifold phase
        this.phase.setPhase(optimIR.manifold.phase);

        // 8. Execute
        const result = await this.vm.execute(optimIR, env);
        result.log.unshift(...pipeline);
        result.durationMs = Date.now() - startTotal;

        return result;
    }

    // ---- Utilities ----

    /** Compile source to IR without executing */
    compile(source: string) {
        const tokens      = tokenize(source);
        const ast         = parse(tokens);
        const { ast: annotated } = analyze(ast);
        const raw         = generateIR(annotated);
        return this.optimizer.optimize(raw);
    }
}
