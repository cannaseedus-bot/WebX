// K'UHUL++ Debugger
// Provides breakpoint-based step-through debugging of K'UHUL++ programs.
// The debugger compiles the source to IR, then executes one instruction at a time.

import { tokenize }     from '../compiler/lexer.js';
import { parse }        from '../compiler/parser.js';
import { analyze }      from '../compiler/semantic-analyzer.js';
import { generateIR }   from '../compiler/ir-generator.js';
import { IROptimizer }  from '../ir/ir-optimizer.js';
import type { GeometricIR, IRInstruction } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

export interface Breakpoint {
    id:       number;
    line?:    number;
    instrIdx: number;
    enabled:  boolean;
}

export interface DebugFrame {
    instrIdx:    number;
    instruction: IRInstruction;
    memory:      Map<string, unknown>;
    phase:       number;
}

// ------------------------------------------------------------------ //
// KuhulDebugger
// ------------------------------------------------------------------ //

/**
 * Interactive debugger for K'UHUL++ programs.
 *
 * @example
 * const dbg = new KuhulDebugger();
 * await dbg.load(source);
 * dbg.setBreakpoint(0, 5);   // break at instruction 5
 * await dbg.run();            // run until breakpoint
 * const frame = dbg.inspect();
 * console.log(frame.memory);
 */
export class KuhulDebugger {
    private ir:          GeometricIR | null = null;
    private instrIdx:    number             = 0;
    private memory:      Map<string, unknown> = new Map();
    private phase:       number             = 0;
    private breakpoints: Breakpoint[]       = [];
    private bpCounter    = 0;
    private paused       = false;

    // ---- Setup ----

    /**
     * Compile and load K'UHUL++ source for debugging.
     *
     * @param source - K'UHUL++ source code
     */
    async load(source: string): Promise<void> {
        const tokens = tokenize(source);
        const ast    = parse(tokens);
        const { ast: annotated } = analyze(ast);
        const rawIR  = generateIR(annotated);
        const opt    = new IROptimizer();
        this.ir      = opt.optimize(rawIR);
        this.instrIdx = 0;
        this.memory   = new Map();
        this.phase    = this.ir.manifold.phase;
        this.paused   = false;
    }

    // ---- Breakpoints ----

    /**
     * Set a breakpoint at a specific instruction index or source line.
     *
     * @param instrIdx - IR instruction index (0-based)
     * @param line     - Optional source line for display
     * @returns Breakpoint id
     */
    setBreakpoint(instrIdx: number, line?: number): number {
        const id = ++this.bpCounter;
        this.breakpoints.push({ id, instrIdx, line, enabled: true });
        return id;
    }

    /** Remove a breakpoint by id */
    removeBreakpoint(id: number): void {
        this.breakpoints = this.breakpoints.filter(bp => bp.id !== id);
    }

    /** Enable / disable a breakpoint */
    toggleBreakpoint(id: number, enabled: boolean): void {
        const bp = this.breakpoints.find(b => b.id === id);
        if (bp) bp.enabled = enabled;
    }

    /** List all breakpoints */
    listBreakpoints(): Breakpoint[] { return [...this.breakpoints]; }

    // ---- Execution control ----

    /**
     * Execute one IR instruction and pause.
     * @returns Current debug frame, or null if execution is complete
     */
    step(): DebugFrame | null {
        if (!this.ir) throw new Error('KuhulDebugger: no program loaded');
        if (this.instrIdx >= this.ir.instructions.length) return null;

        const instr = this.ir.instructions[this.instrIdx];
        this.executeInstruction(instr);
        this.instrIdx++;
        this.paused = true;
        return this.inspect();
    }

    /**
     * Continue execution until the next enabled breakpoint or program end.
     * @returns Debug frame at the breakpoint, or null if program ended
     */
    async run(): Promise<DebugFrame | null> {
        if (!this.ir) throw new Error('KuhulDebugger: no program loaded');
        this.paused = false;

        while (this.instrIdx < this.ir.instructions.length) {
            const bp = this.breakpoints.find(
                b => b.enabled && b.instrIdx === this.instrIdx,
            );
            if (bp) {
                this.paused = true;
                return this.inspect();
            }
            const instr = this.ir.instructions[this.instrIdx];
            this.executeInstruction(instr);
            this.instrIdx++;
        }

        return null;
    }

    // ---- Inspection ----

    /**
     * Inspect the current execution frame.
     * @returns Current debug frame, or null if no program is loaded
     */
    inspect(): DebugFrame | null {
        if (!this.ir) return null;
        const idx   = Math.min(this.instrIdx, this.ir.instructions.length - 1);
        const instr = this.ir.instructions[idx];
        if (!instr) return null;
        return {
            instrIdx:    this.instrIdx,
            instruction: instr,
            memory:      new Map(this.memory),
            phase:       this.phase,
        };
    }

    /** Whether the debugger is currently paused at a breakpoint */
    get isPaused(): boolean { return this.paused; }

    /** Whether the program has finished executing */
    get isDone(): boolean {
        return this.ir === null || this.instrIdx >= this.ir.instructions.length;
    }

    // ---- Instruction execution ----

    private executeInstruction(instr: IRInstruction): void {
        const i = instr as any;
        switch (instr.op) {
            case 'const':  this.memory.set(i.id, i.value); break;
            case 'load':   this.memory.set(i.id, this.memory.get(i.name)); break;
            case 'store':  this.memory.set(i.name, this.memory.get(i.src)); break;
            case 'alloc':  this.memory.set(i.id, new Float32Array(0)); break;
            case 'phase':  this.phase = (this.phase + i.delta) % (2 * Math.PI); break;
            default:
                // Glyph ops and others pass through for display purposes
                break;
        }
    }
}
