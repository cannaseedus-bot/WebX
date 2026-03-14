// K'UHUL++ Virtual Machine
// Executes a Geometric IR program by interpreting instructions one at a time.
// The VM maintains a stack, a memory store (the manifold M), and a phase counter.

import type { GeometricIR, IRInstruction, KuhulType } from './memory-manager.js';

// Re-export ExecutionResult for use by the engine
export { ExecutionResult } from './execution-engine.js';

import type { ExecutionResult } from './execution-engine.js';

// ------------------------------------------------------------------ //
// KuhulVM
// ------------------------------------------------------------------ //

/**
 * Stack-based interpreter for Geometric IR programs.
 *
 * @example
 * const vm = new KuhulVM();
 * const result = await vm.execute(ir);
 */
export class KuhulVM {
    /** Current phase position in [0, 2π] */
    private phase = 0;
    /** Evaluation stack */
    private stack: unknown[]       = [];
    /** Named value store (manifold M bindings) */
    private memory: Map<string, unknown> = new Map();
    /** Execution log entries */
    private log: string[] = [];

    /**
     * Execute a compiled Geometric IR program.
     *
     * @param ir       - Geometric IR from `generateIR()` (optionally optimised)
     * @param env      - Optional host environment bindings
     * @returns Execution result containing outputs and timing
     */
    async execute(
        ir: GeometricIR,
        env: Record<string, unknown> = {},
    ): Promise<ExecutionResult> {
        this.phase  = ir.manifold.phase;
        this.stack  = [];
        this.memory = new Map(Object.entries(env));
        this.log    = [];

        const start = Date.now();

        for (const instr of ir.instructions) {
            await this.step(instr, env);
        }

        return {
            output:    this.memory,
            phase:     this.phase,
            log:       this.log,
            durationMs: Date.now() - start,
        };
    }

    // ---- Single instruction execution ----

    private async step(instr: IRInstruction, env: Record<string, unknown>): Promise<void> {
        const i = instr as any;

        switch (instr.op) {
            case 'const':
                this.push(i.value);
                this.memory.set(i.id, i.value);
                break;

            case 'load': {
                const val = this.memory.get(i.name) ?? env[i.name];
                this.push(val);
                this.memory.set(i.id, val);
                break;
            }

            case 'store':
                this.memory.set(i.name, this.memory.get(i.src));
                break;

            case 'alloc':
                this.memory.set(i.id, new Float32Array(0));
                break;

            case 'invoke': {
                const args = (i.args as string[]).map((a: string) => this.memory.get(a));
                const fn = (env as any)[i.callee];
                let result: unknown;
                if (typeof fn === 'function') {
                    result = await fn(...args);
                } else {
                    result = this.builtinInvoke(i.callee, args);
                }
                this.memory.set(i.id, result);
                this.push(result);
                break;
            }

            case 'phase':
                this.phase = (this.phase + i.delta) % (2 * Math.PI);
                this.log.push(`phase → ${this.phase.toFixed(4)}`);
                break;

            case 'return':
                // Signal end of execution by clearing the stack
                if (i.value !== undefined) {
                    this.push(this.memory.get(i.value));
                }
                break;

            case 'label':
                // Labels are resolved at compile time; no runtime action
                break;

            case 'branch':
            case 'condbranch':
                // Control flow requires basic-block dispatch — simplified here
                this.log.push(`[VM] control flow op "${instr.op}" encountered (linear execution mode)`);
                break;

            default:
                // Glyph operations
                await this.applyGlyph(instr.op, i, env);
                break;
        }
    }

    // ---- Glyph operation execution ----

    private async applyGlyph(op: string, i: any, _env: Record<string, unknown>): Promise<void> {
        const left  = this.resolveOperand(i.left);
        const right = this.resolveOperand(i.right);

        let result: unknown;

        switch (op) {
            case '⊗': result = this.elementWise(left, right, (a, b) => a * b); break;
            case '⊕': result = this.elementWise(left, right, (a, b) => a + b); break;
            case '⊖': result = this.elementWise(left, right, (a, b) => a - b); break;
            case '⊞': result = this.elementWise(left, right, (a, b) => a + b); break;
            case '⊝': result = this.elementWise(left, right, (a, b) => -a);    break;
            case '⊛': result = this.convolve(left, right);                      break;

            case '⤍': result = this.vectorEncrypt(left, right); break;
            case '↻':  result = this.rotCompress(left, right);  break;
            case '⊙':  result = this.radialProject(left, right);break;
            case '⟲':  result = this.sphericalLoop(left);        break;
            case '∿':  result = this.torsionField(left, right);  break;

            default:
                this.log.push(`[VM] unimplemented glyph "${op}" — pass-through`);
                result = left;
                break;
        }

        this.memory.set(i.id, result);
        this.push(result);
    }

    // ---- Math helpers ----

    private resolveOperand(id: string): Float32Array {
        const v = this.memory.get(id);
        if (v instanceof Float32Array) return v;
        if (typeof v === 'number') return new Float32Array([v]);
        return new Float32Array(0);
    }

    private elementWise(
        a: Float32Array,
        b: Float32Array,
        fn: (x: number, y: number) => number,
    ): Float32Array {
        const len = Math.min(a.length, b.length);
        const out = new Float32Array(len);
        for (let i = 0; i < len; i++) out[i] = fn(a[i], b[i]);
        return out;
    }

    private convolve(signal: Float32Array, kernel: Float32Array): Float32Array {
        const out = new Float32Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            let acc = 0;
            for (let j = 0; j < kernel.length; j++) {
                if (i + j < signal.length) acc += signal[i + j] * kernel[j];
            }
            out[i] = acc;
        }
        return out;
    }

    private vectorEncrypt(verts: Float32Array, matrix: Float32Array): Float32Array {
        const out = new Float32Array(verts.length);
        for (let i = 0; i + 2 < verts.length; i += 3) {
            const x = verts[i], y = verts[i+1], z = verts[i+2];
            out[i]   = matrix[0]*x + matrix[4]*y + matrix[8]*z  + matrix[12];
            out[i+1] = matrix[1]*x + matrix[5]*y + matrix[9]*z  + matrix[13];
            out[i+2] = matrix[2]*x + matrix[6]*y + matrix[10]*z + matrix[14];
        }
        return out;
    }

    private rotCompress(verts: Float32Array, params: Float32Array): Float32Array {
        const angleDeg = params[0] ?? 45;
        const rad = (angleDeg * Math.PI) / 180;
        const cos = Math.cos(rad), sin = Math.sin(rad);
        const out = new Float32Array(verts.length);
        for (let i = 0; i + 2 < verts.length; i += 3) {
            out[i]   = verts[i] * cos - verts[i+2] * sin;
            out[i+1] = verts[i+1];
            out[i+2] = verts[i] * sin + verts[i+2] * cos;
        }
        return out;
    }

    private radialProject(points: Float32Array, params: Float32Array): Float32Array {
        const radius = params[0] ?? 1;
        const out = new Float32Array(points.length);
        for (let i = 0; i + 2 < points.length; i += 3) {
            const x = points[i], y = points[i+1], z = points[i+2];
            const len = Math.sqrt(x*x + y*y + z*z) || 1;
            const scale = radius / len;
            out[i] = x*scale; out[i+1] = y*scale; out[i+2] = z*scale;
        }
        return out;
    }

    private sphericalLoop(verts: Float32Array): Float32Array {
        const out = new Float32Array(verts.length);
        for (let i = 0; i + 2 < verts.length; i += 3) {
            const x = verts[i], y = verts[i+1], z = verts[i+2];
            const r     = Math.sqrt(x*x + y*y + z*z) || 1;
            const theta = Math.acos(z / r);
            const phi   = Math.atan2(y, x);
            // Spherical → Cartesian (identity round-trip with phase rotation)
            out[i]   = r * Math.sin(theta + this.phase) * Math.cos(phi);
            out[i+1] = r * Math.sin(theta + this.phase) * Math.sin(phi);
            out[i+2] = r * Math.cos(theta + this.phase);
        }
        return out;
    }

    private torsionField(verts: Float32Array, params: Float32Array): Float32Array {
        const factor = params[0] ?? 0.1;
        const out = new Float32Array(verts.length);
        for (let i = 0; i + 2 < verts.length; i += 3) {
            const twist = factor * verts[i+1]; // twist increases with Y
            const cos = Math.cos(twist), sin = Math.sin(twist);
            out[i]   = verts[i] * cos - verts[i+2] * sin;
            out[i+1] = verts[i+1];
            out[i+2] = verts[i] * sin + verts[i+2] * cos;
        }
        return out;
    }

    // ---- Built-in function dispatch ----

    private builtinInvoke(callee: string, args: unknown[]): unknown {
        switch (callee) {
            case '__array':     return new Float32Array(args as number[]);
            case '__unary_-':   return -(args[0] as number);
            case '__op_/':      return (args[0] as number) / (args[1] as number);
            case '__train':     this.log.push('[VM] Train called'); return null;
            case '__index':     {
                const arr = args[0] as Float32Array;
                const idx = args[1] as number;
                return arr instanceof Float32Array ? arr[idx] : undefined;
            }
            default:
                this.log.push(`[VM] unknown built-in "${callee}"`);
                return undefined;
        }
    }

    // ---- Stack helpers ----

    private push(value: unknown): void { this.stack.push(value); }
    private pop():  unknown            { return this.stack.pop(); }
}
