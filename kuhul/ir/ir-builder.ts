// K'UHUL++ IR Builder
// Builder pattern for constructing Geometric IR programs.
// All instruction creation methods return `this` for chaining.

import type {
    GeometricIR, IRInstruction, KuhulType, GlyphOp,
    ManifoldDef, Phase,
    IRLoad, IRStore, IRAlloc, IRInvoke, IRPhase,
    IRConst, IRGlyphInstr, IRBranch, IRCondBranch,
    IRLabel, IRReturn,
} from './ir-types.js';

// ------------------------------------------------------------------ //
// IRBuilder
// ------------------------------------------------------------------ //

/**
 * Fluent builder for Geometric IR programs.
 *
 * @example
 * const builder = new IRBuilder();
 * builder.beginPhase('main', 0, 2 * Math.PI);
 * builder.addConst('c0', 1.0, { kind: 'scalar', dtype: 'float32' });
 * const ir = builder.build();
 */
export class IRBuilder {
    private instructions: IRInstruction[] = [];
    private phases: Phase[]               = [];
    private symbols: Map<string, KuhulType> = new Map();
    private manifold: ManifoldDef = {
        dimensions: 3,
        metric: 'euclidean',
        phase: 0,
    };

    /** Active phase being built, if any */
    private currentPhase: Phase | null = null;

    // ---- Manifold configuration ----

    /** Set the manifold dimensions and metric */
    setManifold(dimensions: number, metric: ManifoldDef['metric'], initialPhase = 0): this {
        this.manifold = { dimensions, metric, phase: initialPhase };
        return this;
    }

    // ---- Phase management ----

    /** Begin a named execution phase covering [start, end] radians */
    beginPhase(name: string, start: number, end: number): this {
        this.currentPhase = { name, start, end, instructions: [] };
        return this;
    }

    /** Close the current phase and commit it */
    endPhase(): this {
        if (this.currentPhase) {
            this.phases.push(this.currentPhase);
            this.currentPhase = null;
        }
        return this;
    }

    // ---- Instruction factories ----

    private emit(instr: IRInstruction): void {
        if (this.currentPhase) {
            this.currentPhase.instructions.push(instr);
        }
        this.instructions.push(instr);
    }

    /** Emit a `load` instruction */
    addLoad(id: string, name: string, type: KuhulType): this {
        this.symbols.set(name, type);
        this.emit({ op: 'load', id, name, type } satisfies IRLoad);
        return this;
    }

    /** Emit a `store` instruction */
    addStore(id: string, name: string, src: string): this {
        this.emit({ op: 'store', id, name, src } satisfies IRStore);
        return this;
    }

    /** Emit an `alloc` instruction */
    addAlloc(id: string, type: KuhulType): this {
        this.emit({ op: 'alloc', id, type } satisfies IRAlloc);
        return this;
    }

    /** Emit an `invoke` instruction */
    addInvoke(id: string, callee: string, args: string[], returnType: KuhulType): this {
        this.emit({ op: 'invoke', id, callee, args, returnType } satisfies IRInvoke);
        return this;
    }

    /** Emit a `phase` instruction to advance the geometric phase */
    addPhase(id: string, delta: number): this {
        this.emit({ op: 'phase', id, delta } satisfies IRPhase);
        return this;
    }

    /** Emit a `const` instruction */
    addConst(id: string, value: number | string, type: KuhulType): this {
        this.emit({ op: 'const', id, value, type } satisfies IRConst);
        return this;
    }

    /** Emit a glyph operation instruction */
    addGlyphOp(id: string, op: GlyphOp, left: string, right: string, type: KuhulType): this {
        this.emit({ op, id, left, right, type } satisfies IRGlyphInstr);
        return this;
    }

    /** Emit an unconditional branch */
    addBranch(id: string, target: string): this {
        this.emit({ op: 'branch', id, target } satisfies IRBranch);
        return this;
    }

    /** Emit a conditional branch */
    addCondBranch(id: string, cond: string, ifTrue: string, ifFalse: string): this {
        this.emit({ op: 'condbranch', id, cond, ifTrue, ifFalse } satisfies IRCondBranch);
        return this;
    }

    /** Emit a label marking the start of a basic block */
    addLabel(id: string, name: string): this {
        this.emit({ op: 'label', id, name } satisfies IRLabel);
        return this;
    }

    /** Emit a return instruction */
    addReturn(id: string, value?: string): this {
        this.emit({ op: 'return', id, value } satisfies IRReturn);
        return this;
    }

    // ---- Build ----

    /**
     * Finalise and return the complete Geometric IR.
     * Any open phase is automatically closed.
     */
    build(): GeometricIR {
        if (this.currentPhase) this.endPhase();
        return {
            instructions: [...this.instructions],
            manifold:     { ...this.manifold },
            phases:       this.phases.map(p => ({ ...p, instructions: [...p.instructions] })),
            symbols:      new Map(this.symbols),
        };
    }
}
