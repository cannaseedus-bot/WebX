// K'UHUL++ LLM Optimizer
// Analyses a Geometric IR program and suggests optimizations.
// When an LLM provider is available it can suggest novel fusions;
// otherwise rule-based suggestions are returned.

import type { GeometricIR, IRInstruction, GlyphOp } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// OptimizationSuggestion
// ------------------------------------------------------------------ //

export interface OptimizationSuggestion {
    /** Human-readable title */
    title: string;
    /** Detailed explanation */
    description: string;
    /** Estimated speedup (1.0 = no improvement, 2.0 = 2× faster) */
    estimatedSpeedup: number;
    /** Instruction indices involved */
    instructionIndices: number[];
    /** Category */
    category: 'fusion' | 'dead-code' | 'constant-fold' | 'memory' | 'parallelism' | 'glyph';
}

// ------------------------------------------------------------------ //
// LLMOptimizer
// ------------------------------------------------------------------ //

/**
 * LLM-guided optimization suggester for Geometric IR programs.
 * Returns actionable suggestions with speedup estimates.
 *
 * @example
 * const opt = new LLMOptimizer();
 * const suggestions = await opt.suggest(ir);
 */
export class LLMOptimizer {
    private llmProvider?: (prompt: string) => Promise<string>;

    constructor(llmProvider?: (prompt: string) => Promise<string>) {
        this.llmProvider = llmProvider;
    }

    /**
     * Analyse the IR and return optimization suggestions.
     *
     * @param ir - Geometric IR to analyse
     * @returns Array of ranked suggestions (highest speedup first)
     */
    async suggest(ir: GeometricIR): Promise<OptimizationSuggestion[]> {
        const suggestions: OptimizationSuggestion[] = [];

        // Rule-based passes
        suggestions.push(...this.detectTransformFusions(ir));
        suggestions.push(...this.detectDeadConstants(ir));
        suggestions.push(...this.detectSerialGlyphChains(ir));
        suggestions.push(...this.detectPhaseInvariantOps(ir));

        // LLM-assisted (if provider available)
        if (this.llmProvider) {
            try {
                const llmSuggestions = await this.askLLM(ir);
                suggestions.push(...llmSuggestions);
            } catch {
                // LLM unavailable — return rule-based results
            }
        }

        // Sort by estimated speedup descending
        suggestions.sort((a, b) => b.estimatedSpeedup - a.estimatedSpeedup);
        return suggestions;
    }

    // ---- Rule-based detectors ----

    private detectTransformFusions(ir: GeometricIR): OptimizationSuggestion[] {
        const suggestions: OptimizationSuggestion[] = [];
        const fuseable: GlyphOp[] = ['⊗', '⤍', '↻'];

        for (let i = 0; i < ir.instructions.length - 1; i++) {
            const cur  = ir.instructions[i]  as any;
            const next = ir.instructions[i+1] as any;
            if (
                fuseable.includes(cur.op) && fuseable.includes(next.op) &&
                next.left === cur.id
            ) {
                suggestions.push({
                    title:              `Fuse ${cur.op} + ${next.op}`,
                    description:        `Instructions ${i} and ${i+1} can be fused into a single combined transform, eliminating the intermediate tensor allocation.`,
                    estimatedSpeedup:   1.4,
                    instructionIndices: [i, i+1],
                    category:           'fusion',
                });
            }
        }

        return suggestions;
    }

    private detectDeadConstants(ir: GeometricIR): OptimizationSuggestion[] {
        const used = new Set<string>();
        for (const instr of ir.instructions) {
            const i = instr as any;
            if (i.left)  used.add(i.left);
            if (i.right) used.add(i.right);
            if (i.src)   used.add(i.src);
            if (i.args)  (i.args as string[]).forEach((a: string) => used.add(a));
        }

        const suggestions: OptimizationSuggestion[] = [];
        ir.instructions.forEach((instr, idx) => {
            if (instr.op === 'const' && !used.has(instr.id)) {
                suggestions.push({
                    title:              `Remove unused constant "${instr.id}"`,
                    description:        `Constant at index ${idx} is never referenced. Removing it reduces IR size.`,
                    estimatedSpeedup:   1.05,
                    instructionIndices: [idx],
                    category:           'dead-code',
                });
            }
        });

        return suggestions;
    }

    private detectSerialGlyphChains(ir: GeometricIR): OptimizationSuggestion[] {
        const suggestions: OptimizationSuggestion[] = [];
        let chainStart = -1, chainGlyph = '';

        for (let i = 0; i < ir.instructions.length; i++) {
            const op = ir.instructions[i].op;
            const isGlyph = ['⊗','⊕','⊖','⊛'].includes(op);
            if (isGlyph && op === chainGlyph) {
                // Extended chain
            } else if (isGlyph) {
                if (chainStart >= 0 && i - chainStart >= 3) {
                    suggestions.push({
                        title:              `Parallel ${chainGlyph} chain (indices ${chainStart}–${i-1})`,
                        description:        `${i - chainStart} consecutive ${chainGlyph} operations could be dispatched to a GPU compute shader in parallel.`,
                        estimatedSpeedup:   Math.min(i - chainStart, 8),
                        instructionIndices: Array.from({ length: i - chainStart }, (_, k) => chainStart + k),
                        category:           'parallelism',
                    });
                }
                chainStart = i;
                chainGlyph = op;
            } else {
                chainStart = -1; chainGlyph = '';
            }
        }

        return suggestions;
    }

    private detectPhaseInvariantOps(ir: GeometricIR): OptimizationSuggestion[] {
        const phaseOps = ir.instructions.filter(i => i.op === 'phase');
        if (phaseOps.length === 0) return [];
        return [{
            title:       'Hoist phase-invariant operations',
            description: `${phaseOps.length} phase advancement(s) found. Operations between phase updates that don't read the phase could be hoisted before the first phase op.`,
            estimatedSpeedup: 1.1,
            instructionIndices: [],
            category: 'glyph',
        }];
    }

    // ---- LLM call ----

    private async askLLM(ir: GeometricIR): Promise<OptimizationSuggestion[]> {
        const prompt = [
            `Analyze the following K'UHUL++ Geometric IR summary and suggest optimizations.`,
            `Instructions: ${ir.instructions.length}`,
            `Phases: ${ir.phases.map(p => p.name).join(', ')}`,
            `Manifold: ${ir.manifold.dimensions}D ${ir.manifold.metric}`,
            `Glyph ops: ${ir.instructions.filter(i => ['⊗','⊕','⊖','⊛','⤍','↻','⟲','∿','⊙'].includes(i.op)).length}`,
            `Return JSON array of { title, description, estimatedSpeedup, category } objects.`,
        ].join('\n');

        const response = await this.llmProvider!(prompt);
        try {
            const parsed = JSON.parse(response);
            if (!Array.isArray(parsed)) return [];
            return parsed.map((s: any) => ({
                title:              String(s.title ?? ''),
                description:        String(s.description ?? ''),
                estimatedSpeedup:   Number(s.estimatedSpeedup ?? 1),
                instructionIndices: [],
                category:           s.category ?? 'fusion',
            }));
        } catch {
            return [];
        }
    }
}
