// THE MIND BINDER — Unified Native Geometric Runtime (K++ / K'UHUL Plus Plus)
//
// Merges D12WebX's GPU substrate with K'uhul's phase-based control grammar into
// a single orchestrator.  Every K'uhul++ token has a direct counterpart here:
//
//   [Pop]     → constructor / beginScope
//   [Xul]     → endScope
//   [Wo]      → allocateTensor
//   [Yax]     → readTensor
//   [Ch'en]   → writeTensor
//   [Sek]     → applyOperator / applyGlyph
//   [K'ayab'] → beginPhase
//   [Kumk'u]  → endPhase
//   [Muwan]   → executeFold
//
// Phase indices 0–7 correspond to the 8 K'uhul++ phase positions:
//   0  → 0        1  → π/4     2  → π/2     3  → 3π/4
//   4  → π        5  → 5π/4    6  → 3π/2    7  → 7π/4

import D12WebX, { GPU_FLAGS } from './d12webx.js';
import KuhulD12WebX, { applyGlyph } from './kuhul.js';
import {
    GEOMETRIC_OPS,
    matMul,
    translate,
    subtract,
    project,
    validate,
    clamp,
    foldReduce,
    softmax,
} from './geometric-operators.js';
import { encodeToSVG, decodeFromSVG } from './svg3d.js';

// ------------------------------------------------------------------ //
// Phase constants
// ------------------------------------------------------------------ //

/**
 * The 8 phase positions in the K'uhul++ execution cycle (radians).
 * Index 0–7 maps directly to K'ayab' phase arguments.
 */
export const PHASES = Object.freeze([
    0,
    Math.PI / 4,
    Math.PI / 2,
    3 * Math.PI / 4,
    Math.PI,
    5 * Math.PI / 4,
    3 * Math.PI / 2,
    7 * Math.PI / 4,
]);

/** Human-readable labels for each phase. */
export const PHASE_NAMES = Object.freeze([
    '0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4',
]);

// ------------------------------------------------------------------ //
// MindBinder
// ------------------------------------------------------------------ //

/**
 * MindBinder — The K++ unified geometric runtime.
 *
 * Orchestrates GPU memory, phase-based command recording, fold execution,
 * geometric operators, and SVG-3D serialisation in a single coherent API.
 *
 * @example
 * const binder = new MindBinder();
 *
 * // [K'ayab' phase_0]  — allocate tensors
 * binder.beginPhase(0);
 * const X = binder.allocateTensor(1024 * 3);   // [Wo X]
 * binder.writeTensor(X, inputData);             // [Ch'en X]
 * binder.endPhase();                            // [Kumk'u]
 *
 * // [K'ayab' phase_1]  — compute
 * binder.beginPhase(1);
 * const Y = binder.allocateTensor(1024 * 3);
 * const result = binder.applyOperator('⊗', X, W, { aRows: 1024, aCols: 3, bCols: 3 });
 * binder.writeTensor(Y, result);
 * binder.endPhase();
 *
 * await binder.executeAllPhases();
 */
class MindBinder {
    /**
     * @param {object} [opts]
     * @param {number} [opts.heapSize=1073741824] - GPU heap size in bytes (default 1 GB)
     */
    constructor(opts = {}) {
        const { heapSize = 1024 * 1024 * 1024 } = opts;

        // Underlying GPU substrate
        this._gpu   = new D12WebX(heapSize);
        this._kuhul = new KuhulD12WebX();

        // Phase command lists — one per phase (0–7)
        this._phaseCommandLists = Array.from(
            { length: 8 },
            () => this._gpu.createCommandList(),
        );

        // Phase recording state
        this._currentPhaseIdx = null;
        this._phaseActive     = false;

        // Tensor registry: tensorId → { id, buffer, elementCount }
        this._tensors        = new Map();
        this._tensorIdCounter = 0;

        // Fold registry: name → { recordFn }
        this._folds = new Map();

        // Synchronisation fence
        this._fence = this._gpu.createFence();

        // Geometric-operator dispatch table
        this._ops = {
            [GEOMETRIC_OPS.PRODUCT]:    (a, b, meta = {}) =>
                matMul(a, b, meta.aRows || 1, meta.aCols || a.length, meta.bCols || 1),
            [GEOMETRIC_OPS.COMPOSE]:    (a, b)             => translate(a, b),
            [GEOMETRIC_OPS.DIFFERENCE]: (a, b)             => subtract(a, b),
            [GEOMETRIC_OPS.PROJECT]:    (a, _b, meta = {}) => project(a, meta.radius),
            [GEOMETRIC_OPS.VALIDATE]:   (a, _b, meta = {}) =>
                validate(a, meta.minBounds, meta.maxBounds),
            [GEOMETRIC_OPS.CLAMP]:      (a, _b, meta = {}) =>
                clamp(a, meta.min, meta.max),
            [GEOMETRIC_OPS.FOLD]:       (a, _b, meta = {}) =>
                foldReduce(a, meta.stride),
        };
    }

    // ------------------------------------------------------------------ //
    // Phase management  [K'ayab'] / [Kumk'u]
    // ------------------------------------------------------------------ //

    /**
     * Begin recording commands for a phase ([K'ayab']).
     *
     * @param {number} phaseIdx
     *   Phase index 0–7, or a radian value in [0, 2π) which is rounded to the
     *   nearest K'uhul++ phase boundary.
     * @returns {this}
     */
    beginPhase(phaseIdx) {
        if (this._phaseActive) {
            throw new Error("MindBinder: cannot begin a phase while another is active (missing [Kumk'u]?)");
        }

        let idx = phaseIdx;
        // Accept radian values — map to nearest phase index
        if (typeof phaseIdx === 'number' && (phaseIdx < 0 || phaseIdx >= 8 || !Number.isInteger(phaseIdx))) {
            let best = 0, bestDist = Infinity;
            for (let i = 0; i < PHASES.length; i++) {
                const d = Math.abs(PHASES[i] - phaseIdx);
                if (d < bestDist) { bestDist = d; best = i; }
            }
            idx = best;
        }
        this._currentPhaseIdx = Math.max(0, Math.min(7, Math.floor(idx)));
        this._phaseCommandLists[this._currentPhaseIdx].reset();
        this._phaseActive = true;
        return this;
    }

    /**
     * End recording for the current phase ([Kumk'u]).
     * @returns {this}
     */
    endPhase() {
        if (!this._phaseActive) {
            throw new Error('MindBinder: no active phase to end (missing [K\'ayab\'?])');
        }
        this._phaseCommandLists[this._currentPhaseIdx].close();
        this._phaseActive     = false;
        this._currentPhaseIdx = null;
        return this;
    }

    // ------------------------------------------------------------------ //
    // Tensor management  [Wo] / [Yax] / [Ch'en]
    // ------------------------------------------------------------------ //

    /**
     * Allocate a new tensor in manifold M ([Wo]).
     *
     * @param {number} elementCount - Number of float32 scalars to allocate
     * @param {number} [flags=GPU_FLAGS.UAV]
     * @returns {{ id: number, buffer: object, elementCount: number }}
     */
    allocateTensor(elementCount, flags = GPU_FLAGS.UAV) {
        const size   = elementCount * 4; // float32 = 4 bytes
        const buffer = this._gpu.createBuffer(size, flags);
        const id     = ++this._tensorIdCounter;
        const tensor = { id, buffer, elementCount };
        this._tensors.set(id, tensor);
        return tensor;
    }

    /**
     * Read tensor data from M ([Yax]).
     *
     * @param {number|{id:number,buffer:object}} tensorOrId
     * @returns {Float32Array}
     */
    readTensor(tensorOrId) {
        return this._gpu.readBuffer(this._resolveTensor(tensorOrId).buffer);
    }

    /**
     * Write data into a tensor ([Ch'en]).
     *
     * @param {number|{id:number,buffer:object}} tensorOrId
     * @param {ArrayLike<number>} data
     * @returns {this}
     */
    writeTensor(tensorOrId, data) {
        this._gpu.writeBuffer(this._resolveTensor(tensorOrId).buffer, data);
        return this;
    }

    /**
     * Release a tensor and return its memory to the allocator.
     *
     * @param {number|{id:number,buffer:object}} tensorOrId
     * @returns {this}
     */
    releaseTensor(tensorOrId) {
        const tensor = this._resolveTensor(tensorOrId);
        this._gpu.releaseBuffer(tensor.buffer);
        this._tensors.delete(tensor.id);
        return this;
    }

    // ------------------------------------------------------------------ //
    // Geometric operators  [Sek ⊗/⊕/…]
    // ------------------------------------------------------------------ //

    /**
     * Apply a geometric operator to one or two tensors ([Sek]).
     *
     * @param {string} op  - Operator symbol (GEOMETRIC_OPS)
     * @param {number|object} inputA  - Primary input tensor (or its id)
     * @param {number|object|ArrayLike<number>|null} [inputB]
     *   Second input tensor / id, or a raw array for in-place parameters
     *   (e.g. a translation vector for ⊕).  Pass null for unary operators.
     * @param {object} [meta]
     *   Extra parameters used by specific operators:
     *   - ⊗ : { aRows, aCols, bCols }
     *   - ⊛ : { radius }
     *   - ⊜ : { minBounds, maxBounds }
     *   - ⊝ : { min, max }
     *   - ⊞ : { stride }
     * @returns {Float32Array} Result data
     */
    applyOperator(op, inputA, inputB = null, meta = {}) {
        const handler = this._ops[op];
        if (!handler) {
            throw new Error(`MindBinder: unknown geometric operator "${op}"`);
        }
        const A = this.readTensor(inputA);
        // inputB may be a tensor id/object, or a raw array (e.g. translation vector).
        // Use _isTensorRef() to distinguish between the two.
        const B = (inputB !== null && inputB !== undefined)
            ? (this._isTensorRef(inputB) ? this.readTensor(inputB) : inputB)
            : null;
        return handler(A, B, meta);
    }

    /**
     * Apply a KUHUL 3D glyph to a tensor ([Sek] with glyph symbol).
     *
     * @param {string} glyph  - Glyph from GLYPHS
     * @param {number|object} tensorOrId
     * @param {*} param       - Glyph-specific parameter
     * @returns {Promise<{ glyph, verticesProcessed, durationMs }>}
     */
    async applyGlyph(glyph, tensorOrId, param) {
        const tensor = this._resolveTensor(tensorOrId);
        return this._kuhul.executeGlyph(glyph, tensor.buffer, param);
    }

    /**
     * Compute softmax over a tensor's data (geometric normalisation, used for
     * the attention / output layer).
     *
     * @param {number|object} tensorOrId
     * @returns {Float32Array}
     */
    applySoftmax(tensorOrId) {
        return softmax(this.readTensor(tensorOrId));
    }

    // ------------------------------------------------------------------ //
    // Fold registry  [Muwan]
    // ------------------------------------------------------------------ //

    /**
     * Register a named fold (pre-compiled reusable command sequence).
     *
     * The `recordFn` is called each time the fold is executed.  It receives
     * this MindBinder instance so it can allocate tensors, apply operators, etc.
     *
     * @param {string}   name
     * @param {function(MindBinder): Promise<*>} recordFn
     * @returns {this}
     */
    registerFold(name, recordFn) {
        if (typeof recordFn !== 'function') {
            throw new TypeError('MindBinder.registerFold: recordFn must be a function');
        }
        this._folds.set(name, { recordFn });
        return this;
    }

    /**
     * Execute a registered fold ([Muwan]).
     *
     * @param {string} name - Fold name
     * @param {...*}   args - Extra arguments forwarded to the fold function
     * @returns {Promise<*>}
     */
    async executeFold(name, ...args) {
        const fold = this._folds.get(name);
        if (!fold) throw new Error(`MindBinder: unknown fold "${name}"`);
        return fold.recordFn(this, ...args);
    }

    /**
     * Register the standard neural-layer fold.
     *
     * The fold performs: Y = clamp(X ⊗ W ⊕ b)  (linear + bias + ReLU).
     *
     * @returns {this}
     */
    registerNeuralLayerFold() {
        return this.registerFold('neural_layer', async (binder, { X, W, b, aRows, aCols, bCols }) => {
            // Linear: XW = X ⊗ W
            const xw = binder.applyOperator(GEOMETRIC_OPS.PRODUCT, X, W, { aRows, aCols, bCols });

            // Bias: XWb = XW ⊕ b
            const biasVec = binder.readTensor(b);
            const xwb = new Float32Array(xw.length);
            for (let r = 0; r < aRows; r++) {
                for (let c = 0; c < bCols; c++) {
                    xwb[r * bCols + c] = xw[r * bCols + c] + (biasVec[c] || 0);
                }
            }

            // ReLU: Y = ⊝ XWb
            return clamp(xwb);
        });
    }

    // ------------------------------------------------------------------ //
    // Phase execution
    // ------------------------------------------------------------------ //

    /**
     * Execute all recorded and closed phase command lists in parallel.
     * Signals the synchronisation fence when complete.
     *
     * @returns {Promise<Array>} Results from each executed phase
     */
    async executeAllPhases() {
        const active = this._phaseCommandLists.filter(cl => cl.length > 0 && cl.closed);
        const results = await this._gpu.executeParallel(active);
        this._fence.signal();
        return results;
    }

    /**
     * Execute the command list for a single phase.
     *
     * @param {number} phaseIdx - 0–7
     * @returns {Promise<Array>}
     */
    async executePhase(phaseIdx) {
        const cl = this._phaseCommandLists[phaseIdx];
        return this._gpu.executeParallel([cl]);
    }

    // ------------------------------------------------------------------ //
    // SVG-3D serialisation
    // ------------------------------------------------------------------ //

    /**
     * Serialise a tensor to SVG-3D format (on-disk / in-memory storage).
     *
     * @param {number|object} tensorOrId
     * @param {object} [opts] - Forwarded to encodeToSVG
     * @returns {string} SVG markup
     */
    serializeToSVG(tensorOrId, opts = {}) {
        return encodeToSVG(this.readTensor(tensorOrId), opts);
    }

    /**
     * Deserialise an SVG-3D string into a new tensor in M.
     *
     * @param {string} svgString
     * @returns {{ tensor: object, points: Float32Array, phases: Float32Array }}
     */
    deserializeFromSVG(svgString) {
        const { points, phases } = decodeFromSVG(svgString);
        const tensor = this.allocateTensor(points.length);
        this.writeTensor(tensor, points);
        return { tensor, points, phases };
    }

    // ------------------------------------------------------------------ //
    // Accessors
    // ------------------------------------------------------------------ //

    /** @returns {D12WebX} The underlying GPU substrate */
    get gpu()   { return this._gpu; }

    /** @returns {KuhulD12WebX} The underlying KUHUL glyph engine */
    get kuhul() { return this._kuhul; }

    /** @returns {Map<string,object>} Registered folds */
    get folds() { return new Map(this._folds); }

    /** @returns {number} Number of live tensors */
    get tensorCount() { return this._tensors.size; }

    // ------------------------------------------------------------------ //
    // Private helpers
    // ------------------------------------------------------------------ //

    /** @private */
    _resolveTensor(tensorOrId) {
        if (typeof tensorOrId === 'number') {
            const t = this._tensors.get(tensorOrId);
            if (!t) throw new Error(`MindBinder: tensor id=${tensorOrId} not found`);
            return t;
        }
        if (!tensorOrId || !tensorOrId.buffer) {
            throw new TypeError('MindBinder: expected a tensor object or tensor id');
        }
        return tensorOrId;
    }

    /**
     * Returns true if `value` looks like a tensor reference (numeric id or tensor object).
     * Used by applyOperator to distinguish tensors from raw data arrays.
     * @private
     */
    _isTensorRef(value) {
        if (typeof value === 'number') return true;
        return typeof value === 'object' && value !== null && 'id' in value && 'buffer' in value;
    }
}

export default MindBinder;
export { GEOMETRIC_OPS, encodeToSVG, decodeFromSVG };
