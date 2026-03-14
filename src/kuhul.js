// KUHUL 3D - Symbolic Glyph Operations for GPU-Accelerated 3D Transforms
// Each glyph maps to a deterministic geometric operation executed on the manifold M.

/**
 * Symbolic glyph constants for KUHUL operations.
 * Use these with KuhulD12WebX.executeGlyph() or D12WebX CommandList.execute().
 */
export const GLYPHS = {
    VECTOR_ENCRYPT: '⤍',   // Affine transform on a vector field
    ROTATIONAL_COMPRESSION: '↻', // Geometry compression via rotation matrices
    SPHERICAL_LOOP: '⟲',    // Cartesian ↔ spherical coordinate transform
    TORSION_FIELD: '∿',     // Torsion/twisting mesh deformation
    RADIAL_PROJECTION: '⊙', // Radial basis function projection
};

// ------------------------------------------------------------------ //
// Low-level glyph math (operates on Float32Array views directly)
// ------------------------------------------------------------------ //

/**
 * (⤍) Vector Encrypt — apply a 4×4 affine matrix to each vec3 in the buffer.
 * Homogeneous w-component is implicitly 1 (no perspective divide).
 *
 * @param {Float32Array} view  - Buffer view (stride: 3 floats per vertex)
 * @param {number[]|Float32Array} matrix - 16-element column-major 4×4 matrix
 */
function vectorEncrypt(view, matrix) {
    const m = matrix instanceof Float32Array ? matrix : new Float32Array(matrix);
    for (let i = 0; i + 2 < view.length; i += 3) {
        const x = view[i], y = view[i + 1], z = view[i + 2];
        view[i]     = m[0] * x + m[4] * y + m[8]  * z + m[12];
        view[i + 1] = m[1] * x + m[5] * y + m[9]  * z + m[13];
        view[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
}

/**
 * (↻) Rotational Compression — reduce vertex count by merging spatially
 * close vertices after rotating the mesh by `angleDeg` degrees about Y.
 * Returns a new Float32Array containing the compressed vertices.
 *
 * @param {Float32Array} view      - Buffer view (stride: 3 floats per vertex)
 * @param {number}       angleDeg  - Compression rotation angle in degrees
 * @returns {Float32Array} Compressed vertex array
 */
function rotationalCompression(view, angleDeg) {
    const rad = (angleDeg * Math.PI) / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    const n = Math.floor(view.length / 3);

    // Rotate about Y then bucket by grid cell to merge nearby vertices
    const gridStep = 0.05; // merge threshold (world units)
    const merged = new Map();
    const outVerts = [];

    for (let i = 0; i < n; i++) {
        const ix = i * 3;
        const ox =  cos * view[ix]     + sin * view[ix + 2];
        const oy =  view[ix + 1];
        const oz = -sin * view[ix]     + cos * view[ix + 2];

        // Quantise to grid
        const gx = Math.round(ox / gridStep);
        const gy = Math.round(oy / gridStep);
        const gz = Math.round(oz / gridStep);
        const key = `${gx},${gy},${gz}`;

        if (!merged.has(key)) {
            merged.set(key, outVerts.length / 3);
            outVerts.push(ox, oy, oz);
        }
    }

    const compressed = new Float32Array(outVerts);
    // Write back in-place (as much as fits)
    view.set(compressed.subarray(0, Math.min(compressed.length, view.length)));
    return compressed;
}

/**
 * (⟲) Spherical Loop — transform each vec3 between Cartesian and spherical
 * coordinates. Applies `angleDeg` as an azimuthal offset (φ rotation).
 *
 * Cartesian (x,y,z) → Spherical (r, θ, φ) → back to Cartesian with φ offset.
 *
 * @param {Float32Array} view     - Buffer view (stride: 3 floats)
 * @param {number}       angleDeg - Azimuthal offset in degrees
 */
function sphericalLoop(view, angleDeg) {
    const offset = (angleDeg * Math.PI) / 180;
    for (let i = 0; i + 2 < view.length; i += 3) {
        const x = view[i], y = view[i + 1], z = view[i + 2];
        const r     = Math.sqrt(x * x + y * y + z * z);
        if (r === 0) continue;
        const theta = Math.acos(y / r);                    // polar (0…π)
        const phi   = Math.atan2(z, x) + offset;           // azimuth + offset

        view[i]     = r * Math.sin(theta) * Math.cos(phi);
        view[i + 1] = r * Math.cos(theta);
        view[i + 2] = r * Math.sin(theta) * Math.sin(phi);
    }
}

/**
 * (∿) Torsion Field — twist the mesh along the Y axis.
 * Each vertex is rotated about Y by an angle proportional to its Y position
 * scaled by `intensity`.
 *
 * @param {Float32Array} view      - Buffer view (stride: 3 floats)
 * @param {number}       intensity - Torsion strength (radians per world unit)
 */
function torsionField(view, intensity) {
    for (let i = 0; i + 2 < view.length; i += 3) {
        const x = view[i], y = view[i + 1], z = view[i + 2];
        const angle = y * intensity;
        const cos   = Math.cos(angle);
        const sin   = Math.sin(angle);
        view[i]     =  cos * x + sin * z;
        view[i + 2] = -sin * x + cos * z;
    }
}

/**
 * (⊙) Radial Projection — project each point onto the unit sphere (normalise)
 * then scale by the provided `radius`.
 *
 * @param {Float32Array} view   - Buffer view (stride: 3 floats)
 * @param {number}       radius - Target sphere radius
 */
function radialProjection(view, radius) {
    for (let i = 0; i + 2 < view.length; i += 3) {
        const x = view[i], y = view[i + 1], z = view[i + 2];
        const len = Math.sqrt(x * x + y * y + z * z);
        if (len === 0) continue;
        const scale = radius / len;
        view[i]     = x * scale;
        view[i + 1] = y * scale;
        view[i + 2] = z * scale;
    }
}

// ------------------------------------------------------------------ //
// Dispatch table
// ------------------------------------------------------------------ //

const GLYPH_HANDLERS = {
    [GLYPHS.VECTOR_ENCRYPT]:          vectorEncrypt,
    [GLYPHS.ROTATIONAL_COMPRESSION]:  rotationalCompression,
    [GLYPHS.SPHERICAL_LOOP]:          sphericalLoop,
    [GLYPHS.TORSION_FIELD]:           torsionField,
    [GLYPHS.RADIAL_PROJECTION]:       radialProjection,
};

/**
 * Low-level function: apply a glyph directly to a GPU buffer view.
 * Used by D12WebX's internal command executor.
 *
 * @param {string} glyph  - Glyph symbol (see GLYPHS)
 * @param {{ view: Float32Array }} buffer - GPU buffer with Float32Array view
 * @param {*}      param  - Glyph-specific parameter
 * @returns {{ glyph, verticesProcessed, durationMs }}
 */
export function applyGlyph(glyph, buffer, param) {
    const handler = GLYPH_HANDLERS[glyph];
    if (!handler) {
        throw new Error(`Unknown KUHUL glyph: "${glyph}". Valid glyphs: ${Object.values(GLYPHS).join(', ')}`);
    }
    const t0 = performance.now();
    handler(buffer.view, param);
    const durationMs = performance.now() - t0;
    return {
        glyph,
        verticesProcessed: Math.floor(buffer.view.length / 3),
        durationMs,
    };
}

// ------------------------------------------------------------------ //
// High-level KuhulD12WebX class
// ------------------------------------------------------------------ //

/**
 * KuhulD12WebX — symbolic glyph operations for GPU-accelerated 3D transforms.
 *
 * Operates on the geometric manifold M (ℝ³).  All operations are deterministic
 * and verifiable: the same inputs always produce the same outputs.
 *
 * @example
 * import { KuhulD12WebX, GLYPHS } from 'd12webx/kuhul';
 *
 * const kuhul = new KuhulD12WebX();
 * const result = await kuhul.executeGlyph(GLYPHS.VECTOR_ENCRYPT, meshBuffer, matrix);
 * console.log(`Processed ${result.verticesProcessed} vertices in ${result.durationMs.toFixed(2)}ms`);
 */
class KuhulD12WebX {
    constructor() {
        this._history = [];
    }

    /**
     * Execute a KUHUL glyph on a GPU buffer.
     *
     * @param {string} glyph  - Glyph symbol (see GLYPHS constant)
     * @param {{ view: Float32Array }} buffer - GPU buffer returned by D12WebX.createBuffer()
     * @param {*}      param  - Operation parameter:
     *   - (⤍): 16-element column-major 4×4 matrix (number[] | Float32Array)
     *   - (↻): rotation angle in degrees (number)
     *   - (⟲): azimuthal offset in degrees (number)
     *   - (∿): torsion intensity in radians/unit (number)
     *   - (⊙): target sphere radius (number)
     * @returns {Promise<{ glyph, verticesProcessed, durationMs }>}
     */
    async executeGlyph(glyph, buffer, param) {
        const result = applyGlyph(glyph, buffer, param);
        this._history.push({ ...result, timestamp: performance.now() });
        return result;
    }

    /**
     * Execute multiple glyphs in sequence on the same buffer (pipeline).
     *
     * @param {Array<[string, *]>} pipeline - Array of [glyph, param] pairs
     * @param {{ view: Float32Array }} buffer - Target GPU buffer
     * @returns {Promise<Array>} Results for each stage
     */
    async executePipeline(pipeline, buffer) {
        const results = [];
        for (const [glyph, param] of pipeline) {
            results.push(await this.executeGlyph(glyph, buffer, param));
        }
        return results;
    }

    /**
     * Return the execution history (glyph, verticesProcessed, durationMs, timestamp).
     * @returns {Array}
     */
    getHistory() {
        return [...this._history];
    }

    /**
     * Clear execution history.
     */
    clearHistory() {
        this._history = [];
    }
}

export default KuhulD12WebX;
