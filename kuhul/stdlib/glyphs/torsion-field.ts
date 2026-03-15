// K'UHUL++ Standard Library — Torsion Field Glyph (∿)
// Applies a torsion / twisting deformation to a packed vec3 buffer.
// The twist angle increases linearly with the Y coordinate.

/**
 * (∿) Torsion Field — twist mesh vertices around the Y axis by an amount
 * proportional to their Y position.
 *
 * @param data   - Packed vertex buffer (stride: 3 floats per vertex)
 * @param factor - Torsion factor in radians per unit of Y (positive = right-hand twist)
 * @returns New Float32Array with the deformed vertices
 *
 * @example
 * const column = buildColumn(1.0, 2.0, 16);
 * const twisted = torsionField(column, Math.PI / 4); // 45° per unit Y
 */
export function torsionField(data: Float32Array, factor: number): Float32Array {
    const out = new Float32Array(data.length);

    for (let i = 0; i + 2 < data.length; i += 3) {
        const x = data[i], y = data[i + 1], z = data[i + 2];
        const angle = factor * y;
        const cos   = Math.cos(angle);
        const sin   = Math.sin(angle);
        out[i]     = x * cos - z * sin;
        out[i + 1] = y;
        out[i + 2] = x * sin + z * cos;
    }

    return out;
}

/**
 * Apply a spatially varying torsion where the twist is modulated by a
 * sine wave along the Y axis (wave torsion).
 *
 * @param data      - Packed vertex buffer
 * @param amplitude - Maximum twist angle in radians
 * @param frequency - Wave frequency along Y axis
 * @param phase     - Phase offset in radians
 * @returns New Float32Array with wave-deformed vertices
 */
export function waveTorsion(
    data:      Float32Array,
    amplitude: number,
    frequency: number,
    phase    = 0,
): Float32Array {
    const out = new Float32Array(data.length);

    for (let i = 0; i + 2 < data.length; i += 3) {
        const x = data[i], y = data[i + 1], z = data[i + 2];
        const angle = amplitude * Math.sin(frequency * y + phase);
        const cos   = Math.cos(angle);
        const sin   = Math.sin(angle);
        out[i]     = x * cos - z * sin;
        out[i + 1] = y;
        out[i + 2] = x * sin + z * cos;
    }

    return out;
}

/**
 * Compute the torsion curvature magnitude at each vertex (scalar field).
 *
 * @param data   - Packed vertex buffer
 * @param factor - Torsion factor (same as in `torsionField`)
 * @returns Float32Array of per-vertex curvature magnitudes
 */
export function torsionCurvature(data: Float32Array, factor: number): Float32Array {
    const n   = Math.floor(data.length / 3);
    const out = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        const y        = data[i * 3 + 1];
        // Curvature is proportional to the twist angle per unit arc length
        out[i] = Math.abs(factor * y);
    }

    return out;
}
