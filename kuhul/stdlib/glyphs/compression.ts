// K'UHUL++ Standard Library — Rotational Compression Glyph (↻)
// Compresses geometry by rotating then merging spatially close vertices.
// TypeScript version of rotationalCompression() from src/kuhul.js.

/**
 * (↻) Rotational Compression — reduce vertex count by merging spatially
 * close vertices after rotating the mesh by `angleDeg` degrees about the Y axis.
 *
 * @param data      - Packed vertex buffer (stride: 3 floats per vertex)
 * @param angleDeg  - Rotation angle in degrees (used for compression alignment)
 * @param threshold - Distance threshold for merging (default: 0.01)
 * @returns New Float32Array with deduplicated vertices
 *
 * @example
 * const verts = new Float32Array([0, 0, 0,  0.001, 0, 0,  1, 0, 0]);
 * const compressed = rotationalCompression(verts, 45, 0.01);
 */
export function rotationalCompression(
    data:      Float32Array,
    angleDeg:  number,
    threshold = 0.01,
): Float32Array {
    const rad = (angleDeg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const n = Math.floor(data.length / 3);
    const rotated = new Float32Array(data.length);

    // Rotate all vertices about the Y axis
    for (let i = 0; i < n; i++) {
        const bi = i * 3;
        const x  = data[bi], y = data[bi + 1], z = data[bi + 2];
        rotated[bi]     = x * cos - z * sin;
        rotated[bi + 1] = y;
        rotated[bi + 2] = x * sin + z * cos;
    }

    // Merge vertices within threshold distance
    const keep: boolean[] = new Array(n).fill(true);
    const t2 = threshold * threshold;

    for (let i = 0; i < n; i++) {
        if (!keep[i]) continue;
        const ai = i * 3;
        for (let j = i + 1; j < n; j++) {
            if (!keep[j]) continue;
            const bj = j * 3;
            const dx = rotated[ai] - rotated[bj];
            const dy = rotated[ai + 1] - rotated[bj + 1];
            const dz = rotated[ai + 2] - rotated[bj + 2];
            if (dx * dx + dy * dy + dz * dz <= t2) keep[j] = false;
        }
    }

    const survivingIndices = keep.map((k, i) => k ? i : -1).filter(i => i >= 0);
    const out = new Float32Array(survivingIndices.length * 3);

    survivingIndices.forEach((origIdx, newIdx) => {
        out[newIdx * 3]     = rotated[origIdx * 3];
        out[newIdx * 3 + 1] = rotated[origIdx * 3 + 1];
        out[newIdx * 3 + 2] = rotated[origIdx * 3 + 2];
    });

    return out;
}

/**
 * Convenience variant — apply only the rotation without compression.
 *
 * @param data     - Packed vertex buffer
 * @param angleDeg - Rotation angle in degrees
 * @returns Rotated vertex buffer (same size as input)
 */
export function rotateY(data: Float32Array, angleDeg: number): Float32Array {
    const rad = (angleDeg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const out = new Float32Array(data.length);

    for (let i = 0; i + 2 < data.length; i += 3) {
        const x = data[i], z = data[i + 2];
        out[i]     = x * cos - z * sin;
        out[i + 1] = data[i + 1];
        out[i + 2] = x * sin + z * cos;
    }

    return out;
}
