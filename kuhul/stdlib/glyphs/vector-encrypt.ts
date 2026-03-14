// K'UHUL++ Standard Library — Vector Encrypt Glyph (⤍)
// Applies a 4×4 affine matrix transform to a packed vec3 buffer.
// This is the TypeScript version of the vectorEncrypt() function in src/kuhul.js.

/**
 * (⤍) Vector Encrypt — apply a 4×4 affine matrix to each vec3 in the buffer.
 * Vertices are stored with a stride of 3 floats (x, y, z).
 * The homogeneous w-component is implicitly 1 (no perspective divide).
 *
 * @param data   - Packed vertex buffer (length must be a multiple of 3)
 * @param matrix - 16-element column-major 4×4 affine matrix
 * @returns New Float32Array containing the transformed vertices
 *
 * @example
 * const verts  = new Float32Array([1, 0, 0]);
 * const matrix = mat4Identity();
 * const result = vectorEncrypt(verts, matrix);
 */
export function vectorEncrypt(data: Float32Array, matrix: Float32Array): Float32Array {
    if (matrix.length !== 16) {
        throw new RangeError(`vectorEncrypt: matrix must have 16 elements, got ${matrix.length}`);
    }

    const out = new Float32Array(data.length);
    const m   = matrix;

    for (let i = 0; i + 2 < data.length; i += 3) {
        const x = data[i], y = data[i + 1], z = data[i + 2];
        out[i]     = m[0] * x + m[4] * y + m[8]  * z + m[12];
        out[i + 1] = m[1] * x + m[5] * y + m[9]  * z + m[13];
        out[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }

    return out;
}

/**
 * Construct a 4×4 identity matrix (column-major).
 */
export function mat4Identity(): Float32Array {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
}

/**
 * Construct a column-major 4×4 translation matrix.
 *
 * @param tx - Translation along X
 * @param ty - Translation along Y
 * @param tz - Translation along Z
 */
export function mat4Translation(tx: number, ty: number, tz: number): Float32Array {
    const m = mat4Identity();
    m[12] = tx; m[13] = ty; m[14] = tz;
    return m;
}

/**
 * Construct a column-major 4×4 uniform scale matrix.
 *
 * @param s - Scale factor
 */
export function mat4Scale(s: number): Float32Array {
    const m = mat4Identity();
    m[0] = s; m[5] = s; m[10] = s;
    return m;
}
