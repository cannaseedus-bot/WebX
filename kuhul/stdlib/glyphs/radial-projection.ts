// K'UHUL++ Standard Library — Radial Projection Glyph (⊙)
// Projects vertices onto a sphere of given radius centred at an origin point.

/**
 * (⊙) Radial Projection — project each vertex in `data` onto a sphere of
 * `radius` centred at `center`.
 *
 * @param data   - Packed vertex buffer (stride: 3 floats per vertex)
 * @param center - Centre point of the projection sphere [cx, cy, cz]
 * @param radius - Target sphere radius
 * @returns New Float32Array with each vertex moved to the sphere surface
 *
 * @example
 * const cloud = randomPoints(1000);
 * const sphere = radialProjection(cloud, [0, 0, 0], 1.0);
 */
export function radialProjection(
    data:   Float32Array,
    center: [number, number, number],
    radius: number,
): Float32Array {
    const out = new Float32Array(data.length);
    const [cx, cy, cz] = center;

    for (let i = 0; i + 2 < data.length; i += 3) {
        const dx = data[i]     - cx;
        const dy = data[i + 1] - cy;
        const dz = data[i + 2] - cz;

        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-9) {
            out[i] = cx + radius; out[i + 1] = cy; out[i + 2] = cz;
            continue;
        }

        const scale = radius / len;
        out[i]     = cx + dx * scale;
        out[i + 1] = cy + dy * scale;
        out[i + 2] = cz + dz * scale;
    }

    return out;
}

/**
 * Compute the signed distance of each vertex from a sphere surface.
 * Positive = outside the sphere, negative = inside.
 *
 * @param data   - Packed vertex buffer
 * @param center - Sphere centre
 * @param radius - Sphere radius
 * @returns Float32Array of per-vertex signed distances
 */
export function signedDistanceToSphere(
    data:   Float32Array,
    center: [number, number, number],
    radius: number,
): Float32Array {
    const n   = Math.floor(data.length / 3);
    const out = new Float32Array(n);
    const [cx, cy, cz] = center;

    for (let i = 0; i < n; i++) {
        const bi = i * 3;
        const dx = data[bi]     - cx;
        const dy = data[bi + 1] - cy;
        const dz = data[bi + 2] - cz;
        out[i] = Math.sqrt(dx * dx + dy * dy + dz * dz) - radius;
    }

    return out;
}

/**
 * Radial basis function (RBF) weighting — compute the influence weight
 * of each vertex relative to a centre point using a Gaussian falloff.
 *
 * @param data   - Packed vertex buffer
 * @param center - RBF centre
 * @param sigma  - Gaussian standard deviation (controls falloff width)
 * @returns Float32Array of per-vertex RBF weights in [0, 1]
 */
export function rbfWeights(
    data:   Float32Array,
    center: [number, number, number],
    sigma:  number,
): Float32Array {
    const n   = Math.floor(data.length / 3);
    const out = new Float32Array(n);
    const [cx, cy, cz] = center;
    const twoSigSq = 2 * sigma * sigma;

    for (let i = 0; i < n; i++) {
        const bi = i * 3;
        const dx = data[bi]     - cx;
        const dy = data[bi + 1] - cy;
        const dz = data[bi + 2] - cz;
        const d2 = dx * dx + dy * dy + dz * dz;
        out[i] = Math.exp(-d2 / twoSigSq);
    }

    return out;
}
