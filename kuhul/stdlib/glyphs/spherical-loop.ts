// K'UHUL++ Standard Library — Spherical Loop Glyph (⟲)
// Applies a Cartesian ↔ spherical coordinate round-trip transform,
// optionally adding a phase rotation to create looping patterns.

/**
 * (⟲) Spherical Loop — convert each vec3 to spherical coordinates,
 * apply the phase rotation, then convert back to Cartesian space.
 *
 * @param data       - Packed vertex buffer (stride: 3 floats per vertex)
 * @param iterations - Number of spherical loop passes (default: 1)
 * @param phaseDelta - Phase added per iteration in radians (default: 0)
 * @returns New Float32Array with the transformed vertices
 *
 * @example
 * const sphere = new Float32Array([1, 0, 0,  0, 1, 0,  0, 0, 1]);
 * const looped = sphericalLoop(sphere, 3, Math.PI / 6);
 */
export function sphericalLoop(
    data:       Float32Array,
    iterations  = 1,
    phaseDelta  = 0,
): Float32Array {
    let current = new Float32Array(data);

    for (let iter = 0; iter < iterations; iter++) {
        const phaseOffset = phaseDelta * (iter + 1);
        current = sphericalRoundTrip(current, phaseOffset);
    }

    return current;
}

/**
 * Single Cartesian → spherical → Cartesian pass with an optional phase rotation.
 *
 * @param data  - Packed vertex buffer
 * @param phase - Phase angle in radians added to theta during the transform
 */
function sphericalRoundTrip(data: Float32Array, phase: number): Float32Array {
    const out = new Float32Array(data.length);

    for (let i = 0; i + 2 < data.length; i += 3) {
        const x = data[i], y = data[i + 1], z = data[i + 2];

        const r     = Math.sqrt(x * x + y * y + z * z);
        if (r < 1e-9) {
            out[i] = 0; out[i + 1] = 0; out[i + 2] = 0;
            continue;
        }

        const theta = Math.acos(Math.max(-1, Math.min(1, z / r)));
        const phi   = Math.atan2(y, x);

        // Apply phase offset to polar angle
        const newTheta = theta + phase;

        out[i]     = r * Math.sin(newTheta) * Math.cos(phi);
        out[i + 1] = r * Math.sin(newTheta) * Math.sin(phi);
        out[i + 2] = r * Math.cos(newTheta);
    }

    return out;
}

/**
 * Convert packed Cartesian vec3 buffer to spherical (r, θ, φ).
 *
 * @param data - Cartesian vertex buffer
 * @returns Spherical buffer where each triplet is (r, theta, phi)
 */
export function cartesianToSpherical(data: Float32Array): Float32Array {
    const out = new Float32Array(data.length);
    for (let i = 0; i + 2 < data.length; i += 3) {
        const x = data[i], y = data[i + 1], z = data[i + 2];
        const r = Math.sqrt(x * x + y * y + z * z);
        out[i]     = r;
        out[i + 1] = r > 1e-9 ? Math.acos(Math.max(-1, Math.min(1, z / r))) : 0;
        out[i + 2] = Math.atan2(y, x);
    }
    return out;
}

/**
 * Convert packed spherical (r, θ, φ) buffer back to Cartesian.
 *
 * @param data - Spherical buffer (r, theta, phi) triplets
 * @returns Cartesian vertex buffer
 */
export function sphericalToCartesian(data: Float32Array): Float32Array {
    const out = new Float32Array(data.length);
    for (let i = 0; i + 2 < data.length; i += 3) {
        const r = data[i], theta = data[i + 1], phi = data[i + 2];
        out[i]     = r * Math.sin(theta) * Math.cos(phi);
        out[i + 1] = r * Math.sin(theta) * Math.sin(phi);
        out[i + 2] = r * Math.cos(theta);
    }
    return out;
}
