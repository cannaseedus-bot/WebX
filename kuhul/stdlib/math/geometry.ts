// K'UHUL++ Standard Library — Geometry Utilities
// Distance, cross product, spherical coordinate conversion, AABB, and more.

// ------------------------------------------------------------------ //
// Distance and angles
// ------------------------------------------------------------------ //

/**
 * Euclidean distance between two 3-D points.
 *
 * @param a - First point as [x, y, z]
 * @param b - Second point as [x, y, z]
 */
export function distance3D(
    a: [number, number, number],
    b: [number, number, number],
): number {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Squared Euclidean distance (avoids the sqrt — useful for comparisons).
 */
export function distanceSq3D(
    a: [number, number, number],
    b: [number, number, number],
): number {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

/**
 * Angle (in radians) between two 3-D vectors from the origin.
 */
export function angleBetween(
    a: [number, number, number],
    b: [number, number, number],
): number {
    const dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    const lenA = Math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    const lenB = Math.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
    if (lenA < 1e-9 || lenB < 1e-9) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / (lenA * lenB))));
}

// ------------------------------------------------------------------ //
// Cross product
// ------------------------------------------------------------------ //

/**
 * 3-D cross product of vectors a and b.
 */
export function crossProduct(
    a: [number, number, number],
    b: [number, number, number],
): [number, number, number] {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

// ------------------------------------------------------------------ //
// Spherical coordinate conversions
// ------------------------------------------------------------------ //

/**
 * Convert Cartesian (x, y, z) to spherical (r, θ, φ).
 * θ is the polar (inclination) angle from +Z, φ is the azimuthal angle from +X.
 *
 * @returns [r, theta, phi] — radius, polar angle (0..π), azimuthal angle (-π..π)
 */
export function cartesianToSpherical(
    x: number, y: number, z: number,
): [number, number, number] {
    const r     = Math.sqrt(x * x + y * y + z * z);
    const theta = r > 1e-9 ? Math.acos(Math.max(-1, Math.min(1, z / r))) : 0;
    const phi   = Math.atan2(y, x);
    return [r, theta, phi];
}

/**
 * Convert spherical (r, θ, φ) to Cartesian (x, y, z).
 */
export function sphericalToCartesian(
    r: number, theta: number, phi: number,
): [number, number, number] {
    return [
        r * Math.sin(theta) * Math.cos(phi),
        r * Math.sin(theta) * Math.sin(phi),
        r * Math.cos(theta),
    ];
}

// ------------------------------------------------------------------ //
// Axis-Aligned Bounding Box (AABB)
// ------------------------------------------------------------------ //

export interface AABB {
    min: [number, number, number];
    max: [number, number, number];
}

/**
 * Compute the AABB of a packed vec3 buffer.
 *
 * @param data - Packed vertex buffer (stride 3)
 */
export function computeAABB(data: Float32Array): AABB {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i + 2 < data.length; i += 3) {
        if (data[i]     < minX) minX = data[i];
        if (data[i + 1] < minY) minY = data[i + 1];
        if (data[i + 2] < minZ) minZ = data[i + 2];
        if (data[i]     > maxX) maxX = data[i];
        if (data[i + 1] > maxY) maxY = data[i + 1];
        if (data[i + 2] > maxZ) maxZ = data[i + 2];
    }

    return {
        min: [minX, minY, minZ],
        max: [maxX, maxY, maxZ],
    };
}

/**
 * Centre and extent of an AABB.
 */
export function aabbCenterExtent(aabb: AABB): {
    center: [number, number, number];
    extent: [number, number, number];
} {
    return {
        center: [
            (aabb.min[0] + aabb.max[0]) / 2,
            (aabb.min[1] + aabb.max[1]) / 2,
            (aabb.min[2] + aabb.max[2]) / 2,
        ],
        extent: [
            (aabb.max[0] - aabb.min[0]) / 2,
            (aabb.max[1] - aabb.min[1]) / 2,
            (aabb.max[2] - aabb.min[2]) / 2,
        ],
    };
}

// ------------------------------------------------------------------ //
// Surface normal estimation
// ------------------------------------------------------------------ //

/**
 * Compute the face normal of a triangle defined by three vertices.
 */
export function triangleNormal(
    a: [number, number, number],
    b: [number, number, number],
    c: [number, number, number],
): [number, number, number] {
    const ab: [number, number, number] = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    const ac: [number, number, number] = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    const n = crossProduct(ab, ac);
    const len = Math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (len < 1e-9) return [0, 1, 0];
    return [n[0]/len, n[1]/len, n[2]/len];
}
