// K'UHUL++ Standard Library — Linear Algebra
// Core mat4 and vec3 operations used by glyph implementations and shaders.

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

/** Column-major 4×4 matrix stored as a 16-element Float32Array */
export type Mat4 = Float32Array;

/** 3-element vector stored as a 3-element Float32Array */
export type Vec3 = Float32Array;

// ------------------------------------------------------------------ //
// Matrix constructors
// ------------------------------------------------------------------ //

/** Create a 4×4 identity matrix (column-major) */
export function mat4Identity(): Mat4 {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
}

/** Create a translation matrix */
export function mat4Translate(tx: number, ty: number, tz: number): Mat4 {
    const m = mat4Identity();
    m[12] = tx; m[13] = ty; m[14] = tz;
    return m;
}

/** Create a uniform scale matrix */
export function mat4ScaleUniform(s: number): Mat4 {
    const m = mat4Identity();
    m[0] = s; m[5] = s; m[10] = s;
    return m;
}

/** Create a non-uniform scale matrix */
export function mat4Scale(sx: number, sy: number, sz: number): Mat4 {
    const m = mat4Identity();
    m[0] = sx; m[5] = sy; m[10] = sz;
    return m;
}

/** Create a rotation matrix about the X axis */
export function mat4RotateX(rad: number): Mat4 {
    const c = Math.cos(rad), s = Math.sin(rad);
    return new Float32Array([
        1,  0,  0, 0,
        0,  c,  s, 0,
        0, -s,  c, 0,
        0,  0,  0, 1,
    ]);
}

/** Create a rotation matrix about the Y axis */
export function mat4RotateY(rad: number): Mat4 {
    const c = Math.cos(rad), s = Math.sin(rad);
    return new Float32Array([
         c, 0, -s, 0,
         0, 1,  0, 0,
         s, 0,  c, 0,
         0, 0,  0, 1,
    ]);
}

/** Create a rotation matrix about the Z axis */
export function mat4RotateZ(rad: number): Mat4 {
    const c = Math.cos(rad), s = Math.sin(rad);
    return new Float32Array([
        c,  s, 0, 0,
       -s,  c, 0, 0,
        0,  0, 1, 0,
        0,  0, 0, 1,
    ]);
}

// ------------------------------------------------------------------ //
// Matrix operations
// ------------------------------------------------------------------ //

/**
 * Multiply two 4×4 column-major matrices: out = a × b
 */
export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    return out;
}

/**
 * Transpose a 4×4 matrix.
 */
export function mat4Transpose(m: Mat4): Mat4 {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            out[j * 4 + i] = m[i * 4 + j];
        }
    }
    return out;
}

/**
 * Compute the inverse of a 4×4 matrix using the adjugate method.
 * Returns the identity matrix if the input is singular.
 */
export function mat4Inverse(m: Mat4): Mat4 {
    const inv = new Float32Array(16);
    const [
        m0,  m1,  m2,  m3,
        m4,  m5,  m6,  m7,
        m8,  m9,  m10, m11,
        m12, m13, m14, m15,
    ] = m;

    inv[0]  =  m5*m10*m15 - m5*m11*m14 - m9*m6*m15 + m9*m7*m14 + m13*m6*m11 - m13*m7*m10;
    inv[4]  = -m4*m10*m15 + m4*m11*m14 + m8*m6*m15 - m8*m7*m14 - m12*m6*m11 + m12*m7*m10;
    inv[8]  =  m4*m9*m15  - m4*m11*m13 - m8*m5*m15 + m8*m7*m13 + m12*m5*m11 - m12*m7*m9;
    inv[12] = -m4*m9*m14  + m4*m10*m13 + m8*m5*m14 - m8*m6*m13 - m12*m5*m10 + m12*m6*m9;

    inv[1]  = -m1*m10*m15 + m1*m11*m14 + m9*m2*m15 - m9*m3*m14 - m13*m2*m11 + m13*m3*m10;
    inv[5]  =  m0*m10*m15 - m0*m11*m14 - m8*m2*m15 + m8*m3*m14 + m12*m2*m11 - m12*m3*m10;
    inv[9]  = -m0*m9*m15  + m0*m11*m13 + m8*m1*m15 - m8*m3*m13 - m12*m1*m11 + m12*m3*m9;
    inv[13] =  m0*m9*m14  - m0*m10*m13 - m8*m1*m14 + m8*m2*m13 + m12*m1*m10 - m12*m2*m9;

    inv[2]  =  m1*m6*m15  - m1*m7*m14  - m5*m2*m15 + m5*m3*m14 + m13*m2*m7  - m13*m3*m6;
    inv[6]  = -m0*m6*m15  + m0*m7*m14  + m4*m2*m15 - m4*m3*m14 - m12*m2*m7  + m12*m3*m6;
    inv[10] =  m0*m5*m15  - m0*m7*m13  - m4*m1*m15 + m4*m3*m13 + m12*m1*m7  - m12*m3*m5;
    inv[14] = -m0*m5*m14  + m0*m6*m13  + m4*m1*m14 - m4*m2*m13 - m12*m1*m6  + m12*m2*m5;

    inv[3]  = -m1*m6*m11  + m1*m7*m10  + m5*m2*m11 - m5*m3*m10 - m9*m2*m7   + m9*m3*m6;
    inv[7]  =  m0*m6*m11  - m0*m7*m10  - m4*m2*m11 + m4*m3*m10 + m8*m2*m7   - m8*m3*m6;
    inv[11] = -m0*m5*m11  + m0*m7*m9   + m4*m1*m11 - m4*m3*m9  - m8*m1*m7   + m8*m3*m5;
    inv[15] =  m0*m5*m10  - m0*m6*m9   - m4*m1*m10 + m4*m2*m9  + m8*m1*m6   - m8*m2*m5;

    const det = m0 * inv[0] + m1 * inv[4] + m2 * inv[8] + m3 * inv[12];
    if (Math.abs(det) < 1e-12) return mat4Identity();

    const invDet = 1.0 / det;
    for (let i = 0; i < 16; i++) inv[i] *= invDet;
    return inv;
}

// ------------------------------------------------------------------ //
// Vec3 operations
// ------------------------------------------------------------------ //

/** Create a vec3 */
export function vec3(x: number, y: number, z: number): Vec3 {
    return new Float32Array([x, y, z]);
}

/** Add two vec3s */
export function vec3Add(a: Vec3, b: Vec3): Vec3 {
    return new Float32Array([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
}

/** Subtract two vec3s */
export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
    return new Float32Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
}

/** Scale a vec3 by a scalar */
export function vec3Scale(v: Vec3, s: number): Vec3 {
    return new Float32Array([v[0] * s, v[1] * s, v[2] * s]);
}

/** Dot product of two vec3s */
export function vec3Dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Cross product of two vec3s */
export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
    return new Float32Array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]);
}

/** Length of a vec3 */
export function vec3Length(v: Vec3): number {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/** Normalize a vec3 (returns zero-vector if length < epsilon) */
export function vec3Normalize(v: Vec3): Vec3 {
    const len = vec3Length(v);
    if (len < 1e-9) return new Float32Array(3);
    return vec3Scale(v, 1 / len);
}

/** Transform a vec3 by a 4×4 matrix (w=1, no perspective divide) */
export function vec3TransformMat4(v: Vec3, m: Mat4): Vec3 {
    const x = v[0], y = v[1], z = v[2];
    return new Float32Array([
        m[0] * x + m[4] * y + m[8]  * z + m[12],
        m[1] * x + m[5] * y + m[9]  * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    ]);
}
