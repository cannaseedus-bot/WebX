// K'UHUL++ WebGPU Compute Shader Code Generator
// Emits WGSL (WebGPU Shading Language) compute shaders from Geometric IR.
// Each glyph operation maps to a WGSL compute kernel.

import type { GeometricIR, IRInstruction, GlyphOp } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// WGSL templates for each glyph op
// ------------------------------------------------------------------ //

const GLYPH_KERNELS: Record<string, string> = {
    '⊗': `
// ⊗ — Geometric product (matrix multiply, vec3 stride)
@group(0) @binding(0) var<storage, read>       left  : array<f32>;
@group(0) @binding(1) var<storage, read>       right : array<f32>;
@group(0) @binding(2) var<storage, read_write> out   : array<f32>;

@compute @workgroup_size(64)
fn kuhul_geo_product(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    let n   = arrayLength(&left);
    if (idx >= n) { return; }
    // Scalar multiply — extend for matrix dims as needed
    out[idx] = left[idx] * right[idx];
}`,

    '⊕': `
// ⊕ — Translation / bias addition
@group(0) @binding(0) var<storage, read>       left  : array<f32>;
@group(0) @binding(1) var<storage, read>       right : array<f32>;
@group(0) @binding(2) var<storage, read_write> out   : array<f32>;

@compute @workgroup_size(64)
fn kuhul_geo_translate(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&left)) { return; }
    out[idx] = left[idx] + right[idx];
}`,

    '⊖': `
// ⊖ — Difference / subtraction in M
@group(0) @binding(0) var<storage, read>       left  : array<f32>;
@group(0) @binding(1) var<storage, read>       right : array<f32>;
@group(0) @binding(2) var<storage, read_write> out   : array<f32>;

@compute @workgroup_size(64)
fn kuhul_geo_difference(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&left)) { return; }
    out[idx] = left[idx] - right[idx];
}`,

    '⊛': `
// ⊛ — Convolution in M
@group(0) @binding(0) var<storage, read>       signal : array<f32>;
@group(0) @binding(1) var<storage, read>       kernel : array<f32>;
@group(0) @binding(2) var<storage, read_write> out    : array<f32>;

@compute @workgroup_size(64)
fn kuhul_convolution(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    let n   = arrayLength(&signal);
    let k   = arrayLength(&kernel);
    if (idx >= n) { return; }
    var acc : f32 = 0.0;
    for (var j : u32 = 0u; j < k; j++) {
        let si = idx + j;
        if (si < n) { acc += signal[si] * kernel[j]; }
    }
    out[idx] = acc;
}`,

    '⤍': `
// ⤍ — Vector Encrypt (affine 4×4 transform)
@group(0) @binding(0) var<storage, read>       verts  : array<f32>;
@group(0) @binding(1) var<storage, read>       matrix : array<f32>;
@group(0) @binding(2) var<storage, read_write> out    : array<f32>;

@compute @workgroup_size(64)
fn kuhul_vector_encrypt(@builtin(global_invocation_id) gid : vec3<u32>) {
    let vi = gid.x * 3u;
    if (vi + 2u >= arrayLength(&verts)) { return; }
    let x = verts[vi]; let y = verts[vi+1u]; let z = verts[vi+2u];
    out[vi]    = matrix[0]*x + matrix[4]*y + matrix[8]*z  + matrix[12];
    out[vi+1u] = matrix[1]*x + matrix[5]*y + matrix[9]*z  + matrix[13];
    out[vi+2u] = matrix[2]*x + matrix[6]*y + matrix[10]*z + matrix[14];
}`,

    '↻': `
// ↻ — Rotational Compression (rotate then threshold)
@group(0) @binding(0) var<storage, read>       verts : array<f32>;
@group(0) @binding(1) var<storage, read_write> out   : array<f32>;
@group(0) @binding(2) var<uniform>             angle : f32;

@compute @workgroup_size(64)
fn kuhul_rot_compress(@builtin(global_invocation_id) gid : vec3<u32>) {
    let vi = gid.x * 3u;
    if (vi + 2u >= arrayLength(&verts)) { return; }
    let x = verts[vi]; let z = verts[vi+2u];
    out[vi]    = x * cos(angle) - z * sin(angle);
    out[vi+1u] = verts[vi+1u];
    out[vi+2u] = x * sin(angle) + z * cos(angle);
}`,

    '⊙': `
// ⊙ — Radial Projection
@group(0) @binding(0) var<storage, read>       points  : array<f32>;
@group(0) @binding(1) var<storage, read_write> out     : array<f32>;
@group(0) @binding(2) var<uniform>             radius  : f32;

@compute @workgroup_size(64)
fn kuhul_radial_project(@builtin(global_invocation_id) gid : vec3<u32>) {
    let vi = gid.x * 3u;
    if (vi + 2u >= arrayLength(&points)) { return; }
    let x = points[vi]; let y = points[vi+1u]; let z = points[vi+2u];
    let len = sqrt(x*x + y*y + z*z);
    let scale = select(1.0, radius / len, len > 0.0001);
    out[vi] = x * scale; out[vi+1u] = y * scale; out[vi+2u] = z * scale;
}`,
};

// ------------------------------------------------------------------ //
// WebGPUCodegen
// ------------------------------------------------------------------ //

/**
 * Generates WGSL compute shaders from a Geometric IR program.
 * Each glyph operation in the IR becomes a separate WGSL compute kernel.
 *
 * @example
 * const cg = new WebGPUCodegen();
 * const wgsl = cg.generate(ir);
 * // Use wgsl with GPUDevice.createShaderModule({ code: wgsl })
 */
export class WebGPUCodegen {
    /**
     * Generate a WGSL shader source string for the given IR.
     *
     * @param ir - Geometric IR from `generateIR()`
     * @returns WGSL source string
     */
    generate(ir: GeometricIR): string {
        const sections: string[] = [];

        sections.push(this.buildHeader(ir));

        // Emit kernels for every glyph op that appears in the IR
        const emittedGlyphs = new Set<string>();
        for (const instr of ir.instructions) {
            if (this.isGlyphOp(instr.op) && !emittedGlyphs.has(instr.op)) {
                const kernel = GLYPH_KERNELS[instr.op];
                if (kernel) {
                    sections.push(kernel.trim());
                    emittedGlyphs.add(instr.op);
                }
            }
        }

        // If no glyph ops present, emit an identity kernel
        if (emittedGlyphs.size === 0) {
            sections.push(this.buildIdentityKernel());
        }

        return sections.join('\n\n') + '\n';
    }

    // ---- Helpers ----

    private buildHeader(ir: GeometricIR): string {
        return [
            `// K'UHUL++ Generated WGSL Compute Shader`,
            `// Manifold: ${ir.manifold.dimensions}D, metric=${ir.manifold.metric}`,
            `// Phase: ${ir.manifold.phase.toFixed(4)} rad`,
            `// Generated by WebGPUCodegen`,
        ].join('\n');
    }

    private buildIdentityKernel(): string {
        return `
@group(0) @binding(0) var<storage, read>       inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn kuhul_identity(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&inp)) { return; }
    out[idx] = inp[idx];
}`.trim();
    }

    private isGlyphOp(op: string): op is GlyphOp {
        const glyphs: string[] = ['⊗','⊕','⊖','⊛','⊜','⊝','⊞','⤍','↻','⟲','∿','⊙','≋'];
        return glyphs.includes(op);
    }
}
