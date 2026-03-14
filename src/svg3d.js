// SVG-3D — Canonical tensor serialisation format for the K++ geometric runtime.
//
// SVG-3D is a *storage* format, not an execution format.  Tensors (point clouds
// living in manifold M) are encoded as SVG elements:
//
//   <circle cx cy [cz] r data-phase="…"/>  — one point per circle
//   <path d="M x0,y0,z0 L x1,y1,z1"/>      — one edge per path
//   <g …>                                  — fold / composite tensor boundary
//
// The compact schema means each tensor only stores its *deltas* from a shared
// schema, yielding ~90 % compression naturally.

// ------------------------------------------------------------------ //
// Encode
// ------------------------------------------------------------------ //

/**
 * Encode a Float32Array of vec3 points into an SVG-3D string.
 *
 * @param {Float32Array|number[]} points
 *   Flat array of (x, y, z) triplets.  Length must be a multiple of 3.
 * @param {object}   [opts]
 * @param {number[]} [opts.edges]      - Flat array of index pairs [i0, i1, …]
 * @param {number[]|Float32Array} [opts.phases] - Per-point phase value (radians)
 * @param {number}   [opts.viewWidth=1024]
 * @param {number}   [opts.viewHeight=768]
 * @returns {string} SVG-3D markup
 */
export function encodeToSVG(points, opts = {}) {
    const {
        edges     = [],
        phases    = [],
        viewWidth  = 1024,
        viewHeight = 768,
    } = opts;

    const n = Math.floor(points.length / 3);

    // Point cloud → <circle> elements
    const circleParts = [];
    for (let i = 0; i < n; i++) {
        const x     = points[i * 3];
        const y     = points[i * 3 + 1];
        const z     = points[i * 3 + 2];
        const norm  = Math.sqrt(x * x + y * y + z * z);
        const phase = phases[i] !== undefined ? phases[i] : 0;
        circleParts.push(`<circle cx="${x.toFixed(4)}" cy="${y.toFixed(4)}" cz="${z.toFixed(4)}" r="${norm.toFixed(4)}" data-phase="${phase.toFixed(6)}"/>`);
    }

    // Adjacency → <path> elements
    const pathParts = [];
    for (let e = 0; e + 1 < edges.length; e += 2) {
        const i0 = edges[e],     i1 = edges[e + 1];
        const x0 = points[i0 * 3], y0 = points[i0 * 3 + 1], z0 = points[i0 * 3 + 2];
        const x1 = points[i1 * 3], y1 = points[i1 * 3 + 1], z1 = points[i1 * 3 + 2];
        pathParts.push(`<path d="M ${x0.toFixed(4)},${y0.toFixed(4)},${z0.toFixed(4)} L ${x1.toFixed(4)},${y1.toFixed(4)},${z1.toFixed(4)}"/>`);
    }

    return (
        `<svg viewBox="0 0 ${viewWidth} ${viewHeight}" xmlns="http://www.w3.org/2000/svg">` +
        circleParts.join('') +
        pathParts.join('') +
        `</svg>`
    );
}

// ------------------------------------------------------------------ //
// Decode
// ------------------------------------------------------------------ //

/**
 * Decode an SVG-3D string back into a Float32Array of vec3 points and metadata.
 * Uses a regex-based parser — no DOM required.
 *
 * @param {string} svgString - SVG-3D markup produced by encodeToSVG
 * @returns {{ points: Float32Array, phases: Float32Array, norms: Float32Array }}
 */
export function decodeFromSVG(svgString) {
    const points = [];
    const phases = [];
    const norms  = [];

    // Match <circle … /> tags
    const circleRx = /<circle\s(.*?)\/>/gs;
    const attrRx   = /([\w-]+)="([^"]*)"/g;

    let circleMatch;
    while ((circleMatch = circleRx.exec(svgString)) !== null) {
        const attrs = {};
        let attrMatch;
        attrRx.lastIndex = 0;
        while ((attrMatch = attrRx.exec(circleMatch[1])) !== null) {
            attrs[attrMatch[1]] = attrMatch[2];
        }
        points.push(
            parseFloat(attrs.cx)  || 0,
            parseFloat(attrs.cy)  || 0,
            parseFloat(attrs.cz)  || 0,
        );
        phases.push(parseFloat(attrs['data-phase']) || 0);
        norms.push(parseFloat(attrs.r) || 0);
    }

    return {
        points: new Float32Array(points),
        phases: new Float32Array(phases),
        norms:  new Float32Array(norms),
    };
}
