// optical-mesh.js — SH-Wave-Lattice optical processor as a 3D polygon model
//
// JS port of native/optical_processor/geodesic.cpp + optical_processor.cpp.
// Produces the same output as ComputeGeoSphere() from Geometry.cpp:
//   vertices = [{position, normal, uv}]
//   indices  = uint16 triangle list
//
// Also runs the SH wave physics on the CPU so the mesh can be driven by
// the wave field (each vertex carries 9 spherical harmonic phase pairs).
//
// Connection to existing stack:
//   GeoSphereToTokens(pi_kuhul_bridge.h)  → feed vertices as Token array
//   SphericalManifold (geo-weights.js)     → geodesic distances from adjacency
//   EntropyField (entropic-weights.js)     → SH coherence maps to entropy
//   ArcLibrary (replayable-arcs.js)        → wave propagation paths as ARCs
//   KuhulPhysicsSolver (kuhul_physics.py)  → coherence feeds gravity solver
//
// SVG3D program format (tools/data/test_wave_program.svg3d):
//   inject → propagate → memory → read
//   Each instruction dispatches one wave VM opcode on the lattice.

const SH_BANDS = 9;  // l=0..2, 9 coefficients: Y00 Y10 Y11 Y1-1 Y20 Y21 Y2-1 Y22 Y2-2
const TWO_PI   = Math.PI * 2;

// ─── OpticalNode ──────────────────────────────────────────────────────────────

export class OpticalNode {
  constructor() {
    this.pos       = [0, 0, 0];              // 3D position on unit sphere
    this.sh        = new Float32Array(SH_BANDS * 2); // (cos,sin) per band
    this.neighbors = new Uint32Array(6);
    this.neighborCount = 0;
  }

  energy() {
    let e = 0;
    for (let i = 0; i < SH_BANDS; i++) {
      const c = this.sh[i*2], s = this.sh[i*2+1];
      e += Math.sqrt(c*c + s*s);
    }
    return e / SH_BANDS;
  }

  coherence(other) {
    let dot = 0, la = 0, lb = 0;
    for (let i = 0; i < SH_BANDS*2; i++) {
      dot += this.sh[i] * other.sh[i];
      la  += this.sh[i] ** 2;
      lb  += other.sh[i] ** 2;
    }
    return la > 0 && lb > 0 ? dot / Math.sqrt(la * lb) : 0;
  }
}

// ─── Icosphere generation ─────────────────────────────────────────────────────

function norm3(v) {
  const n = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) || 1;
  return [v[0]/n, v[1]/n, v[2]/n];
}

function midpoint(a, b, verts, cache) {
  const key = a < b ? `${a}_${b}` : `${b}_${a}`;
  if (cache.has(key)) return cache.get(key);
  const va = verts[a], vb = verts[b];
  const mid = norm3([(va[0]+vb[0])*.5, (va[1]+vb[1])*.5, (va[2]+vb[2])*.5]);
  const idx = verts.length;
  verts.push(mid);
  cache.set(key, idx);
  return idx;
}

export function generateIcosphere(nodes, subdivisions = 2) {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts = [
    [-1, t,0],[1, t,0],[-1,-t,0],[1,-t,0],
    [0,-1, t],[0, 1, t],[0,-1,-t],[0, 1,-t],
    [t, 0,-1],[t, 0, 1],[-t,0,-1],[-t,0, 1]
  ].map(v => norm3(v));

  let faces = [
    [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
    [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
    [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
    [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
  ];

  for (let s = 0; s < subdivisions; s++) {
    const cache = new Map();
    const newFaces = [];
    for (const [a,b,c] of faces) {
      const ab = midpoint(a,b,verts,cache);
      const bc = midpoint(b,c,verts,cache);
      const ca = midpoint(c,a,verts,cache);
      newFaces.push([a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]);
    }
    faces = newFaces;
  }

  // Build nodes with positions and adjacency
  nodes.length = 0;
  for (let i = 0; i < verts.length; i++) {
    const n = new OpticalNode();
    n.pos = verts[i];
    nodes.push(n);
  }

  for (const [a,b,c] of faces) {
    for (const [src,dst] of [[a,b],[b,c],[c,a],[b,a],[c,b],[a,c]]) {
      const node = nodes[src];
      if (node.neighborCount < 6) {
        let found = false;
        for (let k = 0; k < node.neighborCount; k++) {
          if (node.neighbors[k] === dst) { found = true; break; }
        }
        if (!found) node.neighbors[node.neighborCount++] = dst;
      }
    }
  }
}

// ─── ComputeOpticalMesh ───────────────────────────────────────────────────────
// Outputs a renderable polygon mesh from the node lattice.
// Same format as ComputeGeoSphere() — vertices with normal + UV, uint16 indices.

export function ComputeOpticalMesh(nodes, diameter = 2, rhcoords = true) {
  const radius   = diameter * 0.5;
  const vertices = [];
  const indices  = [];

  // One vertex per node
  for (const node of nodes) {
    const [x,y,z] = node.pos;
    const longitude = Math.atan2(x, -z);
    const latitude  = Math.acos(Math.max(-1, Math.min(1, y)));
    const u = 1 - (longitude / TWO_PI + 0.5);
    const v = latitude / Math.PI;
    vertices.push({
      position: [x*radius, y*radius, z*radius],
      normal:   [x, y, z],
      uv:       [u, v],
      // carry the wave field for downstream use
      sh:       node.sh,
      energy:   node.energy(),
    });
  }

  // Recover triangles from neighbour adjacency (a < b < c, c neighbour of both)
  const n = nodes.length;
  for (let a = 0; a < n; a++) {
    const na = nodes[a];
    for (let bi = 0; bi < na.neighborCount; bi++) {
      const b = na.neighbors[bi];
      if (b <= a) continue;
      const nb = nodes[b];
      for (let ci = 0; ci < nb.neighborCount; ci++) {
        const c = nb.neighbors[ci];
        if (c <= b) continue;
        // verify c is also a neighbour of a
        let ok = false;
        for (let k = 0; k < na.neighborCount; k++) {
          if (na.neighbors[k] === c) { ok = true; break; }
        }
        if (!ok) continue;
        if (rhcoords) { indices.push(a, b, c); }
        else          { indices.push(a, c, b); }
      }
    }
  }

  return { vertices, indices };
}

// ─── SH basis functions ───────────────────────────────────────────────────────
// JS port of projection.cpp evaluateSH()

function evalSH(idx, theta, phi) {
  const c = Math.cos(theta), s = Math.sin(theta);
  switch (idx) {
    case 0: return 0.282095;
    case 1: return 0.488603 * c;
    case 2: return 0.488603 * s * Math.cos(phi);
    case 3: return 0.488603 * s * Math.sin(phi);
    case 4: return 1.092548 * (c*c - 0.333333);
    case 5: return 2.185095 * s * c * Math.cos(phi);
    case 6: return 2.185095 * s * c * Math.sin(phi);
    case 7: return 1.092548 * s * s * Math.cos(2*phi);
    case 8: return 1.092548 * s * s * Math.sin(2*phi);
    default: return 0;
  }
}

// ─── OpticalProcessor ────────────────────────────────────────────────────────

export class OpticalProcessor {
  constructor() {
    this.nodes = [];
  }

  buildGeodesic(level = 2) {
    generateIcosphere(this.nodes, level);
    // Seed wave field from spherical coordinates of each node
    for (let i = 0; i < this.nodes.length; i++) {
      const [x,y,z] = this.nodes[i].pos;
      const theta = Math.acos(Math.max(-1,Math.min(1,y)));
      const phi   = Math.atan2(x,-z) + Math.PI;
      for (let j = 0; j < SH_BANDS; j++) {
        const Y     = evalSH(j, theta, phi);
        const phase = phi * (j+1) * 0.5;
        this.nodes[i].sh[j*2]   = Y * Math.cos(phase) * 0.1;
        this.nodes[i].sh[j*2+1] = Y * Math.sin(phase) * 0.1;
      }
    }
    return this;
  }

  // One wave propagation step (JS port of OP_PROPAGATE kernel)
  propagate() {
    const out = this.nodes.map(() => new Float32Array(SH_BANDS * 2));
    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      for (let band = 0; band < SH_BANDS; band++) {
        let rc = node.sh[band*2]   * 0.7;
        let rs = node.sh[band*2+1] * 0.7;
        // neighbour coupling
        for (let k = 0; k < node.neighborCount; k++) {
          const nb = this.nodes[node.neighbors[k]];
          rc += nb.sh[band*2]   * 0.1;
          rs += nb.sh[band*2+1] * 0.1;
        }
        // harmonic phase rotation
        const l     = Math.floor(Math.sqrt(band));
        const angle = l * 0.1;
        const c = Math.cos(angle), s = Math.sin(angle);
        const rc2 = rc*c - rs*s, rs2 = rc*s + rs*c;
        // cross-band coupling
        const nb2 = (band+1) % SH_BANDS;
        const xc  = node.sh[nb2*2] * 0.02, xs = node.sh[nb2*2+1] * 0.02;
        let mc = rc2+xc, ms = rs2+xs;
        // normalize + inject + decay
        const len = Math.sqrt(mc*mc+ms*ms) || 1e-5;
        mc = mc/len * Math.min(len,1) + 0.002;
        ms = ms/len * Math.min(len,1) + 0.001;
        out[i][band*2]   = mc * 0.99;
        out[i][band*2+1] = ms * 0.99;
      }
    }
    for (let i = 0; i < this.nodes.length; i++) this.nodes[i].sh = out[i];
    return this;
  }

  // Inject energy into one node
  inject(nodeIdx, band, amplitude, phase = 0) {
    if (nodeIdx >= this.nodes.length || band >= SH_BANDS) return this;
    this.nodes[nodeIdx].sh[band*2]   += amplitude * Math.cos(phase);
    this.nodes[nodeIdx].sh[band*2+1] += amplitude * Math.sin(phase);
    return this;
  }

  // Global coherence (like readCoherence() in C++)
  coherence() {
    return this.nodes.reduce((s, n) => s + n.energy(), 0) / this.nodes.length;
  }

  // Get renderable mesh
  mesh(diameter = 2, rhcoords = true) {
    return ComputeOpticalMesh(this.nodes, diameter, rhcoords);
  }

  // Run a simple SVG3D-style program
  runProgram(instructions) {
    for (const ins of instructions) {
      switch (ins.op) {
        case 'inject':    this.inject(ins.node ?? 0, ins.band ?? 0, ins.amplitude ?? 1, ins.phase ?? 0); break;
        case 'propagate': for (let s = 0; s < (ins.steps ?? 1); s++) this.propagate(); break;
        case 'halt':      return this;
      }
    }
    return this;
  }
}

// ─── Canonical SVG3D program (from tools/data/test_wave_program.svg3d) ───────

export const TEST_WAVE_PROGRAM = [
  { op:'inject',    node:0, band:1, amplitude:1.0, phase:0.0 },
  { op:'propagate', steps:5 },
  { op:'halt' },
];
