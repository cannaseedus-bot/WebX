// Intel HD 4600 (Gen7.5 iGPU) optimization constants
//
// Architecture: 20 Execution Units (EUs), SIMD-16 width, 64KB SLM per subslice.
// Shared VRAM: up to ~1.7GB carved from system RAM (driver-controlled).
// Target: cs_5_0 DXBC shaders (NOT DXIL — incompatible with HD 4600).
//
// Thread dispatch rules derived from training runs on the 4600
// (pi_kuhul_train_hd4600.py, D3D11 iGPU confirmed at 100% EU utilization).

// ─── Execution Unit topology ──────────────────────────────────────────────────

export const HD4600 = Object.freeze({
  euCount:          20,    // Execution Units per GPU
  simdWidth:        16,    // SIMD lanes per EU (SIMD-16 dispatch)
  slmPerSubslice:   65536, // 64 KB Shared Local Memory per subslice (bytes)
  subsliceCount:    2,     // 2 subslices on HD 4600
  euPerSubslice:    10,

  // Recommended thread counts to avoid EU stalls on shared VRAM iGPU
  maxThreadsPerEU:  4,     // >4 causes scheduler thrash on shared-memory iGPU
  recommendedGroups:80,    // 20 EUs × 4 threads

  // SIMD bundle size for Mayan 5-digit vectors (pad to 8 for alignment)
  mayanSimdBundle:  4,     // 4 Mayan digits per SIMD unit (kin/uinal/tun/katun; baktun separate)
  mayanPadded:      8,     // align to 8 for SIMD-8 packing compatibility

  // VRAM budget (conservative; driver can allocate up to ~1.7 GB)
  vramBudgetBytes:  1610612736, // 1.5 GB conservative cap
  vramPageBytes:    4096,       // 4 KB VRAM page (matches CPU page for zero-copy)
});

// ─── Morton order (Z-curve) tiling config ─────────────────────────────────────
// Maps 2D tile coordinates to a Z-order index for cache-friendly VRAM access.
// Used by the INT4 GEMM shaders to lay out weight tiles.

export const MORTON_CONFIG = Object.freeze({
  tileW:  8,   // tile width in elements
  tileH:  8,   // tile height in elements
  levels: 3,   // bits per dimension for 3-level Z-curve (max 8×8 = 64-element tile)
});

// Interleave bits of x and y to produce a Morton (Z-order) index.
// Accepts 0-based tile-local coordinates (0..7).
export function mortonEncode(x, y) {
  let z = 0;
  for (let i = 0; i < 8; i++) {
    z |= ((x >> i & 1) << (2 * i)) | ((y >> i & 1) << (2 * i + 1));
  }
  return z;
}

export function mortonDecode(z) {
  let x = 0, y = 0;
  for (let i = 0; i < 8; i++) {
    x |= (z >> (2 * i) & 1) << i;
    y |= (z >> (2 * i + 1) & 1) << i;
  }
  return { x, y };
}

// Build a full 8×8 Morton order lookup table (index → {x,y})
export function buildMortonTable() {
  const table = new Array(64);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    const z = mortonEncode(x, y);
    table[z] = { x, y };
  }
  return Object.freeze(table);
}

export const MORTON_TABLE = buildMortonTable();

// ─── Dispatch sizing helpers ──────────────────────────────────────────────────

// Compute optimal dispatch dimensions for n elements on HD 4600.
// Returns { groupsX, groupsY, threadsPerGroup } for D3D11 Dispatch().
export function hd4600DispatchSize(n, threadsPerGroup = 64) {
  const groups = Math.ceil(n / threadsPerGroup);
  // Spread across both X and Y to avoid >65535 in one dimension (D3D11 limit)
  if (groups <= 65535) return { groupsX: groups, groupsY: 1, threadsPerGroup };
  const groupsY = Math.ceil(Math.sqrt(groups));
  const groupsX = Math.ceil(groups / groupsY);
  return { groupsX, groupsY, threadsPerGroup };
}

// SLM allocation per thread group: how many Float32 elements fit in 64 KB SLM
export function slmFloat32Capacity() {
  return HD4600.slmPerSubslice / 4; // 4 bytes per float32 → 16384 elements
}

// ─── Shader-visible constants (matches HLSL cbuffer layout) ──────────────────

export const HD4600_CBUFFER = Object.freeze({
  euCount:     HD4600.euCount,
  simdWidth:   HD4600.simdWidth,
  tileW:       MORTON_CONFIG.tileW,
  tileH:       MORTON_CONFIG.tileH,
  slmElements: slmFloat32Capacity(),
});
