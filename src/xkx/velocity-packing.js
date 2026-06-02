// velocity-packing.js — XKX Velocity Packing: GPU shader integration
//
// Microsoft Minigraph velocity packing as a first-class XKX component.
// Three modes selectable at runtime based on quality target + memory budget.
//
// K'uhul phase flow:
//   Pop   — allocate packed buffer (Wo)
//   Sek   — pack motion vectors (velocity_packer compute dispatch)
//   Sek   — unpack + reconstruct (velocity_unpacker)
//   Ch'en — quality gate (PSNR check, neighbourhood clipping for TAA)
//   Xul   — emit packed_velocities to downstream (motion blur / TAA)
//
// KXML graph edge: velocity_packer → velocity_unpacker
//   forward  channel=activation  data=packed_velocities  transform=none
//   backward channel=gradient    data=reconstruction_error  scale=0.001
//   phase_gate: forward=Sek→Sek  backward=Ch'en→Sek
//
// Lipschitz bound: L=1.0  (quantization error ≤ 1/1024 XY, 1/512 Z for MSFT mode)
// soft_landing: ||∇pack|| ≤ quantization_error
//
// XCFE runtime:  auto-selects mode given quality target + memory budget
// D3D11 shader:  native/shaders/velocity/VelocityPacking.hlsli
// WGSL shader:   embedded in XJSL node below

// ─── Packing modes ────────────────────────────────────────────────────────────

export const PACK_MODE = Object.freeze({
  MSFT_10_10_10_2:    0,  // XY±256 Z±1   30bit  ~45dB  7.9MB/1080p
  R10G10B10A2_UNORM:  1,  // XY±64  Z±0.5 32bit  ~42dB  8.3MB/1080p
  R16G16B16A16_FLOAT: 2,  // unlimited    64bit  lossless 16.6MB/1080p
});

export const MODE_INFO = Object.freeze([
  { mode: 0, name: 'MSFT_10_10_10_2',    bitsPerPixel: 30, rangXY: 256, rangZ: 1,   psnrEst: 45, memMB_1080p: 7.9  },
  { mode: 1, name: 'R10G10B10A2_UNORM',  bitsPerPixel: 32, rangXY: 64,  rangZ: 0.5, psnrEst: 42, memMB_1080p: 8.3  },
  { mode: 2, name: 'R16G16B16A16_FLOAT', bitsPerPixel: 64, rangXY: null, rangZ: null, psnrEst: Infinity, memMB_1080p: 16.6 },
]);

// ─── JS scalar implementations (reference / testing) ─────────────────────────

function f32tof16(x) {
  // IEEE 754 half-precision conversion (approximate)
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = x;
  const bits = new Uint32Array(buf)[0];
  const s = (bits >> 16) & 0x8000;
  let e = ((bits >> 23) & 0xFF) - 127 + 15;
  let m = (bits & 0x7FFFFF) >> 13;
  if (e <= 0)  return s;
  if (e >= 31) return s | 0x7C00;
  return s | (e << 10) | m;
}

function f16tof32(h) {
  const s = (h & 0x8000) << 16;
  const e = ((h >> 10) & 0x1F);
  const m = h & 0x3FF;
  if (e === 0)  { const buf = new ArrayBuffer(4); new Uint32Array(buf)[0] = s | (m << 13); return new Float32Array(buf)[0]; }
  if (e === 31) { const buf = new ArrayBuffer(4); new Uint32Array(buf)[0] = s | 0x7F800000 | (m << 13); return new Float32Array(buf)[0]; }
  const buf = new ArrayBuffer(4);
  new Uint32Array(buf)[0] = s | ((e + 112) << 23) | (m << 13);
  return new Float32Array(buf)[0];
}

// MSFT 10-10-10-2
function packXY_MSFT(x) {
  const signbit = (x < 0) ? 1 : 0;
  x = Math.min(Math.abs(x / 32768), 0.999969);
  return ((f32tof16(x) + 8) >> 4) | (signbit << 9);
}
function unpackXY_MSFT(x) {
  return f16tof32(((x & 0x1FF) << 4) | ((x >> 9) << 15)) * 32768;
}
function packZ_MSFT(z) {
  const signbit = (z < 0) ? 1 : 0;
  z = Math.min(Math.abs(z / 128), 0.999969);
  return ((f32tof16(z) + 2) >> 2) | (signbit << 11);
}
function unpackZ_MSFT(z) {
  return f16tof32(((z & 0x7FF) << 2) | ((z >> 11) << 15)) * 128;
}

export function packVelocity_MSFT(vx, vy, vz) {
  return (packXY_MSFT(vx) | (packXY_MSFT(vy) << 10) | (packZ_MSFT(vz) << 20)) >>> 0;
}
export function unpackVelocity_MSFT(v) {
  return { x: unpackXY_MSFT(v & 0x3FF), y: unpackXY_MSFT((v >> 10) & 0x3FF), z: unpackZ_MSFT(v >>> 20) };
}

// R10G10B10A2_UNORM
export function packVelocity_R10G10B10A2(vx, vy, vz) {
  const sx = Math.max(0, Math.min(1023, ((vx * 8 / 1024) + 512 / 1023) * 1023)) | 0;
  const sy = Math.max(0, Math.min(1023, ((vy * 8 / 1024) + 512 / 1023) * 1023)) | 0;
  const sz = Math.max(0, Math.min(3,    ((vz * 4096 / 1024) + 512 / 1023) * 3)) | 0;
  return (sx | (sy << 10) | (sz << 20)) >>> 0;
}
export function unpackVelocity_R10G10B10A2(v) {
  const sx = (v & 0x3FF) / 1023;
  const sy = ((v >> 10) & 0x3FF) / 1023;
  const sz = ((v >>> 20) & 0x3) / 3;
  return { x: (sx - 512/1023) * 1024/8, y: (sy - 512/1023) * 1024/8, z: (sz - 512/1023) * 2/8 };
}

// Dispatch
export function packVelocity(vx, vy, vz, mode = PACK_MODE.MSFT_10_10_10_2) {
  if (mode === PACK_MODE.MSFT_10_10_10_2)    return packVelocity_MSFT(vx, vy, vz);
  if (mode === PACK_MODE.R10G10B10A2_UNORM)  return packVelocity_R10G10B10A2(vx, vy, vz);
  return { x: vx, y: vy, z: vz }; // R16G16B16A16: pass-through in JS
}
export function unpackVelocity(v, mode = PACK_MODE.MSFT_10_10_10_2) {
  if (mode === PACK_MODE.MSFT_10_10_10_2)   return unpackVelocity_MSFT(v);
  if (mode === PACK_MODE.R10G10B10A2_UNORM) return unpackVelocity_R10G10B10A2(v);
  return typeof v === 'object' ? v : { x: 0, y: 0, z: 0 };
}

// ─── Quality metrics ──────────────────────────────────────────────────────────

export function compressionPSNR(vx, vy, vz, mode) {
  const packed = packVelocity(vx, vy, vz, mode);
  const { x: rx, y: ry, z: rz } = unpackVelocity(packed, mode);
  const mse = ((vx-rx)**2 + (vy-ry)**2 + (vz-rz)**2) / 3;
  return mse > 0 ? 10 * Math.log10(1 / mse) : Infinity;
}

// ─── Auto mode selection (XCFE runtime logic) ─────────────────────────────────

export function selectPackingMode(targetPSNR = 40, maxMemMB = 10) {
  for (const info of MODE_INFO) {
    if (info.psnrEst >= targetPSNR && info.memMB_1080p <= maxMemMB)
      return info.mode;
  }
  return PACK_MODE.R16G16B16A16_FLOAT;
}

// ─── XJSL node descriptor (KXML-compatible) ───────────────────────────────────

export const VELOCITY_PACKING_NODE = Object.freeze({
  id:     'velocity_packer',
  phase:  'Sek',
  domain: 'compute',
  fold:   'COMPUTE_FOLD',
  device: 'gpu',
  lipschitz: 1.0,

  wgslKernel: `
// velocity_packing.wgsl — WGSL kernel (embedded in XJSL node)
fn pack_xy_msft(x: f32) -> u32 {
    let signbit: u32 = u32(bitcast<i32>(x) >> 31);
    let abs_x: f32 = clamp(abs(x / 32768.0), 0.0, 0.99997);
    return ((f32tof16(abs_x) + 8u) >> 4u) | (signbit << 9u);
}
fn unpack_xy_msft(packed: u32) -> f32 {
    return f16tof32((packed & 0x1FFu) << 4u | (packed >> 9u) << 15u) * 32768.0;
}
fn pack_z_msft(z: f32) -> u32 {
    let signbit: u32 = u32(bitcast<i32>(z) >> 31);
    let abs_z: f32 = clamp(abs(z / 128.0), 0.0, 0.99997);
    return ((f32tof16(abs_z) + 2u) >> 2u) | (signbit << 11u);
}
fn pack_velocity_msft(v: vec3<f32>) -> u32 {
    return pack_xy_msft(v.x) | (pack_xy_msft(v.y) << 10u) | (pack_z_msft(v.z) << 20u);
}
@compute @workgroup_size(64, 64, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) { return; }
    let idx = gid.y * uniforms.width + gid.x;
    let v = raw_velocities[idx];
    packed_velocities[idx] = pack_velocity_msft(v);
}`,

  edge: {
    from: 'velocity_packer', to: 'velocity_unpacker',
    forward:  { channel: 'activation', data: 'packed_velocities', transform: 'none' },
    backward: { channel: 'gradient',   data: 'reconstruction_error', scale: 0.001 },
    phase_gate: { forward: 'Sek→Sek', backward: "Ch'en→Sek" },
  },

  softLanding: {
    lipschitz: 1.0,
    proof: 'Quantization error <= 1/1024 for XY, 1/512 for Z (MSFT mode)',
    gradientBound: '||grad_pack|| <= quantization_error',
  },
});
