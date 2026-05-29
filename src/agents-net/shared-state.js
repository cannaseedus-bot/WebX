// Agents.NET shared-memory state format (Agents.NET.v1.0.0)
//
// Mirrors SharedStateHeader from dotnet-workers/Workers/SharedMemoryStateReader.cs
// Named mapping: Local\KuhulGeometricState  (Windows MMF, not available in browser)
// This module provides the binary layout constants and a pure in-memory state object.
//
// Wire layout (LayoutKind.Sequential, Pack=1):
//   uint32 Version        offset 0
//   uint32 ActiveFold     offset 4
//   uint32 TickCount      offset 8
//   float32 Entropy       offset 12
//   float32 Attention     offset 16
//   float32 Pressure      offset 20
//   float32[10] Reserve   offset 24  (40 bytes)
// Total: 64 bytes

export const SHARED_STATE_VERSION = 1;
export const SHARED_STATE_BYTES   = 64;
export const SHARED_STATE_MMF_NAME = 'Local\\KuhulGeometricState';

export const SHARED_STATE_OFFSETS = Object.freeze({
  version:     0,
  activeFold:  4,
  tickCount:   8,
  entropy:     12,
  attention:   16,
  pressure:    20,
  reserve:     24,
});

export function readSharedState(buffer) {
  const view = new DataView(buffer instanceof ArrayBuffer ? buffer : buffer.buffer);
  const reserve = [];
  for (let i = 0; i < 10; i++) reserve.push(view.getFloat32(24 + i * 4, true));
  return {
    version:    view.getUint32(0,  true),
    activeFold: view.getUint32(4,  true),
    tickCount:  view.getUint32(8,  true),
    entropy:    view.getFloat32(12, true),
    attention:  view.getFloat32(16, true),
    pressure:   view.getFloat32(20, true),
    reserve,
  };
}

export function writeSharedState(state) {
  const buf  = new ArrayBuffer(SHARED_STATE_BYTES);
  const view = new DataView(buf);
  view.setUint32(0,  state.version    || SHARED_STATE_VERSION, true);
  view.setUint32(4,  state.activeFold || 0, true);
  view.setUint32(8,  state.tickCount  || 0, true);
  view.setFloat32(12, state.entropy   || 0, true);
  view.setFloat32(16, state.attention || 0, true);
  view.setFloat32(20, state.pressure  || 0, true);
  const reserve = state.reserve || [];
  for (let i = 0; i < 10; i++) view.setFloat32(24 + i * 4, reserve[i] || 0, true);
  return new Uint8Array(buf);
}

export function createSharedState(overrides = {}) {
  return {
    version:    SHARED_STATE_VERSION,
    activeFold: 0,
    tickCount:  0,
    entropy:    0,
    attention:  0,
    pressure:   0,
    reserve:    new Array(10).fill(0),
    ...overrides,
  };
}
