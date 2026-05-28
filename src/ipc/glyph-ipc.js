// Glyph IPC protocol — binary layout for Windows named file mapping.
// Port of directx/native/glyph.h from v0.2.0-kuhul-directx-native.
//
// Named mapping: "Local\GRAMMAP_<tag>"
// Status flow: caller → READY(1) → engine processes → PROCESSED(2)
//
// Binary layout (packed, no alignment padding):
//   IPCHeader: 28 bytes
//   GlyphEntry[n]: 32 bytes each
//   result GlyphEntry: 32 bytes (written after entries by native engine)

export const GRAM_MAGIC_LO = 0x525F5447; // "GTR_"
export const GRAM_MAGIC_HI = 0x47524D4D; // "MMRG"

export const IPC_STATUS = Object.freeze({ EMPTY: 0, READY: 1, PROCESSED: 2 });

// IPCHeader field offsets (byte offsets, #pragma pack(1))
export const IPC_HEADER = Object.freeze({
  SIZE:        28,  // total byte size of header
  MAGIC_OFF:   0,   // uint64 (8B)
  VERSION_OFF: 8,   // uint32 (4B)
  TOTAL_OFF:   12,  // uint32 (4B) — total mapping bytes
  COUNT_OFF:   16,  // uint32 (4B) — glyph entry count
  STATUS_OFF:  20,  // uint32 (4B) — IPC_STATUS value
  RESV_OFF:    24,  // uint32 (4B) — reserved
});

// GlyphEntry field offsets (byte offsets, #pragma pack(1))
export const GLYPH_ENTRY = Object.freeze({
  SIZE:           32,  // total byte size of one entry
  CODEPOINT_OFF:  0,   // uint32 (4B)
  GLYPH_TYPE_OFF: 4,   // uint16 (2B)
  PAD_OFF:        6,   // uint16 (2B) padding
  FEATURES_OFF:   8,   // uint8[16] (16B)
  RESV1_OFF:      24,  // uint32 (4B)
  RESV2_OFF:      28,  // uint32 (4B)
});

// INT4 Lane ISA opcodes (from glyph_compute.hlsl CS_GlyphExec, cs_6_0)
// Each opcode is a 4-bit nibble packed 8 per uint32 in gramBuffer.
export const INT4_ISA = Object.freeze({
  NOP:   0x0, // No operation
  LOAD:  0x1, // r0 = stateBuffer[lane*4 + 0]
  STORE: 0x2, // stateBuffer[lane*4+0..2] = r0,r1,acc
  ADD:   0x3, // r0 = r0 + r1
  MUL:   0x4, // r0 = r0 * r1
  DOT:   0x5, // acc = r0 * r1
  NORM:  0x6, // RMS normalize r0,r1 in-place
  EXP:   0x8, // r0 = exp(r0)
  SUM:   0x9, // acc += r0
  MAX:   0xA, // r0 = max(r0, r1)
  MIN:   0xB, // r0 = min(r0, r1)
  MOV:   0xC, // r1 = r0
  ESC:   0xF, // signal XCFE routing → outBuffer[lane] |= 0x80000000
});

// Glyph dispatch modes (cbuffer DispatchParams.mode)
export const GLYPH_MODE = Object.freeze({ GRAM: 0, TENSOR: 1 });

// ─── Reader helpers (work on ArrayBuffer / DataView) ─────────────────────────

export function readIPCHeader(view) {
  if (!(view instanceof DataView)) view = new DataView(view);
  return {
    magic_lo:   view.getUint32(IPC_HEADER.MAGIC_OFF,     true),
    magic_hi:   view.getUint32(IPC_HEADER.MAGIC_OFF + 4, true),
    version:    view.getUint32(IPC_HEADER.VERSION_OFF,   true),
    totalSize:  view.getUint32(IPC_HEADER.TOTAL_OFF,     true),
    glyphCount: view.getUint32(IPC_HEADER.COUNT_OFF,     true),
    status:     view.getUint32(IPC_HEADER.STATUS_OFF,    true),
  };
}

export function readGlyphEntry(view, byteOffset = 0) {
  if (!(view instanceof DataView)) view = new DataView(view);
  const off = byteOffset;
  return {
    codepoint:  view.getUint32(off + GLYPH_ENTRY.CODEPOINT_OFF,  true),
    glyphType:  view.getUint16(off + GLYPH_ENTRY.GLYPH_TYPE_OFF, true),
    features:   new Uint8Array(view.buffer, view.byteOffset + off + GLYPH_ENTRY.FEATURES_OFF, 16).slice(),
  };
}

// Read all glyph entries from a mapping buffer.
export function readGlyphEntries(buffer) {
  const view  = new DataView(buffer instanceof ArrayBuffer ? buffer : buffer.buffer);
  const hdr   = readIPCHeader(view);
  const entries = [];
  let off = IPC_HEADER.SIZE;
  for (let i = 0; i < hdr.glyphCount; i++) {
    entries.push(readGlyphEntry(view, off));
    off += GLYPH_ENTRY.SIZE;
  }
  return { header: hdr, entries };
}

// Pack 8 INT4 nibbles into a single uint32 (LSB = first nibble)
export function packNibbles(nibbles) {
  let word = 0;
  for (let i = 0; i < Math.min(8, nibbles.length); i++) {
    word |= (nibbles[i] & 0xf) << (i * 4);
  }
  return word >>> 0;
}

// Encode an INT4 program (array of opcodes) into a Uint32Array
export function encodeINT4Program(opcodes) {
  const wordCount = Math.ceil(opcodes.length / 8);
  const out = new Uint32Array(wordCount);
  for (let w = 0; w < wordCount; w++) {
    out[w] = packNibbles(opcodes.slice(w * 8, (w + 1) * 8));
  }
  return out;
}

export default { IPC_STATUS, IPC_HEADER, GLYPH_ENTRY, INT4_ISA, readIPCHeader, readGlyphEntry, encodeINT4Program };
