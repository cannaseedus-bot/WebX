// SCXQ2 binary format parser — pure JS, browser + Node compatible.
// Port of scxq2_format.cpp from v0.1.0-igpu-trainer.
//
// Wire format:
//   [4B magic: "SCX2"][1B version][1B flags]
//   [varint const_count][varint instr_count]
//   [const_count × constant records]
//   [instr_count × instruction records]
//   [4B CRC32-LE over bytes[6..instr_end-1]]
//
// Constant tags: 0x01 string, 0x02 int32, 0x03 float64,
//                0x04 false, 0x05 true, 0x06 null,
//                0x07 array, 0x08 object, 0x09 ref
//
// Instruction packing: [1B packed: op=bits[5:0], reserved=bits[7:6]]
//   + varint argc + argc×varint args
//   + varint branch_target (only for op 0x02, 0x07, 0x08)

const MAGIC = [0x53, 0x43, 0x58, 0x32]; // "SCX2"
const BRANCH_OPS = new Set([0x02, 0x07, 0x08]);

// ─── CRC32 (IEEE 802.3 polynomial, same as C++ impl) ─────────────────────────

const CRC32_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    t[i] = c;
  }
  return t;
})();

function crc32(bytes, start, end) {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) {
    c = CRC32_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

// ─── Varint reader ────────────────────────────────────────────────────────────

function readVarint(bytes, offset) {
  let value = 0, shift = 0;
  while (offset < bytes.length) {
    const b = bytes[offset++];
    value |= (b & 0x7f) << shift;
    if ((b & 0x80) === 0) return { value: value >>> 0, offset };
    shift += 7;
    if (shift > 28) throw new Error('SCXQ2: varint too large');
  }
  throw new Error('SCXQ2: unexpected EOF while decoding varint');
}

// ─── Constant skip ────────────────────────────────────────────────────────────

function skipConstant(bytes, offset, depth = 0) {
  if (depth > 256) throw new Error('SCXQ2: constant nesting too deep');
  if (offset >= bytes.length) throw new Error('SCXQ2: unexpected EOF reading constant tag');

  const tag = bytes[offset++];
  let v;

  switch (tag) {
    case 0x01: { // string
      ({ value: v, offset } = readVarint(bytes, offset));
      if (offset + v > bytes.length) throw new Error('SCXQ2: unexpected EOF reading constant string');
      return offset + v;
    }
    case 0x02: // int32
      if (offset + 4 > bytes.length) throw new Error('SCXQ2: unexpected EOF reading int32');
      return offset + 4;
    case 0x03: // float64
      if (offset + 8 > bytes.length) throw new Error('SCXQ2: unexpected EOF reading float64');
      return offset + 8;
    case 0x04: case 0x05: case 0x06: // false, true, null
      return offset;
    case 0x07: { // array
      ({ value: v, offset } = readVarint(bytes, offset));
      for (let i = 0; i < v; i++) offset = skipConstant(bytes, offset, depth + 1);
      return offset;
    }
    case 0x08: { // object
      ({ value: v, offset } = readVarint(bytes, offset));
      for (let i = 0; i < v; i++) {
        ({ offset } = readVarint(bytes, offset)); // key index
        offset = skipConstant(bytes, offset, depth + 1);
      }
      return offset;
    }
    case 0x09: { // ref
      ({ offset } = readVarint(bytes, offset));
      return offset;
    }
    default:
      throw new Error(`SCXQ2: unknown constant tag 0x${tag.toString(16)}`);
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Parse and validate an SCXQ2 binary (ArrayBuffer or Uint8Array).
 * Returns { version, flags, constCount, instrCount, constStart, instrStart, instrEnd,
 *           crcExpected, crcComputed, valid } or throws on structural errors.
 */
export function parseScxq2(bufferOrArray) {
  const bytes = bufferOrArray instanceof Uint8Array
    ? bufferOrArray
    : new Uint8Array(bufferOrArray);

  if (bytes.length < 10) throw new Error('SCXQ2: file too small');

  for (let i = 0; i < 4; i++) {
    if (bytes[i] !== MAGIC[i]) throw new Error('SCXQ2: invalid magic (expected "SCX2")');
  }

  const version = bytes[4];
  const flags   = bytes[5];
  let offset    = 6;

  let constCount, instrCount;
  ({ value: constCount, offset } = readVarint(bytes, offset));
  ({ value: instrCount, offset } = readVarint(bytes, offset));

  const constStart = offset;

  for (let i = 0; i < constCount; i++) {
    offset = skipConstant(bytes, offset);
  }

  const instrStart = offset;

  for (let i = 0; i < instrCount; i++) {
    if (offset >= bytes.length) throw new Error('SCXQ2: unexpected EOF reading instruction');
    const packed = bytes[offset++];
    const op     = packed & 0x3f;
    let argc;
    ({ value: argc, offset } = readVarint(bytes, offset));
    for (let a = 0; a < argc; a++) {
      ({ offset } = readVarint(bytes, offset));
    }
    if (BRANCH_OPS.has(op)) {
      ({ offset } = readVarint(bytes, offset)); // branch target
    }
  }

  const instrEnd = offset;

  if (bytes.length < instrEnd + 4) throw new Error('SCXQ2: unexpected EOF reading CRC32');
  if (bytes.length !== instrEnd + 4) throw new Error('SCXQ2: trailing bytes after instructions');

  // CRC is over bytes[6..instrEnd-1] (after version+flags)
  const crcComputed = crc32(bytes, 6, instrEnd);
  const view        = new DataView(bytes.buffer, bytes.byteOffset + instrEnd, 4);
  const crcExpected = view.getUint32(0, true); // little-endian

  return {
    version,
    flags,
    constCount,
    instrCount,
    constStart,
    instrStart,
    instrEnd,
    crcExpected,
    crcComputed,
    valid: crcExpected === crcComputed,
  };
}

/**
 * Parse and validate, throwing if CRC mismatches.
 */
export function parseScxq2OrThrow(buffer) {
  const result = parseScxq2(buffer);
  if (!result.valid) {
    throw new Error(
      `SCXQ2: CRC32 mismatch (expected 0x${result.crcExpected.toString(16).padStart(8,'0')}, `+
      `got 0x${result.crcComputed.toString(16).padStart(8,'0')})`
    );
  }
  return result;
}

/**
 * Read the instruction slice (bytes from instrStart to instrEnd) as Uint8Array.
 * Useful for feeding into the XVM interpreter.
 */
export function extractInstructions(buffer, parsed) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  return bytes.slice(parsed.instrStart, parsed.instrEnd);
}

export default parseScxq2;
