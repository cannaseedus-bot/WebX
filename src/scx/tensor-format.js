// SCXT tensor format — port of scx_tensor.h/c (SCX.v1.0.0)
//
// Binary layout (little-endian):
//   [4B magic "SCXT"][1B version][1B dtype][1B rank][1B stride_mode]
//   [4B×4 dims][4B×4 strides][2B block_size][1B scale_dtype][1B reserved]
//   [4B data_size]
//   → Total header: 4+4+16+16+4+4 = 56 bytes (without #pragma pack(1) gaps — verified from C struct)
//   [data_size bytes data]
//   [32B BLAKE3 hash footer]
//
// Q4_BLOCK data layout:
//   scales first: blocks × 2 bytes (FP16 or BF16 per block)
//   packed nibbles: elements / 2 bytes

export const SCXT_MAGIC = 'SCXT';
export const SCXT_VERSION = 0x01;

export const SCXT_HEADER_SIZE = 56; // #pragma pack(1) struct size
export const SCXT_HASH_SIZE   = 32; // BLAKE3 footer

export const SCX_DTYPE = Object.freeze({
  Q16_16:   0, // 16.16 fixed-point — 4 bytes/element
  INT8:     1, // 8-bit integer     — 1 byte/element
  INT4:     2, // 4-bit packed      — 0.5 byte/element
  Q4_BLOCK: 3, // block-quantized 4-bit + FP16 scale per 64-element block
  BF16:     4, // brain float 16    — 2 bytes/element
  FP16:     5, // float 16          — 2 bytes/element
});

export const SCX_DTYPE_BYTES = Object.freeze({
  [SCX_DTYPE.Q16_16]:   4,
  [SCX_DTYPE.INT8]:     1,
  [SCX_DTYPE.INT4]:     0.5,
  [SCX_DTYPE.Q4_BLOCK]: null, // block-dependent — use calcDataSize()
  [SCX_DTYPE.BF16]:     2,
  [SCX_DTYPE.FP16]:     2,
});

export const SCXT_STRIDE_MODE = Object.freeze({ CONTIGUOUS: 0, EXPLICIT: 1 });

export const Q4_BLOCK_ELEMENTS = 64; // elements per quantization block
export const Q4_SCALE_BYTES    = 2;  // FP16 scale per block

export function calcDataSize(dtype, elementCount, blockSize = Q4_BLOCK_ELEMENTS) {
  switch (dtype) {
    case SCX_DTYPE.Q16_16:   return elementCount * 4;
    case SCX_DTYPE.INT8:     return elementCount;
    case SCX_DTYPE.INT4:     return Math.ceil(elementCount / 2);
    case SCX_DTYPE.BF16:
    case SCX_DTYPE.FP16:     return elementCount * 2;
    case SCX_DTYPE.Q4_BLOCK: {
      const blocks = Math.ceil(elementCount / blockSize);
      return Math.floor(elementCount / 2) + blocks * Q4_SCALE_BYTES;
    }
    default: throw new Error(`SCX: unknown dtype ${dtype}`);
  }
}

export function calcStrides(dims, rank) {
  const strides = new Uint32Array(4);
  let stride = 1;
  for (let i = rank - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}

export function readScxtHeader(buffer) {
  const view = new DataView(ArrayBuffer.isView(buffer) ? buffer.buffer : buffer,
                             ArrayBuffer.isView(buffer) ? buffer.byteOffset : 0);
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== SCXT_MAGIC) throw new Error(`SCXT: bad magic "${magic}"`);

  const version     = view.getUint8(4);
  const dtype       = view.getUint8(5);
  const rank        = view.getUint8(6);
  const stride_mode = view.getUint8(7);

  const dims    = [0, 0, 0, 0].map((_, i) => view.getUint32(8  + i * 4, true));
  const strides = [0, 0, 0, 0].map((_, i) => view.getUint32(24 + i * 4, true));

  const block_size  = view.getUint16(40, true);
  const scale_dtype = view.getUint8(42);
  // reserved byte at 43
  const data_size   = view.getUint32(44, true);

  return { magic, version, dtype, rank, stride_mode, dims, strides, block_size, scale_dtype, data_size };
}

export function writeScxtHeader(opts) {
  const {
    dtype, rank, dims, block_size = Q4_BLOCK_ELEMENTS,
    scale_dtype = 0, stride_mode = SCXT_STRIDE_MODE.CONTIGUOUS, data_size,
  } = opts;

  const strides = opts.strides || calcStrides(dims, rank);
  const buf  = new ArrayBuffer(SCXT_HEADER_SIZE);
  const view = new DataView(buf);

  // magic
  'SCXT'.split('').forEach((c, i) => view.setUint8(i, c.charCodeAt(0)));
  view.setUint8(4, SCXT_VERSION);
  view.setUint8(5, dtype);
  view.setUint8(6, rank);
  view.setUint8(7, stride_mode);

  for (let i = 0; i < 4; i++) {
    view.setUint32(8  + i * 4, dims[i]    || 1, true);
    view.setUint32(24 + i * 4, strides[i] || 0, true);
  }
  view.setUint16(40, block_size,  true);
  view.setUint8(42,  scale_dtype);
  view.setUint8(43,  0); // reserved
  view.setUint32(44, data_size, true);

  return new Uint8Array(buf);
}

// Hash contract — requires BLAKE3 implementation (not bundled here).
// hashFn: (Uint8Array) → Uint8Array[32]
export function hashScxtTensor(header, data, hashFn) {
  if (typeof hashFn !== 'function') throw new Error('SCXT: hashFn is required');
  const dtype     = new Uint8Array([header.dtype]);
  const rank      = new Uint8Array([header.rank]);
  const dims      = new Uint8Array(new Uint32Array(header.dims).buffer);
  const combined  = new Uint8Array(1 + 1 + dims.byteLength + data.byteLength);
  let off = 0;
  combined.set(dtype,  off); off += 1;
  combined.set(rank,   off); off += 1;
  combined.set(dims,   off); off += dims.byteLength;
  combined.set(data,   off);
  return hashFn(combined);
}

// Verify footer hash — returns true if BLAKE3 footer matches data hash
export function verifyScxtBuffer(buffer, hashFn) {
  const ab  = ArrayBuffer.isView(buffer) ? buffer.buffer : buffer;
  const off = ArrayBuffer.isView(buffer) ? buffer.byteOffset : 0;
  const len = ArrayBuffer.isView(buffer) ? buffer.byteLength : ab.byteLength;

  const header   = readScxtHeader(new Uint8Array(ab, off, SCXT_HEADER_SIZE));
  const dataOff  = off + SCXT_HEADER_SIZE;
  const dataLen  = header.data_size;
  const hashOff  = dataOff + dataLen;

  if (hashOff + SCXT_HASH_SIZE > len) return false;

  const data         = new Uint8Array(ab, dataOff, dataLen);
  const storedHash   = new Uint8Array(ab, hashOff, SCXT_HASH_SIZE);
  const computedHash = hashScxtTensor(header, data, hashFn);

  for (let i = 0; i < SCXT_HASH_SIZE; i++) {
    if (storedHash[i] !== computedHash[i]) return false;
  }
  return true;
}
