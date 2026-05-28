// SCXTOK BPE tokenizer — port of SCX_SPECIFICATION.md §2 (SCX.v1.0.0)
//
// Binary header layout (.scxtok):
//   [4B magic "SCXT"][1B version=0x01][1B vocab_type][2B flags]
//   [4B vocab_size][4B merge_count][1B special_count][15B reserved]
//   [32B BLAKE3 hash]
//
// vocab_type: 0=byte-level, 1=BPE, 2=unigram
// special tokens start at ID 256; vocab IDs start at 256+special_count
// byte tokens are always IDs 0-255 (byte-level fallback for OOV)
//
// MergeEntry: [4B left_id][4B right_id][4B new_id][4B priority]
// VocabEntry: [4B id][1B length][length bytes UTF-8] padded to 4-byte boundary

export const SCXTOK_MAGIC   = 'SCXT';
export const SCX_BYTE_OFFSET    = 0;
export const SCX_BYTE_COUNT     = 256;
export const SCX_SPECIAL_OFFSET = 256;

export const SCXTOK_VOCAB_TYPE = Object.freeze({ BYTE_LEVEL: 0, BPE: 1, UNIGRAM: 2 });
export const SCXTOK_SPECIAL    = Object.freeze({ PAD: 0, UNK: 1, BOS: 2, EOS: 3, MASK: 4 });

export class SCXTokenizer {
  constructor() {
    this.merges      = []; // { left_id, right_id, new_id, priority }
    this.vocab       = new Map(); // id → utf8 bytes (Uint8Array)
    this.reverseVocab = new Map(); // utf8 string → id
    this.specialTokens = new Map(); // type → id
    this.vocabSize   = 0;
  }

  addMerge(leftId, rightId, newId, priority) {
    this.merges.push({ left_id: leftId, right_id: rightId, new_id: newId, priority });
    this.merges.sort((a, b) => a.priority - b.priority);
  }

  addVocabEntry(id, bytes) {
    const str = typeof bytes === 'string' ? bytes : new TextDecoder().decode(bytes);
    this.vocab.set(id, typeof bytes === 'string' ? new TextEncoder().encode(bytes) : bytes);
    this.reverseVocab.set(str, id);
    if (id + 1 > this.vocabSize) this.vocabSize = id + 1;
  }

  // Deterministic BPE tokenization matching scx_tokenize() in SCX_SPECIFICATION.md §2.3
  tokenize(text) {
    const utf8 = new TextEncoder().encode(text);

    // Step 1: byte tokens (ID = byte value, 0-255)
    let tokens = new Uint32Array(utf8.length);
    for (let i = 0; i < utf8.length; i++) tokens[i] = utf8[i];
    let count = utf8.length;

    // Step 2: apply merges in priority order (ascending)
    for (const merge of this.merges) {
      const next = new Uint32Array(count);
      let writePos = 0;
      for (let i = 0; i < count; i++) {
        if (i < count - 1 && tokens[i] === merge.left_id && tokens[i + 1] === merge.right_id) {
          next[writePos++] = merge.new_id;
          i++;
        } else {
          next[writePos++] = tokens[i];
        }
      }
      tokens = next;
      count  = writePos;
    }

    return tokens.slice(0, count);
  }

  decode(ids) {
    const parts = [];
    for (const id of ids) {
      if (id < SCX_BYTE_COUNT) {
        parts.push(id); // raw byte token
      } else {
        const entry = this.vocab.get(id);
        if (entry) for (const b of entry) parts.push(b);
      }
    }
    return new TextDecoder().decode(new Uint8Array(parts));
  }

  // Load from a plain-JS descriptor object (for when binary .scxtok is not available)
  static fromDescriptor(desc) {
    const tok = new SCXTokenizer();
    for (const m of (desc.merges || [])) {
      tok.addMerge(m.left_id, m.right_id, m.new_id, m.priority);
    }
    for (const [str, id] of Object.entries(desc.vocab || {})) {
      tok.addVocabEntry(id, str);
    }
    return tok;
  }
}

// Parse binary .scxtok header — returns { vocabType, flags, vocabSize, mergeCount, specialCount }
export function readScxtokHeader(buffer) {
  const view = new DataView(ArrayBuffer.isView(buffer) ? buffer.buffer : buffer,
                             ArrayBuffer.isView(buffer) ? buffer.byteOffset : 0);
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== SCXTOK_MAGIC) throw new Error(`SCXTOK: bad magic "${magic}"`);
  const version      = view.getUint8(4);
  const vocab_type   = view.getUint8(5);
  const flags        = view.getUint16(6, true);
  const vocab_size   = view.getUint32(8, true);
  const merge_count  = view.getUint32(12, true);
  const special_count = view.getUint8(16);
  return { version, vocab_type, flags, vocab_size, merge_count, special_count,
           vocab_offset: SCX_SPECIAL_OFFSET + special_count };
}
