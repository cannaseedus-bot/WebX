# Release Audit: SCX.v1.0.0

**Path:** `releases/SCX.v1.0.0/`
**Audited:** 2026-05-28

---

## Source Files

| File | Purpose |
|------|---------|
| scx_tensor.h | SCXT tensor format structs + API declarations |
| scx_tensor.c | C implementation with BLAKE3 hash verification |
| docs/SCX_SPECIFICATION.md | Complete format spec (tensor, tokenizer, RoPE, sharding, checkpoint) |
| docs/SCX_GRAMMAR_SCHEMA_v3.0.md | EBNF grammar for SCX language |
| docs/SCX_UNARY_ALPHABET_U1.md | U1 glyph alphabet + 3-glyph capsule format |
| cache/sco-cache-index.json | SHA256 manifest of all release files |

---

## Key Innovations

### SCXT Tensor Format (.scxt)

**Header: 56 bytes (packed), little-endian:**

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4B | magic "SCXT" |
| 4 | 1B | version 0x01 |
| 5 | 1B | dtype (0=Q16.16, 1=INT8, 2=INT4, 3=Q4_BLOCK, 4=BF16, 5=FP16) |
| 6 | 1B | rank (1-4) |
| 7 | 1B | stride_mode (0=contiguous, 1=explicit) |
| 8 | 16B | dims[4] uint32 LE |
| 24 | 16B | strides[4] uint32 LE |
| 40 | 2B | quant.block_size |
| 42 | 1B | quant.scale_dtype |
| 43 | 1B | reserved |
| 44 | 4B | data_size |

**Footer:** 32 bytes BLAKE3 hash of (dtype + rank + dims[16] + data + quant_scales_if_Q4_BLOCK)

**Q4_BLOCK layout:** 64 elements per block; data = scales_first (blocks × 2B FP16) + packed nibbles (N/2 bytes)

### SCXTOK Tokenizer (.scxtok)

- BPE merge table + byte-level fallback (token IDs 0-255 = raw bytes, always present)
- Special tokens at IDs 256..256+special_count (PAD=0, UNK=1, BOS=2, EOS=3, MASK=4)
- Vocabulary IDs start at 256 + special_count
- Deterministic tokenization: byte-expand → apply merges ascending by priority
- VocabEntry: [4B id][1B length][N bytes UTF-8] padded to 4-byte boundary
- MergeEntry: [4B left_id][4B right_id][4B new_id][4B priority]

### RoPE Positional Encoding

- Q16.16 fixed-point sin/cos tables (precomputed, deterministic)
- freq_i = 1.0 / (base^(2i/dim)); angle = pos × freq_i
- sin/cos stored as int32: round(sin(angle) × 65536)
- Apply: (x1×cos - x2×sin, x1×sin + x2×cos) in 64-bit fixed-point, shift right 16

### Shard Format (.scxshard)

- Magic "SCXS" + num_shards + SCXShardDescriptor[]
- Each descriptor: tensor_name[64] + shard_index/total_shards + dims[4] + offsets[4] + node_id[16 UUID] + port + shard_hash[32]
- Two parallelism strategies: expert (each expert on a node) + tensor (split attention heads)

### Checkpoint Format (.scxckpt)

- Magic "SCXC" + step (uint64) + timestamp + num_tensors + tensor_names
- Optimizer state per tensor: momentum, variance, step count
- RNG state (32B) for deterministic replay
- loss_scale for mixed precision
- BLAKE3 hash of entire checkpoint

### U1 Unary Alphabet (U1.1.0)

8 baseline glyphs:

| Glyph | Codepoint | Name |
|-------|-----------|------|
| ⟁ | U+27C1 | GLYPH_START |
| ⟦ | U+27E6 | CLUSTER_START |
| ⟧ | U+27E7 | CLUSTER_END |
| ⸬ | U+2E2C | MARKER |
| ⨯ | U+2A2F | REPEAT |
| λ | U+03BB | LAMBDA |
| π | U+03C0 | PI |
| → | U+2192 | ARROW |

Alphabet SHA256: `58d90630b6819f31da8a97e0ea7b97f667a92d1680de1b1fcbf484eb52c57310`

**3-glyph workflow capsule:** `⟁W⟧ ⟁F⟧ ⟁P⟧` → `{@micronaut, verb, fold_lattice[11], program_id}`

**11 top-level folds (version-stable):** micronauts, agents, skills, tools, commands, files, threads, batches, processes, bots, ports

Canonical hash input: `U1:U1.1.0\nALPHABET_SHA256:<hex>\nCODEPOINTS:<NFC-codepoint-stream>\n`

### SCX Grammar (v3.0 EBNF)

Full EBNF for SCX language — `⟦SCX x.y⟧ ⟦DICT⟧ … ⟦/DICT⟧` declarations, schema declarations, expressions, statements. Grammar SHA256 is the U1 alphabet SHA256.

### sco-cache-index.json

SHA256 inventory of all release files — deterministic cache invalidation. Pattern used in SCXRuntime.v1.0.0's sco-cache-index.json as well.

---

## Merge Targets

| Innovation | Target |
|-----------|--------|
| SCXT tensor format | `src/scx/tensor-format.js` |
| SCXTOK BPE tokenizer | `src/scx/tokenizer.js` |
| U1 unary alphabet | `src/scx/u1-alphabet.js` |
| BLAKE3 hash | Not bundled — hashFn param; caller supplies implementation |
| RoPE, sharding, checkpoint | Documented here; JS port deferred to SCXRuntime.v1.0.0 merge |

---

## Notes

- BLAKE3 is not available natively in browsers — the JS tensor format API accepts a `hashFn` parameter so the caller can supply a BLAKE3 WASM binding without making it a hard dependency
- Q4_BLOCK data_size formula from C: `(elements/2) + (blocks × 2)` — note elements/2 truncates (integer division), meaning odd-element tensors lose the last nibble; this matches the C implementation
- The `get_q16` / `set_q16` functions in the C source are stubs ("Placeholder") — Q16.16 indexed access was not implemented in this release; consumers read raw data bytes directly
- sco-cache-index.json SHA256 values are authoritative for cache invalidation in the SCXRuntime dispatch layer
