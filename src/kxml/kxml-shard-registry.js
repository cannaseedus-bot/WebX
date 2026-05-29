// kxml-shard-registry.js — KXML ↔ SCXQ2 DDS fold/shard bridge
//
// Maps KXML node attributes (fold, domain, phase, device) to the existing
// scxq2_dds_folds layout:
//
//   Fold manifest layout (GPT-2 small, 12L×768d):
//     EMBEDDING_FOLD           — wte + wpe, HOT, INT16_SYM
//     ATTENTION_FOLD_L{0-11}   — 4 attn tensors/layer, HOT, INT8_SYM
//                                shard_id = L % 4
//                                DDS_SHARD_{L*4 + tensor_idx}  (0-47)
//     FFN_FOLD_L{0-11}         — 4 ffn tensors/layer, WARM, INT8_SYM
//     KV_CACHE_FOLD            — HOT, ROLLING
//     SESSION_FOLD             — WARM, MUTABLE
//     ADAPTER_FOLD             — STREAMING, DELTA_ONLY
//     DDS_SHARD_{000-047}      — BC7, 64-tile, 4-mip, IMMUTABLE
//
// KXML→SCXQ2 residency table:
//   phase=Sek  + device=gpu  → HOT
//   phase=Pop  + device=cpu  → STREAMING
//   phase=Wo                 → STREAMING
//   phase=Ch'en              → WARM
//   phase=Xul                → STREAMING
//
// Source: C:\Users\canna\.kuhul-v1\models\scxq2_dds_folds\fold_manifest.json

// ─── Static fold → DDS shard mapping ─────────────────────────────────────────
// Built from fold_manifest.json — attn layers cycle shard_id 0-3,
// physical shard = layer * 4 + tensor_index (48 total)

const ATTENTION_TENSOR_NAMES = [
  'attn.c_attn.bias', 'attn.c_attn.weight',
  'attn.c_proj.bias', 'attn.c_proj.weight',
];

const FFN_TENSOR_NAMES = [
  'mlp.c_fc.bias', 'mlp.c_fc.weight',
  'mlp.c_proj.bias', 'mlp.c_proj.weight',
];

function buildFoldIndex(nLayers = 12) {
  const index = new Map();

  // Embedding fold
  index.set('EMBEDDING_FOLD', {
    id: 'EMBEDDING_FOLD', domain: 'EMBEDDING',
    residency: 'HOT', quantization: 'INT16_SYM',
    tensors: ['transformer.wte.weight', 'transformer.wpe.weight'],
    shards: [],
  });

  for (let l = 0; l < nLayers; l++) {
    const shardIdBase = l * 4;

    // Attention fold
    const attnId = `ATTENTION_FOLD_L${l}`;
    index.set(attnId, {
      id: attnId, domain: 'ATTENTION', layer: l,
      residency: 'HOT', quantization: 'INT8_SYM',
      shardIdLogical: l % 4,
      tensors: ATTENTION_TENSOR_NAMES.map(t => `transformer.h.${l}.${t}`),
      shards: ATTENTION_TENSOR_NAMES.map((_, ti) =>
        `DDS_SHARD_${String(shardIdBase + ti).padStart(3, '0')}`),
    });

    // FFN fold (no physical DDS shard — WARM, loaded on demand)
    const ffnId = `FFN_FOLD_L${l}`;
    index.set(ffnId, {
      id: ffnId, domain: 'FFN', layer: l,
      residency: 'WARM', quantization: 'INT8_SYM',
      tensors: FFN_TENSOR_NAMES.map(t => `transformer.h.${l}.${t}`),
      shards: [],
    });
  }

  // Special folds
  for (const [id, extra] of [
    ['KV_CACHE_FOLD', { domain: 'KV_CACHE', residency: 'HOT', policy: 'ROLLING_EVICTION' }],
    ['SESSION_FOLD',  { domain: 'SESSION',  residency: 'WARM', policy: 'SESSION_DELTA_SOURCE' }],
    ['ADAPTER_FOLD',  { domain: 'ADAPTER',  residency: 'STREAMING', mutation: 'DELTA_ONLY' }],
  ]) {
    index.set(id, { id, ...extra, tensors: [], shards: [] });
  }

  return index;
}

// ─── KXML phase → SCXQ2 residency ────────────────────────────────────────────

export const PHASE_RESIDENCY = Object.freeze({
  'Pop':    'STREAMING',  // init — lazy load
  'Wo':     'STREAMING',  // intent declaration — preload hint
  "Sek":    'HOT',        // compute — GPU resident
  "Ch'en":  'WARM',       // render/output — staged to CPU
  'Xul':    'STREAMING',  // termination — evict
});

// ─── KXML fold attribute → candidate SCXQ2 fold IDs ─────────────────────────

export const KXML_FOLD_TO_SCXQ2 = Object.freeze({
  'COMPUTE_FOLD':  ['ATTENTION_FOLD', 'FFN_FOLD'],   // resolved per layer by domain
  'STORAGE_FOLD':  ['KV_CACHE_FOLD', 'SESSION_FOLD'],
  'META_FOLD':     ['SESSION_FOLD'],
  'UI_FOLD':       ['SESSION_FOLD'],
  'ROUTING_FOLD':  ['SESSION_FOLD', 'KV_CACHE_FOLD'],
});

// ─── SCXQ2 ISA opcode table (Python bridge → JS mirror) ─────────────────────

export const SCXQ2_OPCODES = Object.freeze({
  MEM_LOAD:         0x00,
  MEM_STORE:        0x01,
  SCALE_DELTA:      0x02,
  DECOMPRESS:       0x03,
  COMPUTE_ATTENTION: 0x10,
  COMPUTE_MATMUL:   0x10,
  COMPUTE_ADD:      0x12,
  COMPUTE_ACTIVATION: 0x13,
  COMPUTE_LOSS:     0x14,
  FLOW_BARRIER:     0x30,
  VERIFY_HASH:      0x90,
});

export const SCXQ2_DOMAIN = Object.freeze({
  COGNITION: 0x01,
  MEMORY:    0x02,
  GPU:       0x03,
  NETWORK:   0x04,
  SECURITY:  0x05,
});

export const SCXQ2_FOLD_ID = Object.freeze({
  EMBEDDING:  0x00,
  ATTENTION:  0x01,
  FFN:        0x02,
  KV_CACHE:   0x03,
  SESSION:    0x04,
  ADAPTER:    0x05,
  SHARD:      0x0F,
});

// ─── ShardRegistry ────────────────────────────────────────────────────────────

export class ShardRegistry {
  constructor(opts = {}) {
    this._nLayers  = opts.nLayers ?? 12;
    this._baseDir  = opts.baseDir ?? 'C:/Users/canna/.kuhul-v1/models/scxq2_dds_folds';
    this._foldIdx  = buildFoldIndex(this._nLayers);
    this._loaded   = new Map();   // shardId → { residency, hash }
    this._hashMap  = opts.hashMap ?? {};  // from dds_manifest.json .shards
  }

  // ── Load dds_manifest + fold_manifest objects (pre-parsed JSON) ─────────────
  loadManifests(ddsManifest, foldManifest) {
    if (ddsManifest?.shards) {
      for (const [id, info] of Object.entries(ddsManifest.shards)) {
        this._hashMap[id] = info.sha256;
        this._loaded.set(id, { ...info, residency: info.residency ?? 'NOT_LOADED' });
      }
    }
    if (foldManifest?.folds) {
      for (const fold of foldManifest.folds) {
        if (fold.id && this._foldIdx.has(fold.id)) {
          Object.assign(this._foldIdx.get(fold.id), fold);
        }
      }
    }
    return this;
  }

  // ── Resolve KXML node attributes → SCXQ2 fold + shard references ────────────
  resolveFold(kxmlFold, domain, phase, layerHint = null) {
    // Phase-determined residency
    const residency = PHASE_RESIDENCY[phase] ?? 'STREAMING';

    // KXML COMPUTE_FOLD → attention or FFN based on domain
    if (kxmlFold === 'COMPUTE_FOLD') {
      const layer = layerHint ?? 0;
      const isAttn = domain === 'attention' || domain === 'compute';
      const foldId = isAttn ? `ATTENTION_FOLD_L${layer}` : `FFN_FOLD_L${layer}`;
      const fold   = this._foldIdx.get(foldId);
      return fold
        ? { ...fold, residency, ddsShards: fold.shards }
        : { id: foldId, residency, ddsShards: [] };
    }

    // Other KXML folds
    const candidates = KXML_FOLD_TO_SCXQ2[kxmlFold] ?? [];
    for (const prefix of candidates) {
      const found = [...this._foldIdx.values()].find(f => f.id.startsWith(prefix));
      if (found) return { ...found, residency, ddsShards: found.shards };
    }

    return { id: kxmlFold, residency, ddsShards: [] };
  }

  // ── Map KXML node → SCXQ2 ISA instruction ────────────────────────────────────
  nodeToISA(node, phaseIdx) {
    const fold   = this.resolveFold(node.fold, node.domain, node.phase);
    const foldId = SCXQ2_FOLD_ID[fold.domain?.split('_')[0]] ?? SCXQ2_FOLD_ID.SHARD;
    const domain = node.device === 'gpu' ? SCXQ2_DOMAIN.GPU : SCXQ2_DOMAIN.COGNITION;
    const flags  = node.device === 'gpu' ? 0x0040 : 0x0000;

    const instrs = [];

    // Phase barrier at boundary
    instrs.push({
      opcode: SCXQ2_OPCODES.FLOW_BARRIER,
      domain: SCXQ2_DOMAIN.COGNITION,
      phase:  node.phase,
      phaseIdx,
      foldId: 0xFF,
      flags:  0x0001,
    });

    // Op-level instructions
    for (const op of node.ops) {
      instrs.push({
        opcode: opToISACode(op.type),
        domain,
        foldId,
        phase:  node.phase,
        phaseIdx,
        op,
        nodeId: node.id,
        flags,
      });
    }

    return instrs;
  }

  // ── Convert a full KXML graph to SCXQ2 ISA program ───────────────────────────
  graphToISA(kxmlGraph) {
    const PHASE_ORDER = ['Pop', 'Wo', 'Sek', "Ch'en", 'Xul'];
    const instrs = [];

    for (const [phaseIdx, phase] of PHASE_ORDER.entries()) {
      for (const node of kxmlGraph.nodes.values()) {
        if (node.phase !== phase) continue;
        instrs.push(...this.nodeToISA(node, phaseIdx));
      }
    }

    // Verify-hash at end
    instrs.push({
      opcode: SCXQ2_OPCODES.VERIFY_HASH,
      domain: SCXQ2_DOMAIN.SECURITY,
      foldId: 0xFF,
      offset: 0,
      flags:  0x0002,
    });

    return instrs;
  }

  // ── Edge manifest (forward+backward channels) ────────────────────────────────
  edgeManifest(edges) {
    return {
      forward_paths:  edges.filter(e => e.forward).map(e => ({
        from: e.from, to: e.to,
        data: e.forward.data,
        phase: e.phaseGate.forwardRequiresTo ?? 'Sek',
      })),
      backward_paths: edges.filter(e => e.backward).map(e => ({
        from: e.to, to: e.from,
        data: e.backward.data,
        scale: e.backward.scale,
        phase: e.phaseGate.backwardRequiresFrom ?? "Ch'en",
      })),
      phase_gates: Object.fromEntries(
        edges.map(e => [`${e.from}->${e.to}`, e.phaseGate])
      ),
    };
  }

  // ── Fold → shard path (for @load_shard ops) ──────────────────────────────────
  shardPath(shardId) {
    return `${this._baseDir}/shards/${shardId}.dds`;
  }

  shardHash(shardId) {
    return this._hashMap[shardId] ?? null;
  }

  getFoldInfo(foldId) {
    return this._foldIdx.get(foldId) ?? null;
  }

  get foldCount() { return this._foldIdx.size; }
  get shardCount() { return this._loaded.size; }
}

// ─── XCFE @ops[] → SCXQ2 opcode ─────────────────────────────────────────────

function opToISACode(opType) {
  const m = {
    '@load':           SCXQ2_OPCODES.MEM_LOAD,
    '@store':          SCXQ2_OPCODES.MEM_STORE,
    '@load_shard':     SCXQ2_OPCODES.MEM_LOAD,
    '@input':          SCXQ2_OPCODES.MEM_LOAD,
    '@mul':            SCXQ2_OPCODES.COMPUTE_MATMUL,
    '@add':            SCXQ2_OPCODES.COMPUTE_ADD,
    '@gemm':           SCXQ2_OPCODES.COMPUTE_MATMUL,
    '@linear':         SCXQ2_OPCODES.COMPUTE_MATMUL,
    '@activation':     SCXQ2_OPCODES.COMPUTE_ACTIVATION,
    '@softmax':        SCXQ2_OPCODES.COMPUTE_ACTIVATION,
    '@gelu':           SCXQ2_OPCODES.COMPUTE_ACTIVATION,
    '@attention':      SCXQ2_OPCODES.COMPUTE_ATTENTION,
    '@geometric_attention': SCXQ2_OPCODES.COMPUTE_ATTENTION,
    '@loss':           SCXQ2_OPCODES.COMPUTE_LOSS,
    '@barrier':        SCXQ2_OPCODES.FLOW_BARRIER,
    '@scale':          SCXQ2_OPCODES.SCALE_DELTA,
    '@fold_compress':  SCXQ2_OPCODES.DECOMPRESS,
  };
  return m[opType] ?? SCXQ2_OPCODES.MEM_LOAD;
}
