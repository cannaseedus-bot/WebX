// SCX model manifest + config schema (SCXRuntime.v1.0.0)
//
// Model manifest (@kind: "scx.model.manifest.v1") declares:
//   artifacts: { kbc1, dds, config, experts, vocab }
//   hashes: { [path]: sha256 }
//
// ManifestInfo (include/manifest_loader.h):
//   ok, coord_frame, nodes, edges, tensors[], kbc1_bytes, scx_graph
//
// Default model config: scx-moe-16l (from model/config.xjson)

export const MANIFEST_KIND = 'scx.model.manifest.v1';

// Default model config matching model/config.xjson
export const SCX_MOE_16L_CONFIG = Object.freeze({
  '@model':  'scx-moe-16l',
  hidden:    2048,
  layers:    16,
  experts:   32,
  top_k:     2,
  quant:     'int4',
  runtime: Object.freeze({
    gpu:       'd3d12',
    fallback:  'cpu',
    threads:   'auto',
    paging:    true,
    kv_delta:  'int4',
  }),
});

export function createManifestTensor(opts = {}) {
  return Object.freeze({
    id:       opts.id || '',
    shape:    Array.from(opts.shape || []),
    dtype:    opts.dtype || 'int4',
    q_scheme: opts.q_scheme || '',
    source:   opts.source || '',
    data:     opts.data instanceof Uint8Array ? opts.data : new Uint8Array(0),
  });
}

export function createManifestInfo(opts = {}) {
  return {
    ok:          opts.ok !== undefined ? !!opts.ok : true,
    coord_frame: opts.coord_frame || '',
    nodes:       opts.nodes || 0,
    edges:       opts.edges || 0,
    tensors:     Array.from(opts.tensors || []),
    kbc1_bytes:  opts.kbc1_bytes instanceof Uint8Array
                   ? opts.kbc1_bytes
                   : new Uint8Array(opts.kbc1_bytes || []),
    scx_graph:   opts.scx_graph || { ok: false, coord_frame: '', nodes: [], edges: [] },
  };
}

// Parse a model manifest JSON object
export function parseModelManifest(json) {
  if (json['@kind'] !== MANIFEST_KIND) {
    throw new Error(`SCX manifest: expected kind "${MANIFEST_KIND}", got "${json['@kind']}"`);
  }
  const artifacts = json.artifacts || {};
  return Object.freeze({
    kind:      json['@kind'],
    artifacts: Object.freeze({ ...artifacts }),
    hashes:    Object.freeze({ ...(json.hashes || {}) }),
  });
}

// sco-cache-index.json schema (both SCX.v1.0.0 and SCXRuntime.v1.0.0)
export function parseScoIndex(json) {
  const files = (json.files || []).map(f => Object.freeze({
    path:   f.path,
    sha256: (f.sha256 || '').toLowerCase(),
    bytes:  f.bytes || 0,
  }));
  return Object.freeze({
    context:  json['@context'] || json['id'] || '',
    release:  Object.freeze(json.release || {}),
    files,
  });
}

export function validateScoIndex(index, fileContents) {
  const results = [];
  for (const entry of index.files) {
    const content = fileContents[entry.path];
    results.push({ path: entry.path, expected: entry.sha256, found: content !== undefined });
  }
  return results;
}
