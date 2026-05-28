// ScxGraph — computation graph types (SCXRuntime.v1.0.0, include/scxgraph.h + gta1_loader.h)
//
// ScxNode roles: vector, matrix, micronaut-di, intent, dispatch, executor
// ScxEdge types: brain-edge, geodesic-entropy-arc
// Quantization: int4, fp16, fp32
// Device: cpu, gpu

export const SCX_NODE_ROLES = Object.freeze([
  'vector', 'matrix', 'micronaut-di', 'intent', 'dispatch', 'executor',
]);

export const SCX_EDGE_TYPES = Object.freeze([
  'brain-edge',
  'geodesic-entropy-arc',
]);

export const SCX_QUANT = Object.freeze({ INT4: 'int4', FP16: 'fp16', FP32: 'fp32' });
export const SCX_DEVICE = Object.freeze({ CPU: 'cpu', GPU: 'gpu' });

export function createScxNode(opts) {
  const { id, role = 'vector', q = 'fp16', device = 'cpu', pos = [0, 0] } = opts;
  if (!id) throw new Error('ScxNode: id is required');
  return Object.freeze({ id, role, q, device, pos: [pos[0] || 0, pos[1] || 0] });
}

export function createScxEdge(opts) {
  const { id, from, to, type = 'brain-edge', metric = 0, entropy = 0, phase = 0 } = opts;
  if (!id || !from || !to) throw new Error('ScxEdge: id, from, to required');
  return Object.freeze({ id, from, to, type, metric, entropy, phase });
}

export function createScxGraph(opts = {}) {
  const { coord_frame = '', nodes = [], edges = [] } = opts;
  return { ok: true, coord_frame, nodes: Array.from(nodes), edges: Array.from(edges) };
}

// GTA1 binary format types (gta1_loader.h)
// GTA1 contains topology XML + FIELDS tensor metadata; zstd-compressed optional
export function createGta1Tensor(opts = {}) {
  return Object.freeze({
    id:       (opts.id || 0) >>> 0,
    shape:    Array.from(opts.shape || []),
    dtype:    (opts.dtype || 0) & 0xffff,
    q_scheme: (opts.q_scheme || 0) & 0xffff,
    scale:    opts.scale || 0,
  });
}

export function createGta1Node(opts = {}) {
  return Object.freeze({
    id:       opts.id || '',
    role:     opts.role || 'vector',
    shape:    Array.from(opts.shape || []),
    q_scheme: opts.q_scheme || '',
    device:   opts.device || 'cpu',
  });
}

export function createGta1Edge(opts = {}) {
  return Object.freeze({
    id:      opts.id || '',
    from:    opts.from || '',
    to:      opts.to || '',
    type:    opts.type || 'brain-edge',
    metric:  opts.metric || 0,
    entropy: opts.entropy || 0,
    phase:   opts.phase || 0,
  });
}
