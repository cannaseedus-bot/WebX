// .brain binary format reader/writer — port of brain1/compiler.py (v1.0.0-PowerShell-LLM)
//
// File layout (all little-endian):
//   tensor_header.bin  — 5×uint32: [node_count, edge_count, csr_flat_len, K_experts, feat_dim]
//   nodes_time.bin     — float32[node_count]
//   nodes_freq.bin     — float32[node_count]
//   nodes_energy.bin   — float32[node_count]
//   edges.bin          — uint32[edge_count × 2]  (src, dst pairs)
//   csr_index.bin      — uint32[node_count + 1]  (CSR row pointers)
//   csr_neigh.bin      — uint32[csr_flat_len]    (CSR column indices)
//   experts.bin        — uint32[node_count]       (KMeans cluster IDs)
//   routing.bin        — uint32[node_count]       (KMeans cluster IDs, inference-time copy)
//
// K_experts = max(4, floor(node_count / 500))

export const BRAIN_HEADER_BYTES = 20; // 5 × uint32
export const BRAIN_FEAT_DIM     = 3;  // time / freq / energy

export const BRAIN_FIELD = Object.freeze({
  NODE_COUNT:    0,
  EDGE_COUNT:    1,
  CSR_FLAT_LEN:  2,
  K_EXPERTS:     3,
  FEAT_DIM:      4,
});

export function kExpertsFor(nodeCount) {
  return Math.max(4, Math.floor(nodeCount / 500));
}

export function readBrainHeader(buffer) {
  const view = new DataView(ArrayBuffer.isView(buffer) ? buffer.buffer : buffer,
                             ArrayBuffer.isView(buffer) ? buffer.byteOffset : 0,
                             BRAIN_HEADER_BYTES);
  const node_count   = view.getUint32(0,  true);
  const edge_count   = view.getUint32(4,  true);
  const csr_flat_len = view.getUint32(8,  true);
  const K_experts    = view.getUint32(12, true);
  const feat_dim     = view.getUint32(16, true);
  return { node_count, edge_count, csr_flat_len, K_experts, feat_dim };
}

export function writeBrainHeader(header) {
  const buf  = new ArrayBuffer(BRAIN_HEADER_BYTES);
  const view = new DataView(buf);
  view.setUint32(0,  header.node_count,   true);
  view.setUint32(4,  header.edge_count,   true);
  view.setUint32(8,  header.csr_flat_len, true);
  view.setUint32(12, header.K_experts,    true);
  view.setUint32(16, header.feat_dim,     true);
  return new Uint8Array(buf);
}

export function readFloat32Array(buffer, count) {
  const ab = ArrayBuffer.isView(buffer) ? buffer.buffer : buffer;
  const off = ArrayBuffer.isView(buffer) ? buffer.byteOffset : 0;
  return new Float32Array(ab, off, count);
}

export function readUint32Array(buffer, count) {
  const ab = ArrayBuffer.isView(buffer) ? buffer.buffer : buffer;
  const off = ArrayBuffer.isView(buffer) ? buffer.byteOffset : 0;
  return new Uint32Array(ab, off, count);
}

export function readBrainFeatures(header, timeBuffer, freqBuffer, energyBuffer) {
  const n = header.node_count;
  return {
    time:   readFloat32Array(timeBuffer,   n),
    freq:   readFloat32Array(freqBuffer,   n),
    energy: readFloat32Array(energyBuffer, n),
  };
}

export function readBrainTopology(header, edgesBuffer, csrIndexBuffer, csrNeighBuffer) {
  return {
    edges:    readUint32Array(edgesBuffer,    header.edge_count * 2),
    csrIndex: readUint32Array(csrIndexBuffer, header.node_count + 1),
    csrNeigh: readUint32Array(csrNeighBuffer, header.csr_flat_len),
  };
}

export function readBrainRouting(header, expertsBuffer, routingBuffer) {
  return {
    experts: readUint32Array(expertsBuffer, header.node_count),
    routing: readUint32Array(routingBuffer, header.node_count),
  };
}

export function buildBrainManifest(header) {
  return {
    format:      'brain-v1',
    node_count:  header.node_count,
    edge_count:  header.edge_count,
    csr_flat_len: header.csr_flat_len,
    K_experts:   header.K_experts,
    feat_dim:    header.feat_dim,
    files: [
      'tensor_header.bin',
      'nodes_time.bin',
      'nodes_freq.bin',
      'nodes_energy.bin',
      'edges.bin',
      'csr_index.bin',
      'csr_neigh.bin',
      'experts.bin',
      'routing.bin',
    ],
  };
}
