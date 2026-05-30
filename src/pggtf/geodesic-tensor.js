// geodesic-tensor.js — Geodesic Tensor G ∈ ℝ^(N×N)  [sparse CSR]
//
// Connectivity tensor. G[i,j] = geodesic distance between nodes i and j.
// Sparse: only nearby nodes have non-zero entries.
//
// GPU layout (from spec):
//   [source][target][weight]  — COO triples, sorted by source
//
// CSR internal format:
//   data[]    — non-zero weights
//   indices[] — column indices (target nodes)
//   indptr[]  — row start offsets (length = n_nodes + 1)

export class GeodesicTensor {
  /**
   * @param {number} n_nodes
   * @param {number[]} rows    — source node indices (COO)
   * @param {number[]} cols    — target node indices (COO)
   * @param {number[]} weights — geodesic distances
   */
  constructor(n_nodes, rows = [], cols = [], weights = []) {
    this.n_nodes = n_nodes;
    this._build_csr(rows, cols, weights);
  }

  _build_csr(rows, cols, weights) {
    // Count entries per row
    const counts = new Int32Array(this.n_nodes);
    for (const r of rows) counts[r]++;

    // Build indptr (prefix sum)
    this.indptr = new Int32Array(this.n_nodes + 1);
    for (let i = 0; i < this.n_nodes; i++) this.indptr[i + 1] = this.indptr[i] + counts[i];

    const nnz = rows.length;
    this.indices = new Int32Array(nnz);
    this.data    = new Float64Array(nnz);

    // Fill in order
    const next = new Int32Array(this.n_nodes);
    for (let k = 0; k < nnz; k++) {
      const r   = rows[k];
      const pos = this.indptr[r] + next[r]++;
      this.indices[pos] = cols[k];
      this.data[pos]    = weights[k];
    }
  }

  // ── Element access ─────────────────────────────────────────────────────────

  get(i, j) {
    const start = this.indptr[i], end = this.indptr[i + 1];
    for (let k = start; k < end; k++)
      if (this.indices[k] === j) return this.data[k];
    return Infinity;  // not connected → infinite distance
  }

  /** All (j, weight) neighbours of node i */
  neighbours(i) {
    const start = this.indptr[i], end = this.indptr[i + 1];
    const result = [];
    for (let k = start; k < end; k++)
      result.push({ j: this.indices[k], d: this.data[k] });
    return result;
  }

  // ── Geodesic attention weights ────────────────────────────────────────────
  // A_ij = exp(-d(i,j))  — only non-zero for stored edges

  attentionWeights(i, temperature = 1.0) {
    const nbrs = this.neighbours(i);
    if (nbrs.length === 0) return [];
    const raw    = nbrs.map(({ j, d }) => ({ j, a: Math.exp(-d / temperature) }));
    const sumA   = raw.reduce((s, r) => s + r.a, 0);
    if (sumA < 1e-12) return raw.map(r => ({ ...r, a: 1 / raw.length }));
    return raw.map(r => ({ ...r, a: r.a / sumA }));
  }

  // ── Edge mutation (Routing Micronaut) ──────────────────────────────────────

  addEdge(i, j, d) {
    // Rebuild CSR with new edge (simple approach — for large graphs prefer
    // maintaining a COO list and rebuilding lazily)
    const rows = [], cols = [], ws = [];
    for (let n = 0; n < this.n_nodes; n++) {
      for (let k = this.indptr[n]; k < this.indptr[n + 1]; k++) {
        rows.push(n); cols.push(this.indices[k]); ws.push(this.data[k]);
      }
    }
    rows.push(i); cols.push(j); ws.push(d);
    this._build_csr(rows, cols, ws);
  }

  get nnz() { return this.data.length; }

  // ── Factory helpers ────────────────────────────────────────────────────────

  /** k-nearest neighbour graph from Float64Array positions (n_nodes × dim) */
  static fromPositions(positions, n_nodes, dim, k = 8, curvature = 0) {
    const rows = [], cols = [], ws = [];
    for (let i = 0; i < n_nodes; i++) {
      // Compute distances to all other nodes
      const dists = [];
      for (let j = 0; j < n_nodes; j++) {
        if (i === j) continue;
        let sq = 0;
        for (let d = 0; d < dim; d++) {
          const diff = positions[i * dim + d] - positions[j * dim + d];
          sq += diff * diff;
        }
        let dist = Math.sqrt(sq);
        // Apply curvature correction
        if (curvature > 0)  dist = Math.asin(Math.min(dist / (1/Math.sqrt(curvature)), 1)) / Math.sqrt(curvature);
        if (curvature < 0)  dist = Math.asinh(dist * Math.sqrt(-curvature)) / Math.sqrt(-curvature);
        dists.push({ j, dist });
      }
      dists.sort((a, b) => a.dist - b.dist);
      for (let ki = 0; ki < Math.min(k, dists.length); ki++) {
        rows.push(i); cols.push(dists[ki].j); ws.push(dists[ki].dist);
      }
    }
    return new GeodesicTensor(n_nodes, rows, cols, ws);
  }

  toJSON() {
    return { "@tensor": "geodesic", "@shape": [this.n_nodes, this.n_nodes],
             "@sparse": true, nnz: this.nnz,
             data: Array.from(this.data),
             indices: Array.from(this.indices),
             indptr: Array.from(this.indptr) };
  }
}
