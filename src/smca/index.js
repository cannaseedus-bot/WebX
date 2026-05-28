// SMCA — Structural Manifold Cluster Architecture axioms (v1.0.0-PowerShell-LLM)
//
// SMCA defines a 6-layer authority gradient where control flows downward only.
// IDB is append-only / hash-addressed / causally ordered.

export const SMCA_LAYERS = Object.freeze([
  'MATRIX',   // layer 0 — top-level authority
  'CM-1',     // layer 1 — control manifold
  'SCXQ7',    // layer 2 — high-precision quant runtime
  'SCXQ2',    // layer 3 — INT4 quant runtime
  'SCO-1',    // layer 4 — output / settlement
  'IDB',      // layer 5 — immutable distributed base
]);

export const SMCA_AUTHORITY_GRADIENT = Object.freeze({
  order: ['MATRIX', 'CM-1', 'SCXQ7', 'SCXQ2', 'SCO-1', 'IDB'],
  rules: Object.freeze([
    'Authority flows downward only.',
    'No layer may escalate authority upward.',
    'No layer may skip a layer.',
    'No collapse without law.',
  ]),
});

export const SMCA_CLUSTER_ROLES = Object.freeze([
  'Compute',    // active inference / training
  'Proof',      // verification / hash validation
  'Settlement', // commit / checkpoint finalization
  'Ledger',     // append-only event record
  'Relay',      // inter-cluster message passing
]);

export const SMCA_COLLAPSE_CLASSES = Object.freeze([
  'FOLD',       // phase transition — entropy-driven
  'CHAIN',      // sequential dependency collapse
  'MESH',       // peer-validated multi-path collapse
  'STAR',       // hub-and-spoke aggregation
  'TREE',       // hierarchical reduction
  'RING',       // circular pressure propagation
]);

export const IDB_PROPERTIES = Object.freeze({
  append_only:      true,
  hash_addressed:   true,
  causally_ordered: true,
  no_rollback:      true,
  description: 'Immutable Distributed Base — events are addressed by content hash; causal order is enforced at write time; no deletion or amendment is possible.',
});

// Kernel classes forbidden from operating above SCXQ2 layer
export const KXC_FORBID_LIST = Object.freeze([
  'kernel-above-scxq2',       // compute kernels may not escalate past layer 3
  'uav-alias-in-backward',    // SRV/UAV aliasing produces NaN in D3D11 backward pass
  'unlawful-collapse',        // collapse without registered authority is forbidden
]);

export const SMCA_VERSION = '1.0.0';
