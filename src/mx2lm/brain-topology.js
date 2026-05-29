// MX2LM Brain Topology — 7-Organ Geometric Brain Architecture
//
// A conventional model stores intelligence as: weights, tensors, matrices, kv-cache, embeddings.
// The MX2LM brain stores it as geometric organs — each SVG-3D shell IS a brain organ.
// The SVG is the projection. The geometry is the substrate.

// ─── 7-Brain organ registry ───────────────────────────────────────────────────

export const BRAIN_ORGANS = Object.freeze([
  Object.freeze({
    id:         1,
    name:       'Vocabulary Brain',
    shell:      'orbital_halo',
    organ:      'vocabulary_cortex',
    desc:       'Token frequency, embedding distribution, semantic clustering',
    nodeSchema: Object.freeze({ token:true, freq:true, embedding:true, entropy:true, mergeAffinity:true }),
    missing:    false,
  }),
  Object.freeze({
    id:         2,
    name:       'Weight Brain',
    shell:      'stack_grid',
    organ:      'weight_matrix',
    desc:       'Transformer layer weights, attention heads, FFN tensors',
    nodeSchema: Object.freeze({ layer:true, head:true, norm:true, entropy:true, gradient:true }),
    missing:    false,
  }),
  Object.freeze({
    id:         3,
    name:       'Inference Brain',
    shell:      'tunnel_rail',
    organ:      'inference_stream',
    desc:       'Live thought flow — token probabilities, KV-cache, attention scores',
    nodeSchema: Object.freeze({ token:true, probability:true, next:true, energy:true }),
    missing:    false,
  }),
  Object.freeze({
    id:         4,
    name:       'Language Brain',
    shell:      'fractal_tree',
    organ:      'language_tree',
    desc:       'BPE merge lineage — semantic evolution of the tokenizer',
    nodeSchema: Object.freeze({ merge:true, count:true, depth:true, left:true, right:true }),
    missing:    false,
  }),
  Object.freeze({
    id:         5,
    name:       'Executive Brain',
    shell:      'hud_ring',
    organ:      'executive_cortex',
    desc:       'Runtime health, shard load, memory pressure, RLHF state',
    nodeSchema: Object.freeze({ shard:true, load:true, status:true, entropy:true }),
    missing:    false,
  }),
  Object.freeze({
    id:         6,
    name:       'Memory Brain',
    shell:      'memory_constellation',
    organ:      'episodic_memory',
    desc:       'ASX RAM — association graph of memories by strength, age, semantic link',
    nodeSchema: Object.freeze({ id:true, text:true, strength:true, age:true, links:true }),
    missing:    true,  // not yet implemented in the main shell renderer
  }),
  Object.freeze({
    id:         7,
    name:       'Micronaut Brain',
    shell:      'agent_mesh',
    organ:      'micronaut_nervous_system',
    desc:       'Agent neural mesh — Micronauts as neurons with goals, tools, memory refs',
    nodeSchema: Object.freeze({ id:true, goal:true, tools:true, memoryRefs:true, weight:true }),
    missing:    true,  // not yet implemented in the main shell renderer
  }),
]);

export const BRAIN_ORGAN_COUNT = BRAIN_ORGANS.length;

// ─── Shell → organ lookup ─────────────────────────────────────────────────────

export const SHELL_TO_ORGAN = Object.freeze(
  Object.fromEntries(BRAIN_ORGANS.map(o => [o.shell, o]))
);

export const ORGAN_TO_SHELL = Object.freeze(
  Object.fromEntries(BRAIN_ORGANS.map(o => [o.organ, o.shell]))
);

// ─── Compute unit → brain organ mapping (DirectWrite execution) ───────────────
// Each canvas compute kernel maps to a brain organ.

export const COMPUTE_UNIT_TO_ORGAN = Object.freeze({
  orbitalCompute: 'vocabulary_cortex',
  stackCompute:   'weight_matrix',
  tunnelCompute:  'inference_stream',
  treeCompute:    'language_tree',
  hudCompute:     'executive_cortex',
});

// ─── Polygon → kernel mapping (the critical shift: geometry = execution) ──────

export const POLYGON_KERNEL_MAP = Object.freeze({
  halo_rings:    'attention_ring_compute     — halo rings ARE wave propagation lanes',
  stack_blocks:  'layer_dispatch_units       — stack blocks ARE compute dispatch units',
  tunnel_packets:'token_batch_pipeline       — tunnel packets ARE token batch pipelines',
  fractal_nodes: 'recursive_kernel_dispatch  — fractal nodes ARE recursive kernel calls',
  hud_segments:  'residency_controllers      — HUD segments ARE residency controllers',
});

// ─── Brain organ helpers ──────────────────────────────────────────────────────

export function getBrainOrgan(id) {
  return BRAIN_ORGANS.find(o => o.id === id) || null;
}

export function getOrganByShell(shell) {
  return SHELL_TO_ORGAN[shell] || null;
}

export function getMissingOrgans() {
  return BRAIN_ORGANS.filter(o => o.missing);
}

export function getImplementedOrgans() {
  return BRAIN_ORGANS.filter(o => !o.missing);
}

// ─── Brain coherence score ────────────────────────────────────────────────────
// Returns 0–1 based on how many organs are implemented.

export function brainCoherence() {
  const implemented = getImplementedOrgans().length;
  return implemented / BRAIN_ORGAN_COUNT;
}
