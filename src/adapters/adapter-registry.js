// Adapter registry — maps domain names to .scxq2 shard paths.
// LoRA delta weights (rank-8, ~295K params each) load alongside the frozen base model.
// Base: final_3way.scxq2 (52.95M params, INT4, 23.9MB)

export const ADAPTER_REGISTRY = {
  agents:     'agents_adapter.scxq2',
  commands:   'commands_adapter.scxq2',
  micronauts: 'micronauts_adapter.scxq2',
  tools:      'tools_adapter.scxq2',
};

export const BASE_MODEL = {
  path:   'final_3way.scxq2',
  params: 52_950_000,
  step:   27833,
  loss:   0.0003,
  sizeMB: 23.9,
  hash:   '0d56845acd26ebfa',
  dtype:  'INT4',
};

export const ADAPTER_CONFIG = {
  rank:    8,
  params:  295_000,
  sizeMB:  7.8,
  dtype:   'INT4',
  steps:   2000,
  lr:      1e-5,
};

// Resolve adapter path relative to a base directory
export function resolveAdapter(domain, baseDir = '') {
  const filename = ADAPTER_REGISTRY[domain];
  if (!filename) throw new Error(`Unknown adapter domain: ${domain}`);
  return baseDir ? `${baseDir}/${filename}` : filename;
}

export function listAdapters() {
  return Object.keys(ADAPTER_REGISTRY);
}

export default ADAPTER_REGISTRY;
