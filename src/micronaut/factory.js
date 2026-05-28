// Micronaut factory — authority-based instantiation contract (v0.1.0-micronaut-factory)
//
// Port of micronaut_factory.exe design contract (binary-only release; no C++ source).
// Source modules (by .obj): factory_core, factory_registry, evolution_engine,
//   mutation_tracker, python_bridge, service_manager, bigram_merger, cli_handler.
//
// Key invariant: micronauts are bound via explicit SCX command/program bindings only.
// Directory scanning is forbidden — authority requires explicit binding.

export const FACTORY_POLICY = Object.freeze({
  modular_dependency_only:                          true,
  execution_authority_requires_explicit_command_binding: true,
  directory_scan_forbidden:                         true,
  description: 'Micronauts are instantiated only through explicit SCX command/program bindings. Scanning a directory to discover micronauts is a policy violation.',
});

export const FACTORY_VERSION = '0.1.0';

export const MICRONAUT_STATUS = Object.freeze({
  UNBOUND:    'UNBOUND',    // registered but no SCX binding
  BOUND:      'BOUND',      // has a valid SCX command/program binding
  ACTIVE:     'ACTIVE',     // running instance
  MUTATING:   'MUTATING',   // evolution engine is applying a delta
  RETIRED:    'RETIRED',    // no longer active; mutation log preserved
});

// Registry entry shape — matches factory_registry.obj contract
export function createRegistryEntry(opts) {
  const { id, name, binding, version = '0.0.0' } = opts;
  if (!id)      throw new Error('MicronauntFactory: id is required');
  if (!name)    throw new Error('MicronanutFactory: name is required');
  if (!binding) throw new Error('MicronauntFactory: binding (SCX command/program) is required');
  return Object.freeze({
    id,
    name,
    binding,       // SCX command or program identifier — explicit, not path-scanned
    version,
    status: MICRONAUT_STATUS.BOUND,
    createdAt: Date.now(),
    mutationCount: 0,
  });
}

export class MicronanutRegistry {
  constructor() {
    this._entries = new Map();
  }

  register(entry) {
    if (this._entries.has(entry.id)) {
      throw new Error(`MicronauntFactory: duplicate id "${entry.id}"`);
    }
    this._entries.set(entry.id, { ...entry });
    return this;
  }

  get(id) {
    return this._entries.get(id) || null;
  }

  list() {
    return Array.from(this._entries.values());
  }

  unregister(id) {
    return this._entries.delete(id);
  }

  // Validate that no entries were discovered by directory scan (all have explicit bindings)
  validatePolicy() {
    const violations = [];
    for (const [id, entry] of this._entries) {
      if (!entry.binding) violations.push(`id="${id}" has no SCX binding`);
      if (entry.status === MICRONAUT_STATUS.UNBOUND) violations.push(`id="${id}" is UNBOUND`);
    }
    return { ok: violations.length === 0, violations };
  }
}

// Mutation log entry — matches mutation_tracker.obj contract
export function createMutationRecord(micronanutId, delta, reason = '') {
  return Object.freeze({
    micronanutId,
    delta,
    reason,
    timestamp: Date.now(),
    applied: false,
  });
}

export const DEFAULT_REGISTRY = new MicronanutRegistry();
