// ASX RAM Schema suite v1 — frozen JS constants (JSON Schema draft-2020-12)
// Includes: asx_ram, pi_mutation, css_projection schemas + flux_gate_policy.

// ─── ASX RAM Schema v1 ────────────────────────────────────────────────────────

export const ASX_RAM_SCHEMA = Object.freeze({
  $schema:  'https://json-schema.org/draft/2020-12/schema',
  $id:      'asx://schema/ram/asx_ram.v1.schema.json',
  title:    'ASX RAM Schema v1',
  type:     'object',
  additionalProperties: false,
  required: [
    '@context', '@id', '@v',
    '@session', '@tick', '@control',
    '@state', '@pi', '@clusters',
    '@events', '@proof', '@projection',
  ],
  properties: Object.freeze({
    '@context': { const: 'asx://ram/schema/v1' },
    '@id':      { type: 'string', minLength: 1 },
    '@v':       { const: '1.0.0' },

    '@session': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@sid', '@boot_ts', '@tz', '@shard', '@mode', '@build', '@entropy'],
      properties: Object.freeze({
        '@sid':     { type: 'string',  minLength: 6 },
        '@boot_ts': { type: 'integer' },
        '@tz':      { type: 'string',  minLength: 3 },
        '@shard':   { type: 'string',  minLength: 1 },
        '@mode':    { type: 'string',  minLength: 1 },
        '@build':   { type: 'string',  minLength: 1 },
        '@entropy': { type: 'number',  minimum: 0, maximum: 1 },
      }),
    }),

    '@tick': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@n', '@ts', '@phase', '@barriers', '@monotonic_ok'],
      properties: Object.freeze({
        '@n':            { type: 'integer', minimum: 0 },
        '@ts':           { type: 'integer' },
        '@phase':        { type: 'string', enum: ['perceive','represent','reason','decide','act','reflect','collapse'] },
        '@barriers':     { type: 'array',  items: { type: 'string', minLength: 1 } },
        '@monotonic_ok': { type: 'boolean' },
      }),
    }),

    '@control': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@flow', '@allowlist', '@denylist', '@limits'],
      properties: Object.freeze({
        '@flow':      { type: 'string', minLength: 1 },
        '@allowlist': { type: 'array', items: { type: 'string' } },
        '@denylist':  { type: 'array', items: { type: 'string' } },
        '@limits':    Object.freeze({
          type: 'object',
          additionalProperties: false,
          required: ['@max_ops_per_tick', '@max_mutations_per_tick', '@max_payload_bytes'],
          properties: Object.freeze({
            '@max_ops_per_tick':       { type: 'integer', minimum: 1 },
            '@max_mutations_per_tick': { type: 'integer', minimum: 1 },
            '@max_payload_bytes':      { type: 'integer', minimum: 1024 },
          }),
        }),
      }),
    }),

    '@state': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@atoms', '@world', '@ui', '@agents', '@net', '@security'],
      properties: Object.freeze({
        '@atoms':    { type: 'object' },
        '@world':    { type: 'object' },
        '@ui':       { type: 'object' },
        '@agents':   { type: 'object' },
        '@net':      { type: 'object' },
        '@security': { type: 'object' },
      }),
    }),

    '@pi': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@vars', '@signals', '@integrators', '@rng'],
      properties: Object.freeze({
        '@vars':        { type: 'object' },
        '@signals':     { type: 'object' },
        '@integrators': { type: 'object' },
        '@rng': Object.freeze({
          type: 'object',
          additionalProperties: false,
          required: ['@mode', '@seed', '@cursor'],
          properties: Object.freeze({
            '@mode':   { type: 'string', enum: ['deterministic'] },
            '@seed':   { type: ['string', 'integer'] },
            '@cursor': { type: 'integer', minimum: 0 },
          }),
        }),
      }),
    }),

    '@clusters': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@active', '@votes', '@collapse'],
      properties: Object.freeze({
        '@active': { type: 'object' },
        '@votes':  { type: 'object' },
        '@collapse': Object.freeze({
          type: 'object',
          additionalProperties: false,
          required: ['@result', '@confidence', '@method'],
          properties: Object.freeze({
            '@result':     { type: ['object', 'null'] },
            '@confidence': { type: 'number', minimum: 0, maximum: 1 },
            '@method':     { type: 'string', minLength: 1 },
          }),
        }),
      }),
    }),

    '@events': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@queue', '@last'],
      properties: Object.freeze({
        '@queue': { type: 'array', items: { type: 'object' } },
        '@last':  { type: ['object', 'null'] },
      }),
    }),

    '@proof': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@prev_hash', '@tick_hash', '@state_hash', '@emitted'],
      properties: Object.freeze({
        '@prev_hash':  { type: 'string' },
        '@tick_hash':  { type: 'string' },
        '@state_hash': { type: 'string' },
        '@emitted':    { type: 'array', items: { type: 'object' } },
      }),
    }),

    '@projection': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@css', '@dom'],
      properties: Object.freeze({
        '@css': Object.freeze({
          type: 'object',
          additionalProperties: false,
          required: ['@root_vars', '@classes', '@dataset'],
          properties: Object.freeze({
            '@root_vars': { type: 'object', propertyNames: { pattern: '^--[a-z0-9\\-]+$' }, additionalProperties: { type: ['string', 'number', 'integer'] } },
            '@classes':   { type: 'array', items: { type: 'string' } },
            '@dataset':   { type: 'object', propertyNames: { pattern: '^(data\\-[a-z0-9\\-]+)$' }, additionalProperties: { type: 'string' } },
          }),
        }),
        '@dom': Object.freeze({
          type: 'object',
          additionalProperties: false,
          required: ['@diff', '@patch'],
          properties: Object.freeze({
            '@diff':  { type: 'array' },
            '@patch': { type: 'array' },
          }),
        }),
      }),
    }),
  }),
});

// ─── π Mutation Patch Schema v1 ───────────────────────────────────────────────

export const PI_MUTATION_SCHEMA = Object.freeze({
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id:     'asx://schema/ram/pi_mutation.v1.schema.json',
  title:   'ASX π Mutation Patch v1',
  type:    'object',
  additionalProperties: false,
  required: ['@context', '@tick', '@phase', '@ops', '@proof'],
  properties: Object.freeze({
    '@context': { const: 'asx://ram/pi_mutation/v1' },
    '@tick':    { type: 'integer', minimum: 0 },
    '@phase':   { type: 'string', enum: ['perceive','represent','reason','decide','act','reflect','collapse'] },
    '@ops':     { type: 'array', minItems: 1 },
    '@proof': Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@input_hash', '@mut_hash'],
      properties: Object.freeze({
        '@input_hash': { type: 'string', minLength: 16 },
        '@mut_hash':   { type: 'string', minLength: 16 },
      }),
    }),
  }),
});

// ─── CSS Projection Bundle Schema v1 ─────────────────────────────────────────

export const CSS_PROJECTION_SCHEMA = Object.freeze({
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id:     'asx://schema/projection/css_projection.v1.schema.json',
  title:   'ASX CSS Projection Bundle v1',
  type:    'object',
  additionalProperties: false,
  required: ['@context', '@tick', '@root', '@classes', '@dataset'],
  properties: Object.freeze({
    '@context':  { const: 'asx://projection/css/v1' },
    '@tick':     { type: 'integer', minimum: 0 },
    '@root':     { type: 'object', propertyNames: { pattern: '^--[a-z0-9\\-]+$' }, additionalProperties: { type: ['string', 'number', 'integer'] } },
    '@classes':  { type: 'array', items: { type: 'string' } },
    '@dataset':  { type: 'object', propertyNames: { pattern: '^(data\\-[a-z0-9\\-]+)$' }, additionalProperties: { type: 'string' } },
    '@proof':    Object.freeze({
      type: 'object',
      additionalProperties: false,
      required: ['@projection_hash'],
      properties: Object.freeze({
        '@projection_hash': { type: 'string', minLength: 16 },
      }),
    }),
  }),
});

// ─── Flux Gate Policy v1 ──────────────────────────────────────────────────────
// Phase op allowlists + path constraints for π mutation gate enforcement.

export const FLUX_GATE_POLICY = Object.freeze({
  '@context': 'asx://policy/flux_gate/v1',
  '@id':      'flux_gate_policy_asx_ram_v1',
  '@v':       '1.0.0',

  '@phases': Object.freeze({
    perceive:  Object.freeze({ '@allow_ops': ['emit'],                                                                      '@allow_prefixes': ['@events'],                                                            '@deny_prefixes': ['@state', '@security', '@projection', '@proof'] }),
    represent: Object.freeze({ '@allow_ops': ['set', 'merge', 'emit'],                                                     '@allow_prefixes': ['@pi', '@clusters', '@events'],                                        '@deny_prefixes': ['@security'] }),
    reason:    Object.freeze({ '@allow_ops': ['set','merge','inc','dec','add','pop','clamp','swap','emit'],                 '@allow_prefixes': ['@pi', '@clusters', '@events', '@state.@agents'],                      '@deny_prefixes': ['@security'] }),
    decide:    Object.freeze({ '@allow_ops': ['set', 'merge', 'emit'],                                                     '@allow_prefixes': ['@clusters.@collapse', '@events', '@state.@ui', '@state.@agents'],     '@deny_prefixes': ['@security'] }),
    act:       Object.freeze({ '@allow_ops': ['set','merge','del','inc','dec','add','pop','clamp','swap','emit'],           '@allow_prefixes': ['@state.@world', '@state.@ui', '@events', '@pi.@rng'],                  '@deny_prefixes': ['@security', '@proof'] }),
    reflect:   Object.freeze({ '@allow_ops': ['set', 'merge'],                                                            '@allow_prefixes': ['@proof', '@projection'],                                               '@deny_prefixes': ['@security', '@state.@world'] }),
    collapse:  Object.freeze({ '@allow_ops': ['set', 'merge'],                                                            '@allow_prefixes': ['@proof', '@projection', '@clusters.@collapse'],                         '@deny_prefixes': ['@security', '@state'] }),
  }),

  '@global': Object.freeze({
    '@max_ops_per_tick':       4096,
    '@max_mutations_per_tick': 2048,
    '@path_syntax':            'dotpath(@a.@b.@c) with @-segments only',
    '@deny_prefixes':          ['@session'],
    '@hard_deny_ops':          [],
  }),
});

// ─── Tick phase ordered sequence ─────────────────────────────────────────────

export const TICK_PHASES = Object.freeze([
  'perceive', 'represent', 'reason', 'decide', 'act', 'reflect', 'collapse',
]);

// ─── Canonical RAM template (empty/boot state) ───────────────────────────────

export function createEmptyRam(opts = {}) {
  const now = Date.now();
  return {
    '@context': 'asx://ram/schema/v1',
    '@id':      opts.id      || 'asx_ram',
    '@v':       '1.0.0',
    '@session': {
      '@sid':     opts.sid    || `sid_${now}`,
      '@boot_ts': now,
      '@tz':      opts.tz     || 'UTC',
      '@shard':   opts.shard  || 'prime',
      '@mode':    opts.mode   || 'micro-asxr',
      '@build':   opts.build  || 'dev',
      '@entropy': opts.entropy ?? 0.0,
    },
    '@tick': {
      '@n':            0,
      '@ts':           now,
      '@phase':        'perceive',
      '@barriers':     [],
      '@monotonic_ok': true,
    },
    '@control': {
      '@flow':      'PRIME_TICK',
      '@allowlist': [],
      '@denylist':  [],
      '@limits': {
        '@max_ops_per_tick':       4096,
        '@max_mutations_per_tick': 2048,
        '@max_payload_bytes':      262144,
      },
    },
    '@state':    { '@atoms': {}, '@world': {}, '@ui': {}, '@agents': {}, '@net': {}, '@security': {} },
    '@pi':       { '@vars': {}, '@signals': {}, '@integrators': {}, '@rng': { '@mode': 'deterministic', '@seed': opts.seed ?? now, '@cursor': 0 } },
    '@clusters': { '@active': {}, '@votes': {}, '@collapse': { '@result': null, '@confidence': 0, '@method': 'majority' } },
    '@events':   { '@queue': [], '@last': null },
    '@proof':    { '@prev_hash': '', '@tick_hash': '', '@state_hash': '', '@emitted': [] },
    '@projection': {
      '@css': { '@root_vars': {}, '@classes': [], '@dataset': {} },
      '@dom': { '@diff': [],     '@patch': [] },
    },
  };
}

// ─── CSS projection binding table (canonical defaults) ────────────────────────

export const CSS_BINDING_TABLE = Object.freeze([
  { ram: '@tick.@n',                        css: '--asx-tick',           dataset: 'data-asx-tick',    type: 'var+dataset' },
  { ram: '@tick.@phase',                    css: null,                   dataset: 'data-asx-phase',   type: 'class+dataset', classPrefix: 'asx-phase-' },
  { ram: '@pi.@signals.entropy',            css: '--entropy',            dataset: null,               type: 'var' },
  { ram: '@clusters.@collapse.@confidence', css: '--cluster-confidence', dataset: null,               type: 'var' },
  { ram: '@state.@ui.shop.open',            css: '--ui-shop-open',       dataset: null,               type: 'var+class', classOn: 'ui-shop-open' },
  { ram: '@state.@world.player.hp_norm',    css: '--ui-hp',             dataset: null,               type: 'var' },
  { ram: '@state.@world.player.alive',      css: null,                   dataset: null,               type: 'class', classOn: 'player-alive', classOff: 'player-dead' },
  { ram: '@session.@shard',                 css: null,                   dataset: 'data-asx-shard',   type: 'dataset' },
]);
