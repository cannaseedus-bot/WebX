// sk-schema-builder.js — JS port of Microsoft Semantic Kernel's KernelJsonSchemaBuilder
//
// Original: semantic_kernel/schema/kernel_json_schema_builder.py
// Ported to ES module for use in XCFE tool/agent/micronaut auto-registration.
//
// Usage:
//   import { KernelJsonSchemaBuilder, kernelFunction } from './sk-schema-builder.js';
//
//   class KuhulTools {
//     @kernelFunction("Dispatch token IDs through D3D12 MoE GPU pipeline")
//     gpu_dispatch({ token_ids, addon_type = null }) { ... }
//   }
//
//   const schema = KernelJsonSchemaBuilder.buildFromClass(KuhulTools);
//   // → full TOOLS-dict-compatible JSON Schema for every @kernelFunction method
//
// XCFE integration:
//   Every micronaut (SHELL-1, GP-1, FLEET-1, etc.) and every DXSK tool
//   can self-describe via buildFromClass() instead of hand-writing TOOLS entries.
//   The @kernelFunction decorator is the XCFE @function namespace equivalent.

// ─── TYPE_MAPPING (mirrors Python SK) ────────────────────────────────────────

const TYPE_MAPPING = new Map([
  ['number',   'number'],
  ['integer',  'integer'],
  ['int',      'integer'],
  ['string',   'string'],
  ['str',      'string'],
  ['boolean',  'boolean'],
  ['bool',     'boolean'],
  ['array',    'array'],
  ['list',     'array'],
  ['object',   'object'],
  ['dict',     'object'],
  ['null',     'null'],
  ['any',      'object'],
  // JS primitives
  [Number,     'number'],
  [String,     'string'],
  [Boolean,    'boolean'],
  [Array,      'array'],
  [Object,     'object'],
]);

// ─── Type descriptor DSL ──────────────────────────────────────────────────────
// Use these in @kernelFunction param specs instead of raw strings.

export const t = {
  int:     ()      => ({ type: 'integer' }),
  float:   ()      => ({ type: 'number' }),
  str:     ()      => ({ type: 'string' }),
  bool:    ()      => ({ type: 'boolean' }),
  list:    (item)  => ({ type: 'array',  items: item ?? {} }),
  dict:    (val)   => ({ type: 'object', additionalProperties: val ?? {} }),
  optional:(inner) => ({ anyOf: [inner, { type: 'null' }] }),
  enum:    (...vals)=>({ type: typeof vals[0], enum: vals }),
  union:   (...ts) => ({ anyOf: ts }),
  obj:     (props, required) => ({
    type: 'object',
    properties: props,
    ...(required ? { required } : {}),
  }),
};

// ─── KernelJsonSchemaBuilder ──────────────────────────────────────────────────

export class KernelJsonSchemaBuilder {

  // Build a JSON Schema entry from a type descriptor or primitive
  static build(paramType, description = null, structuredOutput = false) {
    if (!paramType) return { type: 'object' };

    let schema;

    if (typeof paramType === 'string') {
      schema = this.buildFromTypeName(paramType);
    } else if (paramType && typeof paramType === 'object' && !Array.isArray(paramType)) {
      // Already a schema object (from t.* helpers or hand-written)
      schema = { ...paramType };
    } else if (paramType === Number)  { schema = { type: 'number' }; }
    else if (paramType === String)    { schema = { type: 'string' }; }
    else if (paramType === Boolean)   { schema = { type: 'boolean' }; }
    else if (paramType === Array)     { schema = { type: 'array' }; }
    else if (paramType === Object)    { schema = { type: 'object' }; }
    else { schema = { type: 'object' }; }

    if (description) schema.description = description;
    if (structuredOutput && schema.type === 'object') schema.additionalProperties = false;
    return schema;
  }

  static buildFromTypeName(name) {
    if (name.includes('|')) {
      // Union: "string|null", "integer|string"
      const parts = name.split('|').map(s => s.trim());
      if (parts.length === 2 && parts.includes('null')) {
        const nonNull = parts.find(p => p !== 'null');
        return { type: [TYPE_MAPPING.get(nonNull) ?? 'object', 'null'] };
      }
      return { anyOf: parts.map(p => ({ type: TYPE_MAPPING.get(p) ?? 'object' })) };
    }
    if (name.startsWith('list[') || name.startsWith('array[')) {
      const inner = name.slice(name.indexOf('[') + 1, -1);
      return { type: 'array', items: this.buildFromTypeName(inner) };
    }
    if (name.startsWith('optional[')) {
      const inner = name.slice(9, -1);
      return { anyOf: [this.buildFromTypeName(inner), { type: 'null' }] };
    }
    return { type: TYPE_MAPPING.get(name) ?? 'object' };
  }

  // Build a full parameters schema from a param-spec object
  // paramSpec: { paramName: { type, description, required?, default? }, ... }
  static buildParameters(paramSpec) {
    if (!paramSpec || Object.keys(paramSpec).length === 0) {
      return { type: 'object', properties: {}, required: [] };
    }
    const properties = {};
    const required   = [];
    for (const [name, spec] of Object.entries(paramSpec)) {
      const typeDesc = spec.type ?? spec;
      properties[name] = this.build(typeDesc, spec.description ?? null);
      if (spec.required !== false && spec.default === undefined) {
        required.push(name);
      }
    }
    return { type: 'object', properties, ...(required.length ? { required } : {}) };
  }

  // Build full TOOLS-dict-compatible entry from a class with @kernelFunction methods
  static buildFromClass(cls) {
    const tools = {};
    const proto  = cls.prototype ?? cls;
    const methods = Object.getOwnPropertyNames(proto)
      .filter(k => k !== 'constructor' && typeof proto[k] === 'function');

    for (const method of methods) {
      const meta = proto[method]?.__kernelFunction;
      if (!meta) continue;
      tools[method] = {
        name:        method,
        description: meta.description ?? `${cls.name}.${method}`,
        parameters:  this.buildParameters(meta.params ?? {}),
      };
    }
    return tools;
  }

  // Build from a plain spec object (for micronauts defined as data, not classes)
  // spec: { name, description, params: { paramName: { type, description, required? } } }
  static buildFromSpec(spec) {
    return {
      name:        spec.name,
      description: spec.description ?? '',
      parameters:  this.buildParameters(spec.params ?? {}),
    };
  }

  // Build ALL tools from a list of specs (replaces hand-written TOOLS dict)
  static buildToolsDict(specs) {
    const tools = {};
    for (const spec of specs) {
      tools[spec.name] = this.buildFromSpec(spec);
    }
    return tools;
  }
}

// ─── @kernelFunction decorator ────────────────────────────────────────────────
// Attach metadata to a method so buildFromClass() can discover it.
//
// Usage (with TC39 stage-3 decorators):
//   @kernelFunction("description", { token_ids: { type: t.list(t.int()), required: true } })
//   gpu_dispatch(args) { ... }
//
// Usage (manual, no decorator syntax):
//   MyClass.prototype.gpu_dispatch.__kernelFunction = { description: "...", params: {...} }

export function kernelFunction(description, params = {}) {
  return function(target, context) {
    // TC39 stage-3 decorator
    if (context && context.kind === 'method') {
      context.addInitializer(function() {
        this[context.name].__kernelFunction = { description, params };
      });
      return target;
    }
    // Legacy decorator (Babel / TypeScript experimentalDecorators)
    if (typeof target === 'object' && typeof context === 'string') {
      target[context].__kernelFunction = { description, params };
      return;
    }
    // Direct function decoration
    if (typeof target === 'function') {
      target.__kernelFunction = { description, params };
      return target;
    }
  };
}

// Attach metadata without decorator syntax (for environments without decorator support)
export function registerKernelFunction(fn, description, params = {}) {
  fn.__kernelFunction = { description, params };
  return fn;
}

// ─── XCFE micronaut auto-registration ────────────────────────────────────────
//
// Given a micronaut spec object (from KUHUL registry), builds the full
// tool definition that toolcall_data.py TOOLS dict would contain.

export function micronautToTool(micronautId, port, skills = []) {
  return KernelJsonSchemaBuilder.buildFromSpec({
    name:        'micronaut_dispatch',
    description: `Dispatch a task to micronaut ${micronautId} at port ${port}. Skills: ${skills.join(', ') || 'none'}.`,
    params: {
      micronaut: { type: t.str(),               description: `Micronaut ID (${micronautId})`, required: true },
      action:    { type: t.str(),               description: 'Action to perform',              required: true },
      args:      { type: t.dict(),              description: 'Action arguments',               required: false },
    },
  });
}

// ─── Build KUHUL tool manifest from the canonical registry CSV ───────────────
// registry: [{id, endpoint, skills}]  (from registry_micronauts.csv)
export function buildMicronautManifest(registry) {
  return registry.map(row => {
    const port   = (row.endpoint ?? '').match(/:(\d+)\/|:(\d+)$/)?.[1] ?? (row.endpoint ?? '').match(/\d+/)?.[0] ?? '0';
    const skills = (row.skills ?? '').split(';').filter(Boolean);
    return micronautToTool(row.id, port, skills);
  });
}
