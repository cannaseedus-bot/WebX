// xjsl-engine.js — XJSL Runtime: XJSON IS the shader language
//
// No separation. XJSON nodes can execute GPU code.
// Every GPU kernel is a string literal in JSON.
// State changes trigger GPU compute.
//
// Core truth:
//   "@shaders"  = shader definitions (JSON + WGSL kernel strings)
//   "@app"      = reactive component tree
//   "@state"    = shared mutable state watched by shaders
//   "@bind"     = maps state paths to shader buffer inputs
//   "@onComplete" = state update actions after GPU dispatch
//
// Architecture:
//   XJSLEngine         top-level: owns state + shader registry
//   XJSLReactiveShader watches state, auto-dispatches bound shaders
//   XJSLD3D11Pipeline  (from xjsl-d3d11.js) — compiles + executes

import { XJSLD3D11Pipeline } from './xjsl-d3d11.js';

// ─── XJSLEngine ───────────────────────────────────────────────────────────────

export class XJSLEngine {
  constructor() {
    this._pipeline = new XJSLD3D11Pipeline();
    this._state    = {};
    this._watchers = new Map(); // path -> Set<fn>
    this._state    = new Proxy(this._state, {
      set: (target, prop, value) => {
        target[prop] = value;
        this._notify(String(prop), value);
        return true;
      }
    });
  }

  async init() { await this._pipeline.init(); return this; }

  get state() { return this._state; }

  // ── Shader registry ──

  async compileShader(def) {
    return this._pipeline.compileShader(def);
  }

  async dispatchShader(name, bindings, dispatchSizes, uniforms = {}) {
    const inputs    = {};
    const outSpec   = {};
    const def       = this._pipeline._compiled.get(name)?.def;
    if (!def) throw new Error(`XJSLEngine: shader not found: ${name}`);

    for (const [k, v] of Object.entries(bindings)) {
      if (k === 'uniforms') continue;
      if (k in (def['@inputs'] ?? {})) inputs[k] = v instanceof Float32Array ? v : new Float32Array(v?.length ?? 0);
      if (k in (def['@outputs'] ?? {})) outSpec[k] = v instanceof Float32Array ? v : new Float32Array(v?.length ?? 0);
    }

    return this._pipeline.executeShader(name, inputs, outSpec, { x: dispatchSizes[0], y: dispatchSizes[1], z: dispatchSizes[2] }, bindings.uniforms ?? uniforms);
  }

  // ── State ──

  setState(path, value) {
    const parts = path.split('.');
    let cur = this._state;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!cur[parts[i]]) cur[parts[i]] = {};
      cur = cur[parts[i]];
    }
    cur[parts[parts.length - 1]] = value;
    this._notify(path, value);
  }

  getState(path) {
    return path.split('.').reduce((o, k) => o?.[k], this._state);
  }

  watch(path, fn) {
    if (!this._watchers.has(path)) this._watchers.set(path, new Set());
    this._watchers.get(path).add(fn);
    return () => this._watchers.get(path)?.delete(fn);
  }

  _notify(path, value) {
    for (const [k, watchers] of this._watchers) {
      if (path === k || path.startsWith(k + '.'))
        for (const fn of watchers) try { fn(value, path); } catch (_) {}
    }
  }

  // ── XJSON app loader ──

  async loadXJSON(def) {
    // Register shaders
    for (const [name, shaderDef] of Object.entries(def['@shaders'] ?? {}))
      await this.compileShader({ name, ...shaderDef });

    // Initialize state from @app/@state
    const appState = def['@app']?.['@state'] ?? {};
    for (const [k, spec] of Object.entries(appState)) {
      if (spec['@init'] === 'random') {
        const shape = spec['@shape'] ?? [64, 64];
        const sz = shape.reduce((a,b) => a*b, 1);
        this.setState(k, new Float32Array(sz).map(() => Math.random() * 2 - 1));
      } else if (spec['@init'] === 'zeros') {
        const shape = spec['@shape'] ?? [64, 64];
        this.setState(k, new Float32Array(shape.reduce((a,b)=>a*b,1)));
      }
    }

    // Mount @children ComputePipeline nodes
    for (const child of def['@app']?.['@children'] ?? []) {
      if (child['@node'] === 'ComputePipeline') await this._mountComputePipeline(child);
    }
  }

  async _mountComputePipeline(node) {
    const shaderName = node['@shader'];
    const dispatch   = node['@dispatch'] ?? [1, 1, 1];

    const resolve = (v) => {
      if (typeof v !== 'string') return v;
      return v.replace(/\{\{\s*([^}]+)\s*\}\}/g, (_, p) => this.getState(p.trim()) ?? '');
    };

    const run = async () => {
      const bindings = {};
      for (const [k, v] of Object.entries(node['@bind'] ?? {}))
        bindings[k] = typeof v === 'string' ? resolve(v) : v;

      const results = await this.dispatchShader(shaderName, bindings, dispatch);

      const onComplete = node['@onComplete'] ?? {};
      for (const [action, spec] of Object.entries(onComplete)) {
        if (action === '@set') {
          for (const [target, src] of Object.entries(spec))
            this.setState(target, results[src.split('.').pop()] ?? results[Object.keys(results)[0]]);
        }
      }
    };

    await run();

    // Watch bound state for reactivity
    for (const v of Object.values(node['@bind'] ?? {})) {
      if (typeof v === 'string' && v.includes('state.'))
        this.watch(v.replace(/.*state\./,'state.').replace(/[^a-z_.]/gi,''), run);
    }
  }

  // ── Template interpolation ──

  interpolate(template, scope = {}) {
    if (typeof template !== 'string') return template;
    return template.replace(/\{\{\s*([^}]+)\s*\}\}/g, (_, expr) => {
      const parts = expr.trim().split('.');
      let val = { state: this._state, ...scope };
      for (const p of parts) val = val?.[p];
      return val ?? '';
    });
  }
}

// ─── XJSLReactiveShader ───────────────────────────────────────────────────────

export class XJSLReactiveShader {
  constructor(engine) {
    this._engine = engine;
    this._bindings = [];
  }

  bind(sourcePath, shaderName, outputPath, inputMapping) {
    const unsub = this._engine.watch(sourcePath, async (value) => {
      const inputs = {};
      for (const [inputName, inputPath] of Object.entries(inputMapping)) {
        if (inputName === 'uniforms') { inputs.uniforms = inputPath; continue; }
        inputs[inputName] = this._engine.getState(inputPath) ?? value;
      }
      const dispatch = inputMapping['@dispatch'] ?? [64, 1, 1];
      const results  = await this._engine.dispatchShader(shaderName, inputs, dispatch);
      const outKey   = outputPath.split('.').pop();
      const outVal   = results[outKey] ?? results[Object.keys(results)[0]];
      if (outVal !== undefined) this._engine.setState(outputPath, outVal);
    });
    this._bindings.push(unsub);
    return this;
  }

  dispose() { for (const u of this._bindings) u(); this._bindings = []; }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

export async function createXJSLEngine() {
  const engine = new XJSLEngine();
  await engine.init();
  return engine;
}
