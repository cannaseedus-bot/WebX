// flux-ir.js — FLUX Intermediate Representation
//
// The bridge between timeless math and physical execution.
//
//   Layer 0  Math functions  — no time, no state, pure mapping
//   Layer 1  FLUX IR         — time made explicit via actions + reducers
//   Layer 2  FLUX Runtime    — lives in time (flux-runtime.js)
//   Layer 3  CPU / OS / Net  — physical reality
//
// FLUX IR constructs (one-to-one with the formal semantics):
//   pure(name, params, body)          — timeless math function
//   store(name, fields)               — state container
//   action(name, fields)              — event type
//   reduce(storeName, handlers)       — state transition (pure math function over time)
//   query(name, storeName, body)      — pure read of store state
//   effectCreator(name, params, body) — async side-effect declaration
//   view(name, storeName, render)     — reactive UI binding
//
// Operational semantics:
//   Global state: < stores:S, queue:Q, effects:E >
//   Action step:  action in Q, reducer(S,action)=S'  =>  <S', Q-action, E>
//   Effect step:  effect.done, result_action         =>  <S, Q+result_action, E-effect>
//   View step:    store changed                      =>  UI updated (side effect)
//
// K'UHUL phase mapping:
//   Pop   = dequeue + snapshot state_before
//   Wo    = route action to all store reducers
//   Sek   = reducer computes new state (pure)
//   Ch'en = commit new state + notify views + tick effects
//
// µMODEL bridge:
//   fromDescriptor(descriptor)  builds a FLUX IR program from a µPY descriptor
//   The CDATA kuhul capsule IS the reducer body.
//   SemanticReader topology folds = store declarations.
//   Geodesics = action routing weights.

// ─── IR Node constructors ─────────────────────────────────────────────────────

export function pureFn(name, params, body) {
  return { kind: 'pure', name, params, body };
}

export function storeDef(name, fields) {
  return { kind: 'store', name, fields };
}

export function actionDef(name, fields = []) {
  return { kind: 'action', name, fields };
}

export function reduceDef(storeName, handlers) {
  return { kind: 'reduce', storeName, handlers };
}

export function queryDef(name, storeName, body) {
  return { kind: 'query', name, storeName, body };
}

export function effectCreatorDef(name, params, effectType, body) {
  return { kind: 'effect_creator', name, params, effectType, body };
}

export function viewDef(name, storeNames, render) {
  return { kind: 'view', name, storeNames: [].concat(storeNames), render };
}

export function moduleDef(name, nodes) {
  return { kind: 'module', name, nodes };
}

// ─── FLUX IR Interpreter ─────────────────────────────────────────────────────
//
// Compiles a FLUX IR module into a live FluxRuntime instance.

import { FluxRuntime, PromiseEffect, TimerEffect } from './flux-runtime.js';

export class FluxIRInterpreter {
  constructor() {
    this._pureFns   = new Map();
    this._queries   = new Map();
    this._effects   = new Map();
    this._views     = new Map();
    this._runtime   = null;
  }

  compile(irModule) {
    this._runtime = new FluxRuntime({ enableTimeTravel: true });

    for (const node of irModule.nodes) {
      switch (node.kind) {
        case 'pure':           this._compilePure(node);    break;
        case 'store':          this._compileStore(node);   break;
        case 'action':         /* just a type declaration */ break;
        case 'reduce':         this._compileReduce(node);  break;
        case 'query':          this._compileQuery(node);   break;
        case 'effect_creator': this._compileEffect(node);  break;
        case 'view':           this._compileView(node);    break;
      }
    }

    return this._runtime;
  }

  _compilePure(node) {
    this._pureFns.set(node.name, (...args) => node.body(...args));
  }

  _compileStore(node) {
    const initial = {};
    for (const { name, defaultValue } of node.fields) initial[name] = defaultValue;
    // Placeholder reducer — replaced when reduce() node is processed
    this._runtime.registerStore(node.name, initial, s => s);
  }

  _compileReduce(node) {
    const handlers = node.handlers; // { [actionType]: (state, payload) => newState }
    const reducer = (state, action) => {
      const h = handlers[action.type];
      return h ? h(state, action.payload ?? action) : state;
    };
    // Replace the placeholder reducer
    const store = this._runtime._stores._stores.get(node.storeName);
    if (store) store.reducer = reducer;
  }

  _compileQuery(node) {
    this._queries.set(node.name, () => {
      const state = this._runtime.getState(node.storeName);
      return node.body(state);
    });
  }

  _compileEffect(node) {
    this._effects.set(node.name, (...args) => {
      const spec = node.body(...args);
      if (spec.type === 'promise') {
        this._runtime.addEffect(new PromiseEffect(spec.promise, spec.onSuccess, spec.onError));
      } else if (spec.type === 'timer') {
        this._runtime.addEffect(new TimerEffect(spec.delayMs, spec.onComplete));
      }
    });
  }

  _compileView(node) {
    const unsubs = node.storeNames.map(name =>
      this._runtime.subscribe(name, state => node.render(
        Object.fromEntries(node.storeNames.map(n => [n, this._runtime.getState(n)]))
      ))
    );
    this._views.set(node.name, { node, unsubscribe: () => unsubs.forEach(u => u()) });
  }

  query(name) {
    const fn = this._queries.get(name);
    if (!fn) throw new Error(`FluxIR: query '${name}' not found`);
    return fn();
  }

  call(effectName, ...args) {
    const fn = this._effects.get(effectName);
    if (!fn) throw new Error(`FluxIR: effect creator '${effectName}' not found`);
    fn(...args);
    return this;
  }

  get runtime() { return this._runtime; }
}

// ─── Code generator: FLUX IR → JavaScript ────────────────────────────────────
//
// Compiles a FLUX IR module to idiomatic JavaScript source code.
// Demonstrates the "universal translator" concept from the docs.

export function generateJS(irModule) {
  const lines = [
    `// Generated from FLUX IR module: ${irModule.name}`,
    `// Math → FLUX IR → JavaScript`,
    '',
  ];

  for (const node of irModule.nodes) {
    switch (node.kind) {
      case 'pure':
        lines.push(`const ${node.name} = (${node.params.join(', ')}) => {`, `  return ${node.body};`, `};`, '');
        break;

      case 'store': {
        const fields = node.fields.map(f => `  ${f.name}: ${JSON.stringify(f.defaultValue)}`).join(',\n');
        lines.push(`const ${node.name}_INITIAL = {\n${fields}\n};`, '');
        break;
      }

      case 'action':
        lines.push(`// Action: ${node.name}(${node.fields.map(f => f.name).join(', ')})`, '');
        break;

      case 'reduce': {
        const cases = Object.entries(node.handlers).map(([type, handler]) =>
          `    case '${type}': return (${handler.toString()})(state, action.payload ?? action);`
        ).join('\n');
        lines.push(
          `const ${node.storeName}_reducer = (state = ${node.storeName}_INITIAL, action) => {`,
          `  switch (action.type) {`,
          cases,
          `    default: return state;`,
          `  }`,
          `};`, ''
        );
        break;
      }

      case 'query':
        lines.push(`const ${node.name} = (state) => {`, `  return (${node.body.toString()})(state);`, `};`, '');
        break;

      case 'effect_creator':
        lines.push(
          `const ${node.name} = (${node.params.join(', ')}) => async (dispatch) => {`,
          `  // effectType: ${node.effectType}`,
          `  const spec = (${node.body.toString()})(${node.params.join(', ')});`,
          `  try { dispatch(await spec.promise.then(spec.onSuccess)); }`,
          `  catch (e) { dispatch(spec.onError(e)); }`,
          `};`, ''
        );
        break;

      case 'view':
        lines.push(
          `// View: ${node.name} subscribes to [${node.storeNames.join(', ')}]`,
          `// Render function: ${node.render.toString().slice(0, 80)}...`, ''
        );
        break;
    }
  }

  return lines.join('\n');
}

// ─── µMODEL → FLUX IR bridge ──────────────────────────────────────────────────
//
// A µPY descriptor IS a FLUX IR program:
//   descriptor.topology_folds     → store declarations
//   descriptor.cdata.kuhul_programs → reducer bodies (phase programs)
//   descriptor.capabilities        → action types
//   descriptor.cdata.policies      → query-time constraints
//   descriptor.cdata.semantic_grams → pure function vocabulary
//   descriptor.routing             → effect creator (async dispatch to specialist)

export function fromDescriptor(descriptor) {
  const name = descriptor.domain;
  const nodes = [];

  // 1. Pure functions from semantic grams (vocabulary)
  for (const gram of (descriptor.cdata?.semantic_grams ?? [])) {
    const fnName = gram.replace(/\./g, '_');
    nodes.push(pureFn(fnName, ['x'], `x`)); // identity placeholder
  }

  // 2. Store declarations from topology folds
  const folds = descriptor.topology_folds ?? [];
  if (folds.length === 0) {
    // Default: single domain store
    nodes.push(storeDef(name, [
      { name: 'active',   defaultValue: descriptor.activation?.active ?? false },
      { name: 'pressure', defaultValue: descriptor.activation?.pressure ?? 0.5 },
      { name: 'phase',    defaultValue: descriptor.phase ?? 'Sek' },
      { name: 'gravity',  defaultValue: descriptor.gravity ?? 'Normal' },
      { name: 'result',   defaultValue: null },
    ]));
  } else {
    for (const fold of folds) {
      nodes.push(storeDef(fold.id ?? fold, [
        { name: 'phase',   defaultValue: fold.phase   ?? descriptor.phase },
        { name: 'gravity', defaultValue: fold.gravity ?? descriptor.gravity },
        { name: 'active',  defaultValue: false },
      ]));
    }
  }

  // 3. Action types from capabilities
  for (const cap of (descriptor.capabilities ?? [])) {
    nodes.push(actionDef(`${name}_${cap}`, [{ name: 'payload' }]));
  }
  nodes.push(actionDef(`${name}_ACTIVATE`));
  nodes.push(actionDef(`${name}_DEACTIVATE`));
  nodes.push(actionDef(`${name}_RESULT`, [{ name: 'result' }]));

  // 4. Reducer — built from K'UHUL phase program in CDATA
  const storeName = folds.length > 0 ? (folds[0].id ?? folds[0]) : name;
  const handlers = {
    [`${name}_ACTIVATE`]:   (state) => ({ ...state, active: true }),
    [`${name}_DEACTIVATE`]: (state) => ({ ...state, active: false }),
    [`${name}_RESULT`]:     (state, payload) => ({ ...state, result: payload }),
  };
  // Phase transitions from K'UHUL program
  for (const prog of (descriptor.cdata?.kuhul_programs ?? [])) {
    for (const marker of (prog.phase_markers ?? [])) {
      const phaseM = marker.match(/^(Pop|Wo|Sek|Ch'en|Chen|Xul):/i);
      if (phaseM) {
        const phase = phaseM[1];
        handlers[`${name}_PHASE_${phase.toUpperCase()}`] =
          (state) => ({ ...state, phase });
      }
    }
  }
  nodes.push(reduceDef(storeName, handlers));

  // 5. Queries from policy constraints
  for (const policy of (descriptor.cdata?.policies ?? [])) {
    nodes.push(queryDef(`${name}_policy_${policy}`, storeName,
      (state) => ({ policy, active: state.active, phase: state.phase })));
  }

  // 6. Effect creator — dispatches to specialist model
  const trigger  = descriptor.routing?.trigger ?? '';
  const fallback = descriptor.routing?.fallback ?? 'base_gpt2';
  nodes.push(effectCreatorDef(
    `${name}_run`,
    ['prompt'],
    'http',
    (prompt) => ({
      type:      'promise',
      promise:   Promise.resolve({ text: `[${name}] ${prompt}` }),
      onSuccess: (r) => ({ type: `${name}_RESULT`, payload: r }),
      onError:   (e) => ({ type: `${name}_DEACTIVATE`, error: String(e) }),
    })
  ));

  return moduleDef(name, nodes);
}

// ─── Example: canonical counter program in FLUX IR ───────────────────────────

export const COUNTER_PROGRAM = moduleDef('counter', [
  pureFn('double', ['x'], 'x * 2'),
  storeDef('Counter', [{ name: 'count', defaultValue: 0 }]),
  actionDef('INCREMENT'),
  actionDef('DECREMENT'),
  actionDef('ADD', [{ name: 'amount' }]),
  reduceDef('Counter', {
    INCREMENT: (state)        => ({ count: state.count + 1 }),
    DECREMENT: (state)        => ({ count: state.count - 1 }),
    ADD:       (state, { amount }) => ({ count: state.count + (amount ?? 1) }),
  }),
  queryDef('isEven', 'Counter', (state) => state.count % 2 === 0),
  effectCreatorDef('delayedIncrement', ['ms'], 'timer', (ms) => ({
    type:       'timer',
    delayMs:    ms,
    onComplete: () => ({ type: 'INCREMENT' }),
  })),
]);
