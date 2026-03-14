// K'UHUL++ v2.0 Runtime
// Executes a K'UHUL++ AST using the D12WebX GPU infrastructure.

import { NodeKind } from '../compiler/parser.js';
import { applyGlyph, GLYPHS as KUHUL_GLYPHS } from '../kuhul.js';

// ------------------------------------------------------------------ //
// RuntimeError
// ------------------------------------------------------------------ //

export class RuntimeError extends Error {
    /**
     * @param {string} message
     * @param {object|null} astNode - Node being executed when the error occurred
     */
    constructor(message, astNode = null) {
        super(`RuntimeError — ${message}`);
        this.name    = 'RuntimeError';
        this.astNode = astNode;
    }
}

// ------------------------------------------------------------------ //
// Environment (execution scope with variable bindings)
// ------------------------------------------------------------------ //

class Environment {
    constructor(parent = null) {
        this.parent   = parent;
        this.bindings = new Map();
    }

    define(name, value) {
        this.bindings.set(name, value);
    }

    set(name, value) {
        if (this.bindings.has(name)) {
            this.bindings.set(name, value);
        } else if (this.parent) {
            this.parent.set(name, value);
        } else {
            // Auto-create at top level
            this.bindings.set(name, value);
        }
    }

    get(name) {
        if (this.bindings.has(name)) return this.bindings.get(name);
        if (this.parent) return this.parent.get(name);
        return undefined;
    }

    child() {
        return new Environment(this);
    }
}

// ------------------------------------------------------------------ //
// KuhulRuntime
// ------------------------------------------------------------------ //

/**
 * K'UHUL++ Runtime — executes a parsed K'UHUL++ AST.
 *
 * @example
 * import { KuhulRuntime } from './runtime/runtime.js';
 * import { tokenize } from './compiler/lexer.js';
 * import { parse } from './compiler/parser.js';
 *
 * const tokens = tokenize(source);
 * const ast    = parse(tokens);
 * const rt     = new KuhulRuntime();
 * const result = await rt.execute(ast);
 */
export class KuhulRuntime {
    /**
     * @param {object}  [options]
     * @param {object}  [options.gpu]        - D12WebX GPU instance (optional, uses mock if absent)
     * @param {boolean} [options.strict]     - Throw on every RuntimeError (default: false)
     * @param {number}  [options.maxLoops]   - Max loop iterations to prevent infinite loops (default: 10000)
     */
    constructor(options = {}) {
        this._gpu      = options.gpu ?? null;
        this._strict   = options.strict ?? false;
        this._maxLoops = options.maxLoops ?? 10_000;
        this._log      = [];
    }

    /** Return the execution log (each entry: { kind, detail }) */
    get log() {
        return [...this._log];
    }

    // ------------------------------------------------------------------ //
    // Public API
    // ------------------------------------------------------------------ //

    /**
     * Execute a K'UHUL++ AST.
     *
     * @param {{ kind: 'Program', body: object[] }} ast
     * @param {object} [globals] - Additional global bindings
     * @returns {Promise<{ log: object[], env: object }>}
     */
    async execute(ast, globals = {}) {
        const env = new Environment();
        this._installBuiltins(env);
        for (const [k, v] of Object.entries(globals)) {
            env.define(k, v);
        }
        await this._execNode(ast, env);
        return {
            log: this.log,
            env: Object.fromEntries(env.bindings),
        };
    }

    // ------------------------------------------------------------------ //
    // Node executor (recursive interpreter)
    // ------------------------------------------------------------------ //

    async _execNode(n, env) {
        if (!n || typeof n !== 'object') return undefined;

        switch (n.kind) {
            case NodeKind.Program:      return this._execProgram(n, env);
            case NodeKind.TensorDecl:
            case NodeKind.ClusterDecl:
            case NodeKind.ModelDecl:
            case NodeKind.PipelineDecl: return this._execDecl(n, env);
            case NodeKind.Assignment:   return this._execAssignment(n, env);
            case NodeKind.GlyphOp:      return this._execGlyphOp(n, env);
            case NodeKind.NativeBlock:  return this._execNativeBlock(n, env);
            case NodeKind.IfStmt:       return this._execIf(n, env);
            case NodeKind.ForStmt:      return this._execFor(n, env);
            case NodeKind.ParallelFor:  return this._execParallelFor(n, env);
            case NodeKind.WhileStmt:    return this._execWhile(n, env);
            case NodeKind.ForEachGlyph: return this._execForEachGlyph(n, env);
            case NodeKind.TrainStmt:    return this._execTrain(n, env);
            case NodeKind.FunctionCall: return this._evalCall(n, env);
            case NodeKind.MethodCall:   return this._evalMethodCall(n, env);
            case NodeKind.Block:        return this._execBlock(n, env);

            // Expressions (return values)
            case NodeKind.BinaryExpr:   return this._evalBinary(n, env);
            case NodeKind.UnaryExpr:    return this._evalUnary(n, env);
            case NodeKind.MemberExpr:   return this._evalMember(n, env);
            case NodeKind.IndexExpr:    return this._evalIndex(n, env);
            case NodeKind.Identifier:   return env.get(n.name);
            case NodeKind.NumberLiteral: return n.value;
            case NodeKind.PiExpr:       return n.coefficient * Math.PI;
            case NodeKind.StringLiteral: return n.value;
            case NodeKind.ArrayLiteral: return this._evalArray(n, env);
            case NodeKind.ObjectLiteral: return this._evalObject(n, env);
            case NodeKind.Vector3Literal: return this._evalVector3(n, env);

            default:
                return undefined;
        }
    }

    // ---- statement executors ----

    async _execProgram(n, env) {
        let last;
        for (const stmt of n.body) {
            last = await this._execNode(stmt, env);
        }
        return last;
    }

    async _execDecl(n, env) {
        const value = await this._execNode(n.init, env);
        env.define(n.name, value);
        this._logEntry('declare', { name: n.name, kind: n.kind, value });
        return value;
    }

    async _execAssignment(n, env) {
        const value = await this._execNode(n.value, env);
        const name  = this._resolveName(n.target);
        if (name) {
            env.set(name, value);
        }
        return value;
    }

    async _execGlyphOp(n, env) {
        const glyph = n.glyph;

        // Resolve parameter values
        const params = {};
        for (const [k, v] of Object.entries(n.params)) {
            params[k] = await this._execNode(v, env);
        }

        // Resolve target (buffer or named tensor)
        let targetBuffer = null;
        if (n.target) {
            const targetVal = await this._execNode(n.target, env);
            if (targetVal && targetVal.view instanceof Float32Array) {
                targetBuffer = targetVal;
            }
        }

        // Execute the glyph operation if we have a valid buffer
        let result = null;
        if (targetBuffer) {
            const param = params.angle ?? params.radius ?? params.intensity ?? params.degrees ?? 0;
            try {
                result = applyGlyph(glyph, targetBuffer, param);
            } catch (err) {
                this._handleError(err.message, n);
            }
        }

        this._logEntry('glyph', { glyph, params, result });
        return result;
    }

    async _execNativeBlock(n, env) {
        const dx12Env = env.child();
        const results = [];
        for (const stmt of n.stmts ?? []) {
            const r = await this._execNode(stmt, dx12Env);
            results.push(r);
        }
        this._logEntry('dx12', { statementCount: n.stmts?.length ?? 0 });
        return results;
    }

    async _execIf(n, env) {
        const cond = await this._execNode(n.condition, env);
        if (cond) {
            return this._execNode(n.consequent, env);
        } else if (n.alternate) {
            return this._execNode(n.alternate, env);
        }
        return undefined;
    }

    async _execFor(n, env) {
        const range  = await this._execNode(n.range, env);
        const loopEnv = env.child();
        let count = 0;

        if (Array.isArray(range)) {
            for (const val of range) {
                if (++count > this._maxLoops) break;
                loopEnv.set(n.ident, val);
                await this._execNode(n.body, loopEnv);
            }
        } else if (typeof range === 'number') {
            for (let i = 0; i < range && count <= this._maxLoops; i++, count++) {
                loopEnv.set(n.ident, i);
                await this._execNode(n.body, loopEnv);
            }
        }
        return undefined;
    }

    async _execParallelFor(n, env) {
        // In the browser runtime we execute sequentially (true parallelism
        // would require Workers); the semantics are the same.
        return this._execFor(n, env);
    }

    async _execWhile(n, env) {
        let count = 0;
        while (count++ < this._maxLoops) {
            const cond = await this._execNode(n.condition, env);
            if (!cond) break;
            await this._execNode(n.body, env);
        }
        return undefined;
    }

    async _execForEachGlyph(n, env) {
        const loopEnv = env.child();
        for (const g of n.glyphs) {
            loopEnv.set('glyph', g);
            await this._execNode(n.body, loopEnv);
        }
        return undefined;
    }

    async _execTrain(n, env) {
        const model = env.get(n.model);
        const data  = env.get(n.data);
        this._logEntry('train', { model: n.model, data: n.data, hasConfig: !!n.body });
        if (n.body) await this._execNode(n.body, env);
        return { model, data, trained: true };
    }

    async _execBlock(n, env) {
        let last;
        for (const stmt of n.stmts ?? []) {
            last = await this._execNode(stmt, env);
        }
        return last;
    }

    // ---- expression evaluators ----

    async _evalCall(n, env) {
        const builtin = this._getBuiltin(n.callee, env);
        if (builtin) {
            const args = await this._evalArgs(n.args, env);
            return builtin(...args);
        }
        const fn = env.get(n.callee);
        if (typeof fn === 'function') {
            const args = await this._evalArgs(n.args, env);
            return fn(...args);
        }
        this._logEntry('call', { callee: n.callee, resolved: false });
        return null;
    }

    async _evalMethodCall(n, env) {
        const obj = typeof n.object === 'string'
            ? env.get(n.object)
            : await this._execNode(n.object, env);

        const args = await this._evalArgs(n.args, env);

        if (obj && typeof obj[n.method] === 'function') {
            return obj[n.method](...args);
        }

        // GPU built-in dispatch
        if (n.object === 'GPU' || (n.object && n.object.name === 'GPU')) {
            this._logEntry('gpu', { method: n.method, args });
            return this._simulateGPUCall(n.method, args);
        }

        this._logEntry('methodCall', { method: n.method, resolved: false });
        return null;
    }

    async _evalBinary(n, env) {
        const left  = await this._execNode(n.left,  env);
        const right = await this._execNode(n.right, env);
        switch (n.op) {
            case '+':  return left + right;
            case '-':  return left - right;
            case '*':  return left * right;
            case '/':  return left / right; // standard JS semantics (Infinity/NaN for div by zero)
            case '>':  return left > right;
            case '<':  return left < right;
            case '>=': return left >= right;
            case '<=': return left <= right;
            case '|':  return left | right;
            default:   return undefined;
        }
    }

    async _evalUnary(n, env) {
        const val = await this._execNode(n.operand, env);
        return n.op === '-' ? -val : val;
    }

    async _evalMember(n, env) {
        const obj = await this._execNode(n.object, env);
        if (obj && typeof obj === 'object') return obj[n.member];
        return undefined;
    }

    async _evalIndex(n, env) {
        const obj   = await this._execNode(n.object, env);
        const index = await this._execNode(n.index,  env);
        if (Array.isArray(obj)) return obj[index];
        if (obj && typeof obj === 'object') return obj[index];
        return undefined;
    }

    async _evalArray(n, env) {
        const elements = [];
        for (const el of n.elements) {
            elements.push(await this._execNode(el, env));
        }
        return elements;
    }

    async _evalObject(n, env) {
        const obj = {};
        for (const [k, v] of Object.entries(n.properties)) {
            obj[k] = await this._execNode(v, env);
        }
        return obj;
    }

    async _evalVector3(n, env) {
        return {
            x: await this._execNode(n.x, env),
            y: await this._execNode(n.y, env),
            z: await this._execNode(n.z, env),
        };
    }

    async _evalArgs(argList, env) {
        const values = [];
        for (const arg of argList ?? []) {
            values.push(await this._execNode(arg.value, env));
        }
        return values;
    }

    // ---- built-in functions ----

    _installBuiltins(env) {
        // Geometric construction built-ins
        env.define('GeometricTensor', (options = {}) => ({
            type:     'GeometricTensor',
            view:     options.data instanceof Float32Array ? options.data : new Float32Array(options.stride ?? 32),
            phase:    options.phase  ?? 0,
            symmetry: options.symmetry ?? 1,
            stride:   options.stride   ?? 32,
        }));

        env.define('TensorCluster', (options = {}) => ({
            type:         'TensorCluster',
            tensors:      options.tensors      ?? [],
            center:       options.center       ?? { x: 0, y: 0, z: 0 },
            plane:        options.plane        ?? 'flat',
            relationship: options.relationship ?? 'none',
        }));

        env.define('GeometricModel', (options = {}) => ({
            type:       'GeometricModel',
            modelType:  options.type        ?? 'classification',
            dimensions: options.dimensions  ?? [],
            piPhase:    options.pi_phase    ?? 0,
            symmetry:   options.symmetry    ?? 1,
        }));

        env.define('generate_spiral', (n = 100) => {
            const data = new Float32Array(n * 3);
            for (let i = 0; i < n; i++) {
                const t   = (i / n) * 2 * Math.PI;
                data[i * 3]     = Math.cos(t) * (i / n);
                data[i * 3 + 1] = Math.sin(t) * (i / n);
                data[i * 3 + 2] = i / n;
            }
            return data;
        });

        env.define('load_dataset', (path = '') => ({
            type: 'dataset',
            path,
            tensors: [],
        }));

        env.define('split_tensors', (cluster, count, index) => ({
            type:    'TensorCluster',
            slice:   index,
            tensors: (cluster?.tensors ?? []).filter((_, i) => i % (count ?? 1) === (index ?? 0)),
        }));

        env.define('merge_clusters', (clusters = []) => ({
            type:    'TensorCluster',
            tensors: clusters.flatMap(c => c?.tensors ?? []),
        }));

        env.define('GPU', {
            count: 1,
            Dispatch: (threads, shader) => this._simulateGPUCall('Dispatch', [threads, shader]),
            Draw:     (tensors, pipeline) => this._simulateGPUCall('Draw', [tensors, pipeline]),
            Present:  () => this._simulateGPUCall('Present', []),
            CreateSwapChain: (opts) => this._simulateGPUCall('CreateSwapChain', [opts]),
            CreateFence: () => this._simulateGPUCall('CreateFence', []),
        });
    }

    _getBuiltin(name, env) {
        const val = env.get(name);
        return typeof val === 'function' ? val : null;
    }

    // ---- GPU simulation ----

    _simulateGPUCall(method, args) {
        if (this._gpu) {
            switch (method) {
                case 'Dispatch': {
                    const cmdList = this._gpu.createCommandList();
                    const [threads] = args;
                    const [x = 1, y = 1, z = 1] = Array.isArray(threads) ? threads : [threads ?? 1, 1, 1];
                    cmdList.dispatch(x, y, z);
                    return this._gpu.executeParallel([cmdList]);
                }
                case 'CreateFence':
                    return this._gpu.createFence();
                default:
                    break;
            }
        }
        const result = { gpuCall: method, args, executedAt: Date.now() };
        this._logEntry('gpu', result);
        return result;
    }

    // ---- utilities ----

    _resolveName(n) {
        if (!n) return null;
        if (n.kind === NodeKind.Identifier) return n.name;
        if (n.kind === NodeKind.MemberExpr) return this._resolveName(n.object);
        return null;
    }

    _logEntry(kind, detail) {
        this._log.push({ kind, detail, timestamp: Date.now() });
    }

    _handleError(message, node) {
        const err = new RuntimeError(message, node);
        if (this._strict) throw err;
        this._logEntry('error', { message });
    }
}

// ------------------------------------------------------------------ //
// Convenience: compile + run in one step
// ------------------------------------------------------------------ //

/**
 * Compile and execute K'UHUL++ source code.
 *
 * @param {string}  source    - .KPP source code
 * @param {object}  [options] - Options forwarded to KuhulRuntime constructor
 * @param {object}  [globals] - Additional global bindings
 * @returns {Promise<{ log: object[], env: object, errors: import('../compiler/semantic.js').SemanticError[], warnings: string[] }>}
 */
export async function run(source, options = {}, globals = {}) {
    const { tokenize }  = await import('../compiler/lexer.js');
    const { parse }     = await import('../compiler/parser.js');
    const { analyze }   = await import('../compiler/semantic.js');

    const tokens  = tokenize(source);
    const ast     = parse(tokens);
    const { errors, warnings } = analyze(ast);

    const rt     = new KuhulRuntime(options);
    const result = await rt.execute(ast, globals);

    return { ...result, errors, warnings };
}
