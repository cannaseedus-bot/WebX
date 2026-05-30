// agent-runtime.js — @agent / @skill / @micronaut / @command / @tool namespace handlers
//
// Full @ hierarchy:
//   @agent      WHO  — autonomous entity with capabilities + goals
//   @skill      WHAT — reusable, composable capability set
//   @micronaut  HOW  — lightweight stateful service runtime
//   @command    DO   — single executable action with parameters
//   @tool       USE  — external system binding (http, db, cli, gpu)
//
// Vertical composition:  agent → skill → micronaut → command → tool
// Horizontal composition: multiple agents in @orchestrator.pipeline
//
// K'UHUL opcode alignment:
//   @agent       ≡ ⟁Nod⟁  0x41  define node
//   @skill       ≡ ⟁Wo⟁   0x05  call/invoke
//   @micronaut   ≡ ⟁Clu⟁  0x40  define cluster
//   @command     ≡ ⟁Sek⟁  0x03  + ⟁Wo⟁ 0x05 (set + call)
//   @tool        ≡ ⟁Ch'en⟁ 0x06 store/persist (external binding)
//   @wait_for    ≡ ⟁Sync⟁ 0x46  synchronize
//   @orchestrator ≡ ⟁Dist⟁ 0x44 distribute

import { EventEmitter } from 'node:events';

// ─── Registry ─────────────────────────────────────────────────────────────────

class Registry extends Map {
  define(name, spec) { this.set(name, spec); return this; }
  lookup(name)        { return this.get(name); }
}

export const AGENT_REGISTRY      = new Registry();
export const SKILL_REGISTRY      = new Registry();
export const MICRONAUT_REGISTRY  = new Registry();
export const COMMAND_REGISTRY    = new Registry();
export const TOOL_REGISTRY       = new Registry();

// ─── Agent ────────────────────────────────────────────────────────────────────

export class Agent extends EventEmitter {
  constructor(name, spec = {}) {
    super();
    this.name         = name;
    this.capabilities = spec['@capabilities'] ?? [];
    this.skills       = {};
    this.state        = {};
    this._spec        = spec;
  }

  async init(rt) {
    // Wire declared skills
    for (const [k, v] of Object.entries(this._spec)) {
      if (k.startsWith('@skill.')) {
        const skillName = k.slice(7);
        this.skills[skillName] = await rt.resolveSkill(skillName, v);
      }
      if (k.startsWith('@micronaut.')) {
        const mn = k.slice(11);
        await rt.execMicronaut(mn, v);
      }
    }
    this.emit('ready');
    return this;
  }

  async execute(commandName, params, rt) {
    this.emit('executing', commandName, params);
    const skill = Object.values(this.skills).find(s => s.hasCommand(commandName));
    if (skill) return skill.execute(commandName, params, rt);
    // Direct command lookup
    const cmd = COMMAND_REGISTRY.lookup(commandName);
    if (cmd) return rt.execCommand(commandName, { ...cmd, ...params });
    throw new Error(`Agent ${this.name}: unknown command "${commandName}"`);
  }
}

// ─── Skill ────────────────────────────────────────────────────────────────────

export class Skill {
  constructor(name, spec = {}) {
    this.name     = name;
    this.version  = spec['@version'] ?? '1.0.0';
    this.commands = {};
    this._spec    = spec;
  }

  hasCommand(name) { return name in this.commands; }

  async execute(name, params, rt) {
    const cmd = this.commands[name];
    if (!cmd) throw new Error(`Skill ${this.name}: no command "${name}"`);
    return rt.execCommand(name, { ...cmd, ...params });
  }
}

// ─── Micronaut ────────────────────────────────────────────────────────────────

export class Micronaut extends EventEmitter {
  constructor(name, spec = {}) {
    super();
    this.name   = name;
    this.port   = spec['@port'] ?? 0;
    this.routes = {};
    this.state  = {};
    this._spec  = spec;
  }

  route(method, path, handler) {
    this.routes[`${method.toUpperCase()} ${path}`] = handler;
    return this;
  }

  async handle(method, path, body) {
    const key = `${method.toUpperCase()} ${path}`;
    const handler = this.routes[key];
    if (!handler) return { status: 404, body: 'Not found' };
    return handler(body, this.state);
  }
}

// ─── Tool ─────────────────────────────────────────────────────────────────────

export class Tool {
  constructor(name, spec = {}) {
    this.name       = name;
    this.provider   = spec['@provider'] ?? 'builtin';
    this.operations = {};
    this._spec      = spec;
    this._retry     = spec['@retry_policy'] ?? { attempts: 1, backoff: 'none' };
  }

  register(opName, fn) { this.operations[opName] = fn; return this; }

  async run(opName, params, attempt = 1) {
    const op = this.operations[opName];
    if (!op) throw new Error(`Tool ${this.name}: unknown op "${opName}"`);
    try {
      return await op(params);
    } catch (e) {
      if (attempt < (this._retry.attempts ?? 1)) {
        const delay = this._retry.backoff === 'exponential' ? 2 ** attempt * 100 : 100;
        await new Promise(r => setTimeout(r, delay));
        return this.run(opName, params, attempt + 1);
      }
      throw e;
    }
  }
}

// ─── XCFE Agent Runtime ───────────────────────────────────────────────────────

export class XCFEAgentRuntime {
  constructor(nodeRuntime) {
    this._node    = nodeRuntime;      // base node-runtime
    this._agents  = new Map();
    this._skills  = new Map();
    this._mnts    = new Map();        // micronauts
    this._tools   = new Map();
    this._waiting = new Map();        // @wait_for promises
  }

  // ─── @ dispatch extension ───────────────────────────────────────────────────

  async execute(block) {
    for (const [key, val] of Object.entries(block)) {
      await this._dispatch(key, val);
    }
  }

  async _dispatch(key, val) {
    if (!key.startsWith('@')) return;
    const parts = key.slice(1).split('.');
    switch (parts[0]) {
      case 'agent':        return this.execAgent(parts.slice(1).join('.'), val);
      case 'skill':        return this.execSkill(parts.slice(1).join('.'), val);
      case 'micronaut':    return this.execMicronaut(parts.slice(1).join('.'), val);
      case 'command':      return this.execCommand(parts.slice(1).join('.'), val);
      case 'tool':         return this.execTool(parts.slice(1).join('.'), val);
      case 'orchestrator': return this.execOrchestrator(parts.slice(1).join('.'), val);
      case 'wait_for':     return this.execWaitFor(val);
      case 'schedule':     return this.execSchedule(key, val);
      default:             return this._node?._dispatch(key, val);
    }
  }

  // ─── @agent ─────────────────────────────────────────────────────────────────

  async execAgent(name, spec) {
    if (this._agents.has(name)) {
      // Execute on existing agent
      const agent = this._agents.get(name);
      for (const [k, v] of Object.entries(spec)) {
        if (k.startsWith('@command.')) {
          await agent.execute(k.slice(9), v, this);
        }
      }
      return agent;
    }

    const agent = new Agent(name, spec);
    AGENT_REGISTRY.define(name, spec);
    this._agents.set(name, agent);
    await agent.init(this);

    // Execute immediate @command blocks
    for (const [k, v] of Object.entries(spec)) {
      if (k.startsWith('@command.')) {
        await this.execCommand(k.slice(9), v);
      }
    }

    // Signal availability to any @wait_for listeners
    this._resolve(name, agent);
    return agent;
  }

  // ─── @skill ──────────────────────────────────────────────────────────────────

  async execSkill(name, spec) {
    const skill = new Skill(name, spec);
    SKILL_REGISTRY.define(name, spec);

    // Register commands declared under @commands
    const cmds = spec['@commands'] ?? {};
    for (const [k, v] of Object.entries(cmds)) {
      if (k.startsWith('@command.')) {
        skill.commands[k.slice(9)] = v;
      }
    }
    // Also register direct @command.* keys on the skill
    for (const [k, v] of Object.entries(spec)) {
      if (k.startsWith('@command.')) skill.commands[k.slice(9)] = v;
    }

    this._skills.set(name, skill);
    return skill;
  }

  resolveSkill(name, spec) {
    return this._skills.has(name)
      ? this._skills.get(name)
      : this.execSkill(name, spec ?? {});
  }

  // ─── @micronaut ──────────────────────────────────────────────────────────────

  async execMicronaut(name, spec) {
    const mn = new Micronaut(name, spec);
    MICRONAUT_REGISTRY.define(name, spec);

    // Parse @routes
    const routes = spec['@routes'] ?? {};
    for (const [k, v] of Object.entries(routes)) {
      if (k.startsWith('@route.')) {
        // @route.post."/login"
        const rParts = k.slice(7).split('.');
        const method = rParts[0];
        const path   = rParts.slice(1).join('.').replace(/^"|"$/g, '');
        mn.route(method, path, async (body) => {
          const result = {};
          for (const [ck, cv] of Object.entries(v)) {
            if (ck.startsWith('@command.')) {
              result[ck.slice(9)] = await this.execCommand(ck.slice(9), { ...cv, ...body });
            }
          }
          return result;
        });
      }
    }

    this._mnts.set(name, mn);
    this._resolve(name, mn);
    return mn;
  }

  // ─── @command ────────────────────────────────────────────────────────────────

  async execCommand(name, spec) {
    COMMAND_REGISTRY.define(name, spec);
    const ctx  = this._node?._ctx ?? {};
    const steps = spec['@steps'] ?? {};
    const results = {};

    for (const [k, v] of Object.entries(steps)) {
      if (k.startsWith('@command.')) {
        results[k.slice(9)] = await this.execCommand(k.slice(9), v);
      }
      if (k.startsWith('@tool.')) {
        results[k.slice(6)] = await this.execTool(k.slice(6), v);
      }
    }

    // Also execute inline @tool.* and @command.* on this spec
    for (const [k, v] of Object.entries(spec)) {
      if (k.startsWith('@tool.')) {
        results[k.slice(6)] = await this.execTool(k.slice(6), v);
      }
    }

    // @rollback spec stored but not executed now
    const rollback = spec['@rollback'];

    this._resolve(name, results);
    return results;
  }

  // ─── @tool ──────────────────────────────────────────────────────────────────

  async execTool(name, spec) {
    TOOL_REGISTRY.define(name, spec);

    // Look up existing tool instance
    if (!this._tools.has(name)) {
      const tool = new Tool(name, spec);
      // Auto-register operations from @operations
      const ops = spec['@operations'] ?? {};
      for (const [k, v] of Object.entries(ops)) {
        const opName = k.startsWith('@tool.') ? k.slice(6) : k;
        tool.register(opName, async (params) => ({ ...params, _op: opName, _tool: name }));
      }
      this._tools.set(name, tool);
    }

    const tool = this._tools.get(name);
    const opName = spec['@op'] || 'default';
    const params = this._resolve(spec);

    if (tool.operations[opName]) return tool.run(opName, params);
    return { _tool: name, _op: opName, params };
  }

  // ─── @orchestrator ───────────────────────────────────────────────────────────

  async execOrchestrator(name, spec) {
    const parallel = [];
    const sequential = [];

    for (const [k, v] of Object.entries(spec)) {
      if (k.startsWith('@agent.')) {
        // Each agent in orchestrator runs as a stage
        const agentName  = k.slice(7);
        const waitFor    = v['@wait_for'];
        if (waitFor) {
          sequential.push({ agent: agentName, spec: v, waitFor });
        } else {
          parallel.push(this.execAgent(agentName, v));
        }
      }
    }

    // Run parallel agents
    const parallelResults = await Promise.all(parallel);

    // Run sequential agents (respecting @wait_for)
    for (const { agent, spec, waitFor } of sequential) {
      await this.execWaitFor(waitFor);
      await this.execAgent(agent, spec);
    }

    return parallelResults;
  }

  // ─── @wait_for ────────────────────────────────────────────────────────────────

  async execWaitFor(val) {
    const key = typeof val === 'string' ? val : val.key ?? val;
    if (this._waiting.has(key)) {
      return this._waiting.get(key).promise;
    }
    // Check if already resolved in context
    const ctx = this._node?._ctx ?? {};
    if (ctx[key]) return ctx[key];

    let resolve;
    const promise = new Promise(r => { resolve = r; });
    this._waiting.set(key, { promise, resolve });
    return promise;
  }

  _resolve(key, value) {
    if (this._waiting.has(key)) {
      this._waiting.get(key).resolve(value);
      this._waiting.delete(key);
    }
    if (this._node) this._node._ctx[key] = value;
  }

  // ─── @schedule ────────────────────────────────────────────────────────────────

  execSchedule(key, spec) {
    const cron = spec['@schedule'] ?? spec;
    console.log(`[xcfe] @schedule ${key} → "${cron}" (use node-cron or setInterval)`);
    // Real cron: import cron from 'node-cron'; cron.schedule(cron, () => execute(spec))
    return { scheduled: true, cron };
  }

  // ─── Template helper ────────────────────────────────────────────────────────

  _resolve(val) {
    if (this._node) return this._node._resolve(val);
    return val;
  }
}

// ─── @ → K'UHUL opcode alignment table ───────────────────────────────────────

export const AGENT_OPCODE_MAP = Object.freeze({
  '@agent':        { kuhul: '⟁Nod⟁ 0x41',  description: 'define/spawn node' },
  '@skill':        { kuhul: '⟁Wo⟁ 0x05',   description: 'invoke capability' },
  '@micronaut':    { kuhul: '⟁Clu⟁ 0x40',  description: 'define cluster member' },
  '@command':      { kuhul: '⟁Sek⟁+⟁Wo⟁', description: 'assign then call' },
  '@tool':         { kuhul: '⟁Ch\'en⟁ 0x06',description: 'external binding' },
  '@orchestrator': { kuhul: '⟁Dist⟁ 0x44', description: 'distribute across nodes' },
  '@wait_for':     { kuhul: '⟁Sync⟁ 0x46', description: 'synchronize' },
  '@schedule':     { kuhul: '⟁Mon⟁ 0x62',  description: 'monitor / periodic' },
  '@route':        { kuhul: '⟁Path⟁ 0x65', description: 'path select' },
  '@rollback':     { kuhul: '⟁Rec⟁ 0x4B',  description: 'recover / rollback' },
});
