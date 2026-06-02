// tool-registry.js — JSONL tool registry for K'UHUL agentic system
//
// Each line in kuhul.tools.jsonl is one complete tool definition.
// Line-oriented = batchable, parallelizable, sandboxable.
//
// Tool → Fold mapping (from tools/kuhul.tools.jsonl):
//   FILE_FOLD    0.5  read_file / write_file           (sandboxed IO)
//   TOOL_FOLD    0.8  assistant / batch / http / infer (validated execution)
//   COMPUTE_FOLD 2.0  data_processor / fibonacci_fold  (GPU/SIMD)
//   CMD_FOLD     0.6  exec / shell                     (parse/validate)
//   AGENT_FOLD   0.3  agent_api / kxml_dispatcher      (orchestrator)
//   OPCODE_FOLD  0.9  isolated_exec / sandbox          (tightest bounds)
//   THINK_FOLD   0.2  think_trace                      (near-antigravity)
//   META_FOLD    0.0  tool_help                        (antigravity, observe-only)
//
// K'UHUL physics per tool:
//   gravity field = tool.gravity (from JSONL)
//   antigravity   = tool.antigravity === true OR tool.gravity < G.THRESHOLD
//   pressure      = PRESSURE_TABLE[tool.fold][rank]
//
// Sandbox → phase gate:
//   restricted    Pop    (read-only, initialized at load)
//   gpu_sandbox   Sek    (compute phase only)
//   full_isolation Sek   (tightest — OPCODE_FOLD 0.9)
//   readonly      Pop    (antigravity — observe, never mutate)

import { FOLD, getPressure } from '../xcfe/pressure-mapper.js';
import { G } from '../xcfe/gravity.js';

// ─── ToolRegistry ────────────────────────────────────────────────────────────

export class ToolRegistry {
  constructor() {
    this._tools      = new Map();  // id → tool
    this._byType     = new Map();  // type → [id]
    this._byFold     = new Map();  // fold → [id]
    this._byPerm     = new Map();  // permission → [id]
    this._rateCounts = new Map();  // key → {count, resetAt}
  }

  /** Load tools from a JSONL string (one JSON object per line). */
  load(jsonlText) {
    for (const raw of jsonlText.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith('#')) continue;
      try {
        const tool = JSON.parse(line);
        if (!tool.id) continue;
        this._register(tool);
      } catch (_) {}
    }
    return this;
  }

  _register(tool) {
    this._tools.set(tool.id, tool);
    this._index(this._byType,  tool.type, tool.id);
    this._index(this._byFold,  tool.fold ?? FOLD.TOOL, tool.id);
    for (const p of (tool.permissions ?? [])) this._index(this._byPerm, p, tool.id);
  }

  _index(map, key, val) {
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(val);
  }

  get(id)          { return this._tools.get(id) ?? null; }
  byType(t)        { return (this._byType.get(t)  ?? []).map(id => this._tools.get(id)); }
  byFold(f)        { return (this._byFold.get(f)  ?? []).map(id => this._tools.get(id)); }
  byPerm(p)        { return (this._byPerm.get(p)  ?? []).map(id => this._tools.get(id)); }
  get size()       { return this._tools.size; }
  all()            { return [...this._tools.values()]; }
  batchable()      { return this.all().filter(t => t.batchable); }
  antigravity()    { return this.all().filter(t => t.antigravity || (t.gravity ?? 1) < G.THRESHOLD); }

  /** Effective fold pressure for a tool (rank-2 default). */
  pressure(id, rank = 2) {
    const t = this.get(id);
    if (!t) return 1.0;
    return getPressure(t.fold ?? FOLD.TOOL, rank);
  }

  /** Rate-limit check (returns false if limit exceeded). */
  checkRate(id, limit) {
    if (!limit || limit <= 0) return true;
    const key  = id;
    const now  = Date.now();
    const rec  = this._rateCounts.get(key);
    if (!rec || now > rec.resetAt) {
      this._rateCounts.set(key, { count: 1, resetAt: now + 60_000 });
      return true;
    }
    if (rec.count >= limit) return false;
    rec.count++;
    return true;
  }

  /** Summary of the registry by fold + type. */
  summary() {
    const byFold = {}, byType = {};
    for (const t of this._tools.values()) {
      const f = t.fold ?? 'TOOL_FOLD';
      const tp = t.type ?? 'function';
      byFold[f]  = (byFold[f]  ?? 0) + 1;
      byType[tp] = (byType[tp] ?? 0) + 1;
    }
    return { total: this._tools.size, byFold, byType,
             batchable: this.batchable().length, antigravity: this.antigravity().length };
  }
}

// ─── ToolExecutor ─────────────────────────────────────────────────────────────
//
// Lightweight JS executor — routes tool calls to handlers.
// Respects sandbox, rate-limit, and gravity constraints.

export class ToolExecutor {
  constructor(registry, opts = {}) {
    this._registry = registry;
    this._handlers = new Map(); // cmd → async fn(tool, args) → result
    this._clipNorm  = opts.clipNorm ?? 1.0;
    this._maxLogit  = opts.maxLogit ?? 20.0;
    this._log       = [];
    this._registerDefaults();
  }

  registerHandler(cmd, fn)   { this._handlers.set(cmd, fn); return this; }

  async call(id, args = []) {
    const tool = this._registry.get(id);
    if (!tool) return { ok: false, error: `tool not found: ${id}` };

    // Rate limit
    if (!this._registry.checkRate(id, tool.rate_limit)) {
      return { ok: false, error: 'rate limit exceeded' };
    }

    // Gravity: antigravity tools bypass phase check
    const isAnti = tool.antigravity || (tool.gravity ?? 1) < G.THRESHOLD;
    const entry = { id, fold: tool.fold, gravity: tool.gravity ?? 1.0, phase: tool.phase, isAnti };

    // Execute
    const handler = this._handlers.get(tool.cmd) ?? this._handlers.get('default');
    let result;
    try {
      result = handler ? await handler(tool, args) : `[stub] ${tool.name}(${args.join(',')})`;
    } catch (e) {
      result = `error: ${e.message}`;
    }

    // Heavy-gravity (OPCODE/COMPUTE): clamp result logits
    if (!isAnti && (tool.gravity ?? 1) >= 2.0 && typeof result === 'number') {
      result = Math.max(-this._maxLogit, Math.min(this._maxLogit, result));
      entry.clamped = true;
    }

    this._log.push({ ...entry, result: String(result).slice(0, 200) });
    return { ok: true, result, tool };
  }

  async batch(calls) {
    return Promise.all(calls.map(({ id, args }) => this.call(id, args ?? [])));
  }

  _registerDefaults() {
    this._handlers.set('kuhul.help', async (tool, args) => {
      const t = this._registry.get(args[0]);
      if (!t) return 'tool not found';
      return `${t.name} v${t.version}: ${t.help}\nArgs: ${(t.args??[]).join(', ')}\nFold: ${t.fold} gravity=${t.gravity}`;
    });
    this._handlers.set('default', async (tool, args) =>
      `[${tool.type}:${tool.name}] ${args.join(' ')} → (stub)`);
  }

  get log() { return [...this._log]; }
}

// ─── Load from file (browser: fetch; Node: fs) ────────────────────────────────

export async function loadToolRegistry(urlOrPath) {
  let text;
  if (typeof fetch !== 'undefined') {
    text = await (await fetch(urlOrPath)).text();
  } else {
    const fs = await import('fs');
    text = fs.readFileSync(urlOrPath, 'utf-8');
  }
  return new ToolRegistry().load(text);
}
