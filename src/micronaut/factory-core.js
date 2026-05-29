// factory-core.js — K'UHUL Micronaut Factory (JS port of micronaut-factory)
//
// Source: C:\Users\canna\.kuhul-v1\micronaut-factory\src\factory_core.cpp
// Backs micronaut_factory.exe — authority-based micronaut instantiation.
//
// MicronauntFactory: scans directories for .micronaut folders, loads
//   personality.json from each, registers them in an authority-based registry.
//
// Key design: authority not directory scanning — micronauts are created
//   via SCX command/program bindings, not filesystem enumeration.
//   Bug documented: hardcoded path C:\public_html\MX2LM\codex\AS-XCFE\micronaut
//   Fix: configurable root via opts.root or MICRONAUT_ROOT env var.
//
// Commands (mirrors micronaut_factory.exe CLI):
//   scan           — discover .micronaut directories
//   create <domain>— scaffold new micronaut in root
//   list           — enumerate registry
//   help           — show usage

// ─── Micronaut record schema ──────────────────────────────────────────────────

export function createMicronaut(id, opts = {}) {
  return {
    id,
    domain:      opts.domain     ?? id,
    version:     opts.version    ?? "v1.0.0",
    base_path:   opts.base_path  ?? "",
    personality: opts.personality ?? null,
    status:      opts.status     ?? "registered",
    created_at:  opts.created_at ?? new Date().toISOString(),
    tools:       opts.tools      ?? [],
    authority:   opts.authority  ?? "factory",
  };
}

// ─── Personality template (mirrors personality.json) ─────────────────────────

export function defaultPersonality(domain) {
  return {
    name:    `K'UHUL ${domain.charAt(0).toUpperCase() + domain.slice(1)} Expert`,
    version: "1.0",
    style:   "professional",
    temperature: 0.7,
    focus_areas: [domain],
    capabilities: [`${domain}_analysis`, `${domain}_execution`, `${domain}_review`],
  };
}

// ─── MicronauntFactory ────────────────────────────────────────────────────────

export class MicronauntFactory {
  constructor(opts = {}) {
    // Configurable root — fixes the hardcoded path bug from v3.2.0-supernaut
    this._root    = opts.root ?? (
      typeof process !== "undefined"
        ? (process.env.MICRONAUT_ROOT ?? "~/.micronauts")
        : "~/.micronauts"
    );
    this._registry = new Map();   // id → micronaut record
    this._stats    = { scanned: 0, created: 0, errors: 0 };
  }

  get root()     { return this._root; }
  get registry() { return this._registry; }

  // ── scan(): discover .micronaut entries ──────────────────────────────────────
  // In browser: scan is a no-op (no filesystem). Use register() directly.
  // In Node 18+: optionally pass a list of folder paths to scan.
  async scan(paths = []) {
    let count = 0;
    for (const p of paths) {
      try {
        const m = await this._loadFromPath(p);
        if (m) { this._registry.set(m.id, m); count++; }
      } catch (e) {
        this._stats.errors++;
      }
    }
    this._stats.scanned += count;
    return count;
  }

  async _loadFromPath(folderPath) {
    // Extract id from folder name: "git-agent.micronaut" → "git-agent"
    const folderName = folderPath.split(/[/\\]/).pop() ?? "";
    const id = folderName.endsWith(".micronaut")
      ? folderName.slice(0, -".micronaut".length)
      : folderName;
    if (!id) return null;

    // Load personality.json if accessible
    let personality = null;
    try {
      if (typeof fetch !== "undefined") {
        const r = await fetch(folderPath + "/personality.json");
        if (r.ok) personality = await r.json();
      }
    } catch (_) { /* personality stays null */ }

    return createMicronaut(id, {
      base_path:   folderPath,
      personality: personality ?? defaultPersonality(id),
      status:      "registered",
    });
  }

  // ── register(): authority-based (no filesystem required) ────────────────────
  register(id, opts = {}) {
    const m = createMicronaut(id, opts);
    this._registry.set(id, m);
    this._stats.created++;
    return m;
  }

  // ── create(): scaffold new micronaut ────────────────────────────────────────
  // Returns the scaffold template as a plain object (no filesystem write in browser)
  create(domain, opts = {}) {
    const id = opts.id ?? domain;
    const scaffold = {
      folder:      `${id}.micronaut`,
      personality: defaultPersonality(domain),
      service:     {
        name: id,
        port: opts.port ?? 3300,
        route: `/dispatch`,
        effect: "agent",
      },
      readme: `# ${id}\nK'UHUL micronaut — domain: ${domain}\n`,
    };
    const m = this.register(id, {
      domain, base_path: `${this._root}/${id}.micronaut`,
      personality: scaffold.personality, ...opts
    });
    return { micronaut: m, scaffold };
  }

  // ── list(): enumerate registry ───────────────────────────────────────────────
  list(filter = {}) {
    const all = [...this._registry.values()];
    if (filter.domain)  return all.filter(m => m.domain === filter.domain);
    if (filter.status)  return all.filter(m => m.status === filter.status);
    if (filter.authority) return all.filter(m => m.authority === filter.authority);
    return all;
  }

  // ── get/has ──────────────────────────────────────────────────────────────────
  get(id)    { return this._registry.get(id) ?? null; }
  has(id)    { return this._registry.has(id); }
  unregister(id) { this._registry.delete(id); }

  // ── stats ────────────────────────────────────────────────────────────────────
  get stats() {
    return {
      ...this._stats,
      registered: this._registry.size,
      root: this._root,
    };
  }
}

// ─── Singleton factory (configurable root) ───────────────────────────────────

let _default = null;

export function getFactory(opts) {
  if (!_default) _default = new MicronauntFactory(opts ?? {});
  return _default;
}
