// ASX RAM Replay Verifier v1 — deterministic re-apply + proof check
//
// Contract:
//   replayVerify({ ram_snapshot, patch, policy, prev_hash? })
//   → { @context, @ok, @tick, @phase, @proof_hash, @hashes, @ram_out? }
//
// Hashing: SHA-256 via WebCrypto when available; pluggable fallback via setSha256.
// Path syntax: "@a.@b.@c" — all segments must start with "@".

// ─── SHA-256 adapter ──────────────────────────────────────────────────────────

let _sha256Impl = null;

export function setSha256(impl) {
  _sha256Impl = impl;
}

export async function sha256Hex(str) {
  if (_sha256Impl) return _sha256Impl(str);
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const bytes  = new TextEncoder().encode(str);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
  }
  throw new Error('No WebCrypto available; call setSha256(impl) to provide a fallback.');
}

// ─── Canonical JSON ───────────────────────────────────────────────────────────

export function stableStringify(x) {
  if (x === null || typeof x !== 'object') return JSON.stringify(x);
  if (Array.isArray(x)) return '[' + x.map(stableStringify).join(',') + ']';
  const keys = Object.keys(x).sort();
  return '{' + keys.map(k => JSON.stringify(k) + ':' + stableStringify(x[k])).join(',') + '}';
}

// ─── Deep clone ───────────────────────────────────────────────────────────────

export function deepClone(x) {
  return JSON.parse(JSON.stringify(x));
}

// ─── Dot-path helpers ─────────────────────────────────────────────────────────

export function getByDotPath(obj, path) {
  let cur = obj;
  for (const seg of path.split('.')) {
    if (cur === null || typeof cur !== 'object') return undefined;
    cur = cur[seg];
  }
  return cur;
}

export function setByDotPath(obj, path, value) {
  const segs = path.split('.');
  let cur = obj;
  for (let i = 0; i < segs.length - 1; i++) {
    const s = segs[i];
    if (!cur[s] || typeof cur[s] !== 'object') cur[s] = {};
    cur = cur[s];
  }
  cur[segs[segs.length - 1]] = value;
}

export function delByDotPath(obj, path) {
  const segs = path.split('.');
  let cur = obj;
  for (let i = 0; i < segs.length - 1; i++) {
    cur = cur?.[segs[i]];
    if (!cur || typeof cur !== 'object') return;
  }
  delete cur[segs[segs.length - 1]];
}

// ─── Phase gate enforcement ───────────────────────────────────────────────────

export function assertGate(policy, phase, op) {
  const p = policy['@phases']?.[phase];
  if (!p) throw new Error(`Unknown phase: ${phase}`);
  if (!p['@allow_ops'].includes(op['@op'])) {
    throw new Error(`Op not allowed in phase ${phase}: ${op['@op']}`);
  }

  const path  = op['@path'] || '';
  const allow = p['@allow_prefixes'] || [];
  const deny  = [...(p['@deny_prefixes'] || []), ...(policy['@global']?.['@deny_prefixes'] || [])];

  if (deny.some(prefix => path.startsWith(prefix))) {
    throw new Error(`Path denied in phase ${phase}: ${path}`);
  }
  if (allow.length && !allow.some(prefix => path.startsWith(prefix))) {
    throw new Error(`Path not in allowlist for phase ${phase}: ${path}`);
  }
}

// ─── Op executor ─────────────────────────────────────────────────────────────

export function applyOp(ram, op) {
  const p = op['@path'];
  switch (op['@op']) {
    case 'set':
      setByDotPath(ram, p, op['@value']);
      return;

    case 'merge': {
      const cur  = getByDotPath(ram, p);
      const next = (cur && typeof cur === 'object' && !Array.isArray(cur))
        ? { ...cur, ...op['@value'] }
        : { ...op['@value'] };
      setByDotPath(ram, p, next);
      return;
    }

    case 'del':
      delByDotPath(ram, p);
      return;

    case 'inc': {
      const cur = Number(getByDotPath(ram, p) ?? 0);
      setByDotPath(ram, p, cur + Number(op['@by']));
      return;
    }

    case 'dec': {
      const cur = Number(getByDotPath(ram, p) ?? 0);
      setByDotPath(ram, p, cur - Number(op['@by']));
      return;
    }

    case 'add': {
      const cur = getByDotPath(ram, p);
      const arr = Array.isArray(cur) ? cur.slice() : [];
      arr.push(op['@value']);
      setByDotPath(ram, p, arr);
      return;
    }

    case 'pop': {
      const cur = getByDotPath(ram, p);
      if (!Array.isArray(cur) || cur.length === 0) return;
      const arr = cur.slice();
      if (typeof op['@index'] === 'number') arr.splice(op['@index'], 1);
      else arr.pop();
      setByDotPath(ram, p, arr);
      return;
    }

    case 'clamp': {
      const cur = Number(getByDotPath(ram, p) ?? 0);
      setByDotPath(ram, p, Math.min(Number(op['@max']), Math.max(Number(op['@min']), cur)));
      return;
    }

    case 'swap': {
      const va = getByDotPath(ram, op['@a']);
      const vb = getByDotPath(ram, op['@b']);
      setByDotPath(ram, op['@a'], vb);
      setByDotPath(ram, op['@b'], va);
      return;
    }

    case 'emit': {
      const q   = getByDotPath(ram, '@events.@queue');
      const arr = Array.isArray(q) ? q.slice() : [];
      arr.push(op['@event']);
      setByDotPath(ram, '@events.@queue', arr);
      setByDotPath(ram, '@events.@last', op['@event']);
      return;
    }

    default:
      throw new Error(`Unknown op: ${op['@op']}`);
  }
}

// ─── Proof hash builders ──────────────────────────────────────────────────────

export async function computeInputHash(ramSnapshot, patch) {
  const selected = {
    '@tick':     ramSnapshot['@tick'],
    '@pi':       { '@rng': ramSnapshot?.['@pi']?.['@rng'] },
    '@state':    ramSnapshot['@state'],
    '@clusters': ramSnapshot['@clusters'],
  };
  return sha256Hex(stableStringify({
    selected,
    '@patch_meta': { '@tick': patch['@tick'], '@phase': patch['@phase'] },
  }));
}

export async function computeMutHash(patch) {
  return sha256Hex(stableStringify({ '@ops': patch['@ops'] }));
}

export async function computeStateHash(ram) {
  const core = deepClone(ram);
  delete core['@projection'];
  return sha256Hex(stableStringify(core));
}

export async function computeTickHash(prevHash, tickN, stateHash, inputHash, mutHash) {
  return sha256Hex(stableStringify({
    '@prev_hash':  prevHash || '',
    '@tick':       tickN,
    '@state_hash': stateHash,
    '@input_hash': inputHash,
    '@mut_hash':   mutHash,
  }));
}

// ─── Main verifier ────────────────────────────────────────────────────────────
//
// Stages (rotation-style):
//   rotation_validate   — structural checks + gate enforcement
//   rotation_hash_check — recompute + compare input/mut hashes
//   rotation_apply      — deep clone + ordered op application
//   rotation_seal       — state/tick hash computation + result block

export async function replayVerify({
  ram_snapshot,
  patch,
  policy,
  prev_hash = '',
}) {
  try {
    // ── rotation_validate ───────────────────────────────────────────────────
    if (!ram_snapshot || typeof ram_snapshot !== 'object') throw new Error('rotation_validate: ram_snapshot missing');
    if (!patch        || typeof patch        !== 'object') throw new Error('rotation_validate: patch missing');
    if (!policy       || typeof policy       !== 'object') throw new Error('rotation_validate: policy missing');

    const phase = patch['@phase'];
    const ops   = patch['@ops'] || [];
    const maxOps = policy['@global']?.['@max_ops_per_tick'] ?? 4096;
    if (ops.length > maxOps) throw new Error(`rotation_validate: too_many_ops (${ops.length} > ${maxOps})`);

    for (const op of ops) assertGate(policy, phase, op);

    // ── rotation_hash_check ─────────────────────────────────────────────────
    const input_hash = await computeInputHash(ram_snapshot, patch);
    const mut_hash   = await computeMutHash(patch);

    if (patch?.['@proof']?.['@input_hash'] !== input_hash) throw new Error('rotation_hash_check: input_hash_mismatch');
    if (patch?.['@proof']?.['@mut_hash']   !== mut_hash)   throw new Error('rotation_hash_check: mut_hash_mismatch');

    // ── rotation_apply ──────────────────────────────────────────────────────
    const ram = deepClone(ram_snapshot);
    for (const op of ops) applyOp(ram, op);

    // ── rotation_seal ───────────────────────────────────────────────────────
    const state_hash = await computeStateHash(ram);
    const tick_hash  = await computeTickHash(prev_hash, patch['@tick'], state_hash, input_hash, mut_hash);

    return {
      '@context':    'asx://verify/replay_result/v1',
      '@ok':         true,
      '@tick':       patch['@tick'],
      '@phase':      phase,
      '@proof_hash': tick_hash,
      '@hashes': {
        '@input_hash': input_hash,
        '@mut_hash':   mut_hash,
        '@state_hash': state_hash,
        '@tick_hash':  tick_hash,
      },
      '@ram_out': ram,
    };

  } catch (e) {
    return {
      '@context':       'asx://verify/replay_result/v1',
      '@ok':            false,
      '@tick':          patch?.['@tick'] ?? -1,
      '@failure_stage': String(e?.message || e),
      '@proof_hash':    '',
      '@hashes':        null,
    };
  }
}

// ─── CSS projection compiler ──────────────────────────────────────────────────
// Given ASX RAM + an optional binding table, emits exactly one CSS projection bundle per tick.

export async function compileProjection(ram, bindingTable = [], opts = {}) {
  const tick     = ram['@tick']?.['@n'] ?? 0;
  const phase    = ram['@tick']?.['@phase'] ?? 'perceive';
  const rootVars = { '--asx-tick': tick, '--asx-phase': phase };
  const classes  = [`asx-phase-${phase}`];
  const dataset  = { 'data-asx-tick': String(tick), 'data-asx-phase': phase };

  for (const binding of bindingTable) {
    const val = getByDotPath(ram, binding.ram);
    if (val === undefined || val === null) continue;

    if (binding.css)          rootVars[binding.css]          = val;
    if (binding.dataset)      dataset[binding.dataset]        = String(val);
    if (binding.classPrefix)  classes.push(binding.classPrefix + val);
    if (binding.classOn && val)   classes.push(binding.classOn);
    if (binding.classOff && !val) classes.push(binding.classOff);
  }

  const projStr          = stableStringify({ rootVars, classes, dataset });
  const projection_hash  = await sha256Hex(projStr);

  return {
    '@context':  'asx://projection/css/v1',
    '@tick':      tick,
    '@root':      rootVars,
    '@classes':   classes,
    '@dataset':   dataset,
    '@proof':    { '@projection_hash': projection_hash },
  };
}
