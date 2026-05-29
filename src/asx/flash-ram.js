// ASX FLASH RAM v1 — deterministic semi-persistent memory plane
//
// Position in Ω stack:
//   XJSON → ATOMIC → XCFE → π → ASX RAM → ASX FLASH RAM → K'UHUL STORAGE BRIDGE → ARCHIVE
//
// ASX RAM  = reality while alive  (volatile, mutable, live)
// FLASH RAM = reality after death (frozen, verified, resurrectable)
//
// FLASH does not think. FLASH does not execute. FLASH remembers enough truth to rebuild the world.

import { stableStringify, sha256Hex, deepClone } from './replay-verifier.js';

// ─── Memory hierarchy constants ───────────────────────────────────────────────

export const MEMORY_HIERARCHY = Object.freeze([
  { level: 0, name: 'CPU Registers',      desc: 'Execution registers — ephemeral, not persistent' },
  { level: 1, name: 'Execution Cache',    desc: 'Runtime caches — ephemeral, not persistent' },
  { level: 2, name: 'ASX RAM',            desc: 'Live reality — volatile, mutable, tick-evolving' },
  { level: 3, name: 'ASX FLASH RAM',      desc: 'Resurrection state — frozen, verified, persistent' },
  { level: 4, name: 'IDB / LocalStorage', desc: 'Browser-side durable storage' },
  { level: 5, name: 'K\'uhul Storage Bridge', desc: 'Cross-session storage bridge' },
  { level: 6, name: 'Archive',            desc: 'Permanent immutable record' },
]);

// ─── Flash snapshot modes ─────────────────────────────────────────────────────

export const FLASH_MODES = Object.freeze({
  FULL:       'full',        // complete RAM state — best recovery, largest size
  DELTA:      'delta',       // only changes — smallest storage, requires replay
  CHECKPOINT: 'checkpoint',  // milestone states — ideal for agents and games
});

// ─── What FLASH RAM is allowed to store ──────────────────────────────────────
// Only durable reality — no live handles, no projection, no runtime state.

export const FLASH_ALLOWED_KEYS = Object.freeze([
  '@world',    // persistent world entities, player state, inventory
  '@agents',   // agent relationship graphs, history
  '@ngrams',   // learned language structures
  '@clusters', // cluster knowledge (not votes/queue)
  '@memories', // episodic memory nodes
]);

export const FLASH_FORBIDDEN_KEYS = Object.freeze([
  '@dom',       // DOM is rebuilt — never persist
  '@css',       // CSS/projection is derived — never persist
  '@projection',// projection is compiled — never persist
  '@net',       // network sockets — never persist
  '@rng',       // RNG handles — reseed on boot
  '@timers',    // timer handles — never persist
  '@gpu',       // GPU buffers — never persist
  '@threads',   // thread handles — never persist
  '@transactions', // open transactions — never persist
]);

// ─── FLASH RAM snapshot creator ───────────────────────────────────────────────

export async function createFlashSnapshot(ram, opts = {}) {
  const mode    = opts.mode     || FLASH_MODES.FULL;
  const prevHash = opts.prevHash || '';

  // Extract only the durable subset of @state
  const durableState = {};
  for (const key of FLASH_ALLOWED_KEYS) {
    const val = ram['@state']?.[key] ?? ram[key];
    if (val !== undefined) durableState[key] = deepClone(val);
  }

  const snapshot = {
    '@context': 'asx://flash_ram/v1',
    '@id':       `flash_${Date.now()}`,
    '@mode':     mode,
    '@snapshot': {
      '@tick':       ram['@tick']?.['@n'] ?? 0,
      '@state_hash': '',
      '@timestamp':  Date.now(),
    },
    '@state': durableState,
    '@proof': {
      '@snapshot_hash': '',
      '@prev_hash':      prevHash,
    },
  };

  // Hash the durable state
  const stateStr      = stableStringify(durableState);
  const stateHash     = await sha256Hex(stateStr);
  snapshot['@snapshot']['@state_hash'] = stateHash;

  const snapshotStr   = stableStringify({ '@snapshot': snapshot['@snapshot'], '@state': durableState });
  const snapshotHash  = await sha256Hex(snapshotStr);
  snapshot['@proof']['@snapshot_hash'] = snapshotHash;

  return snapshot;
}

// ─── FLASH verification ───────────────────────────────────────────────────────

export async function verifyFlashSnapshot(flash) {
  try {
    if (flash?.['@context'] !== 'asx://flash_ram/v1') {
      return { ok: false, reason: 'wrong_context' };
    }

    const durableState  = flash['@state'] || {};
    const stateStr      = stableStringify(durableState);
    const expectedState = await sha256Hex(stateStr);

    if (flash['@snapshot']?.['@state_hash'] !== expectedState) {
      return { ok: false, reason: 'state_hash_mismatch' };
    }

    const snapshotStr     = stableStringify({ '@snapshot': flash['@snapshot'], '@state': durableState });
    const expectedSnap    = await sha256Hex(snapshotStr);

    if (flash['@proof']?.['@snapshot_hash'] !== expectedSnap) {
      return { ok: false, reason: 'snapshot_hash_mismatch' };
    }

    return { ok: true, tick: flash['@snapshot']['@tick'] };

  } catch (e) {
    return { ok: false, reason: String(e?.message || e) };
  }
}

// ─── Resurrection: FLASH → RAM rebuild ───────────────────────────────────────
//
// Resurrection sequence:
//   BOOT → LOAD FLASH → VERIFY HASH → REBUILD RAM → REPLAY DELTAS → ENTER TICK LOOP

export function rebuildRamFromFlash(flash, baseRam) {
  if (!baseRam || typeof baseRam !== 'object') {
    throw new Error('rebuildRamFromFlash: baseRam (empty template) is required');
  }
  const ram = deepClone(baseRam);

  // Restore durable state into the base RAM template
  for (const key of FLASH_ALLOWED_KEYS) {
    if (flash['@state']?.[key] !== undefined) {
      if (!ram['@state']) ram['@state'] = {};
      ram['@state'][key] = deepClone(flash['@state'][key]);
    }
  }

  // Restore tick counter to the snapshot tick (caller bumps to +1 on first real tick)
  if (flash['@snapshot']?.['@tick'] !== undefined) {
    ram['@tick'] = {
      ...ram['@tick'],
      '@n':       flash['@snapshot']['@tick'],
      '@phase':   'perceive',
      '@barriers': [],
      '@monotonic_ok': true,
    };
  }

  return ram;
}

// ─── Delta FLASH — store only changed keys ───────────────────────────────────

export async function createDeltaFlash(prevFlash, ram, opts = {}) {
  const prevState = prevFlash?.['@state'] || {};
  const curState  = {};

  for (const key of FLASH_ALLOWED_KEYS) {
    const cur = ram['@state']?.[key] ?? ram[key];
    if (cur !== undefined) curState[key] = cur;
  }

  // Find changed keys
  const deltaState = {};
  for (const key of FLASH_ALLOWED_KEYS) {
    if (stableStringify(curState[key]) !== stableStringify(prevState[key])) {
      deltaState[key] = deepClone(curState[key]);
    }
  }

  return createFlashSnapshot(
    { '@tick': ram['@tick'], '@state': deltaState },
    { mode: FLASH_MODES.DELTA, prevHash: prevFlash?.['@proof']?.['@snapshot_hash'] || '', ...opts }
  );
}

// ─── Flash chain verifier ─────────────────────────────────────────────────────
// Verifies a linked list of FLASH snapshots (each referencing the previous).

export async function verifyFlashChain(flashList) {
  const results = [];
  let expectedPrev = '';

  for (let i = 0; i < flashList.length; i++) {
    const flash  = flashList[i];
    const result = await verifyFlashSnapshot(flash);

    if (!result.ok) {
      results.push({ index: i, ok: false, reason: result.reason });
      break;
    }
    if (flash['@proof']?.['@prev_hash'] !== expectedPrev) {
      results.push({ index: i, ok: false, reason: 'chain_prev_hash_mismatch' });
      break;
    }

    expectedPrev = flash['@proof']['@snapshot_hash'];
    results.push({ index: i, ok: true, tick: result.tick });
  }

  const allOk = results.every(r => r.ok);
  return { ok: allOk, results };
}
