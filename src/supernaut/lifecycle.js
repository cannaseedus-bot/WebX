// Supernaut Lifecycle — monotonic 7-state machine (v3.2.0-supernaut)
//
// Mirrors dispatch.py LifecycleState + DispatchContext + ExpertRoute.
// Rollback always throws — lifecycle is monotonically increasing.
//
// REGISTERED → VALIDATED → BOUND → ACTIVE → EXECUTED → OBSERVED → ARCHIVED

export const LifecycleState = Object.freeze({
  REGISTERED: 'REGISTERED',
  VALIDATED:  'VALIDATED',
  BOUND:      'BOUND',
  ACTIVE:     'ACTIVE',
  EXECUTED:   'EXECUTED',
  OBSERVED:   'OBSERVED',
  ARCHIVED:   'ARCHIVED',
});

export const LIFECYCLE_ORDER = Object.freeze([
  'REGISTERED', 'VALIDATED', 'BOUND', 'ACTIVE', 'EXECUTED', 'OBSERVED', 'ARCHIVED',
]);

export function lifecycleIndex(state) {
  const i = LIFECYCLE_ORDER.indexOf(state);
  if (i === -1) throw new Error(`Unknown lifecycle state: ${state}`);
  return i;
}

export const EXPERT_MAP = Object.freeze({
  geometry:  0,
  temporal:  1,
  amplify:   2,
  compress:  3,
  focus:     4,
  integrate: 5,
  pattern:   6,
  novelty:   7,
});

export const ADDON_EXPERT = 2;  // amplify — all addon/shard deltas route here

export function createExpertRoute(type, strength = 0.10) {
  const expertId   = EXPERT_MAP[type] ?? ADDON_EXPERT;
  const expertName = type in EXPERT_MAP ? type : 'amplify';
  return { expertId, expertName, addonShard: null, strength };
}

export class DispatchContext {
  constructor(tokenIds, route) {
    if (!tokenIds || tokenIds.length === 0) throw new Error('tokenIds must not be empty');
    this.tokenIds = tokenIds;
    this.route    = route;
    this.state    = LifecycleState.REGISTERED;
    this.logits   = null;
    this.report   = {};
  }

  advance(nextState) {
    if (lifecycleIndex(nextState) <= lifecycleIndex(this.state)) {
      throw new Error(`Lifecycle rollback forbidden: ${this.state} -> ${nextState}`);
    }
    this.state = nextState;
  }
}

// Drive a DispatchContext through REGISTERED → OBSERVED in one call.
// On GPU-capable systems, replace the logits stub with actual GPU forward output.
export function runDispatch(tokenIds, route, opts = {}) {
  const ctx = new DispatchContext(tokenIds, route);

  if (!tokenIds.every(t => Number.isInteger(t) && t >= 0)) {
    throw new Error('tokenIds must be non-negative integers');
  }
  ctx.advance(LifecycleState.VALIDATED);
  ctx.advance(LifecycleState.BOUND);
  ctx.advance(LifecycleState.ACTIVE);

  // CPU stub — replace with GPU forward when sxme_host.dll is available
  const outSize = opts.outSize || 50257;
  ctx.logits = new Float32Array(outSize);  // all zeros
  ctx.report.forward_rc = -1;              // -1 = CPU-only mode

  ctx.advance(LifecycleState.EXECUTED);
  ctx.report.expert_id   = route.expertId;
  ctx.report.expert_name = route.expertName;
  ctx.report.seq_len     = tokenIds.length;
  ctx.report.out_size    = outSize;
  ctx.advance(LifecycleState.OBSERVED);

  return ctx;
}

export function archiveContext(ctx) {
  ctx.advance(LifecycleState.ARCHIVED);
  return ctx;
}
