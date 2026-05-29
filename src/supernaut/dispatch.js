// Supernaut Dispatch — SkillRunResult, RoundContext, Mayan fold (v3.2.0-supernaut)
//
// Mirrors dispatch.py SkillRunResult + RoundContext + _mayan_fold.
// Sigil action control vectors: @load @route @dispatch @merge @validate @report
// Transitions: @next @accept @reject

export const SIGIL_ACTIONS = Object.freeze([
  '@load', '@route', '@dispatch', '@merge', '@validate', '@report',
]);

export const SIGIL_TRANSITIONS = Object.freeze(['@next', '@accept', '@reject']);

export const STOCK_SKILLS = Object.freeze([
  'as-xcfe-stack-intel',
  'codex-agent',
  'micronaut',
  'micronaut-agent-factory',
]);

// MayanFold = Σ level_value[n] × 20^n  (n = 0..19, vigesimal accumulation)
export function mayanFold(levelValues) {
  return levelValues.reduce((acc, v, n) => acc + v * Math.pow(20, n), 0);
}

export class SkillRunResult {
  constructor(skillName, roundNum) {
    this.skillName        = skillName;
    this.roundNum         = roundNum;
    this.vigesimalAddr    = roundNum - 1;
    this.actionsExecuted  = [];
    this.results          = [];
    this.status           = 'ok';   // 'ok' | 'skipped' | 'error'
    this.durationMs       = 0;
  }

  get levelValue() {
    if (this.status === 'ok' && this.actionsExecuted.length > 0) {
      return Math.min(this.actionsExecuted.length, 19);
    }
    return 0;
  }
}

export class RoundContext {
  constructor(roundNum, opts = {}) {
    this.roundNum   = roundNum;
    this.skillsOrder = opts.skillsOrder || [];
    this.completed  = [];
    this.activeSkill = null;
    this.results    = [];
    this.status     = 'running';  // 'running' | 'complete'
    this.telemetry  = opts.telemetry || {};
    this.activeTodo = opts.activeTodo || null;
    this.todosDone  = [];
  }

  get vigesimalAddr() { return this.roundNum - 1; }

  get levelValue() {
    return Math.min(this.results.filter(r => r.status === 'ok').length, 19);
  }
}

export function parseSigilActions(doc, skillResolver = null) {
  const resolved = {};
  for (const [key, value] of Object.entries(doc)) {
    if (!key.startsWith('@')) continue;
    const actionName = key.slice(1);
    const handler    = skillResolver ? skillResolver(actionName) : null;
    resolved[actionName] = handler;
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      const nested = parseSigilActions(value, skillResolver);
      for (const [k, v] of Object.entries(nested)) {
        resolved[`${actionName}.${k}`] = v;
      }
    }
  }
  return resolved;
}

export function foldReport(rounds) {
  const levelValues = rounds.map(r => r.levelValue);
  let ok = 0, skipped = 0, errors = 0;
  for (const r of rounds) {
    for (const s of r.results) {
      if (s.status === 'ok') ok++;
      else if (s.status === 'skipped') skipped++;
      else if (s.status === 'error') errors++;
    }
  }
  return { rounds: rounds.length, levelValues, mayanFold: mayanFold(levelValues), ok, skipped, errors };
}
