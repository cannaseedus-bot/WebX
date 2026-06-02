// mu-brain.js — µBRAIN: Atomic Cognitive Architecture
//
// Four differentiable cognitive modules, each backed by a µJSONL grammar:
//   µTHINK    Pop   — perception, attention, semantic fold
//   µRESEARCH Wo    — memory retrieval, knowledge graph, ranking
//   µREASON   Sek   — deduction, arithmetic, Bayesian belief update
//   µPLAN     Ch'en — goal decomposition, Q-learning, execution + replan
//
// Backward pass propagates error through all four modules (Ch'en phase).
// Each module's weights update independently via Lipschitz-clipped gradient.
//
// FLUX IR connection:
//   µBRAIN.process()  = one full Pop->Wo->Sek->Ch'en->Xul cycle
//   µBRAIN.learn()    = backward through the FLUX store chain
//   Each module's backward() dispatches a weight-update action to FluxRuntime

import { MuJSONLCompiler } from './mu-jsonl.js';

// ─── µJSONL grammars (embedded — one per module) ─────────────────────────────

const THINK_GRAMMAR = `
{"id":"think.num","µmodel":"mu_pattern","pattern":"\\\\b(\\\\d+(?:\\\\.\\\\d+)?)\\\\b","semantic_type":"quantity","weight":1.0,"phase":"Pop","lipschitz":1.0}
{"id":"think.noun","µmodel":"mu_pattern","pattern":"(farmer|apple|book|child|bird|dollar|orange|pear|cat|dog|ball|coin|box|bag|car|tree)","semantic_type":"object","weight":0.95,"phase":"Pop","lipschitz":1.0}
{"id":"think.action","µmodel":"mu_pattern","pattern":"(gives?|sells?|buys?|eats?|finds?|loses?|adds?|removes?|gains?|spends?|receives?|takes?|drops?)","semantic_type":"verb","weight":0.98,"phase":"Pop","lipschitz":1.0}
{"id":"think.possess","µmodel":"mu_pattern","pattern":"(\\\\w+) has (\\\\d+)","semantic_type":"has","weight":0.9,"phase":"Wo","lipschitz":1.0}
{"id":"think.question","µmodel":"mu_pattern","pattern":"how many","semantic_type":"query","weight":1.0,"phase":"Pop","lipschitz":1.0}
`;

const RESEARCH_GRAMMAR = `
{"id":"research.wm","µmodel":"mu_pattern","pattern":".*","source":"working_memory","weight":1.0,"phase":"Sek","lipschitz":1.0}
{"id":"research.lts","µmodel":"mu_pattern","pattern":"(remember|recall|know|fact|history)","source":"long_term","weight":0.9,"phase":"Wo","lipschitz":1.0}
{"id":"research.graph","µmodel":"mu_pattern","pattern":"(related|similar|connected|linked)","source":"graph","weight":0.8,"phase":"Sek","lipschitz":1.0}
`;

const REASON_GRAMMAR = `
{"id":"reason.add","µmodel":"mu_add","pattern":"(?:adds?|gains?|receives?|finds?|buys?) (\\\\d+)","weight":1.0,"phase":"Sek","lipschitz":1.0}
{"id":"reason.sub","µmodel":"mu_subtract","pattern":"(?:sells?|gives? away|loses?|spends?|removes?) (\\\\d+)","weight":1.0,"phase":"Sek","lipschitz":1.0}
{"id":"reason.mul","µmodel":"mu_multiply","pattern":"(\\\\d+) (?:times|groups? of|multiplied by) (\\\\d+)","weight":0.95,"phase":"Sek","lipschitz":1.5}
{"id":"reason.div","µmodel":"mu_divide","pattern":"split (\\\\d+) (?:among|between) (\\\\d+)","weight":0.9,"phase":"Sek","lipschitz":1.0}
{"id":"reason.chain","µmodel":"mu_chain","pattern":"has (\\\\d+).*?(?:sells?|gives?) (\\\\d+).*?(?:buys?|receives?) (\\\\d+)","steps":["reason.sub","reason.add"],"weight":1.0,"phase":"Sek","lipschitz":2.0}
`;

const PLAN_GRAMMAR = `
{"id":"plan.sequential","µmodel":"mu_chain","pattern":"first.*then","weight":1.0,"phase":"Ch'en","lipschitz":2.0}
{"id":"plan.goal_compute","µmodel":"mu_pattern","pattern":"how many|what is|calculate|find the","weight":0.95,"phase":"Ch'en","lipschitz":1.0}
{"id":"plan.monitor","µmodel":"mu_pattern","pattern":"(progress|status|check|verify)","weight":0.8,"phase":"Xul","lipschitz":1.0}
{"id":"plan.replan","µmodel":"mu_pattern","pattern":"(failed|wrong|error|retry|again)","weight":0.9,"phase":"Xul","lipschitz":1.0}
`;

// ─── µTHINK ───────────────────────────────────────────────────────────────────

export class MuThink {
  constructor() {
    this._compiler = new MuJSONLCompiler();
    this._compiler.load(THINK_GRAMMAR);
    this._attnWeights = new Map();
  }

  process(input) {
    const tokens = [];
    for (const p of this._compiler.compile()) {
      const m = input.match(p.regex);
      if (!m) continue;
      const captures = m.slice(1).filter(Boolean);
      tokens.push({
        type:      p.entry?.semantic_type ?? 'generic',
        value:     captures[0] ?? m[0],
        captures,
        weight:    p.weight,
        patternId: p.id,
      });
    }

    const attention = this._softmaxAttention(tokens);
    const folded    = this._fold(tokens, attention);
    const confidence = this._calibrate(folded);

    return { tokens, attention, folded, confidence };
  }

  _softmaxAttention(tokens) {
    if (tokens.length === 0) return [];
    const scores = tokens.map(t => t.weight * (t.type === 'quantity' ? 1.2 : 1.0));
    const max    = Math.max(...scores);
    const exps   = scores.map(s => Math.exp(s - max));
    const sum    = exps.reduce((a,b) => a+b, 0);
    return tokens.map((t, i) => ({ ...t, attn: exps[i] / sum }));
  }

  _fold(tokens, attention) {
    const folded = { quantities: [], objects: [], actions: [], other: [] };
    for (const t of attention) {
      if (t.attn > 0.05) {
        const key = t.type === 'quantity' ? 'quantities'
                  : t.type === 'object'   ? 'objects'
                  : t.type === 'verb'     ? 'actions'
                  : 'other';
        folded[key].push(t);
      }
    }
    return folded;
  }

  _calibrate(folded) {
    const all  = Object.values(folded).flat();
    const high = all.filter(t => t.attn > 0.15).length;
    return all.length > 0 ? high / all.length : 0;
  }

  backward(errorSignal, lr = 0.01) {
    for (const p of this._compiler.compile()) {
      this._compiler.backward(errorSignal, p.id, lr);
    }
  }

  get grammar() { return this._compiler; }
}

// ─── µRESEARCH ────────────────────────────────────────────────────────────────

export class MuResearch {
  constructor() {
    this._compiler    = new MuJSONLCompiler();
    this._compiler.load(RESEARCH_GRAMMAR);
    this._workingMem  = new Map();
    this._episodic    = [];
    this._sourceWeights = { working_memory: 1.0, long_term: 0.9, graph: 0.8 };
  }

  query(tokens, context = {}) {
    const queryKey = tokens.map(t => t.value).join(' ');

    // Working memory first
    const wmResults = [...this._workingMem.values()]
      .filter(v => this._jaccard(v.key, queryKey) > 0.2)
      .map(v => ({ ...v, source: 'working_memory', relevance: this._jaccard(v.key, queryKey) * this._sourceWeights.working_memory }));

    // Graph lookup (simple token overlap)
    const graphResults = this._episodic.slice(0, 50)
      .filter(ep => ep.tokens.some(t => tokens.some(q => q.value === t.value)))
      .map(ep => ({ key: ep.key, data: ep.result, source: 'graph', relevance: 0.7 * this._sourceWeights.graph }));

    const all = [...wmResults, ...graphResults]
      .sort((a,b) => b.relevance - a.relevance)
      .slice(0, 10);

    // Store episode
    this._episodic.unshift({ key: queryKey, tokens, result: all[0] ?? null, ts: Date.now() });
    if (this._episodic.length > 500) this._episodic.pop();

    return {
      results:    all,
      count:      all.length,
      confidence: all.length > 0 ? all[0].relevance : 0,
    };
  }

  store(key, data) { this._workingMem.set(key, { key, data, ts: Date.now() }); }

  _jaccard(a, b) {
    const sa = new Set(a.toLowerCase().split(/\s+/));
    const sb = new Set(b.toLowerCase().split(/\s+/));
    const inter = [...sa].filter(x => sb.has(x)).length;
    const union = new Set([...sa, ...sb]).size;
    return union > 0 ? inter / union : 0;
  }

  backward(errorSignal, lr = 0.01) {
    for (const k in this._sourceWeights) {
      this._sourceWeights[k] = Math.max(0.1, Math.min(1.5,
        this._sourceWeights[k] + lr * errorSignal * 0.1));
    }
  }

  get grammar() { return this._compiler; }
}

// ─── µREASON ──────────────────────────────────────────────────────────────────

export class MuReason {
  constructor() {
    this._compiler = new MuJSONLCompiler();
    this._compiler.load(REASON_GRAMMAR);
    this._beliefs  = new Map();
    this._lr       = 0.1;
  }

  infer(input, knowl = []) {
    // Forward through µJSONL patterns
    const { prediction, patternId, numbers, mumodel } = this._compiler.forward(input);

    // Deductive rules on quantities from knowledge
    const deductions = this._deduce(knowl);

    // Bayesian belief update
    const key = patternId ?? 'unknown';
    const prior = this._beliefs.get(key) ?? 0.5;
    const posterior = prior + this._lr * ((prediction !== null ? 0.9 : 0.1) - prior);
    this._beliefs.set(key, Math.min(1, Math.max(0, posterior)));

    const conclusions = [];
    if (prediction !== null) conclusions.push({ type: mumodel, value: prediction, confidence: this._beliefs.get(key) });
    conclusions.push(...deductions);

    const contradictions = this._detectContradictions(conclusions);
    const overallConf = conclusions.reduce((s, c) => s + (c.confidence ?? 0.5), 0) / Math.max(1, conclusions.length);

    return { prediction, patternId, numbers, conclusions, contradictions, overallConfidence: overallConf };
  }

  _deduce(knowledge) {
    const facts = [];
    const nums  = knowledge.flatMap(k => k.captures ?? []).map(Number).filter(n => !isNaN(n));
    if (nums.length >= 2) {
      facts.push({ type: 'sum',        value: nums.reduce((a,b)=>a+b,0), confidence: 0.9 });
      facts.push({ type: 'difference', value: nums[0] - nums[1],         confidence: 0.9 });
    }
    return facts;
  }

  _detectContradictions(conclusions) {
    const conts = [];
    for (let i = 0; i < conclusions.length; i++)
      for (let j = i+1; j < conclusions.length; j++) {
        const a = conclusions[i], b = conclusions[j];
        if (a.type === b.type && a.value !== b.value && (a.confidence??0) > 0.7 && (b.confidence??0) > 0.7)
          conts.push({ a, b, severity: ((a.confidence??0) + (b.confidence??0)) / 2 });
      }
    return conts;
  }

  backward(errorSignal, patternId = null, lr = 0.01) {
    if (patternId) this._compiler.backward(errorSignal, patternId, lr);
    for (const [k, b] of this._beliefs)
      this._beliefs.set(k, Math.min(1, Math.max(0, b + lr * errorSignal * 0.1)));
  }

  get grammar() { return this._compiler; }
}

// ─── µPLAN ────────────────────────────────────────────────────────────────────

export class MuPlan {
  constructor() {
    this._compiler = new MuJSONLCompiler();
    this._compiler.load(PLAN_GRAMMAR);
    this._qTable  = new Map();       // stateKey|actionKey -> Q-value
    this._history = [];
    this.epsilon  = 0.1;
    this.gamma    = 0.95;
    this.lr       = 0.1;
  }

  plan(goal, inferences) {
    // Decompose goal into action steps from inferences
    const actions = [];
    for (const c of inferences.conclusions) {
      if (c.value !== undefined) actions.push({ name: c.type, value: c.value, confidence: c.confidence ?? 0.9 });
    }

    // Q-value scoring
    const stateKey = JSON.stringify(goal);
    const scored = actions.map(a => {
      const ak  = `${a.name}:${a.value}`;
      const qv  = this._qTable.get(`${stateKey}|${ak}`) ?? 0;
      return { ...a, qValue: qv, score: a.confidence * 0.6 + Math.max(0, qv) * 0.4 };
    }).sort((a,b) => b.score - a.score);

    const best = {
      actions:    scored,
      score:      scored.length > 0 ? scored[0].score : 0,
      goal,
      initialState: inferences,
    };

    this._history.push({ goal, plan: best, ts: Date.now() });
    return { plan: best, alternatives: scored.slice(1, 4), confidence: best.score };
  }

  execute(plan) {
    const log = [];
    let finalValue = null;

    for (const action of plan.actions) {
      const success = Math.random() < (action.confidence ?? 0.9);
      if (!success) { log.push({ action, success: false }); continue; }

      finalValue = action.value;
      log.push({ action, success: true, value: action.value });

      // Q-update
      const stateKey = JSON.stringify(plan.goal);
      const ak       = `${action.name}:${action.value}`;
      const key      = `${stateKey}|${ak}`;
      const qOld     = this._qTable.get(key) ?? 0;
      this._qTable.set(key, qOld + this.lr * (action.score - qOld));
    }

    return {
      success:      log.some(l => l.success),
      finalValue,
      executionLog: log,
      steps:        log.length,
    };
  }

  backward(errorSignal, lr = 0.01) {
    for (const [k, q] of this._qTable)
      this._qTable.set(k, Math.max(-1, Math.min(1, q + lr * errorSignal * 0.1)));
    this.epsilon = errorSignal < -0.5
      ? Math.min(0.5, this.epsilon + 0.01)
      : Math.max(0.01, this.epsilon - 0.01);
  }

  get grammar() { return this._compiler; }
}

// ─── µBRAIN ───────────────────────────────────────────────────────────────────
//
// Pipeline: input -> µTHINK -> µRESEARCH -> µREASON -> µPLAN -> execution
// Backward: error -> µPLAN -> µREASON -> µRESEARCH -> µTHINK

export class MuBrain {
  constructor() {
    this.think    = new MuThink();
    this.research = new MuResearch();
    this.reason   = new MuReason();
    this.plan     = new MuPlan();
    this.phase    = 'idle';
    this.history  = [];
  }

  process(input) {
    // Pop — perception
    this.phase = 'Pop';
    const perception = this.think.process(input);

    // Wo — retrieval
    this.phase = 'Wo';
    const knowledge = this.research.query(perception.tokens, { input });

    // Sek — inference
    this.phase = 'Sek';
    const inferences = this.reason.infer(input, [
      ...perception.tokens,
      ...knowledge.results.map(r => r.data).filter(Boolean),
    ]);

    // Ch'en — plan
    this.phase = "Ch'en";
    const goal    = this._inferGoal(perception, inferences);
    const planned = this.plan.plan(goal, inferences);

    // Xul — execute
    this.phase = 'Xul';
    const execution = this.plan.execute(planned.plan);

    const result = {
      input,
      perception,
      knowledge,
      inferences,
      plan:      planned,
      execution,
      answer:    execution.finalValue,
      phase:     this.phase,
      success:   execution.success,
      metrics:   this._metrics(perception, knowledge, inferences, planned, execution),
    };

    this.history.unshift(result);
    if (this.history.length > 200) this.history.pop();

    this.phase = 'idle';
    return result;
  }

  learn(expected, actual) {
    const errorSignal = expected - actual;
    // Backward through all four modules (Ch'en phase)
    this.plan.backward(errorSignal);
    this.reason.backward(errorSignal);
    this.research.backward(errorSignal);
    this.think.backward(errorSignal);
    return { errorSignal, updated: ['plan','reason','research','think'] };
  }

  _inferGoal(perception, inferences) {
    const hasQuery    = perception.tokens.some(t => t.type === 'query');
    const hasQuantity = perception.folded.quantities.length > 0;
    return {
      type:        hasQuery && hasQuantity ? 'compute' : 'understand',
      description: hasQuery ? 'Calculate result from perceived quantities' : 'Comprehend and respond',
    };
  }

  _metrics(perception, knowledge, inferences, planned, execution) {
    return {
      perceptionConfidence: perception.confidence,
      knowledgeConfidence:  knowledge.confidence,
      inferenceConfidence:  inferences.overallConfidence,
      planQuality:          planned.plan?.score ?? 0,
      executionSuccess:     execution.success,
      totalSteps:           execution.executionLog?.length ?? 0,
      contradictions:       inferences.contradictions.length,
      answer:               execution.finalValue,
    };
  }

  // Expose grammars for external weight inspection / fine-tuning
  get grammars() {
    return {
      think:    this.think.grammar,
      research: this.research.grammar,
      reason:   this.reason.grammar,
      plan:     this.plan.grammar,
    };
  }

  /** Checkpoint all four grammars. */
  checkpoint() {
    return {
      think:    this.think.grammar.checkpoint(),
      research: this.research.grammar.checkpoint(),
      reason:   this.reason.grammar.checkpoint(),
      plan:     this.plan.grammar.checkpoint(),
    };
  }

  restore(ckpt) {
    if (ckpt.think)    this.think.grammar.restore(ckpt.think);
    if (ckpt.research) this.research.grammar.restore(ckpt.research);
    if (ckpt.reason)   this.reason.grammar.restore(ckpt.reason);
    if (ckpt.plan)     this.plan.grammar.restore(ckpt.plan);
  }
}

// ─── µBRAIN µMODEL KXML spec ─────────────────────────────────────────────────

export const MUPY_BRAIN_KXML = `<?xml version="1.0" encoding="utf-8"?>
<kxml domain="mu_brain" phase="Pop" gravity="Normal">

  <fold id="think"    domain="PERCEPTION"/>
  <fold id="research" domain="RETRIEVAL"/>
  <fold id="reason"   domain="INFERENCE"/>
  <fold id="plan"     domain="EXECUTION"/>

  <geodesic from="think"    to="research" cost="0.2"/>
  <geodesic from="research" to="reason"   cost="0.3"/>
  <geodesic from="reason"   to="plan"     cost="0.2"/>
  <geodesic from="plan"     to="think"    cost="0.8"/>

  <lane id="pop"  type="perception"  permission="inherit"/>
  <lane id="wo"   type="retrieval"   permission="inherit"/>
  <lane id="sek"  type="inference"   permission="inherit"/>
  <lane id="chen" type="planning"    permission="inherit"/>
  <lane id="xul"  type="execution"   permission="inherit"/>

  <![CDATA[
    Pop:   muTHINK.process(input)  -- perceive, attend, fold, calibrate
    Wo:    muRESEARCH.query(tokens) -- working-mem, episodic, graph lookup
    Sek:   muREASON.infer(input)   -- deduct, compute, Bayesian update
    Ch'en: muPLAN.plan(goal)       -- decompose, Q-score, select best
    Xul:   muPLAN.execute(plan)    -- run actions, monitor, replan if needed
    backward(error): plan->reason->research->think (gradient descent)
  ]]>

  <![CDATA[
    mu_brain.think mu_brain.research mu_brain.reason mu_brain.plan
    mu_brain.perception mu_brain.attention mu_brain.fold mu_brain.calibrate
    mu_brain.retrieval mu_brain.episodic mu_brain.graph
    mu_brain.deduction mu_brain.syllogism mu_brain.bayesian
    mu_brain.decompose mu_brain.qlearning mu_brain.replan
    mujsonl.think mujsonl.research mujsonl.reason mujsonl.plan
    glyph.NEURAL_PATH glyph.TENSOR_CORE glyph.PARALLEL_GEO
  ]]>

  <shader type="brain_projection">
    <![CDATA[
      // field_optimizer.hlsl — mu_brain pipeline projection
      // Four folds chained: think->research->reason->plan
      // Backward error flows opposite: plan->reason->research->think
      float perception_pressure = think_confidence * 1.0;
      float inference_pressure  = reason_confidence * gravity_scale;
      float plan_score          = q_value_max * navigation_force_scale(lr);
    ]]>
  </shader>

</kxml>`;

export const MUPY_BRAIN_SPEC = `
[model]
domain   = "mu_brain"
phase    = "Pop"
gravity  = "Normal"

[training]
steps  = 5000
batch  = 4
lr     = 1e-5
block  = 256

[routing]
trigger  = "think,reason,plan,perceive,infer,deduce,remember,goal,execute"
fallback = "base_gpt2"

[capabilities]
tools = "mu_brain,math_tool,coder_tool,kuhul_agent,micronaut_dispatch,kxml_run"
`;
