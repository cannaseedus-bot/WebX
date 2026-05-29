// MATRIX DAG Runtime v0.1 — UnifiedAPI + K'uhul/XCFE scheduling
//
// Executes a compiled MATRIX DAG (.mxb structural graph) by:
//   1. Wrapping each node as a MicronautJob / UnifiedAPI unit
//   2. Running topological execution (⟁Pop execute_graph ... ⟁Xul⟁)
//   3. Dispatching parallel-capable nodes as batch units, sequential as thread units

// ─── Lane types ───────────────────────────────────────────────────────────────

export const LANE_TYPES = Object.freeze({
  batch:   'batch',   // parallel computation
  thread:  'thread',  // sequential computation
  process: 'process', // source / sink I/O
});

// ─── Node → UnifiedAPI unit mapping ──────────────────────────────────────────

export function nodeToUnit(node) {
  let laneType;
  if ((node.type === 'computation' || node.type === 'transform') && node.parallel) {
    laneType = LANE_TYPES.batch;
  } else if (node.type === 'source' || node.type === 'sink') {
    laneType = LANE_TYPES.process;
  } else {
    laneType = LANE_TYPES.thread;
  }

  return {
    unitId:   node.id,
    laneType,
    nodeType: node.type,
    label:    node.label || node.id,
    work:     node.work  || null,
    meta:     node.meta  || {},
  };
}

// ─── MicronautJob state machine ───────────────────────────────────────────────
// States: pending → ready → running → done | error

export const JOB_STATE = Object.freeze({
  PENDING: 'pending',
  READY:   'ready',
  RUNNING: 'running',
  DONE:    'done',
  ERROR:   'error',
});

export function createMicronautJob(node, unit) {
  return {
    jobId:  node.id,
    unit,
    state:  JOB_STATE.PENDING,
    deps:   node.deps  ? [...node.deps]  : [],
    result: null,
    error:  null,
  };
}

// ─── .mxb graph loader ────────────────────────────────────────────────────────
// .mxb is a plain JSON structural graph: { nodes: MxbNode[], edges: MxbEdge[] }
// MxbNode: { id, type, label?, parallel?, work?, meta?, deps? }
// MxbEdge: { from, to, label? }
//
// loadMxb accepts: a parsed object (for browser/Node) or a JSON string.
// For file-system loading in Node, caller must read the file and pass the text.

export function loadMxb(graphOrText) {
  const graph = typeof graphOrText === 'string'
    ? JSON.parse(graphOrText)
    : graphOrText;

  if (!graph || !Array.isArray(graph.nodes)) {
    throw new Error('loadMxb: invalid .mxb — expected { nodes: [], edges: [] }');
  }

  // Build dep lists from edges if not inline
  const depMap = {};
  for (const node of graph.nodes) depMap[node.id] = [];
  for (const edge of (graph.edges || [])) {
    if (depMap[edge.to]) depMap[edge.to].push(edge.from);
  }
  const nodes = graph.nodes.map(n => ({
    ...n,
    deps: n.deps ?? depMap[n.id] ?? [],
  }));

  return { nodes, edges: graph.edges || [] };
}

// ─── Build runtime: graph → job map ──────────────────────────────────────────

export function buildRuntime(graph) {
  const jobs = {};
  for (const node of graph.nodes) {
    const unit = nodeToUnit(node);
    jobs[node.id] = createMicronautJob(node, unit);
  }
  return jobs;
}

// ─── Topological ready-set computation ───────────────────────────────────────

function computeReady(jobs) {
  const ready = [];
  for (const job of Object.values(jobs)) {
    if (job.state !== JOB_STATE.PENDING) continue;
    const allDepsDone = job.deps.every(depId => jobs[depId]?.state === JOB_STATE.DONE);
    if (allDepsDone) ready.push(job);
  }
  return ready;
}

// ─── UnifiedAPI submit stub ───────────────────────────────────────────────────
// Actual work function signature: (unit, env) => Promise<any> | any

async function submitUnit(unit, env) {
  if (typeof unit.work === 'function') {
    return unit.work(unit, env);
  }
  // Default: resolve immediately with unit meta
  return { unitId: unit.unitId, laneType: unit.laneType, meta: unit.meta };
}

// ─── runGraph: ⟁Pop execute_graph(graph) ... ⟁Xul⟁ ─────────────────────────
//
// ⟁Pop execute_graph(graph)
//   ⟁Wo⟁  ready = nodes with no dependencies
//   ⟁Sek⟁ while exists unfinished jobs:
//            for node in ready: UnifiedAPI.submit(unit.work, {...})
//            wait for completions
//            update jobState, recompute ready set
// ⟁Xul⟁

export async function runGraph(graph, env = {}) {
  const jobs = buildRuntime(graph);

  // ⟁Wo⟁ — initial ready set
  let ready = computeReady(jobs);

  const results = {};

  // ⟁Sek⟁ — execute until all done
  while (ready.length > 0) {
    // Mark all ready jobs as running
    for (const job of ready) job.state = JOB_STATE.RUNNING;

    // Dispatch by lane type: batch → concurrent, thread → sequential, process → sequential
    const batchJobs  = ready.filter(j => j.unit.laneType === LANE_TYPES.batch);
    const serialJobs = ready.filter(j => j.unit.laneType !== LANE_TYPES.batch);

    // Batch: all in parallel
    const batchPromises = batchJobs.map(async job => {
      try {
        job.result = await submitUnit(job.unit, { ...env, deps: job.deps, jobs });
        job.state  = JOB_STATE.DONE;
        results[job.jobId] = job.result;
      } catch (err) {
        job.state = JOB_STATE.ERROR;
        job.error = err;
      }
    });

    // Serial: one at a time
    const serialRun = async () => {
      for (const job of serialJobs) {
        try {
          job.result = await submitUnit(job.unit, { ...env, deps: job.deps, jobs });
          job.state  = JOB_STATE.DONE;
          results[job.jobId] = job.result;
        } catch (err) {
          job.state = JOB_STATE.ERROR;
          job.error = err;
        }
      }
    };

    await Promise.all([...batchPromises, serialRun()]);

    // Recompute ready set — any error halts further scheduling
    const hasError = Object.values(jobs).some(j => j.state === JOB_STATE.ERROR);
    if (hasError) {
      const failed = Object.values(jobs).filter(j => j.state === JOB_STATE.ERROR);
      const msg    = failed.map(j => `${j.jobId}: ${j.error?.message || j.error}`).join('; ');
      throw new Error(`runGraph: job(s) failed — ${msg}`);
    }

    ready = computeReady(jobs);
  }

  // ⟁Xul⟁ — seal: return final job states + results
  return { jobs, results };
}

// ─── K'uhul scheduling annotation ─────────────────────────────────────────────
// Annotate a .mxb graph with K'uhul fold metadata for XCFE routing.

export function annotateKuhulFolds(graph) {
  const annotated = graph.nodes.map(node => {
    let kuhul_phase;
    switch (node.type) {
      case 'source':      kuhul_phase = 'Pop'; break;
      case 'transform':   kuhul_phase = 'Yax'; break;
      case 'computation': kuhul_phase = 'Sek'; break;
      case 'combine':     kuhul_phase = "Ch'en"; break;
      case 'sink':        kuhul_phase = 'Xul'; break;
      default:            kuhul_phase = 'Wo';
    }
    return { ...node, kuhul_phase };
  });
  return { ...graph, nodes: annotated };
}

// ─── Graph introspection helpers ───────────────────────────────────────────────

export function getNodesByType(graph, type) {
  return graph.nodes.filter(n => n.type === type);
}

export function getNodeById(graph, id) {
  return graph.nodes.find(n => n.id === id) || null;
}

export function graphSummary(graph) {
  const counts = {};
  for (const node of graph.nodes) {
    counts[node.type] = (counts[node.type] || 0) + 1;
  }
  return {
    nodeCount: graph.nodes.length,
    edgeCount: (graph.edges || []).length,
    byType:    counts,
  };
}
