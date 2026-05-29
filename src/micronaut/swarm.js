// K'UHUL Swarm — port of kuhul_swarm_server.cpp (KUHUL.EXE.v3.0.0)
//
// Pure in-memory port (no HTTP server, no fs hot-reload — those are Windows-native).
// PhaseArray, Agent, Skill, Swarm, SwarmManager fully functional in browser/Node.
//
// Swarm strategies: hierarchical, mesh, star, broadcast, consensus
// Phase routing: π-geodesic signature (16-byte) computed from content hash XOR

export const PHASE_ARRAY_SIZE = 16;
export const DEFAULT_PORT     = 8080;
export const WEBSOCKET_PORT   = 8081;
export const SWARM_STRATEGIES = Object.freeze(['hierarchical', 'mesh', 'star', 'broadcast', 'consensus']);

// ── PhaseArray ───────────────────────────────────────────────────────────────

export class PhaseArray {
  constructor(source = null) {
    this.bytes = new Uint8Array(PHASE_ARRAY_SIZE);
    if (typeof source === 'string') {
      this._fromString(source);
    } else if (source instanceof Uint8Array || Array.isArray(source)) {
      this.bytes.set(Array.from(source).slice(0, PHASE_ARRAY_SIZE));
    }
  }

  _fromString(content) {
    // Replicate std::hash<std::string> via djb2 + XOR with content bytes
    let hash = 5381;
    for (let i = 0; i < content.length; i++) {
      hash = ((hash << 5) + hash + content.charCodeAt(i)) >>> 0;
    }
    for (let i = 0; i < PHASE_ARRAY_SIZE; i++) {
      this.bytes[i] = (hash >>> (i * 8)) & 0xFF;
      if (i < content.length) this.bytes[i] ^= content.charCodeAt(i) & 0xFF;
    }
  }

  distanceTo(other) {
    let sum = 0;
    for (let i = 0; i < PHASE_ARRAY_SIZE; i++) {
      const diff = this.bytes[i] - other.bytes[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  toArray() { return Array.from(this.bytes); }
  toJSON()  { return this.toArray(); }
}

// ── Markdown frontmatter parser ──────────────────────────────────────────────

function parseFrontmatter(content) {
  const result = { fields: {}, body: content };
  const m = content.match(/^---\n([\s\S]*?)\n---/);
  if (!m) return result;

  const fm = m[1]; let body = content.slice(m[0].length).replace(/^\n+/, '');
  result.body = body;
  const fields = {};

  // Parse key: value lines and list blocks
  const lines = fm.split('\n');
  let currentListKey = null;
  for (const line of lines) {
    const listItem = line.match(/^\s+-\s+(.+)/);
    if (listItem && currentListKey) {
      if (!Array.isArray(fields[currentListKey])) fields[currentListKey] = [];
      fields[currentListKey].push(listItem[1].trim());
      continue;
    }
    currentListKey = null;

    const kv = line.match(/^(\w[\w'-]*):\s*(.*)$/);
    if (!kv) continue;
    const key = kv[1]; const val = kv[2].trim();
    if (val === '') {
      fields[key] = [];
      currentListKey = key;
    } else {
      const n = parseFloat(val);
      fields[key] = isNaN(n) ? val.replace(/^"(.*)"$/, '$1') : n;
    }
  }
  result.fields = fields;
  return result;
}

// ── Skill ────────────────────────────────────────────────────────────────────

export function createSkill(id, markdown = '') {
  const { fields, body } = parseFrontmatter(markdown);
  return {
    id,
    name:          fields.name          || id,
    description:   fields.description   || '',
    category:      fields.category      || '',
    weight:        typeof fields.weight === 'number' ? fields.weight : 0.5,
    prerequisites: Array.isArray(fields.prerequisites) ? fields.prerequisites : [],
    triggers:      Array.isArray(fields.triggers)      ? fields.triggers      : [],
    action:        body,
    phase:         new PhaseArray(markdown),
    useCount:      0,
    lastUsed:      null,
  };
}

// ── Agent ────────────────────────────────────────────────────────────────────

export function createAgent(id, markdown = '') {
  const { fields } = parseFrontmatter(markdown);
  return {
    id,
    name:        fields.name        || id,
    role:        fields.role        || '',
    personality: fields.personality || '',
    skills:      Array.isArray(fields.skills) ? fields.skills : [],
    swarmIds:    Array.isArray(fields.swarms) ? fields.swarms : [],
    customParams: {},
    phase:       new PhaseArray(markdown + (fields.name || id)),
    active:      true,
  };
}

// ── Swarm ────────────────────────────────────────────────────────────────────

export function createSwarm(id, markdown = '', agentMap = {}) {
  const { fields } = parseFrontmatter(markdown);
  const agentIds = Array.isArray(fields.agents) ? fields.agents : [];
  const agents   = agentIds.map(aid => agentMap[aid]).filter(Boolean);
  return {
    id,
    name:            fields.name     || id,
    strategy:        fields.strategy || 'mesh',
    agentIds,
    agents,
    sharedState:     {},
    tasksCompleted:  0,
    avgResponseTime: 0,
    swarmCoherence:  0,
  };
}

export function computeSwarmCoherence(swarm) {
  const agents = swarm.agents;
  if (agents.length < 2) { swarm.swarmCoherence = 1; return 1; }
  let total = 0; let pairs = 0;
  for (let i = 0; i < agents.length; i++) {
    for (let j = i + 1; j < agents.length; j++) {
      total += agents[i].phase.distanceTo(agents[j].phase);
      pairs++;
    }
  }
  swarm.swarmCoherence = 1 - (total / pairs / 255);
  return swarm.swarmCoherence;
}

// ── SwarmManager ─────────────────────────────────────────────────────────────

export class SwarmManager {
  constructor() {
    this._agents = new Map();
    this._skills = new Map();
    this._swarms = new Map();
  }

  registerAgent(agent)  { this._agents.set(agent.id, agent); return this; }
  registerSkill(skill)  { this._skills.set(skill.id, skill); return this; }
  registerSwarm(swarm)  { this._swarms.set(swarm.id, swarm); return this; }

  getAgent(id)  { return this._agents.get(id) || null; }
  getSkill(id)  { return this._skills.get(id) || null; }
  getSwarm(id)  { return this._swarms.get(id) || null; }
  listAgents()  { return Array.from(this._agents.values()); }
  listSkills()  { return Array.from(this._skills.values()); }
  listSwarms()  { return Array.from(this._swarms.values()); }

  // Route task to agent with minimum PhaseArray distance
  routeByPhase(phaseArray) {
    let bestId = null; let bestDist = Infinity;
    for (const [id, agent] of this._agents) {
      const dist = phaseArray.distanceTo(agent.phase);
      if (dist < bestDist) { bestDist = dist; bestId = id; }
    }
    return { agentId: bestId, distance: bestDist };
  }

  executeSkill(agentId, skillId, params = {}) {
    const agent = this._agents.get(agentId);
    const skill = this._skills.get(skillId);
    if (!agent) return { error: `Agent not found: ${agentId}` };
    if (!skill) return { error: `Skill not found: ${skillId}` };
    if (!agent.skills.includes(skillId)) return { error: `Agent ${agentId} does not have skill ${skillId}` };
    skill.useCount++;
    skill.lastUsed = Date.now();
    return { success: true, agent: agent.name, skill: skill.name, params, phaseDistance: agent.phase.distanceTo(skill.phase) };
  }

  executeSwarm(swarmId, task, params = {}) {
    const swarm = this._swarms.get(swarmId);
    if (!swarm) return { error: `Swarm not found: ${swarmId}` };
    const results = swarm.agents.map(a => ({ agentId: a.id, task, params }));
    swarm.tasksCompleted++;
    const coherence = computeSwarmCoherence(swarm);
    return { success: true, swarm: swarm.name, strategy: swarm.strategy, task, agentCount: swarm.agents.length, swarmCoherence: coherence, results, tasksCompleted: swarm.tasksCompleted };
  }

  createDynamicSwarm(agentIds, strategy = 'mesh') {
    const id    = `dynamic_${Date.now()}`;
    const agents = agentIds.map(aid => this._agents.get(aid)).filter(Boolean);
    const swarm  = { id, name: 'Dynamic Swarm', strategy, agentIds, agents, sharedState: {}, tasksCompleted: 0, avgResponseTime: 0, swarmCoherence: 0 };
    this._swarms.set(id, swarm);
    return swarm;
  }

  getStats() {
    let totalCaps = 0;
    for (const a of this._agents.values()) totalCaps += a.skills.length;
    return { agents: this._agents.size, skills: this._skills.size, swarms: this._swarms.size, total_capabilities: totalCaps };
  }
}

// REST API route table (informational — matches kuhul_swarm_server.cpp routes)
export const SWARM_API_ROUTES = Object.freeze({
  'GET /api':             'API info',
  'GET /api/agents':      'List all agents',
  'GET /api/agents/{id}': 'Get agent details',
  'GET /api/skills':      'List all skills',
  'GET /api/skills/{id}': 'Get skill details',
  'GET /api/swarms':      'List all swarms',
  'POST /api/swarms/{id}':'Execute swarm task',
  'POST /api/execute':    'Execute agent skill',
  'POST /api/route':      'Route task by phase',
  'POST /api/swarm/create':'Create dynamic swarm',
  'GET /api/stats':       'System statistics',
});
