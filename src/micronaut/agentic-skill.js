// Agentic Micronaut skill manifest — fleet agents + XCFE ops (v3.5.0-WebX)
// Ported from .gpu_trainer/skills/agentic-micronaut/skill.matrix.toml

export const AGENTIC_MICRONAUT_ACTIONS = Object.freeze({
  arbitrate:               'AgenticMicronautActions.arbitrate',
  project_skill:           'AgenticMicronautActions.project_skill',
  resolve_agent:           'AgenticMicronautActions.resolve_agent',
  list_registry:           'AgenticMicronautActions.list_registry',
  pack_shard:              'AgenticMicronautActions.pack_shard',
  wire_fleet_agent:        'AgenticMicronautActions.wire_fleet_agent',
  configure_agent_pipeline:'AgenticMicronautActions.configure_agent_pipeline',
  emit_agent_event:        'AgenticMicronautActions.emit_agent_event',
  validate_agent_contract: 'AgenticMicronautActions.validate_agent_contract',
  export_agent_definition: 'AgenticMicronautActions.export_agent_definition',
});

// 5 fleet agent types — each maps to a UNIFIED_SERVICES port range (25101-25105)
export const FLEET_AGENTS = Object.freeze({
  planner:      { type:'planning',     port:25101, skills:['requirement-analysis','phase-decomposition','dependency-mapping','resource-estimation','risk-assessment'] },
  executor:     { type:'execution',    port:25102, skills:['task-execution','result-validation','progress-tracking','error-handling','tool-calling'] },
  coordinator:  { type:'coordination', port:25103, skills:['multi-turn-coordination','consensus-building','debate','semantic-search','delegation'] },
  responder:    { type:'response',     port:25104, skills:['semantic-search','memory-recall','prompt-engineering','response-formatting','token-counting'] },
  diagnostician:{ type:'health',       port:25105, skills:['tool-calling','result-validation','memory-store','memory-retrieve','fetch-url'] },
});

// XCFE opcode table — micronaut DI/lifecycle operations
export const MICRONAUT_XCFE_OPS = Object.freeze({
  start_project:      { action:'micronaut.startProject',      xcfe_op:'PROJECT_INIT' },
  configure_endpoint: { action:'micronaut.configureEndpoint', xcfe_op:'ENDPOINT_CONFIG' },
  secure_endpoint:    { action:'micronaut.secureEndpoint',    xcfe_op:'AUTH_GATE' },
  inject_bean:        { action:'micronaut.injectBean',        xcfe_op:'DI_INJECT' },
  schedule_task:      { action:'micronaut.scheduleTask',      xcfe_op:'SCHEDULE' },
  event_bridge:       { action:'micronaut.eventBridge',       xcfe_op:'EVENT_ROUTE' },
  tool_runtime_step:  { action:'micronaut.toolRuntimeStep',   xcfe_op:'TICK' },
  tool_state_set:     { action:'micronaut.toolStateSet',      xcfe_op:'STATE_SNAPSHOT' },
  tool_schedule_task: { action:'micronaut.toolScheduleTask',  xcfe_op:'SCHEDULE_TASK' },
  tool_event_emit:    { action:'micronaut.toolEventEmit',     xcfe_op:'EVENT_EMIT' },
});

// Alias routes under the 'super' namespace (from super_routes.toml)
export const SUPER_ALIAS_ROUTES = Object.freeze({
  'super.micronaut_init':           'AgenticMicronautAddon.init',
  'super.micronaut_configure':      'AgenticMicronautAddon.configure',
  'super.micronaut_secure':         'AgenticMicronautAddon.secure',
  'super.micronaut_inject':         'AgenticMicronautAddon.inject',
  'super.micronaut_schedule':       'AgenticMicronautAddon.schedule',
  'super.micronaut_event':          'AgenticMicronautAddon.event',
  'super.stack_audit':              'ASXCFEStackIntel.audit',
  'super.stack_topology':           'ASXCFEStackIntel.topology',
  'super.stack_gaps':               'ASXCFEStackIntel.gaps',
  'super.stack_next_path':          'ASXCFEStackIntel.next_path',
  'super.stack_summary':            'ASXCFEStackIntel.summary',
  'super.dataset_manifest':         'DatasetTraining.generate_manifest',
  'super.dataset_token_estimate':   'DatasetTraining.token_estimate',
  'super.dataset_prepare':          'DatasetTraining.prepare_for_training',
  'super.dataset_publish':          'DatasetTraining.publish_artifacts',
});

export function getFleetAgent(role) {
  return FLEET_AGENTS[role] || null;
}

export function getXcfeOp(action) {
  return MICRONAUT_XCFE_OPS[action] || null;
}
