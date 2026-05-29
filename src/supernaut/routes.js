// Supernaut skill route manifest + skill matrix — canonical route table (v3.5.0-WebX)
// Ported from .gpu_trainer/skills/supernaut/supernaut_manifest.json + skill.matrix.toml
// Maps HTTP path → { action, skill_intent } for all Supernaut dispatch surfaces.

// 35 merged skills that compose the Supernaut runtime pack
export const SUPERNAUT_MERGED_SKILLS = Object.freeze([
  'workflow-orchestrator', 'sql-skill', 'supernaut', 'wordpad', 'micronaut',
  'codex-agent', 'asx-verifier', 'agentic-micronaut', 'as-xcfe-stack-intel',
  'micronaut-model', 'windows-sdk', 'scx2-runtime', 'scxqdds', 'scxq2-vector',
  'vs-native-tools', 'sk-coordinator', 'mx2lex', 'netfx-sdk', 'pipx-compiler',
  'powershell-ise', 'vs2019-tools', 'vs2022-tools', 'vs-insiders', 'team-lead',
  'dataset-training', 'cloudflare-deploy', 'project-factory', 'commands', 'doc',
  'sqlite3', 'msbuild-nuget', 'figma', 'figma-implement-design', 'vsdevcmd', 'winkit-81',
]);

export const SUPERNAUT_SKILL_MATRIX = Object.freeze({
  name:             'supernaut',
  version:          '1.0.0',
  runtime:          'xcfe',
  shard_checkpoint: true,
  shard_registry:   'artifacts/training/shard-manifest.json',
  shard_index:      'artifacts/training/shard-index.json',
  shard_format:     'scxqdds',
  merged_skills:    SUPERNAUT_MERGED_SKILLS,
  actions: Object.freeze({
    compose:         'SupernautActions.compose',
    log_action:      'SupernautActions.log_action',
    run_pipx:        'SupernautActions.run_pipx',
    arbitrate:       'SupernautActions.arbitrate',
    list_skills:     'SupernautActions.list_skills',
    merge_manifests: 'SupernautActions.merge_manifests',
    pack_shard:      'SupernautActions.pack_shard',
  }),
});

export const SUPERNAUT_MANIFEST = Object.freeze({
  assetRoots: {
    codex: {
      class:   'codex/class/supernaut.schema.xjson',
      control: 'codex/control/supernaut.control.asx',
    },
  },
  sourceRoots: { codex: 'codex/' },
});

// Full route table — path → { action, skill_intent }
export const SUPERNAUT_ROUTES = Object.freeze({
  '/skill/supernaut/health':            { action:'SupernautActions.health',                      skill_intent:'health' },
  '/skill/supernaut/model':             { action:'SupernautActions.modelProbe',                  skill_intent:'model_probe' },
  '/skill/supernaut/model/select':      { action:'SupernautActions.modelSelect',                 skill_intent:'model_select' },

  '/skill/init':                        { action:'MicronautActions.startProject',                skill_intent:'init' },
  '/skill/configure':                   { action:'MicronautActions.configureEndpoint',           skill_intent:'configure' },
  '/skill/secure':                      { action:'MicronautActions.secureEndpoint',              skill_intent:'secure' },
  '/skill/inject':                      { action:'MicronautActions.injectBean',                  skill_intent:'inject' },
  '/skill/schedule':                    { action:'MicronautActions.scheduleTask',                skill_intent:'schedule' },
  '/skill/event':                       { action:'MicronautActions.eventBridge',                 skill_intent:'event' },
  '/skill/tool/di/resolve':             { action:'MicronautActions.toolDiResolve',               skill_intent:'tool_di_resolve' },
  '/skill/tool/di/inject':              { action:'MicronautActions.toolDiInject',                skill_intent:'tool_di_inject' },
  '/skill/tool/state/get':              { action:'MicronautActions.toolStateGet',                skill_intent:'tool_state_get' },
  '/skill/tool/state/set':              { action:'MicronautActions.toolStateSet',                skill_intent:'tool_state_set' },
  '/skill/tool/schedule/task':          { action:'MicronautActions.toolScheduleTask',            skill_intent:'tool_schedule_task' },
  '/skill/tool/event/emit':             { action:'MicronautActions.toolEventEmit',               skill_intent:'tool_event_emit' },
  '/skill/tool/runtime/step':           { action:'MicronautActions.toolRuntimeStep',             skill_intent:'tool_runtime_step' },
  '/skill/tool/runtime/run':            { action:'MicronautActions.toolRuntimeRun',              skill_intent:'tool_runtime_run' },
  '/skill/tool/cluster/migrate':        { action:'MicronautActions.toolClusterMigrate',          skill_intent:'tool_cluster_migrate' },
  '/skill/tool/cluster/replicate':      { action:'MicronautActions.toolClusterReplicate',        skill_intent:'tool_cluster_replicate' },

  '/skill/learn':                       { action:'SemanticPlannerActions.generate_learning_plan',skill_intent:'learn_topic' },
  '/skill/orchestrate':                 { action:'SemanticPlannerActions.build_multi_step_plan', skill_intent:'orchestrate_multi_call' },
  '/skill/plan':                        { action:'LearningSchedulerActions.schedule_learning',   skill_intent:'schedule_learning' },
  '/skill/di':                          { action:'MicronautDIActions.inject_component',          skill_intent:'micronaut_di_inject' },
  '/skill/micronaut/init':              { action:'MicronautAddonActions.start_project',          skill_intent:'micronaut_init' },
  '/skill/micronaut/configure':         { action:'MicronautAddonActions.configure_endpoint',     skill_intent:'micronaut_configure' },
  '/skill/micronaut/secure':            { action:'MicronautAddonActions.secure_endpoint',        skill_intent:'micronaut_secure' },
  '/skill/micronaut/inject':            { action:'MicronautAddonActions.inject_bean',            skill_intent:'micronaut_inject' },
  '/skill/micronaut/schedule':          { action:'MicronautAddonActions.schedule_task',          skill_intent:'micronaut_schedule' },
  '/skill/micronaut/event':             { action:'MicronautAddonActions.event_bridge',           skill_intent:'micronaut_event' },

  '/skill/semantic/intent':             { action:'SKCoordinatorActions.parse_intent',            skill_intent:'semantic_parse_intent' },
  '/skill/semantic/learn':              { action:'SKCoordinatorActions.schedule_learning_goal',  skill_intent:'schedule_learning_goal' },
  '/skill/semantic/di':                 { action:'SKCoordinatorActions.route_to_micronaut',      skill_intent:'orchestrate_micronaut' },
  '/skill/semantic/goalgraph':          { action:'SKCoordinatorActions.build_goal_graph',        skill_intent:'meta_plan_goal_graph' },

  '/skill/workflow/plan':               { action:'WorkflowOrchestratorActions.plan_project',     skill_intent:'plan_project' },
  '/skill/workflow/automate':           { action:'WorkflowOrchestratorActions.automate_workflow',skill_intent:'automate_workflow' },
  '/skill/workflow/scaffold':           { action:'WorkflowOrchestratorActions.scaffold_kuhul_app',skill_intent:'scaffold_kuhul_app' },

  '/skill/sql/connect':                 { action:'SqlActions.connect',                           skill_intent:'connect' },
  '/skill/sql/tables':                  { action:'SqlActions.listTables',                        skill_intent:'list_tables' },
  '/skill/sql/schema':                  { action:'SqlActions.describeTable',                     skill_intent:'describe_table' },
  '/skill/sql/query':                   { action:'SqlActions.query',                             skill_intent:'query' },
  '/skill/sql/upsert':                  { action:'SqlActions.upsert',                            skill_intent:'upsert' },

  '/skill/code/read':                   { action:'CodeSkillActions.read',                        skill_intent:'code_read' },
  '/skill/code/patch':                  { action:'CodeSkillActions.patch',                       skill_intent:'code_patch' },
  '/skill/code/test':                   { action:'CodeSkillActions.test',                        skill_intent:'code_test' },
  '/skill/code/review':                 { action:'CodeSkillActions.review',                      skill_intent:'code_review' },
  '/skill/dolphin/catalog':             { action:'DolphinCoderActions.catalog',                  skill_intent:'dolphin_catalog' },
  '/skill/dolphin/refine-plan':         { action:'DolphinCoderActions.refinePlan',               skill_intent:'dolphin_refine_plan' },
  '/skill/ping':                        { action:'SkillActions.ping',                            skill_intent:'ping' },
  '/skill/echo':                        { action:'SkillActions.echo',                            skill_intent:'echo' },

  '/skill/powershell/health':           { action:'PowerShellAgentActions.health',                skill_intent:'powershell_health' },
  '/skill/powershell/layout':           { action:'PowerShellAgentActions.scaffoldLayout',        skill_intent:'powershell_layout' },
  '/skill/powershell/catalog':          { action:'PowerShellAgentActions.commandCatalog',        skill_intent:'powershell_catalog' },
  '/skill/powershell/run':              { action:'PowerShellAgentActions.runCommand',            skill_intent:'powershell_run' },

  '/skill/supernaut/asx/verify':        { action:'AsxVerifierActions.verify',                   skill_intent:'asx_verify' },
  '/skill/supernaut/asx/verify/bad':    { action:'AsxVerifierActions.verifyBad',                skill_intent:'asx_verify_bad' },
  '/skill/supernaut/scx2/smoke':        { action:'Scx2RuntimeActions.smoke',                    skill_intent:'scx2_smoke' },
  '/skill/supernaut/scxq2/run':         { action:'Scxq2VectorActions.run',                      skill_intent:'scxq2_run' },
  '/skill/supernaut/scxqdds/decode':    { action:'ScxqddsActions.decode',                       skill_intent:'scxqdds_decode' },
  '/skill/supernaut/scxqdds/selftest':  { action:'ScxqddsActions.selftest',                     skill_intent:'scxqdds_selftest' },
  '/skill/supernaut/scxqdds/vector':    { action:'ScxqddsActions.vectorRun',                    skill_intent:'scxqdds_vector_run' },
  '/skill/supernaut/mx2lex/compile':    { action:'Mx2lexActions.compile',                       skill_intent:'mx2lex_compile' },
  '/skill/supernaut/mx2lex/oracle':     { action:'Mx2lexActions.oracle',                        skill_intent:'mx2lex_oracle' },
  '/skill/supernaut/mx2lex/vector':     { action:'Mx2lexActions.vectorRun',                     skill_intent:'mx2lex_vector_run' },
  '/skill/supernaut/pipx/run':          { action:'PipxCompilerActions.run',                     skill_intent:'pipx_run' },

  '/skill/supernaut/intel/diff':        { action:'AsXcfeStackIntelActions.diffStacks',          skill_intent:'intel_diff' },
  '/skill/supernaut/intel/agents':      { action:'AsXcfeStackIntelActions.listAgents',          skill_intent:'intel_list_agents' },
  '/skill/supernaut/intel/validate':    { action:'AsXcfeStackIntelActions.validateSync',        skill_intent:'intel_validate' },
  '/skill/supernaut/intel/gaps':        { action:'AsXcfeStackIntelActions.reportGaps',          skill_intent:'intel_gaps' },
  '/skill/supernaut/intel/sync':        { action:'AsXcfeStackIntelActions.syncSkill',           skill_intent:'intel_sync' },

  '/skill/supernaut/agentic/arbitrate': { action:'AgenticMicronautActions.arbitrate',           skill_intent:'micronaut_arbitrate' },
  '/skill/supernaut/agentic/project':   { action:'AgenticMicronautActions.projectSkill',        skill_intent:'micronaut_project' },
  '/skill/supernaut/agentic/resolve':   { action:'AgenticMicronautActions.resolveAgent',        skill_intent:'micronaut_resolve' },
  '/skill/supernaut/agentic/registry':  { action:'AgenticMicronautActions.listRegistry',        skill_intent:'micronaut_registry' },
  '/skill/supernaut/agentic/pack':      { action:'AgenticMicronautActions.packShard',           skill_intent:'micronaut_pack' },

  '/skill/supernaut/model/backends':    { action:'MicronautModelActions.backends',              skill_intent:'model_backends' },
  '/skill/supernaut/model/catalog':     { action:'MicronautModelActions.catalog',               skill_intent:'model_catalog' },
  '/skill/supernaut/model/xjson':       { action:'MicronautModelActions.xjsonFormat',           skill_intent:'model_xjson' },
  '/skill/supernaut/model/fold':        { action:'MicronautModelActions.foldMap',               skill_intent:'model_fold_map' },
  '/skill/supernaut/model/new':         { action:'MicronautModelActions.new',                   skill_intent:'model_new' },
  '/skill/supernaut/model/register':    { action:'MicronautModelActions.register',              skill_intent:'model_register' },
  '/skill/supernaut/model/list':        { action:'MicronautModelActions.list',                  skill_intent:'model_list' },
  '/skill/supernaut/model/fetch':       { action:'MicronautModelActions.fetchWeights',          skill_intent:'model_fetch_weights' },
  '/skill/supernaut/model/native':      { action:'MicronautModelActions.nativeBridge',          skill_intent:'model_native_bridge' },

  '/skill/supernaut/team/assign':       { action:'TeamLeadActions.assign_tasks',                skill_intent:'team_assign' },
  '/skill/supernaut/team/dispatch':     { action:'TeamLeadActions.dispatch_agents',             skill_intent:'team_dispatch' },
  '/skill/supernaut/team/board/update': { action:'TeamLeadActions.update_board',                skill_intent:'team_board_update' },
  '/skill/supernaut/team/board':        { action:'TeamLeadActions.get_board',                   skill_intent:'team_board_get' },
  '/skill/supernaut/team/synthesize':   { action:'TeamLeadActions.synthesize',                  skill_intent:'team_synthesize' },
  '/skill/supernaut/team/task/done':    { action:'TeamLeadActions.complete_task',               skill_intent:'team_task_done' },
  '/skill/supernaut/team/task/block':   { action:'TeamLeadActions.block_task',                  skill_intent:'team_task_block' },

  '/skill/supernaut/vs/x64':            { action:'VsNativeToolsActions.x64',                   skill_intent:'vs_x64' },
  '/skill/supernaut/vs/x86':            { action:'VsNativeToolsActions.x86',                   skill_intent:'vs_x86' },
  '/skill/supernaut/vs/x86_x64':        { action:'VsNativeToolsActions.x86x64',                skill_intent:'vs_x86_x64' },
  '/skill/supernaut/vs/x64_x86':        { action:'VsNativeToolsActions.x64x86',                skill_intent:'vs_x64_x86' },

  '/skill/supernaut/winsdk/header':     { action:'WindowsSdkActions.resolveHeader',             skill_intent:'winsdk_header' },
  '/skill/supernaut/winsdk/cmake':      { action:'WindowsSdkActions.cmakeFlags',                skill_intent:'winsdk_cmake' },

  '/skill/supernaut/dataset/manifest':  { action:'DatasetTrainingActions.generateManifest',     skill_intent:'dataset_manifest' },
  '/skill/supernaut/dataset/discover':  { action:'DatasetTrainingActions.discover',             skill_intent:'dataset_discover' },
  '/skill/supernaut/dataset/ingest':    { action:'DatasetTrainingActions.ingest',               skill_intent:'dataset_ingest' },
  '/skill/supernaut/dataset/status':    { action:'DatasetTrainingActions.status',               skill_intent:'dataset_status' },

  '/skill/supernaut/factory/build':     { action:'FactoryMicronautActions.build',              skill_intent:'factory_build' },
  '/skill/supernaut/factory/templates': { action:'FactoryMicronautActions.templates',           skill_intent:'factory_templates' },
  '/skill/supernaut/factory/health':    { action:'FactoryMicronautActions.health',             skill_intent:'factory_health' },
  '/skill/supernaut/factory/metrics':   { action:'FactoryMicronautActions.metrics',            skill_intent:'factory_metrics' },

  '/skill/supernaut/pool/list':         { action:'MicronautPoolActions.list',                   skill_intent:'pool_list' },
  '/skill/supernaut/pool/dispatch':     { action:'MicronautPoolActions.dispatch',               skill_intent:'pool_dispatch' },
  '/skill/supernaut/pool/invoke':       { action:'MicronautPoolActions.invoke',                 skill_intent:'pool_invoke' },
  '/skill/supernaut/pool/health/all':   { action:'MicronautPoolActions.healthAll',              skill_intent:'pool_health_all' },
  '/skill/supernaut/pool/health':       { action:'MicronautPoolActions.health',                 skill_intent:'pool_health' },
  '/skill/supernaut/pool/eval':         { action:'MicronautPoolActions.eval',                   skill_intent:'pool_eval' },
  '/skill/supernaut/pool/quarantine':   { action:'MicronautPoolActions.quarantine',             skill_intent:'pool_quarantine' },
  '/skill/supernaut/pool/pending':      { action:'MicronautPoolActions.pending',                skill_intent:'pool_pending' },
  '/skill/supernaut/pool/evict':        { action:'MicronautPoolActions.confirmEvict',           skill_intent:'pool_evict' },
  '/skill/supernaut/pool/allow':        { action:'MicronautPoolActions.confirmAllow',           skill_intent:'pool_allow' },
  '/skill/supernaut/pool/replay':       { action:'MicronautPoolActions.replay',                 skill_intent:'pool_replay' },
  '/skill/supernaut/pool/metrics/all':  { action:'MicronautPoolActions.metricsAll',             skill_intent:'pool_metrics_all' },

  '/skill/supernaut/sys/spec':          { action:'SystemSpecActions.spec',                      skill_intent:'sys_spec' },
  '/skill/supernaut/sys/stack':         { action:'SystemSpecActions.stack',                     skill_intent:'sys_stack' },
  '/skill/supernaut/sys/models':        { action:'SystemSpecActions.models',                    skill_intent:'sys_models' },
  '/skill/supernaut/sys/agents':        { action:'SystemSpecActions.agents',                    skill_intent:'sys_agents' },
  '/skill/supernaut/sys/igpu':          { action:'SystemSpecActions.igpuSpec',                  skill_intent:'sys_igpu' },
  '/skill/supernaut/sys/shard':         { action:'SystemSpecActions.shardSpec',                 skill_intent:'sys_shard_spec' },
  '/skill/supernaut/sys/shard/stack':   { action:'SystemSpecActions.shardStackSpec',            skill_intent:'sys_shard_stack' },
  '/skill/supernaut/sys/tiktok':        { action:'SystemSpecActions.tiktokSpec',                skill_intent:'sys_tiktok' },
  '/skill/supernaut/sys/vocab':         { action:'SystemSpecActions.vocabSpec',                 skill_intent:'sys_vocab' },
  '/skill/supernaut/sys/copilot':       { action:'SystemSpecActions.copilotSpec',               skill_intent:'sys_copilot' },

  '/skill/supernaut/web/search':        { action:'WebSearchActions.search',                     skill_intent:'web_search' },
  '/skill/supernaut/web/answer':        { action:'WebSearchActions.answer',                     skill_intent:'web_answer' },
  '/skill/supernaut/web/compose':       { action:'WebSearchActions.compose',                    skill_intent:'web_compose' },
  '/skill/supernaut/web/policy':        { action:'WebSearchActions.policy',                     skill_intent:'web_policy' },
  '/skill/supernaut/web/health':        { action:'WebSearchActions.health',                     skill_intent:'web_health' },

  '/skill/supernaut/vs/insiders/devshell':             { action:'VsInsidersActions.devshellInsiders',      skill_intent:'vs_insiders_devshell' },
  '/skill/supernaut/vs/insiders/devcmd':               { action:'VsInsidersActions.devcmdInsiders',       skill_intent:'vs_insiders_devcmd' },
  '/skill/supernaut/vs/insiders/buildtools/devshell':  { action:'VsInsidersActions.devshellBuildtools',   skill_intent:'vs_insiders_buildtools_devshell' },
  '/skill/supernaut/vs/insiders/buildtools/devcmd':    { action:'VsInsidersActions.devcmdBuildtools',     skill_intent:'vs_insiders_buildtools_devcmd' },

  '/skill/supernaut/vs/2019/devshell':          { action:'Vs2019Actions.devshell',   skill_intent:'vs2019_devshell' },
  '/skill/supernaut/vs/2019/devcmd':            { action:'Vs2019Actions.devcmd',     skill_intent:'vs2019_devcmd' },
  '/skill/supernaut/vs/2019/x64':               { action:'Vs2019Actions.x64',        skill_intent:'vs2019_x64' },
  '/skill/supernaut/vs/2019/x86':               { action:'Vs2019Actions.x86',        skill_intent:'vs2019_x86' },
  '/skill/supernaut/vs/2019/cross/x86_x64':     { action:'Vs2019Actions.x86_x64',   skill_intent:'vs2019_cross_x86_x64' },
  '/skill/supernaut/vs/2019/cross/x64_x86':     { action:'Vs2019Actions.x64_x86',   skill_intent:'vs2019_cross_x64_x86' },

  '/skill/supernaut/vs/2022/community/devshell':  { action:'Vs2022Actions.devshellCommunity',  skill_intent:'vs2022_community_devshell' },
  '/skill/supernaut/vs/2022/buildtools/devshell': { action:'Vs2022Actions.devshellBuildtools', skill_intent:'vs2022_buildtools_devshell' },
  '/skill/supernaut/vs/2022/community/devcmd':    { action:'Vs2022Actions.devcmdCommunity',    skill_intent:'vs2022_community_devcmd' },
  '/skill/supernaut/vs/2022/buildtools/devcmd':   { action:'Vs2022Actions.devcmdBuildtools',   skill_intent:'vs2022_buildtools_devcmd' },
  '/skill/supernaut/vs/2022/debuggable/pkg':      { action:'Vs2022Actions.debuggablePackageManager', skill_intent:'vs2022_debuggable_pkg' },

  '/skill/supernaut/netfx/clr/host':    { action:'NetfxSdkActions.clrHost',    skill_intent:'netfx_clr_host' },
  '/skill/supernaut/netfx/clr/debug':   { action:'NetfxSdkActions.clrDebug',   skill_intent:'netfx_clr_debug' },
  '/skill/supernaut/netfx/clr/profile': { action:'NetfxSdkActions.clrProfile', skill_intent:'netfx_clr_profile' },

  '/skill/supernaut/cf/worker/deploy':  { action:'CloudflareDeployActions.deploy_worker',   skill_intent:'cf_deploy_worker' },
  '/skill/supernaut/cf/pages/deploy':   { action:'CloudflareDeployActions.deploy_pages',    skill_intent:'cf_deploy_pages' },
  '/skill/supernaut/cf/kv/publish':     { action:'CloudflareDeployActions.publish_kv',      skill_intent:'cf_publish_kv' },
  '/skill/supernaut/cf/r2/upload':      { action:'CloudflareDeployActions.upload_r2',       skill_intent:'cf_upload_r2' },
  '/skill/supernaut/cf/d1/migrate':     { action:'CloudflareDeployActions.migrate_d1',      skill_intent:'cf_migrate_d1' },
  '/skill/supernaut/cf/deployments':    { action:'CloudflareDeployActions.list_deployments',skill_intent:'cf_list_deployments' },

  '/skill/supernaut/codex/generate':    { action:'CodexAgentActions.generate_code',  skill_intent:'codex_generate' },
  '/skill/supernaut/codex/refactor':    { action:'CodexAgentActions.refactor',       skill_intent:'codex_refactor' },
  '/skill/supernaut/codex/tests':       { action:'CodexAgentActions.write_tests',    skill_intent:'codex_write_tests' },
  '/skill/supernaut/codex/explain':     { action:'CodexAgentActions.explain_code',   skill_intent:'codex_explain' },
  '/skill/supernaut/codex/review':      { action:'CodexAgentActions.review_pr',      skill_intent:'codex_review_pr' },
  '/skill/supernaut/codex/shard':       { action:'CodexAgentActions.pack_shard',     skill_intent:'codex_pack_shard' },

  '/skill/supernaut/micronaunt/create':  { action:'MicronauntFactoryActions.CreateMicronaut',   skill_intent:'micronaunt_create' },
  '/skill/supernaut/micronaunt/publish': { action:'MicronauntFactoryActions.PublishMicronaut',  skill_intent:'micronaunt_publish' },
  '/skill/supernaut/micronaunt/resolve': { action:'MicronauntFactoryActions.ResolveMicronaut',  skill_intent:'micronaunt_resolve' },
  '/skill/supernaut/micronaunt/compose': { action:'MicronauntFactoryActions.ComposeMicronauts', skill_intent:'micronaunt_compose' },
  '/skill/supernaut/micronaunt/list':    { action:'MicronauntFactoryActions.ListMicronauts',    skill_intent:'micronaunt_list' },
  '/skill/supernaut/micronaunt/stats':   { action:'MicronauntFactoryActions.GetFactoryStats',   skill_intent:'micronaunt_stats' },

  '/skill/supernaut/commands/run':       { action:'JsonRuntime.CommandsActions.run',            skill_intent:'commands_run' },

  '/skill/supernaut/doc/readme':         { action:'JsonRuntime.DocActions.generate_readme',     skill_intent:'doc_generate_readme' },
  '/skill/supernaut/doc/api-ref':        { action:'JsonRuntime.DocActions.update_api_ref',      skill_intent:'doc_update_api_ref' },
  '/skill/supernaut/doc/changelog':      { action:'JsonRuntime.DocActions.write_changelog',     skill_intent:'doc_write_changelog' },
  '/skill/supernaut/doc/diagram':        { action:'JsonRuntime.DocActions.draw_diagram',        skill_intent:'doc_draw_diagram' },
  '/skill/supernaut/doc/validate-links': { action:'JsonRuntime.DocActions.validate_links',      skill_intent:'doc_validate_links' },

  '/skill/supernaut/sqlite3/open':       { action:'JsonRuntime.Sqlite3Actions.open',                  skill_intent:'sqlite3_open' },
  '/skill/supernaut/sqlite3/query':      { action:'JsonRuntime.Sqlite3Actions.query',                 skill_intent:'sqlite3_query' },
  '/skill/supernaut/sqlite3/fts5':       { action:'JsonRuntime.Sqlite3Actions.fts5_query',            skill_intent:'sqlite3_fts5_query' },
  '/skill/supernaut/sqlite3/backup':     { action:'JsonRuntime.Sqlite3Actions.backup',                skill_intent:'sqlite3_backup' },
  '/skill/supernaut/sqlite3/shard':      { action:'JsonRuntime.Sqlite3Actions.pack_checkpoint_shard', skill_intent:'sqlite3_pack_shard' },

  '/skill/supernaut/msbuild-nuget/query':         { action:'JsonRuntime.MsbuildNugetActions.query',         skill_intent:'msbuild_nuget_query' },
  '/skill/supernaut/msbuild-nuget/debug-restore':  { action:'JsonRuntime.MsbuildNugetActions.debug_restore', skill_intent:'msbuild_nuget_debug_restore' },

  '/skill/supernaut/figma/file':         { action:'JsonRuntime.FigmaActions.get_file',          skill_intent:'figma_get_file' },
  '/skill/supernaut/figma/assets':       { action:'JsonRuntime.FigmaActions.export_assets',     skill_intent:'figma_export_assets' },
  '/skill/supernaut/figma/components':   { action:'JsonRuntime.FigmaActions.list_components',   skill_intent:'figma_list_components' },
  '/skill/supernaut/figma/tokens':       { action:'JsonRuntime.FigmaActions.get_tokens',        skill_intent:'figma_get_tokens' },
  '/skill/supernaut/figma/frame':        { action:'JsonRuntime.FigmaActions.inspect_frame',     skill_intent:'figma_inspect_frame' },
  '/skill/supernaut/figma/styles':       { action:'JsonRuntime.FigmaActions.get_styles',        skill_intent:'figma_get_styles' },

  '/skill/supernaut/figma-impl/react':     { action:'JsonRuntime.FigmaImplementDesignActions.frame_to_react',  skill_intent:'figma_impl_frame_to_react' },
  '/skill/supernaut/figma-impl/html':      { action:'JsonRuntime.FigmaImplementDesignActions.frame_to_html',   skill_intent:'figma_impl_frame_to_html' },
  '/skill/supernaut/figma-impl/tokens':    { action:'JsonRuntime.FigmaImplementDesignActions.map_tokens',      skill_intent:'figma_impl_map_tokens' },
  '/skill/supernaut/figma-impl/component': { action:'JsonRuntime.FigmaImplementDesignActions.emit_component',  skill_intent:'figma_impl_emit_component' },
  '/skill/supernaut/figma-impl/styles':    { action:'JsonRuntime.FigmaImplementDesignActions.sync_styles',     skill_intent:'figma_impl_sync_styles' },

  '/skill/supernaut/vsdevcmd/setup':     { action:'JsonRuntime.VsDevCmdActions.setup_env',     skill_intent:'vsdevcmd_setup_env' },
  '/skill/supernaut/vsdevcmd/debug':     { action:'JsonRuntime.VsDevCmdActions.debug_env',     skill_intent:'vsdevcmd_debug_env' },

  '/skill/supernaut/winkit-81/winmd':    { action:'JsonRuntime.Winkit81Actions.query_winmd',   skill_intent:'winkit81_query_winmd' },
});

export const SUPERNAUT_ROUTE_COUNT = Object.keys(SUPERNAUT_ROUTES).length;

// Resolve a path to its route descriptor (exact match only)
export function resolveRoute(path) {
  return SUPERNAUT_ROUTES[path] || null;
}

// All routes for a given action namespace prefix (e.g. 'TeamLeadActions')
export function getRoutesByActionNs(ns) {
  return Object.entries(SUPERNAUT_ROUTES)
    .filter(([, r]) => r.action.startsWith(ns))
    .map(([path, r]) => ({ path, ...r }));
}

// All routes for a given skill_intent prefix (e.g. 'pool_')
export function getRoutesByIntent(prefix) {
  return Object.entries(SUPERNAUT_ROUTES)
    .filter(([, r]) => r.skill_intent.startsWith(prefix))
    .map(([path, r]) => ({ path, ...r }));
}
