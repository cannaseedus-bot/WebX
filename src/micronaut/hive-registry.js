// Hive Micronaut Atlas — canonical 85-agent registry
// Ported from releases/index.html inline registry data.
// Hive micronauts: ports 3167-3207 + KUX-1:4200
// Unified services: coordinator (25100) + 12 services (25101-25200)
// Special packs: supernaut, agentic-micronaut

export const HIVE_MICRONAUTS = Object.freeze([
  { id:'ST-1',    name:'StudioAssistant',     fold:'UNASSIGNED', port:3167, invoke_via:'http', backend:'Studio assistance surface',                  experts:[] },
  { id:'CH-1',    name:'ChatAssistant',        fold:'UNASSIGNED', port:3168, invoke_via:'http', backend:'Chat-oriented assistant surface',             experts:[] },
  { id:'MM-1',    name:'ModelManager',         fold:'UNASSIGNED', port:3169, invoke_via:'http', backend:'Model lifecycle and registry',                experts:[] },
  { id:'PK-1',    name:'Packager',             fold:'UNASSIGNED', port:3170, invoke_via:'http', backend:'Packaging/bundle operations',                 experts:[] },
  { id:'CL-1',    name:'ClusterManager',       fold:'UNASSIGNED', port:3171, invoke_via:'http', backend:'Cluster management',                          experts:[] },
  { id:'BR-1',    name:'BrainRouter',          fold:'COMPUTE',    port:3172, invoke_via:'http', backend:'qwen2-q8 | inference: http://127.0.0.1:8000/chat/completions', experts:['scxq2_brains','cm1_gate','inference_dispatch'] },
  { id:'VM-1',    name:'VisualMusicEngine',    fold:'UI',         port:3173, invoke_via:'http', backend:'cp1-brain | model.json',                      experts:['entropy_arcs','geodesic_bridge','svg_render','audio_synthesis'] },
  { id:'OV-1',    name:'OscillatorVM',         fold:'COMPUTE',    port:3174, invoke_via:'http', backend:'oscillator runtime',                          experts:['attractor_circle','attractor_polygon','attractor_torus','attractor_strange','attractor_spiral'] },
  { id:'KX-1',    name:'KuhulExecutor',        fold:'STATE',      port:3175, invoke_via:'http', backend:"K'UHUL core engine",                          experts:['kuhul_programs','mayan_fold','trig_brain','kshtml_parser'] },
  { id:'FG-1',    name:'FabricGraph',          fold:'CONTROL',    port:3176, invoke_via:'http', backend:'Frame router',                                experts:['worm_compiler','frame_router','law_engine','trace_log'] },
  { id:'DX-1',    name:'DirectXMathBridge',    fold:'COMPUTE',    port:3177, invoke_via:'http', backend:'DirectXMath core',                            experts:['matrix_ops','stereo3d','sh_math','xdsp_fft'] },
  { id:'IM-1',    name:'InferenceMicronaut',   fold:'COMPUTE',    port:3178, invoke_via:'http', backend:'phi3-mini.scxqdds',                            experts:['phi3_inference','scxqdds_decode','payload_stream'] },
  { id:'CM-1',    name:'CompressionMicronaut', fold:'META',       port:3179, invoke_via:'http', backend:'Compression stack',                           experts:['scxq2_lane','kv_delta','merkle_tree','gzip_hybrid'] },
  { id:'PM-1',    name:'PlanModeMicronaut',    fold:'META',       port:8001, invoke_via:'http', backend:'phi3-q2 | inference',                         experts:['planning','reasoning','web_research','math_ops','scratch_pad'] },
  { id:'PSISE-1', name:'PowerShellISE',        fold:'META',       port:3180, invoke_via:'http', backend:'PowerShell 5.1 toolchain',                    experts:['ps_codegen','ps_ise_launch','script_create','addon_catalog'] },
  { id:'PYIDE-1', name:'PythonIDE',            fold:'META',       port:3181, invoke_via:'http', backend:'Python 3.14/3.12',                             experts:['py_codegen','py_runtime_launch','script_create','version_resolution'] },
  { id:'BATCH-1', name:'BatchCMD',             fold:'META',       port:3182, invoke_via:'http', backend:'cmd.exe runner',                              experts:['bat_codegen','cmd_execution','script_create','env_management'] },
  { id:'SHELL-1', name:'UniversalShell',       fold:'META',       port:3183, invoke_via:'http', backend:'cmd/ps/bash replay engine',                   experts:['cmd_execution','ps_execution','bash_execution','replay_catalog','session_recording'] },
  { id:'FM-1',    name:'FileManager',          fold:'META',       port:3184, invoke_via:'http', backend:'Object server http://127.0.0.1:3185/objects',  experts:['build','update','save','location','list','tree','read','write','copy','move','delete','search','info','tag','link'] },
  { id:'DQ-1',    name:'DataQuery',            fold:'DATA',       port:3186, invoke_via:'http', backend:'SQLite + JSON1',                              experts:['sql','jsonpath','xpath','fts','sharding','idb'] },
  { id:'D3D-1',   name:'DirectXWorld',         fold:'COMPUTE',    port:3187, invoke_via:'http', backend:'world.bin + D3D12DynamicLOD.exe',              experts:['world_gen','mesh_gen','instance_export','spatial_query','pipeline'] },
  { id:'WB-1',    name:'WorldBuilder',         fold:'UI',         port:3188, invoke_via:'http', backend:'Three.js UI builder',                         experts:['world_gen','xml_io','d3d1_bridge','three_js_preview'] },
  { id:'WSL-1',   name:'WSLBridge',            fold:'META',       port:3194, invoke_via:'http', backend:'wsl.exe',                                     experts:['wsl_launch','linux_command','distro_inventory'] },
  { id:'AR-1',    name:'AgentRuntime',         fold:'META',       port:3195, invoke_via:'http', backend:'SpaceAgent.exe | AgentService.exe',            experts:['space_agent','agent_service','mmga_server','shell_app_runtime'] },
  { id:'BC-1',    name:'ByteCodeForge',        fold:'COMPUTE',    port:3196, invoke_via:'http', backend:'ByteCodeGenerator.exe',                       experts:['bytecode_generator','ntfs_compact','binary_compare'] },
  { id:'DT-1',    name:'DesktopTools',         fold:'UI',         port:3197, invoke_via:'http', backend:'notepad.exe | calc.exe | mspaint.exe',         experts:['notepad_launch','paint_launch','snipping','calculator','character_map'] },
  { id:'SH-1',    name:'ScriptHost',           fold:'META',       port:3198, invoke_via:'http', backend:'cscript.exe | wscript.exe',                   experts:['cscript_host','wscript_host','script_file_launch'] },
  { id:'SCM-1',   name:'SourceControlMesh',    fold:'META',       port:3199, invoke_via:'http', backend:'git.exe + gh.exe + git-gui',                  experts:['git_cli','bash_shell','github_cli','git_gui','repository_transport'] },
  { id:'SSH-1',   name:'GitSecurityBootstrap', fold:'META',       port:3200, invoke_via:'http', backend:'ssh-agent bootstrap',                         experts:['ssh_agent_bootstrap','pageant_bridge','aslr_policy'] },
  { id:'WK-1',    name:'WorkspaceKeeper',      fold:'META',       port:3201, invoke_via:'http', backend:'workspace inventory',                         experts:['workspace_inventory','powershell_runtime_tree','shell_replay_catalog','batch_and_python_workspace'] },
  { id:'GX-1',    name:'GraphicsWorkspace',    fold:'COMPUTE',    port:3202, invoke_via:'http', backend:'Direct3D | Win2D',                            experts:['direct3d_tree','directui_tree','directxmath_tree','win2d_tree'] },
  { id:'SCX-8',   name:'ScxExpertEight',       fold:'COMPUTE',    port:3203, invoke_via:'http', backend:'SCX_EXPERT_8.scxp',                           experts:['scxp_capsule','mgguf_variant','resource_control'] },
  { id:'SMG-1',   name:'SmgmModelCore',        fold:'COMPUTE',    port:3204, invoke_via:'http', backend:'weights.int4.bin + metrics',                  experts:['weights_catalog','mu_adapter','shard_map','metrics_tensor'] },
  { id:'SXME-1',  name:'SxmeShaderCompute',    fold:'COMPUTE',    port:3205, invoke_via:'http', backend:'dxc.exe + HLSL pipeline',                     experts:['shader_compile','dxil_validate','sxme_demo','hlsl_pipeline'] },
  { id:'DST-1',   name:'DatasetFoundry',       fold:'DATA',       port:3206, invoke_via:'http', backend:'train_val_test splits + shards',               experts:['dataset_inventory','prompt_layer','response_layer','prompt_code','prompt_math','train_val_test_split','shard_logging'] },
  { id:'S7-1',    name:'S7SupernautStack',     fold:'COMPUTE',    port:3207, invoke_via:'http', backend:'supernaut.exe + weights_v2',                  experts:['compiled_weights','agent_contracts','atomic_brain_maps','supernaut_cpp_runtime','scxq2_vectors'] },
  { id:'KUX-1',   name:'KuxVerifier',          fold:'META',       port:4200, invoke_via:'http', backend:'SHA256 | K-UX v1 deterministic projection',   experts:['sha256_phase','collapse_hash','projection_hash','deterministic_layout','c_code_emitter'] },
]);

export const UNIFIED_SERVICES = Object.freeze([
  { id:'coordinator',    name:'Coordinator (CO-1)',        port:25100, role:'routing',   type:'router',    domain:'coordinator', startup_order:1 },
  { id:'factory',        name:'Factory (FG-1)',            port:25101, role:'creation',  type:'factory',   domain:'factory',     startup_order:2 },
  { id:'responder',      name:'Responder (RP-1)',          port:25102, role:'response',  type:'formatter', domain:'response',    startup_order:2 },
  { id:'executor',       name:'Executor (EX-1)',           port:25103, role:'execution', type:'executor',  domain:'execution',   startup_order:3 },
  { id:'manager',        name:'Manager (MG-1)',            port:25104, role:'lifecycle', type:'manager',   domain:'lifecycle',   startup_order:2 },
  { id:'coder',          name:'Coder (CD-1)',              port:25105, role:'coding',    type:'agent',     domain:'coding',      startup_order:3, capabilities:['code_review','code_generation','refactoring'] },
  { id:'planner',        name:'Planner (PL-1)',            port:25106, role:'planning',  type:'agent',     domain:'planning',    startup_order:3, capabilities:['planning','task_breakdown','orchestration'] },
  { id:'skills-router',  name:'Skills Router (SK-1)',      port:25107, role:'skills',    type:'router',    domain:'skills',      startup_order:2, capabilities:['skill_routing','manifest_loading'] },
  { id:'web-research',   name:'Web Research Bot (WR-1)',   port:25108, role:'research',  type:'agent',     domain:'research',    startup_order:3, capabilities:['web_scraping','ngram_extraction','xql_query','gguf_inference'] },
  { id:'session-memory', name:'Session Memory (SM-1)',     port:25109, role:'memory',    type:'store',     domain:'memory',      startup_order:2 },
  { id:'pm1-plan',       name:'Plan Service (PM-1)',       port:25110, role:'plans',     type:'store',     domain:'plans',       startup_order:2 },
  { id:'gguf-server',    name:'GGUF Bridge (GB-1)',        port:5000,  role:'inference', type:'agent',     domain:'inference',   startup_order:2 },
  { id:'splash-server',  name:'Splash Cache Server (SS-1)',port:25200, role:'static',    type:'static',    domain:'splash',      startup_order:3 },
]);

export const HIVE_SKILL_PACKS = Object.freeze([
  {
    id: 'supernaut',
    name: 'Supernaut Skill Pack',
    kind: 'skill-pack',
    fold: 'PACK',
    invoke_via: 'xcfe',
    backend_runtime: 'xcfe',
    model_preference: 'scxqdds',
    replay_state_path: 'skills/supernaut/shard-manifest.json',
  },
  {
    id: 'agentic-micronaut',
    name: 'Agentic Micronaut Pack',
    kind: 'skill-pack',
    fold: 'PACK',
    invoke_via: 'xcfe',
    backend_runtime: 'xcfe',
    model_preference: 'scxqdds',
    replay_state_path: 'skills/agentic-micronaut/agentic_micronaut_manifest.json',
  },
]);

export const HIVE_AGENT_COUNT = HIVE_MICRONAUTS.length + UNIFIED_SERVICES.length + HIVE_SKILL_PACKS.length;

export const HIVE_FOLDS = Object.freeze([
  'COMPUTE', 'CONTROL', 'DATA', 'META', 'PACK', 'STATE', 'UI', 'UNASSIGNED',
]);

export function getHiveMicronaut(id) {
  return HIVE_MICRONAUTS.find(m => m.id === id) || null;
}

export function getHiveService(id) {
  return UNIFIED_SERVICES.find(s => s.id === id) || null;
}

export function getHiveMicronautsByFold(fold) {
  return HIVE_MICRONAUTS.filter(m => m.fold === fold);
}

export function getHiveMicronautsByExpert(expert) {
  return HIVE_MICRONAUTS.filter(m => m.experts.includes(expert));
}

export function buildHiveAtlas() {
  const micronauts = HIVE_MICRONAUTS.map(m => ({ ...m, kind: 'micronaut', source: 'hive-registry' }));
  const services   = UNIFIED_SERVICES.map(s => ({ ...s, kind: s.type === 'agent' ? 'agent' : 'service', source: 'unified-registry', invoke_via: 'http' }));
  const packs      = HIVE_SKILL_PACKS.map(p => ({ ...p, source: 'skill-pack' }));
  return [...micronauts, ...services, ...packs];
}
