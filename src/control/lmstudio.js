// LMStudio SDK binding types — informational constants (v3.3.0-scx-control-flow)
//
// Mirrors control/lmstudio-sdk.js binding factories.
// Actual SDK (CJS/ESM) is external; these types describe the binding protocol.
// Use createDelta() from delta.js to emit control deltas for each binding.

export const LMSTUDIO_SDK_VERSION = '1.5.0';

export const LMSTUDIO_SURFACE_BY_KIND = Object.freeze({
  agent:     'agent',
  chat:      'agent',
  command:   'command',
  config:    'function',
  extension: 'program',
  file:      'file',
  model:     'micronaut',
  plugin:    'program',
  skill:     'skill',
  status:    'action',
  tool:      'tool',
  tools:     'tool',
});

export const LMSTUDIO_EFFECTS_BY_SURFACE = Object.freeze({
  action:    ['execute'],
  agent:     ['execute', 'memory'],
  command:   ['execute'],
  file:      ['read'],
  function:  ['execute', 'memory'],
  micronaut: ['execute', 'network', 'memory'],
  program:   ['execute', 'model'],
  skill:     ['execute', 'memory'],
  tool:      ['execute', 'network'],
});

export function lmStudioDeltaParams(kind, name, operation = 'invoke', inputs = {}, result = {}) {
  const surface = LMSTUDIO_SURFACE_BY_KIND[kind] || 'function';
  const effects = LMSTUDIO_EFFECTS_BY_SURFACE[surface] || ['execute'];
  return {
    surface,
    target: `lmstudio:${kind}:${name}`,
    operation,
    inputs: { kind, name, sdk_version: LMSTUDIO_SDK_VERSION, ...inputs },
    effects,
    result,
    authority: { domain: 'LMSTUDIO_SDK', lane: String(kind).toUpperCase(), effects },
  };
}

// Binding descriptor factories — pure data, no SDK import required
export function createToolBinding({ name, description = '', parameters = {}, source = 'scx' }) {
  if (!name) throw new Error('LMStudio tool binding requires a name');
  return { type: 'lmstudio_tool_binding', name, description, parameters, source };
}

export function createAgentBinding({ name, model, tools = [], extensions = [], prompt = '', policy = {} }) {
  if (!name) throw new Error('LMStudio agent binding requires a name');
  return { type: 'lmstudio_agent_binding', name, model, tools, extensions, prompt, policy };
}

export function createSkillBinding({ name, description = '', command = null, args = {}, verbs = [], tools = [] }) {
  if (!name) throw new Error('LMStudio skill binding requires a name');
  return { type: 'lmstudio_skill_binding', name, description, command, args, verbs, tools };
}

export function createCommandBinding({ name, executable, args = [], verbs = ['exe'] }) {
  if (!name || !executable) throw new Error('LMStudio command binding requires name and executable');
  return { type: 'lmstudio_command_binding', name, executable, args: args.map(String), verbs };
}

export function createChatActBinding({ model, chat = [], tools = [], structured = false, options = {} }) {
  if (!model) throw new Error('LMStudio chat/act binding requires a model id');
  return { type: 'lmstudio_chat_act_binding', model, chat, tools, structured, options };
}
