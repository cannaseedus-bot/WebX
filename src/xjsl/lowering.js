// XJSL lowering — JSON IR → WGSL / HLSL
// Port of xjsl-lowering.mjs from v0.1.1-igpu-trainer-xjsl.
// This is the pure in-memory version (no Node.js fs); works in browser + Node.
//
// XJSL document format:
//   @xjson_version, @paradigm, @shader_language, @target_backends
//   @shaders: { <name>: { @type, @workgroup, @inputs, @outputs, @uniforms, @kernel } }

const TYPE_HLSL = {
  'f32':       'float',
  'i32':       'int',
  'u32':       'uint',
  'vec2<f32>': 'float2',
  'vec3<f32>': 'float3',
  'vec4<f32>': 'float4',
};

const TYPE_WGSL = {
  'f32':       'f32',
  'i32':       'i32',
  'u32':       'u32',
  'vec2<f32>': 'vec2<f32>',
  'vec3<f32>': 'vec3<f32>',
  'vec4<f32>': 'vec4<f32>',
};

function wgslType(t) { return TYPE_WGSL[t] || t || 'f32'; }
function hlslType(t) { return TYPE_HLSL[t] || t || 'float'; }

function uniformStructWGSL(uniforms = {}) {
  const entries = Object.entries(uniforms);
  if (!entries.length) return '';
  const fields = entries.map(([n, t]) => `  ${n}: ${wgslType(t)},`).join('\n');
  return `struct Params {\n${fields}\n};\n@group(0) @binding(0) var<uniform> params: Params;\n\n`;
}

function uniformStructHLSL(uniforms = {}) {
  const entries = Object.entries(uniforms);
  if (!entries.length) return '';
  const fields = entries.map(([n, t]) => `  ${hlslType(t)} ${n};`).join('\n');
  return `cbuffer Params : register(b0) {\n${fields}\n};\n\n`;
}

function wgslBufferDecls(shader) {
  const lines = [];
  let binding = Object.keys(shader['@uniforms'] || {}).length > 0 ? 1 : 0;
  for (const [name, def] of Object.entries(shader['@inputs'] || {})) {
    lines.push(`@group(0) @binding(${binding++}) var<storage, read> ${name}: array<${wgslType(def['@type'])}>;`);
  }
  for (const [name, def] of Object.entries(shader['@outputs'] || {})) {
    const access = def['@layout'] === 'write' ? 'read_write' : (def['@layout'] || 'read_write');
    lines.push(`@group(0) @binding(${binding++}) var<storage, ${access}> ${name}: array<${wgslType(def['@type'])}>;`);
  }
  return lines.join('\n') + '\n\n';
}

function hlslBufferDecls(shader) {
  const lines = [];
  let srv = 0, uav = 0;
  for (const [name, def] of Object.entries(shader['@inputs'] || {})) {
    lines.push(`StructuredBuffer<${hlslType(def['@type'])}> ${name} : register(t${srv++});`);
  }
  for (const [name, def] of Object.entries(shader['@outputs'] || {})) {
    lines.push(`RWStructuredBuffer<${hlslType(def['@type'])}> ${name} : register(u${uav++});`);
  }
  return lines.join('\n') + '\n\n';
}

// Translate WGSL-style body to HLSL (same heuristics as original xjsl-lowering.mjs)
function translateBodyToHLSL(body) {
  return body
    .replace(/\blet\s+(row|col|idx|j|k|local_j|global_j)\s*=/g, 'uint $1 =')
    .replace(/\blet\s+(p|q|d)\s*=/g, 'float3 $1 =')
    .replace(/\blet\s+(dist|scale|weight|allowed|influence)\s*=/g, 'float $1 =')
    .replace(/\blet\s+([A-Za-z_]\w*)\s*=/g, 'uint $1 =')
    .replace(/\bvar\s+([A-Za-z_]\w*)\s*:\s*f32\s*=/g, 'float $1 =')
    .replace(/\bvar\s+([A-Za-z_]\w*)\s*:\s*u32\s*=/g, 'uint $1 =')
    .replace(/\bvar\s+([A-Za-z_]\w*)\s*=/g, 'auto $1 =')
    .replace(/\bu32\(/g, '(uint)(')
    .replace(/\bf32\(/g, '(float)(')
    .replace(/\bi32\(/g, '(int)(')
    .replace(/\bparams\./g, '');
}

export function generateWGSL(name, shader) {
  const wg = shader['@workgroup'] || [1, 1, 1];
  return (
    `// Generated WGSL from XJSL: ${name}\n` +
    uniformStructWGSL(shader['@uniforms']) +
    wgslBufferDecls(shader) +
    `@compute @workgroup_size(${wg[0]}, ${wg[1]}, ${wg[2]})\n` +
    `fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>,\n` +
    `           @builtin(local_invocation_id)  local_id:  vec3<u32>,\n` +
    `           @builtin(workgroup_id)         workgroup_id: vec3<u32>) {\n` +
    `${shader['@kernel'] || ''}\n}\n`
  );
}

export function generateHLSL(name, shader) {
  const wg = shader['@workgroup'] || [1, 1, 1];
  return (
    `// Generated HLSL/D3D11 from XJSL: ${name}\n` +
    uniformStructHLSL(shader['@uniforms']) +
    hlslBufferDecls(shader) +
    `[numthreads(${wg[0]}, ${wg[1]}, ${wg[2]})]\n` +
    `void CSMain(uint3 global_id : SV_DispatchThreadID,\n` +
    `            uint3 local_id  : SV_GroupThreadID,\n` +
    `            uint3 workgroup_id : SV_GroupID) {\n` +
    `${translateBodyToHLSL(shader['@kernel'] || '')}\n}\n`
  );
}

// Pure in-memory lowering — returns { shaders: [{ name, wgsl, hlsl, workgroup }] }
export function lowerXJSLDoc(doc) {
  const result = {
    source:       doc['@paradigm'] || 'xjsl',
    generatedAt:  new Date().toISOString(),
    shaders:      [],
  };
  for (const [name, shader] of Object.entries(doc['@shaders'] || {})) {
    result.shaders.push({
      name,
      workgroup: shader['@workgroup'] || [1, 1, 1],
      wgsl:      generateWGSL(name, shader),
      hlsl:      generateHLSL(name, shader),
    });
  }
  return result;
}

export default lowerXJSLDoc;
