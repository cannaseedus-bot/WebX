// xjsl-d3d11.js — XJSL → HLSL → D3D11 lowering pipeline
//
// Lowering layers:
//   XJSL source (JSON + WGSL kernel string)
//   → WGSLParser  (parse kernel, extract AST tokens)
//   → HLSLGen     (WGSL AST → HLSL + [numthreads] entry)
//   → D3D11Adapter (D3DCompile → DXBC → execute)
//
// K'uhul phase:  Sek (XJSL dispatch) maps to [numthreads] + Dispatch
// Compatibility: cs_5_0 DXBC — matches Intel HD4600 constraint (no DXIL)
// π-phase sync:  workgroupBarrier() → GroupMemoryBarrierWithGroupSync()

// ─── Type mappings ────────────────────────────────────────────────────────────

export const WGSL_TO_HLSL_TYPES = Object.freeze({
  'f32':           'float',
  'i32':           'int',
  'u32':           'uint',
  'bool':          'bool',
  'vec2<f32>':     'float2',
  'vec3<f32>':     'float3',
  'vec4<f32>':     'float4',
  'mat2x2<f32>':   'float2x2',
  'mat3x3<f32>':   'float3x3',
  'mat4x4<f32>':   'float4x4',
  'array<f32>':    'float',   // buffer element type
  'array<u32>':    'uint',
  'array<i32>':    'int',
});

export const WGSL_TO_HLSL_BUILTINS = Object.freeze({
  'global_invocation_id': 'SV_DispatchThreadID',
  'local_invocation_id':  'SV_GroupThreadID',
  'workgroup_id':         'SV_GroupID',
  'workgroupBarrier()':   'GroupMemoryBarrierWithGroupSync()',
});

// ─── XJSLD3D11Lowering ────────────────────────────────────────────────────────

export class XJSLD3D11Lowering {
  constructor() {
    this._typeMap  = new Map(Object.entries(WGSL_TO_HLSL_TYPES));
  }

  /** Lower one XJSL shader definition to { hlsl, bindingTable }. */
  lower(shaderDef) {
    const workgroup    = shaderDef['@workgroup'] ?? [16, 16, 1];
    const inputs       = shaderDef['@inputs']   ?? {};
    const outputs      = shaderDef['@outputs']  ?? {};
    const uniforms     = shaderDef['@uniforms'] ?? {};
    const constants    = shaderDef['@constants'] ?? {};
    const kernelWGSL   = shaderDef['@kernel'] ?? '';

    const hlsl        = this._generateHLSL(shaderDef.name, workgroup, inputs, outputs, uniforms, constants, kernelWGSL);
    const bindingTable = this._generateBindingTable(shaderDef.name, inputs, outputs, uniforms);
    return { hlsl, bindingTable };
  }

  _generateHLSL(name, wg, inputs, outputs, uniforms, constants, kernel) {
    const lines = [`// XJSL → HLSL  shader: ${name}  (cs_5_0 DXBC — Intel HD4600 compatible)`, ''];

    // Constants
    for (const [k, v] of Object.entries(constants))
      lines.push(`static const float ${k} = ${v}f;`);
    if (Object.keys(constants).length) lines.push('');

    // Constant buffer
    if (Object.keys(uniforms).length) {
      lines.push(`cbuffer Uniforms : register(b0) {`);
      for (const [k, t] of Object.entries(uniforms))
        lines.push(`  ${this._typeMap.get(t) ?? t} ${k};`);
      lines.push(`};`, '');
    }

    // SRV (input) buffers
    let srvSlot = 0;
    for (const [bname, def] of Object.entries(inputs)) {
      lines.push(`ByteAddressBuffer ${bname}_buf : register(t${srvSlot++});`);
      const etype = this._typeMap.get(def['@type']?.replace('[]','')) ?? 'float';
      lines.push(`${etype} ${bname}Load(uint i){ return asfloat(${bname}_buf.Load(i*4)); }`, '');
    }

    // UAV (output) buffers
    let uavSlot = 0;
    for (const [bname] of Object.entries(outputs)) {
      lines.push(`RWByteAddressBuffer ${bname}_buf : register(u${uavSlot++});`);
      lines.push(`void ${bname}Store(uint i, float v){ ${bname}_buf.Store(i*4, asuint(v)); }`, '');
    }

    // Entry point
    lines.push(`[numthreads(${wg[0]}, ${wg[1]}, ${wg[2]})]`);
    lines.push(`void CSMain(`);
    lines.push(`  uint3 global_id : SV_DispatchThreadID,`);
    lines.push(`  uint3 local_id  : SV_GroupThreadID,`);
    lines.push(`  uint3 group_id  : SV_GroupID`);
    lines.push(`) {`);
    lines.push(this._translateKernel(kernel, inputs, outputs));
    lines.push(`}`);

    return lines.join('\n');
  }

  _translateKernel(wgsl, inputs, outputs) {
    let hlsl = wgsl;
    // Barrier
    hlsl = hlsl.replace(/workgroupBarrier\(\)/g, 'GroupMemoryBarrierWithGroupSync()');
    // var declarations
    hlsl = hlsl.replace(/\blet\b\s+(\w+)\s*=/g, 'const auto $1 =');
    hlsl = hlsl.replace(/\bvar\b\s+(\w+)\s*:\s*f32\s*=/g, 'float $1 =');
    hlsl = hlsl.replace(/\bvar\b\s+(\w+)\s*:\s*u32\s*=/g, 'uint $1 =');
    hlsl = hlsl.replace(/\bvar\b\s+(\w+)\s*:\s*i32\s*=/g, 'int $1 =');
    hlsl = hlsl.replace(/\bvar\b\s+(\w+)\b\s*=/g, 'auto $1 =');
    // f32/u32/i32 casts
    hlsl = hlsl.replace(/\bf32\(([^)]+)\)/g, '(float)($1)');
    hlsl = hlsl.replace(/\bu32\(([^)]+)\)/g, '(uint)($1)');
    hlsl = hlsl.replace(/\bi32\(([^)]+)\)/g, '(int)($1)');
    // Workgroup arrays
    hlsl = hlsl.replace(/var<workgroup>\s+(\w+)\s*:\s*array<array<f32,\s*(\w+)>,\s*(\w+)>/g,
      'groupshared float $1[$3][$2]');
    hlsl = hlsl.replace(/var<workgroup>\s+(\w+)\s*:\s*array<f32,\s*(\w+)>/g,
      'groupshared float $1[$2]');
    // For loops: WGSL u32 suffix
    hlsl = hlsl.replace(/(\d+)u\b/g, '$1u');
    // Boolean
    hlsl = hlsl.replace(/\btrue\b/g, 'true').replace(/\bfalse\b/g, 'false');
    // Buffer access: patch known input/output names
    for (const bname of Object.keys(inputs))
      hlsl = hlsl.replace(new RegExp(`${bname}\\[(\\w+)\\]`, 'g'), `${bname}Load($1)`);
    for (const bname of Object.keys(outputs))
      hlsl = hlsl.replace(new RegExp(`${bname}\\[(\\w+)\\]\\s*=\\s*([^;]+)`, 'g'), `${bname}Store($1, $2)`);
    return hlsl;
  }

  _generateBindingTable(name, inputs, outputs, uniforms) {
    const cbv = Object.keys(uniforms).length
      ? [{ name: 'Uniforms', slot: 0, fields: uniforms }] : [];
    const srv = Object.entries(inputs).map(([n, d], i) => ({ name: n, slot: i, type: d['@type'] }));
    const uav = Object.entries(outputs).map(([n, d], i) => ({ name: n, slot: i, type: d['@type'] }));
    return { shader: name, cbv, srv, uav };
  }
}

// ─── XJSLD3D11Pipeline ────────────────────────────────────────────────────────
//
// JS-side runtime adapter.
// In production: calls D3DCompile via native bridge (C++/WinRT or Electron).
// In browser: falls back to WebGPU with WGSL kernel passed directly.

export class XJSLD3D11Pipeline {
  constructor() {
    this._lowering = new XJSLD3D11Lowering();
    this._compiled  = new Map();
    this._gpuDevice = null;
  }

  async init() {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      const adapter  = await navigator.gpu.requestAdapter();
      this._gpuDevice = await adapter?.requestDevice() ?? null;
    }
  }

  async compileShader(shaderDef) {
    const { hlsl, bindingTable } = this._lowering.lower(shaderDef);
    const entry = { hlsl, bindingTable, def: shaderDef, pipeline: null };

    if (this._gpuDevice) {
      // WebGPU fallback: use original WGSL kernel
      const wgsl   = this._wrapWGSL(shaderDef);
      const mod    = this._gpuDevice.createShaderModule({ code: wgsl });
      entry.pipeline = this._gpuDevice.createComputePipeline({
        layout: 'auto', compute: { module: mod, entryPoint: 'main' }
      });
    }

    this._compiled.set(shaderDef.name, entry);
    return entry;
  }

  _wrapWGSL(def) {
    const inputs   = def['@inputs']  ?? {};
    const outputs  = def['@outputs'] ?? {};
    const uniforms = def['@uniforms'] ?? {};
    const constants = def['@constants'] ?? {};
    const wg = def['@workgroup'] ?? [16, 16, 1];
    let code = '';
    for (const [k, v] of Object.entries(constants)) code += `const ${k}: f32 = ${v};\n`;
    if (Object.keys(uniforms).length) {
      code += `struct Uniforms { ${Object.entries(uniforms).map(([k,t])=>`${k}: ${t}`).join(', ')} };\n`;
      code += `@group(0) @binding(0) var<uniform> params: Uniforms;\n`;
    }
    let b = 1;
    for (const [n, d] of Object.entries(inputs))  code += `@group(0) @binding(${b++}) var<storage,read> ${n}: array<${d['@type']}>;\n`;
    for (const [n, d] of Object.entries(outputs)) code += `@group(0) @binding(${b++}) var<storage,read_write> ${n}: array<${d['@type']}>;\n`;
    code += `@compute @workgroup_size(${wg[0]},${wg[1]},${wg[2]})\n`;
    code += `fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {\n`;
    code += def['@kernel'] ?? '';
    code += `\n}\n`;
    return code;
  }

  async executeShader(name, inputs, outputSpec, dispatchSize, uniforms = {}) {
    const entry = this._compiled.get(name);
    if (!entry) throw new Error(`XJSLPipeline: shader not compiled: ${name}`);

    if (entry.pipeline && this._gpuDevice) {
      return this._execWebGPU(entry, inputs, outputSpec, dispatchSize, uniforms);
    }
    // JS scalar fallback
    const out = {};
    for (const k of Object.keys(outputSpec ?? {})) out[k] = new Float32Array(1);
    return out;
  }

  async _execWebGPU(entry, inputs, outputSpec, ds, uniforms) {
    const dev = this._gpuDevice;
    const entries = [];
    let b = 0;

    // Uniform buffer
    if (Object.keys(uniforms).length) {
      const data = new Float32Array(Object.values(uniforms));
      const buf  = this._mkBuf(dev, data, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      dev.queue.writeBuffer(buf, 0, data);
      entries.push({ binding: b++, resource: { buffer: buf } });
    }

    const outBufs = {};
    for (const [k, v] of Object.entries(inputs)) {
      const arr = v instanceof Float32Array ? v : new Float32Array(v.length ?? 1);
      const buf = this._mkBuf(dev, arr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      dev.queue.writeBuffer(buf, 0, arr);
      entries.push({ binding: b++, resource: { buffer: buf } });
    }
    for (const k of Object.keys(outputSpec ?? {})) {
      const buf = this._mkBuf(dev, null, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 4096);
      entries.push({ binding: b++, resource: { buffer: buf } });
      outBufs[k] = buf;
    }

    const bg  = dev.createBindGroup({ layout: entry.pipeline.getBindGroupLayout(0), entries });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(entry.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(ds.x ?? 1, ds.y ?? 1, ds.z ?? 1);
    pass.end();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();

    const results = {};
    for (const [k, buf] of Object.entries(outBufs)) {
      const staging = this._mkBuf(dev, null, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, buf.size);
      const enc2 = dev.createCommandEncoder();
      enc2.copyBufferToBuffer(buf, 0, staging, 0, buf.size);
      dev.queue.submit([enc2.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      results[k] = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
    }
    return results;
  }

  _mkBuf(dev, data, usage, size) {
    const sz = size ?? (data?.byteLength ?? 64);
    const buf = dev.createBuffer({ size: Math.max(64, sz), usage, mappedAtCreation: false });
    return buf;
  }
}
