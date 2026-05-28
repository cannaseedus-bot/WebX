// XJSL WebGPU runtime — WebGPU dispatch adapter for XJSL shaders.
// Port of xjsl-wgpu-runtime.mjs from v0.1.1-igpu-trainer-xjsl.
// Browser-only: requires navigator.gpu (WebGPU API).

export class XJSLWGPURuntime {
  constructor() {
    this.device      = null;
    this.pipelines   = new Map();
    this.definitions = new Map();
  }

  async initialize() {
    if (!globalThis.navigator?.gpu) {
      throw new Error('WebGPU is not available in this environment');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter available');
    this.device = await adapter.requestDevice();
    return this;
  }

  // Compile and register a WGSL shader by name.
  registerWGSL(name, wgslCode, entryPoint = 'cs_main') {
    const module   = this.device.createShaderModule({ code: wgslCode, label: `xjsl:${name}` });
    const pipeline = this.device.createComputePipeline({
      layout:  'auto',
      compute: { module, entryPoint },
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }

  // Register all shaders in an XJSL document.
  // lowerer(name, shaderDef) must return a WGSL string (e.g. generateWGSL from lowering.js).
  async registerXJSLDocument(doc, lowerer) {
    for (const [name, shader] of Object.entries(doc['@shaders'] || {})) {
      const wgsl = lowerer(name, shader);
      this.definitions.set(name, shader);
      this.registerWGSL(name, wgsl);
    }
  }

  // Allocate a STORAGE buffer pre-filled with `data` (TypedArray or ArrayBuffer).
  createStorageBuffer(data, extraUsage = 0) {
    const bytes  = data.byteLength ?? data.length * 4;
    const buffer = this.device.createBuffer({
      size:  Math.max(4, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
    });
    this.device.queue.writeBuffer(buffer, 0, data instanceof ArrayBuffer ? data : data.buffer ?? data);
    return buffer;
  }

  // Allocate a UNIFORM buffer pre-filled with `data`. Size is padded to 16-byte alignment.
  createUniformBuffer(data) {
    const bytes  = data.byteLength ?? data.length * 4;
    const buffer = this.device.createBuffer({
      size:  Math.max(16, (bytes + 15) & ~15),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data instanceof ArrayBuffer ? data : data.buffer ?? data);
    return buffer;
  }

  // Dispatch a registered shader.
  // entries: GPUBindGroupEntry[] (binding + resource)
  // dispatch: [x, y?, z?] workgroup counts
  async dispatch(shaderName, entries, dispatch) {
    const pipeline = this.pipelines.get(shaderName);
    if (!pipeline) throw new Error(`shader not registered: ${shaderName}`);

    const bindGroup = this.device.createBindGroup({
      layout:  pipeline.getBindGroupLayout(0),
      entries,
    });

    const encoder = this.device.createCommandEncoder();
    const pass    = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatch[0], dispatch[1] ?? 1, dispatch[2] ?? 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  // Read `length` float32 values from a GPU buffer into a Float32Array.
  async readFloat32(buffer, length) {
    const bytes   = length * 4;
    const staging = this.device.createBuffer({
      size:  bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    return new Float32Array(copy);
  }
}

export default XJSLWGPURuntime;
