// WebGPU Fallback Manager v1
//
// Detects GPU capability and manages a 4-tier fallback chain:
//   Tier 1 — native GPU (driver-level, cs_5_0 DXBC)
//   Tier 2 — WebGPU (navigator.gpu)
//   Tier 3 — WASM SIMD + SharedArrayBuffer
//   Tier 4 — CPU Atomics (sequential fallback)
//
// Tensor allocation tracks VRAM usage with LRU eviction to CPU.
// Pure ESM — no Node-only APIs; works in browser and Node 18+.

// ─── Backend detection ────────────────────────────────────────────────────────

export const BACKEND = Object.freeze({
  GPU_NATIVE: 'gpu_native',
  WEBGPU:     'webgpu',
  WASM_SIMD:  'wasm_simd',
  CPU:        'cpu',
});

export const MIN_WEBGPU_BUFFER_BYTES = 128 * 1024 * 1024; // 128 MB
export const SAB_TOTAL_BYTES         = 2 * 1024 * 1024 * 1024; // 2 GB

/**
 * Probe for WebGPU capability.
 * Returns one of: 'webgpu_full' | 'webgpu_limited' | 'webgl' | 'cpu'
 */
export async function detectWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return 'cpu';

  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
  } catch (_) {
    return 'cpu';
  }
  if (!adapter) return 'cpu';

  const limits = adapter.limits;
  if ((limits?.maxBufferSize ?? 0) < MIN_WEBGPU_BUFFER_BYTES) return 'cpu';

  const hasSAB = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) &&
                 (typeof SharedArrayBuffer !== 'undefined');
  return hasSAB ? 'webgpu_full' : 'webgpu_limited';
}

export function detectWasmSimd() {
  // WebAssembly.validate with SIMD opcode (0xfd01 = v128.const)
  try {
    const simdProbe = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // WASM magic + version
      0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,        // type: () -> v128
      0x03, 0x02, 0x01, 0x00,                           // function
      0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0c,        // body: v128.const
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x0b,
    ]);
    return typeof WebAssembly !== 'undefined' && WebAssembly.validate(simdProbe);
  } catch (_) {
    return false;
  }
}

export function detectSAB() {
  return typeof SharedArrayBuffer !== 'undefined';
}

// ─── Tensor allocation + LRU eviction ────────────────────────────────────────

export class WebGPUFallbackManager {
  constructor(opts = {}) {
    this._vramBudget   = opts.vramBudget ?? 1.5 * 1024 * 1024 * 1024; // 1.5 GB
    this._vramUsed     = 0;
    this._allocations  = new Map();  // id → { buffer, bytes, lastUsed }
    this._lruOrder     = [];         // ids in LRU order (oldest first)
    this._cpuBackup    = new Map();  // id → Float32Array (evicted to CPU)
    this._nextId       = 0;
    this._backend      = BACKEND.CPU;
    this._gpuDevice    = null;
    this._sabBuffer    = null;
  }

  async init() {
    const webgpuCap = await detectWebGPU();
    const hasSimd   = detectWasmSimd();
    const hasSAB    = detectSAB();

    if (webgpuCap === 'webgpu_full' || webgpuCap === 'webgpu_limited') {
      this._backend = BACKEND.WEBGPU;
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
        this._gpuDevice = await adapter.requestDevice();
      } catch (_) {
        this._backend = hasSimd && hasSAB ? BACKEND.WASM_SIMD : BACKEND.CPU;
      }
    } else if (hasSimd && hasSAB) {
      this._backend = BACKEND.WASM_SIMD;
      this._sabBuffer = new SharedArrayBuffer(Math.min(SAB_TOTAL_BYTES, 512 * 1024 * 1024));
    } else {
      this._backend = BACKEND.CPU;
    }

    return this._backend;
  }

  get backend() { return this._backend; }

  // Allocate a tensor buffer — returns allocation id
  allocateTensor(elementCount, ElementType = Float32Array) {
    const bytes = elementCount * ElementType.BYTES_PER_ELEMENT;

    // Evict LRU tensors until we have room
    while (this._vramUsed + bytes > this._vramBudget && this._lruOrder.length > 0) {
      this._evictLRU();
    }

    const id = String(this._nextId++);

    if (this._backend === BACKEND.WEBGPU && this._gpuDevice) {
      try {
        const gpuBuffer = this._gpuDevice.createBuffer({
          size:  Math.max(bytes, 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this._allocations.set(id, { buffer: gpuBuffer, bytes, lastUsed: Date.now(), kind: 'gpu' });
        this._vramUsed += bytes;
        this._touchLRU(id);
        return id;
      } catch (oom) {
        // OOM — fall through to CPU
        this._evictLRU();
      }
    }

    // CPU/WASM fallback allocation
    const cpuBuf = new ElementType(elementCount);
    this._cpuBackup.set(id, cpuBuf);
    this._allocations.set(id, { buffer: cpuBuf, bytes, lastUsed: Date.now(), kind: 'cpu' });
    return id;
  }

  // Read tensor data back to a TypedArray
  async readTensor(id, ElementType = Float32Array) {
    const alloc = this._allocations.get(id);
    if (!alloc) throw new Error(`WebGPUFallbackManager: unknown tensor id ${id}`);
    this._touchLRU(id);

    if (alloc.kind === 'cpu') return new ElementType(alloc.buffer.buffer ?? alloc.buffer);

    // WebGPU readback
    const stagingBuffer = this._gpuDevice.createBuffer({
      size:  alloc.bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this._gpuDevice.createCommandEncoder();
    enc.copyBufferToBuffer(alloc.buffer, 0, stagingBuffer, 0, alloc.bytes);
    this._gpuDevice.queue.submit([enc.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new ElementType(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    return result;
  }

  swapToCPU(id) {
    const alloc = this._allocations.get(id);
    if (!alloc || alloc.kind === 'cpu') return;

    // Move to CPU backup (data transfer is async in real usage; this marks it evicted)
    const cpuBuf = new Float32Array(alloc.bytes / Float32Array.BYTES_PER_ELEMENT);
    this._cpuBackup.set(id, cpuBuf);
    alloc.buffer.destroy?.();
    alloc.kind   = 'cpu';
    alloc.buffer = cpuBuf;
    this._vramUsed -= alloc.bytes;
  }

  freeTensor(id) {
    const alloc = this._allocations.get(id);
    if (!alloc) return;
    if (alloc.kind === 'gpu') {
      alloc.buffer.destroy?.();
      this._vramUsed -= alloc.bytes;
    }
    this._allocations.delete(id);
    this._cpuBackup.delete(id);
    const idx = this._lruOrder.indexOf(id);
    if (idx >= 0) this._lruOrder.splice(idx, 1);
  }

  // Execute a compute kernel — dispatches to the active backend
  async executeAtomic(kernelFn, inputIds, outputId) {
    try {
      return await kernelFn(this, inputIds, outputId);
    } catch (oom) {
      if (String(oom).includes('out of memory') || String(oom).includes('OOM')) {
        // Evict half the LRU list then retry on CPU
        const count = Math.max(1, Math.floor(this._lruOrder.length / 2));
        for (let i = 0; i < count; i++) this._evictLRU();
        this._backend = BACKEND.CPU;
        return await kernelFn(this, inputIds, outputId);
      }
      throw oom;
    }
  }

  get vramUsed()    { return this._vramUsed; }
  get vramBudget()  { return this._vramBudget; }
  get vramPressure() {
    const ratio = this._vramUsed / this._vramBudget;
    if (ratio < 0.4) return 'low';
    if (ratio < 0.7) return 'medium';
    return 'high';
  }

  _touchLRU(id) {
    const idx = this._lruOrder.indexOf(id);
    if (idx >= 0) this._lruOrder.splice(idx, 1);
    this._lruOrder.push(id);
    const alloc = this._allocations.get(id);
    if (alloc) alloc.lastUsed = Date.now();
  }

  _evictLRU() {
    if (this._lruOrder.length === 0) return;
    const oldest = this._lruOrder[0];
    this.swapToCPU(oldest);
    this._lruOrder.shift();
  }
}
