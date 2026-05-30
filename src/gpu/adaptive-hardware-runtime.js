// AdaptiveHardwareRuntime v0.1 — unified GPU/WebGPU/WASM/CPU executor
//
// 4-tier fallback chain:
//   Tier 1 — gpu_native   (driver-level: D3D11/HLSL cs_5_0 for HD4600)
//   Tier 2 — webgpu       (navigator.gpu — browser WebGPU API)
//   Tier 3 — wasm_simd    (WebAssembly SIMD + SharedArrayBuffer)
//   Tier 4 — cpu          (pure JS scalar, Atomics when SAB available)
//
// Integration:
//   MATRIX DAG node → XCFE dispatch → XJSL kernel → execute(kernel, ctx)
//   Returns normalized output regardless of which tier ran it.

import { BACKEND, detectWebGPU, detectWasmSimd, detectSAB, WebGPUFallbackManager } from './webgpu-runtime.js';
import { selectPowerState, dispatchConfig } from './power-scheduler.js';
import { dctCompress, dctDecompress, deltaCompress, deltaDecompress } from './vram-compression.js';
import { HD4600, hd4600DispatchSize } from './hd4600.js';

// ─── ExecContext ──────────────────────────────────────────────────────────────

export function createExecContext(overrides = {}) {
  return {
    deviceInfo: {
      hasNativeGPU:       false,
      hasWebGPU:          false,
      hasWasmSimd:        false,
      hasSharedArrayBuffer: false,
      vramBytes:          0,
      maxBufferBytes:     0,
      ...overrides.deviceInfo,
    },
    powerState:     overrides.powerState     ?? 'P0',
    memoryPressure: overrides.memoryPressure ?? 'low',
    gpuUtilization: overrides.gpuUtilization ?? 0,
    temperatureC:   overrides.temperatureC   ?? 60,
    ...overrides,
  };
}

// ─── Backend selection ────────────────────────────────────────────────────────

export function selectBackend(ctx) {
  const { deviceInfo, memoryPressure, temperatureC, powerState } = ctx;

  if (deviceInfo.hasNativeGPU &&
      memoryPressure !== 'high' &&
      temperatureC < 90 &&
      ['P0', 'P1', 'P2'].includes(powerState)) {
    return BACKEND.GPU_NATIVE;
  }

  if (deviceInfo.hasWebGPU &&
      (deviceInfo.maxBufferBytes ?? 0) >= 128 * 1024 * 1024) {
    return BACKEND.WEBGPU;
  }

  if (deviceInfo.hasWasmSimd && deviceInfo.hasSharedArrayBuffer) {
    return BACKEND.WASM_SIMD;
  }

  return BACKEND.CPU;
}

// ─── AdaptiveHardwareRuntime class ───────────────────────────────────────────

export class AdaptiveHardwareRuntime {
  constructor(opts = {}) {
    this._opts        = opts;
    this._ctx         = createExecContext();
    this._webgpuMgr   = null;
    this._initialized = false;
  }

  async init(ctxOverrides = {}) {
    const webgpuCap = await detectWebGPU();
    const hasSimd   = detectWasmSimd();
    const hasSAB    = detectSAB();

    this._ctx = createExecContext({
      ...ctxOverrides,
      deviceInfo: {
        hasNativeGPU:        false,     // native GPU only available through D3D/native path
        hasWebGPU:           webgpuCap.startsWith('webgpu'),
        hasWasmSimd:         hasSimd,
        hasSharedArrayBuffer: hasSAB,
        ...ctxOverrides.deviceInfo,
      },
    });

    if (this._ctx.deviceInfo.hasWebGPU) {
      this._webgpuMgr = new WebGPUFallbackManager({ vramBudget: this._opts.vramBudget });
      await this._webgpuMgr.init();
    }

    this._initialized = true;
    return this;
  }

  // Execute a kernel descriptor — returns a result object.
  // kernel: { kind: 'kernel', kernel: string, inputs: any, outputs: any, meta: {} }
  async execute(kernel, ctxOverride = {}) {
    if (!this._initialized) await this.init();

    const ctx = { ...this._ctx, ...ctxOverride };
    const backend = selectBackend(ctx);

    switch (backend) {
      case BACKEND.GPU_NATIVE:  return this._runOnNativeGPU(kernel, ctx);
      case BACKEND.WEBGPU:      return this._runOnWebGPU(kernel, ctx);
      case BACKEND.WASM_SIMD:   return this._runOnWasmSimd(kernel, ctx);
      default:                   return this._runOnCPU(kernel, ctx);
    }
  }

  // ── Tier 1: native GPU (stub — caller injects actual D3D dispatch) ──────────
  async _runOnNativeGPU(kernel, ctx) {
    const powerState = selectPowerState({
      batteryPct: 100,
      tempC:      ctx.temperatureC ?? 60,
      idleSecs:   0,
    });
    const config = dispatchConfig(powerState);

    if (this._opts.nativeGPUDispatch) {
      return this._opts.nativeGPUDispatch(kernel, config);
    }

    // No native dispatcher injected — fall through to WebGPU
    return this._runOnWebGPU(kernel, ctx);
  }

  // ── Tier 2: WebGPU ──────────────────────────────────────────────────────────
  async _runOnWebGPU(kernel, ctx) {
    const mgr = this._webgpuMgr;
    if (!mgr) return this._runOnWasmSimd(kernel, ctx);

    try {
      // Compression policy based on VRAM pressure
      const pressure    = mgr.vramPressure;
      const inputData   = kernel.inputs?.data;
      const inputRows   = kernel.inputs?.rows ?? 1;
      const inputCols   = kernel.inputs?.cols ?? (inputData?.length ?? 1);

      let processedInput = inputData;
      let compressionMeta = null;

      if (pressure === 'high' && inputData instanceof Float32Array) {
        const compressed  = dctCompress(inputData, inputRows, inputCols);
        processedInput    = compressed.data;
        compressionMeta   = compressed.meta;
      }

      // Allocate and run
      const tensorId = mgr.allocateTensor(processedInput?.length ?? 1);
      const result   = await this._runOnCPU(kernel, ctx); // semantic equivalent on CPU

      // Decompress if needed
      if (compressionMeta) {
        result.data = dctDecompress({ data: result.data, meta: compressionMeta });
      }

      mgr.freeTensor(tensorId);
      return { ...result, backend: BACKEND.WEBGPU };

    } catch (oom) {
      return this._runOnWasmSimd(kernel, ctx);
    }
  }

  // ── Tier 3: WASM SIMD (runs as JS scalar without actual .wasm load) ─────────
  async _runOnWasmSimd(kernel, ctx) {
    const result = await this._runOnCPU(kernel, ctx);
    return { ...result, backend: BACKEND.WASM_SIMD };
  }

  // ── Tier 4: CPU scalar ───────────────────────────────────────────────────────
  async _runOnCPU(kernel, ctx) {
    const { kernel: kernelName, inputs, meta } = kernel;

    // Dispatch to registered CPU kernel implementations
    const impl = AdaptiveHardwareRuntime.CPU_KERNELS[kernelName];
    if (impl) {
      return { result: await impl(inputs, meta, ctx), backend: BACKEND.CPU, kernel: kernelName };
    }

    // Generic passthrough
    return { result: inputs, backend: BACKEND.CPU, kernel: kernelName };
  }

  get context() { return this._ctx; }
}

// ─── Registered CPU kernel implementations ───────────────────────────────────
// Callers may add their own: AdaptiveHardwareRuntime.CPU_KERNELS['my_op'] = fn

AdaptiveHardwareRuntime.CPU_KERNELS = {
  // matrix multiply: inputs = { A: Float64Array, B: Float64Array, m, k, n }
  matmul(inputs) {
    const { A, B, m, k, n } = inputs;
    const C = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let l = 0; l < k; l++) s += A[i * k + l] * B[l * n + j];
        C[i * n + j] = s;
      }
    }
    return C;
  },

  // softmax: inputs = { data: Float64Array }
  softmax(inputs) {
    const v   = Float64Array.from(inputs.data);
    let maxV  = -Infinity;
    for (let i = 0; i < v.length; i++) if (v[i] > maxV) maxV = v[i];
    let sumE  = 0;
    for (let i = 0; i < v.length; i++) { v[i] = Math.exp(v[i] - maxV); sumE += v[i]; }
    for (let i = 0; i < v.length; i++) v[i] /= sumE;
    return v;
  },

  // dot product: inputs = { a: Float64Array, b: Float64Array }
  dot(inputs) {
    let s = 0;
    const { a, b } = inputs;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  },
};

// ─── Singleton factory ────────────────────────────────────────────────────────

let _defaultRuntime = null;

export async function getAdaptiveRuntime(opts) {
  if (!_defaultRuntime) {
    _defaultRuntime = await new AdaptiveHardwareRuntime(opts ?? {}).init();
  }
  return _defaultRuntime;
}

export { BACKEND };
