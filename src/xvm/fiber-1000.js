// XVM 1000-fiber CUDA emulator
// Port of xvm_cuda_complete.cpp + xvm_cluster_manager.cpp from .gpu_trainer/kuhul/xvm/
// Extends CPUCluster32 with warp vote, shuffle, tensor cores, dynamic parallelism, streams.

import CPUCluster32, { FLAG, OP as BASE_OP } from './cpu-cluster.js';

export const WARP_SIZE = 32;

export const OP_EXT = Object.freeze({
  // Warp vote (0x30-0x32)
  VOTE_ALL:       0x30,
  VOTE_ANY:       0x31,
  BALLOT:         0x32,
  // Warp shuffle (0x33-0x35)
  SHUFFLE_XOR:    0x33,
  SHUFFLE_UP:     0x34,
  SHUFFLE_DOWN:   0x35,
  // Tensor core emulation (0x42)
  WMMA_COMPUTE:   0x42,
  // Dynamic parallelism (0x50-0x51)
  LAUNCH_KERNEL:  0x50,
  LAUNCH_GRID:    0x51,
  // Async streams (0x60-0x62)
  STREAM_CREATE:  0x60,
  STREAM_LAUNCH:  0x61,
  STREAM_SYNC:    0x62,
});

// Default cluster configuration matching xvm_cluster_manager.cpp DEFAULT_CLUSTER
export const DEFAULT_CONFIG = Object.freeze({
  totalCores:    1000,
  cpuThreads:    8,
  sharedMemoryKB: 1024,
  ticksPerCore:  1000000,
  synchronize:   true,
});

// Stream state
class XVMStream {
  constructor(id) {
    this.id = id;
    this.completed = true;
    this._promise = null;
  }
}

export class FiberPool1000 extends CPUCluster32 {
  constructor(config = {}) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    // totalCores fibers, 1 cluster per warp (groups of 32)
    super(Math.ceil(cfg.totalCores / WARP_SIZE));
    this.config = cfg;

    // Allocate exactly totalCores fibers (clusterCount * 32 might be > totalCores)
    this.fibers = this.fibers.slice(0, cfg.totalCores);
    // Set each fiber's r0 to its core index (like CUDA threadIdx)
    for (let i = 0; i < this.fibers.length; i++) {
      this.fibers[i].r0 = i;
    }

    this.streams = new Map();
    this._pendingKernels = [];
    this.metrics = { totalInstructions: 0, startTime: 0, elapsedMs: 0 };
  }

  // ─── Warp vote ────────────────────────────────────────────────────────────

  _warpBase(fiberId) {
    return Math.floor(fiberId / WARP_SIZE) * WARP_SIZE;
  }

  _warpFibers(fiberId) {
    const base = this._warpBase(fiberId);
    return this.fibers.slice(base, base + WARP_SIZE);
  }

  voteAll(fiberId, predReg, dstReg) {
    const f = this.fibers[fiberId];
    const allTrue = this._warpFibers(fiberId)
      .every(wf => wf.flags !== FLAG.HALTED && wf.readReg(predReg) !== 0);
    f.writeReg(dstReg, allTrue ? 1 : 0);
  }

  voteAny(fiberId, predReg, dstReg) {
    const f = this.fibers[fiberId];
    const anyTrue = this._warpFibers(fiberId)
      .some(wf => wf.flags !== FLAG.HALTED && wf.readReg(predReg) !== 0);
    f.writeReg(dstReg, anyTrue ? 1 : 0);
  }

  ballot(fiberId, predReg, dstReg) {
    const f = this.fibers[fiberId];
    let mask = 0;
    const base = this._warpBase(fiberId);
    for (let i = 0; i < WARP_SIZE; i++) {
      const wf = this.fibers[base + i];
      if (wf && wf.flags !== FLAG.HALTED && wf.readReg(predReg) !== 0) {
        mask |= (1 << i);
      }
    }
    f.writeReg(dstReg, mask >>> 0);
  }

  // ─── Warp shuffle ─────────────────────────────────────────────────────────

  shuffleXor(fiberId, srcReg, laneMask, dstReg) {
    const f = this.fibers[fiberId];
    const lane = fiberId % WARP_SIZE;
    const targetLane = (lane ^ laneMask) % WARP_SIZE;
    const targetFiber = this._warpBase(fiberId) + targetLane;
    const tf = this.fibers[targetFiber];
    f.writeReg(dstReg, tf && tf.flags !== FLAG.HALTED
      ? tf.readReg(srcReg)
      : f.readReg(srcReg));
  }

  shuffleUp(fiberId, srcReg, delta, dstReg) {
    const f = this.fibers[fiberId];
    const lane = fiberId % WARP_SIZE;
    const targetLane = Math.max(0, lane - delta);
    const targetFiber = this._warpBase(fiberId) + targetLane;
    const tf = this.fibers[targetFiber];
    f.writeReg(dstReg, tf && tf.flags !== FLAG.HALTED
      ? tf.readReg(srcReg)
      : f.readReg(srcReg));
  }

  shuffleDown(fiberId, srcReg, delta, dstReg) {
    const f = this.fibers[fiberId];
    const lane = fiberId % WARP_SIZE;
    const targetLane = Math.min(WARP_SIZE - 1, lane + delta);
    const targetFiber = this._warpBase(fiberId) + targetLane;
    const tf = this.fibers[targetFiber];
    f.writeReg(dstReg, tf && tf.flags !== FLAG.HALTED
      ? tf.readReg(srcReg)
      : f.readReg(srcReg));
  }

  // ─── Tensor core emulation ────────────────────────────────────────────────

  // 16×16×16 matmul-accumulate: C += A * B
  // A, B, C are Float32Array[256] stored in this.shared starting at byte offsets r0, r1, r2
  wmmaCompute(fiberId) {
    const f = this.fibers[fiberId];
    const aBase = f.r0;
    const bBase = f.r1;
    const cBase = f.r2;
    const M = 16, N = 16, K = 16;

    // Interpret shared memory as float (shared is Uint32Array, reinterpret as Float32)
    const sharedBuf = this.shared.buffer;
    const A = new Float32Array(sharedBuf, aBase * 4, M * K);
    const B = new Float32Array(sharedBuf, bBase * 4, K * N);
    const C = new Float32Array(sharedBuf, cBase * 4, M * N);

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = C[i * N + j];
        for (let k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  }

  // ─── Dynamic parallelism ─────────────────────────────────────────────────

  launchKernel(parentFiberId, kernelAddr, numFibers) {
    const startId = this.fibers.length;
    for (let i = 0; i < numFibers; i++) {
      const nf = {
        pc: kernelAddr, sp: 0, phase: 0, flags: FLAG.ACTIVE,
        r0: i, r1: 0, r2: 0, r3: 0, entropy: 0, pressure: 0,
        readReg(idx) { return [this.r0, this.r1, this.r2, this.r3][idx & 3] || 0; },
        writeReg(idx, v) {
          const key = ['r0','r1','r2','r3'][idx & 3];
          this[key] = v >>> 0;
        },
      };
      this.fibers.push(nf);
    }
    this._pendingKernels.push({ startId, count: numFibers });
  }

  // ─── Async streams ────────────────────────────────────────────────────────

  streamCreate(id) {
    if (!this.streams.has(id)) this.streams.set(id, new XVMStream(id));
    return this.streams.get(id);
  }

  streamLaunch(id, task) {
    const s = this.streamCreate(id);
    s.completed = false;
    s._promise = Promise.resolve().then(() => { task(); s.completed = true; });
    return s._promise;
  }

  async streamSync(id) {
    const s = this.streams.get(id);
    if (s && s._promise) await s._promise;
  }

  // ─── Extended step dispatcher ─────────────────────────────────────────────

  step(fiberId) {
    const f = this.fibers[fiberId];
    if (!f || f.flags === FLAG.HALTED || f.flags === FLAG.BARRIER_WAIT) return;
    if (f.pc >= this.code.length) { f.flags = FLAG.HALTED; return; }

    const op = this.code[f.pc] & 0xff;

    switch (op) {
      case OP_EXT.VOTE_ALL: {
        f.pc++;
        const pred = this.code[f.pc++];
        const dst  = this.code[f.pc++];
        this.voteAll(fiberId, pred, dst);
        break;
      }
      case OP_EXT.VOTE_ANY: {
        f.pc++;
        const pred = this.code[f.pc++];
        const dst  = this.code[f.pc++];
        this.voteAny(fiberId, pred, dst);
        break;
      }
      case OP_EXT.BALLOT: {
        f.pc++;
        const pred = this.code[f.pc++];
        const dst  = this.code[f.pc++];
        this.ballot(fiberId, pred, dst);
        break;
      }
      case OP_EXT.SHUFFLE_XOR: {
        f.pc++;
        const src  = this.code[f.pc++];
        const mask = this.code[f.pc++];
        const dst  = this.code[f.pc++];
        this.shuffleXor(fiberId, src, mask, dst);
        break;
      }
      case OP_EXT.SHUFFLE_UP: {
        f.pc++;
        const src   = this.code[f.pc++];
        const delta = this.code[f.pc++];
        const dst   = this.code[f.pc++];
        this.shuffleUp(fiberId, src, delta, dst);
        break;
      }
      case OP_EXT.SHUFFLE_DOWN: {
        f.pc++;
        const src   = this.code[f.pc++];
        const delta = this.code[f.pc++];
        const dst   = this.code[f.pc++];
        this.shuffleDown(fiberId, src, delta, dst);
        break;
      }
      case OP_EXT.WMMA_COMPUTE:
        f.pc++;
        this.wmmaCompute(fiberId);
        break;
      case OP_EXT.LAUNCH_KERNEL: {
        f.pc++;
        const addr = this.code[f.pc++];
        const cnt  = this.code[f.pc++];
        this.launchKernel(fiberId, addr, cnt);
        break;
      }
      case OP_EXT.LAUNCH_GRID: {
        f.pc++;
        const addr   = this.code[f.pc++];
        const blocks = this.code[f.pc++];
        const threads = this.code[f.pc++];
        this.launchKernel(fiberId, addr, blocks * threads);
        break;
      }
      case OP_EXT.STREAM_CREATE:
        f.pc++;
        this.streamCreate(this.code[f.pc++]);
        break;
      case OP_EXT.STREAM_LAUNCH: {
        f.pc++;
        const streamId = this.code[f.pc++];
        const taskAddr = this.code[f.pc++];
        this.streamLaunch(streamId, () => {
          // Snapshot code at task address and run a minimal fiber for it
          const tf = { pc: taskAddr, sp: 0, flags: FLAG.ACTIVE,
                       r0: 0, r1: 0, r2: 0, r3: 0, phase: 0,
                       entropy: 0, pressure: 0,
                       readReg(i) { return [this.r0,this.r1,this.r2,this.r3][i&3]||0; },
                       writeReg(i, v) { const k=['r0','r1','r2','r3'][i&3]; this[k]=v>>>0; } };
          this.fibers.push(tf);
        });
        break;
      }
      case OP_EXT.STREAM_SYNC:
        f.pc++;
        this.streamSync(this.code[f.pc++]);
        break;
      default:
        // Delegate to base 32-fiber implementation for all other opcodes
        super.step(fiberId);
        return;
    }
  }

  // Launch: distribute fibers across cpuThreads (simulated; JS is single-threaded)
  launch() {
    this.metrics.startTime = performance.now();
    const ticks = this.config.ticksPerCore;
    for (let t = 0; t < ticks; t++) {
      let anyActive = false;
      for (let i = 0; i < this.fibers.length; i++) {
        if (this.fibers[i].flags !== FLAG.HALTED) {
          this.step(i);
          anyActive = true;
        }
      }
      this.releaseClusterBarriers();
      this.tickCount++;
      if (!anyActive) break;
    }
    this.metrics.elapsedMs = performance.now() - this.metrics.startTime;
    this.metrics.totalInstructions = this.fibers.length * this.config.ticksPerCore;
  }

  get totalFibers() {
    return this.fibers.length;
  }

  get ips() {
    return this.metrics.elapsedMs > 0
      ? this.metrics.totalInstructions / (this.metrics.elapsedMs / 1000)
      : 0;
  }

  // Sum of r1 (accumulator) across all cores — mirrors GetAggregatedResult()
  get aggregatedResult() {
    return this.fibers.reduce((sum, f) => sum + f.r1, 0);
  }

  getMetrics() {
    return {
      totalInstructions: this.metrics.totalInstructions,
      executionTimeMs:   this.metrics.elapsedMs,
      mips:              this.ips / 1e6,
      activeCores:       this.activeFiberCount,
      aggregatedResult:  this.aggregatedResult,
    };
  }
}

export default FiberPool1000;
