// XVM 32-fiber CPU cluster VM
// Port of xvm_runtime.h/cpp + xvm_compute.hlsl with manifold opcodes.
// Source: v0.1.0-xvm-cpu-thread-cluster

const CLUSTER_SIZE = 32;
const SHARED_SIZE = 1024;
const PHASE_COUNT = 6;
const TRACE_CAPACITY = 65536;

// CPU cluster training optimum: low batch / high epoch reaches best ML loss.
// 1000 batch × 4 epoch or 500 batch × 8 epoch are the validated sweet spots.
export const XVM_TRAINING_OPTIMUM = Object.freeze({
  configs: [
    { batch_size: 1000, epochs: 4 },
    { batch_size: 500,  epochs: 8 },
  ],
  rationale: 'Phase-based sync + manifold ops provide better gradient signal at small batch size; high epoch count compensates for CPU throughput limits.',
});

export const FLAG = Object.freeze({ HALTED: 0, ACTIVE: 1, BARRIER_WAIT: 2 });

export const OP = Object.freeze({
  LOAD_CONST:          0x01,
  MOV:                 0x02,
  ADD:                 0x03,
  SUB:                 0x04,
  MUL:                 0x05,
  DIV:                 0x06,
  ATOMIC_ADD:          0x10,
  JMP:                 0x20,
  JMP_IF:              0x21,
  CMP_EQ:              0x22,
  LOAD_SHARED:         0x30,
  STORE_SHARED:        0x31,
  BARRIER:             0x3e,
  RETURN:              0x3f,
  // Manifold opcodes — phase-space geometry
  GEODESIC:            0x40,
  ENTROPY_GRADIENT:    0x41,
  RIEMANN_CURVATURE:   0x42,
  FOLD_ENTER:          0x43,
  FOLD_EXIT:           0x44,
  PRESSURE_PROPAGATE:  0x45,
});

class XVMFiber {
  constructor() {
    this.pc = 0;
    this.sp = 0;
    this.phase = 0;           // [0, PHASE_COUNT)
    this.flags = FLAG.ACTIVE;
    this.r0 = 0;
    this.r1 = 0;
    this.r2 = 0;
    this.r3 = 0;
    this.entropy = 0;         // u8, drives manifold ops
    this.pressure = 0;        // u8, propagates to neighbors
  }

  readReg(idx) {
    switch (idx & 3) {
      case 0: return this.r0;
      case 1: return this.r1;
      case 2: return this.r2;
      default: return this.r3;
    }
  }

  writeReg(idx, value) {
    const v = value >>> 0;
    switch (idx & 3) {
      case 0: this.r0 = v; break;
      case 1: this.r1 = v; break;
      case 2: this.r2 = v; break;
      default: this.r3 = v; break;
    }
  }
}

export class CPUCluster32 {
  constructor(clusterCount = 1) {
    this.clusterCount = clusterCount;
    this.fibers = Array.from({ length: CLUSTER_SIZE * clusterCount }, () => new XVMFiber());
    this.shared = new Uint32Array(SHARED_SIZE);
    this.code = new Uint32Array(0);
    this.constPool = new Uint8Array(0);
    this.tickCount = 0;
    this.trace = [];
    this._traceEnabled = false;
  }

  loadProgram(code, constPool = new Uint8Array(0)) {
    if (code instanceof Uint32Array) {
      this.code = code;
    } else if (ArrayBuffer.isView(code) || code instanceof ArrayBuffer) {
      this.code = new Uint32Array(ArrayBuffer.isView(code) ? code.buffer : code);
    } else {
      this.code = new Uint32Array(code);
    }
    this.constPool = constPool instanceof Uint8Array ? constPool : new Uint8Array(constPool);
    for (const f of this.fibers) {
      f.pc = 0;
      f.sp = 0;
      f.flags = FLAG.ACTIVE;
    }
  }

  enableTrace(on = true) {
    this._traceEnabled = on;
    if (on) this.trace = [];
  }

  _writeTrace(fid, op, f) {
    if (!this._traceEnabled || this.trace.length >= TRACE_CAPACITY) return;
    this.trace.push({ fid, op, r0: f.r0, r1: f.r1, r2: f.r2, r3: f.r3, pc: f.pc });
  }

  step(fiberId) {
    const f = this.fibers[fiberId];
    if (f.flags === FLAG.HALTED || f.flags === FLAG.BARRIER_WAIT) return;
    if (f.pc >= this.code.length) { f.flags = FLAG.HALTED; return; }

    const op = this.code[f.pc++] & 0x3f;

    switch (op) {
      case OP.LOAD_CONST: {
        const reg = this.code[f.pc++];
        const imm = this.code[f.pc++];
        f.writeReg(reg, imm);
        break;
      }
      case OP.MOV: {
        const dst = this.code[f.pc++];
        const src = this.code[f.pc++];
        f.writeReg(dst, f.readReg(src));
        break;
      }
      case OP.ADD: {
        const dst = this.code[f.pc++];
        const src = this.code[f.pc++];
        f.writeReg(dst, (f.readReg(dst) + f.readReg(src)) >>> 0);
        break;
      }
      case OP.SUB: {
        const dst = this.code[f.pc++];
        const src = this.code[f.pc++];
        f.writeReg(dst, (f.readReg(dst) - f.readReg(src)) >>> 0);
        break;
      }
      case OP.MUL: {
        const dst = this.code[f.pc++];
        const src = this.code[f.pc++];
        f.writeReg(dst, Math.imul(f.readReg(dst), f.readReg(src)) >>> 0);
        break;
      }
      case OP.DIV: {
        const dst = this.code[f.pc++];
        const src = this.code[f.pc++];
        const div = f.readReg(src);
        f.writeReg(dst, div === 0 ? 0 : ((f.readReg(dst) / div) >>> 0));
        break;
      }
      case OP.ATOMIC_ADD: {
        const idx = this.code[f.pc++];
        const val = this.code[f.pc++];
        if (idx < SHARED_SIZE) this.shared[idx] = (this.shared[idx] + val) >>> 0;
        break;
      }
      case OP.JMP: {
        f.pc = this.code[f.pc];
        break;
      }
      case OP.JMP_IF: {
        const target = this.code[f.pc++];
        if (f.r0 !== 0) f.pc = target;
        break;
      }
      case OP.CMP_EQ: {
        const a = this.code[f.pc++];
        const b = this.code[f.pc++];
        f.r0 = f.readReg(a) === f.readReg(b) ? 1 : 0;
        break;
      }
      case OP.LOAD_SHARED: {
        const reg = this.code[f.pc++];
        const idx = this.code[f.pc++];
        f.writeReg(reg, idx < SHARED_SIZE ? this.shared[idx] : 0);
        break;
      }
      case OP.STORE_SHARED: {
        const idx = this.code[f.pc++];
        const reg = this.code[f.pc++];
        if (idx < SHARED_SIZE) this.shared[idx] = f.readReg(reg);
        break;
      }
      case OP.BARRIER:
        // Hold until all 32 fibers in the cluster reach the same phase.
        f.flags = FLAG.BARRIER_WAIT;
        break;
      case OP.RETURN:
        f.flags = FLAG.HALTED;
        break;

      // Manifold opcodes — phase-space geometry
      case OP.GEODESIC: {
        // dst ← scaled Euclidean distance between this fiber and target in (phase, entropy, pressure) space
        const dst = this.code[f.pc++];
        const targetId = this.code[f.pc++] % this.fibers.length;
        const tf = this.fibers[targetId];
        const dPhase = Math.abs(f.phase - tf.phase) % PHASE_COUNT;
        const dist = Math.sqrt(dPhase * dPhase + (f.entropy - tf.entropy) ** 2 + (f.pressure - tf.pressure) ** 2);
        f.writeReg(dst, (dist * 1000 + 0.5) >>> 0);
        break;
      }
      case OP.ENTROPY_GRADIENT: {
        // r0 ← unsigned diff vs lane-rotated neighbor; bump local entropy by 1
        const neighbor = (fiberId + 1) % this.fibers.length;
        f.r0 = ((f.entropy - this.fibers[neighbor].entropy) >>> 0) & 0xFF;
        f.entropy = (f.entropy + 1) & 0xFF;
        break;
      }
      case OP.RIEMANN_CURVATURE: {
        // dst ← entropy XOR pressure (cheap curvature estimate)
        const dst = this.code[f.pc++];
        f.writeReg(dst, (f.entropy ^ f.pressure) >>> 0);
        break;
      }
      case OP.FOLD_ENTER: {
        // Advance phase if entropy ≥ threshold; reset entropy on transition
        const threshold = this.code[f.pc++];
        if (f.entropy >= threshold) {
          f.phase = (f.phase + 1) % PHASE_COUNT;
          f.entropy = 0;
        }
        break;
      }
      case OP.FOLD_EXIT: {
        // Step phase back (floor at 0)
        if (f.phase > 0) f.phase--;
        break;
      }
      case OP.PRESSURE_PROPAGATE: {
        // Push pressure delta to ring neighbors; self loses 2× what each neighbor gains
        const strength = this.code[f.pc++] & 0xFF;
        const left  = (fiberId - 1 + this.fibers.length) % this.fibers.length;
        const right = (fiberId + 1) % this.fibers.length;
        const delta = ((f.pressure * strength) / 255 + 0.5) >>> 0;
        this.fibers[left].pressure  = (this.fibers[left].pressure  + delta) & 0xFF;
        this.fibers[right].pressure = (this.fibers[right].pressure + delta) & 0xFF;
        f.pressure = Math.max(0, f.pressure - delta * 2) & 0xFF;
        break;
      }
      default:
        f.flags = FLAG.HALTED;
        break;
    }

    this._writeTrace(fiberId, op, f);
  }

  // Release barrier-waiting fibers in each cluster once all 32 are waiting at the same phase.
  releaseClusterBarriers() {
    for (let c = 0; c < this.clusterCount; c++) {
      const base = c * CLUSTER_SIZE;
      const refPhase = this.fibers[base].phase;
      let allBlocked = true;
      for (let i = 0; i < CLUSTER_SIZE; i++) {
        const f = this.fibers[base + i];
        if (f.flags === FLAG.ACTIVE) { allBlocked = false; break; }
        if (f.flags === FLAG.BARRIER_WAIT && f.phase !== refPhase) { allBlocked = false; break; }
      }
      if (allBlocked) {
        for (let i = 0; i < CLUSTER_SIZE; i++) {
          if (this.fibers[base + i].flags === FLAG.BARRIER_WAIT) {
            this.fibers[base + i].flags = FLAG.ACTIVE;
          }
        }
      }
    }
  }

  run(tickCount = 1) {
    for (let t = 0; t < tickCount; t++) {
      for (let i = 0; i < this.fibers.length; i++) {
        if (this.fibers[i].flags !== FLAG.HALTED) this.step(i);
      }
      this.releaseClusterBarriers();
      this.tickCount++;
    }
  }

  get activeFiberCount() {
    return this.fibers.filter(f => f.flags !== FLAG.HALTED).length;
  }

  reset() {
    for (const f of this.fibers) {
      f.pc = 0; f.sp = 0; f.phase = 0;
      f.flags = FLAG.ACTIVE;
      f.r0 = f.r1 = f.r2 = f.r3 = 0;
      f.entropy = 0; f.pressure = 0;
    }
    this.shared.fill(0);
    this.tickCount = 0;
    this.trace = [];
  }

  snapshot() {
    return {
      tickCount: this.tickCount,
      activeFibers: this.activeFiberCount,
      shared: Array.from(this.shared.slice(0, 16)),
      fibers: this.fibers.map((f, i) => ({
        id: i, pc: f.pc, phase: f.phase, flags: f.flags,
        r0: f.r0, r1: f.r1, r2: f.r2, r3: f.r3,
        entropy: f.entropy, pressure: f.pressure,
      })),
    };
  }
}

export const CLUSTER_SIZE_CONST = CLUSTER_SIZE;
export const PHASE_COUNT_CONST = PHASE_COUNT;
export default CPUCluster32;
