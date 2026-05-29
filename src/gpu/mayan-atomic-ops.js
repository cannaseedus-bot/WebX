// Mayan Atomic Ops v0.1 — unified base-20 atomic semantics across all backends
//
// Canonical interface: AtomicMayanOps
//   Backend 1 — GPU / WebGPU (WGSL atomicCompareExchangeWeak equivalent)
//   Backend 2 — WASM SIMD + SharedArrayBuffer (JS Atomics on Int32Array)
//   Backend 3 — CPU Atomics (JS Atomics or sequential fallback)
//
// All backends share the same logical contract:
//   Digit d ∈ {0,…,19}; carry ∈ {0,1}; CAS loop guarantees lock-free progress.
//
// Ordering:
//   Per-digit: linearized by CAS.
//   Per-vector: logically sequential, least-significant → most-significant digit.
//   Cross-backend: higher-level synchronization handled by DAG scheduler.

// ─── SAB-backed backend (SharedArrayBuffer + JS Atomics) ─────────────────────

export class AtomicMayanOpsSAB {
  constructor(buffer, byteOffset = 0) {
    // buffer must be a SharedArrayBuffer or ArrayBuffer
    this._view = new Int32Array(buffer, byteOffset);
  }

  load(ptr)             { return Atomics.load(this._view, ptr); }
  store(ptr, val)       { Atomics.store(this._view, ptr, val); }
  exchange(ptr, val)    { return Atomics.exchange(this._view, ptr, val); }
  add(ptr, val)         { return Atomics.add(this._view, ptr, val); }

  cas(ptr, expected, replacement) {
    return Atomics.compareExchange(this._view, ptr, expected, replacement);
  }

  // Base-20 atomic add on one digit.
  // Returns the new digit value. Propagates carry to carry_ptr if provided.
  mayan_add_digit(digit_ptr, value, carry_ptr = null) {
    for (;;) {
      const old = Atomics.load(this._view, digit_ptr);
      const sum = old + value;

      if (sum < 20) {
        if (Atomics.compareExchange(this._view, digit_ptr, old, sum) === old) return sum;
        continue;
      }

      const new_digit = sum - 20;
      if (Atomics.compareExchange(this._view, digit_ptr, old, new_digit) !== old) continue;

      if (carry_ptr !== null) this.mayan_add_digit(carry_ptr, 1, null);
      return new_digit;
    }
  }

  // Vector add: add scalar value to a multi-digit buffer (LSB at index base_ptr).
  mayan_add_vec(base_ptr, len, value) {
    let carry = value;
    for (let i = 0; i < len && carry > 0; i++) {
      const old = Atomics.load(this._view, base_ptr + i);
      this.mayan_add_digit(base_ptr + i, carry, null);
      const updated = Atomics.load(this._view, base_ptr + i);
      carry = (old + carry >= 20) ? 1 : 0;
      void updated;
    }
  }
}

// ─── CPU sequential backend (plain Int32Array, no Atomics) ───────────────────
// Used when SharedArrayBuffer is unavailable. Single-threaded only.

export class AtomicMayanOpsCPU {
  constructor(capacity = 256) {
    this._mem = new Int32Array(capacity);
  }

  load(ptr)          { return this._mem[ptr]; }
  store(ptr, val)    { this._mem[ptr] = val; }
  exchange(ptr, val) { const old = this._mem[ptr]; this._mem[ptr] = val; return old; }
  add(ptr, val)      { const old = this._mem[ptr]; this._mem[ptr] = old + val; return old; }

  cas(ptr, expected, replacement) {
    const old = this._mem[ptr];
    if (old === expected) this._mem[ptr] = replacement;
    return old;
  }

  mayan_add_digit(digit_ptr, value, carry_ptr = null) {
    const old = this._mem[digit_ptr];
    const sum = old + value;
    if (sum < 20) {
      this._mem[digit_ptr] = sum;
      return sum;
    }
    const new_digit = sum - 20;
    this._mem[digit_ptr] = new_digit;
    if (carry_ptr !== null) this.mayan_add_digit(carry_ptr, 1, null);
    return new_digit;
  }

  mayan_add_vec(base_ptr, len, value) {
    let carry = value;
    for (let i = 0; i < len && carry > 0; i++) {
      const old = this._mem[base_ptr + i];
      const sum = old + carry;
      if (sum < 20) { this._mem[base_ptr + i] = sum; carry = 0; }
      else          { this._mem[base_ptr + i] = sum - 20; carry = 1; }
    }
  }
}

// ─── Backend factory ─────────────────────────────────────────────────────────

export function createAtomicMayanOps(opts = {}) {
  if (opts.sab instanceof SharedArrayBuffer) {
    return new AtomicMayanOpsSAB(opts.sab, opts.byteOffset ?? 0);
  }
  if (opts.buffer instanceof ArrayBuffer) {
    return new AtomicMayanOpsSAB(opts.buffer, opts.byteOffset ?? 0);
  }
  return new AtomicMayanOpsCPU(opts.capacity ?? 256);
}

// ─── Mayan digit buffer initializer ──────────────────────────────────────────
// Fill a SAB-backed Int32Array region with Long Count digits [kin,uinal,…,baktun].

export function initMayanBuffer(ops, basePtr, digits) {
  for (let i = 0; i < digits.length; i++) {
    ops.store(basePtr + i, digits[i]);
  }
}

export function readMayanBuffer(ops, basePtr, len) {
  const out = new Array(len);
  for (let i = 0; i < len; i++) out[i] = ops.load(basePtr + i);
  return out;
}
