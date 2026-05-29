// Agents.NET SyncWorker — in-memory port (Agents.NET.v1.0.0)
//
// Mirrors SyncWorker.cs tick-based execution model.
// Real implementation reads from Windows named MMF (Local\KuhulGeometricState).
// This port accepts a state supplier function instead, enabling browser + Node use.

export class SyncWorker {
  constructor(stateSupplier, taskHandler = null) {
    this._getState  = stateSupplier;
    this._onTask    = taskHandler || defaultTaskHandler;
    this._lastTick  = 0;
    this._running   = false;
    this._intervalId = null;
  }

  start(pollMs = 1) {
    if (this._running) return;
    this._running = true;
    this._intervalId = setInterval(() => {
      const state = this._getState();
      if (state && state.tickCount !== this._lastTick) {
        this._lastTick = state.tickCount;
        Promise.resolve(this._onTask(state)).catch(() => {});
      }
    }, pollMs);
  }

  stop() {
    this._running = false;
    if (this._intervalId !== null) {
      clearInterval(this._intervalId);
      this._intervalId = null;
    }
  }

  get lastTick() { return this._lastTick; }
  get running()  { return this._running; }
}

function defaultTaskHandler(state) {
  if (state.entropy > 0.5)   console.log('[⟁] High entropy — manifold stabilization needed');
  if (state.attention > 0.8) console.log('[*] High attention — specialized task active');
}

// SecurityPlugin op names (from SecurityPlugin.cs KernelFunction attrs)
export const SECURITY_PLUGIN_OPS = Object.freeze([
  'VerifyIdentityAsync',
  'IssueSecuroLink',
  'RevokeIdentity',
]);
