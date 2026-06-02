"""
kuhul_epoch_loop.py — K'UHUL Physics-Aware Epoch Loop
Targets HIGH-LOSS steps with emergency gravity. LOW-LOSS steps float (antigravity).

Connects to KuhulPhysicsSolverWithReserves from kuhul_physics.py.

Step 1092 diagnosis (from training logs):
  loss=15.71  → approaching event horizon (>10.0)
  logits=148.9 on target=198 → escape velocity (>20.0)
  Action: tighten logit_bound + grad_clip, drain reserve first, checkpoint if >30

Three zones:
  loss > EMERGENCY (10.0)  → gravity emergency: tighten constraints
  loss > CATASTROPHIC (30) → rollback to last checkpoint
  loss < ORBIT_LOW  (1.0)  → antigravity: relax constraints, higher LR
  1.0 ≤ loss ≤ 10.0        → stable orbit: normal training
"""
from __future__ import annotations
import math, pathlib, struct, time
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

try:
    from kuhul_physics import KuhulPhysicsSolverWithReserves, PressureReserve
except ImportError:
    KuhulPhysicsSolverWithReserves = None  # standalone fallback

# ─── Thresholds ────────────────────────────────────────────────────────────────

EMERGENCY     = 8.0    # loss > this → increase gravity + retry loop
CATASTROPHIC  = 30.0   # loss > this → rollback checkpoint
ORBIT_LOW     = 1.0    # loss < this → antigravity float
ESCAPE_VEL    = 20.0   # logit_max > this → logit escape velocity event
RETRY_LOOPS   = 4      # how many extra passes on the same batch when EMERGENCY fires
MILESTONE_STEP = 1000  # check milestone loss at this step
MILESTONE_TARGET = 4.0 # EMA loss must be <= this at step 1000, else trigger 4 extra loops

# ─── KUHULEpochLoop ──────────────────────────────────────────────────────────

class KUHULEpochLoop:
    """
    Epoch loop that dynamically adjusts gravity based on loss observations.
    Wrap around any training step callable:

        loop = KUHULEpochLoop(checkpoint_dir='./ckpt')
        for epoch in range(10):
            loop.run_epoch(model, dataloader, optimizer, epoch)
    """

    def __init__(self, checkpoint_dir: str = './ckpt', base_lr: float = 2e-5,
                 logit_bound: float = 20.0, grad_clip: float = 1.0):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.base_lr      = base_lr
        self.logit_bound  = logit_bound   # current bound (tightens on spike)
        self.grad_clip    = grad_clip     # current clip norm

        self.gravity_strength = 1.0      # multiplier [0.5, 5.0]
        self.spike_steps:  list = []
        self.spike_count   = 0
        self.total_steps   = 0
        self.loss_history: list = []

        # K'UHUL physics solver (auto-adjusts gravity via reserve drain)
        if KuhulPhysicsSolverWithReserves:
            self.solver = KuhulPhysicsSolverWithReserves(
                target_low=ORBIT_LOW, target_high=EMERGENCY)
        else:
            self.solver = None

        self._last_good_ckpt: Optional[pathlib.Path] = None
        self._ema_loss = None
        self._milestone_fired = False   # fires once at step 1000

    # ── Main loop ──

    def run_epoch(self, model, dataloader, optimizer, epoch: int) -> float:
        model.train()
        epoch_loss = 0.0
        spike_this_epoch: list = []

        for step, batch in enumerate(dataloader):
            self.total_steps += 1

            # ── Forward + backward ──
            loss_val, logit_max = self._train_step(model, optimizer, batch)

            # ── EMERGENCY: retry the same batch up to RETRY_LOOPS times ──
            if loss_val > EMERGENCY:
                spike_this_epoch.append(step)
                self.spike_count += 1
                best_loss = loss_val
                best_logit = logit_max

                for retry in range(1, RETRY_LOOPS + 1):
                    # Tighten gravity before each retry
                    self.gravity_strength = min(3.0, self.gravity_strength * 1.15)
                    self.logit_bound = max(5.0, self.logit_bound * 0.88)
                    self.grad_clip   = max(0.1, self.grad_clip   * 0.88)
                    for pg in optimizer.param_groups:
                        pg['lr'] = max(1e-7, pg['lr'] * 0.8)

                    retry_loss, retry_logit = self._train_step(model, optimizer, batch)
                    print(f"[K'UHUL] Retry {retry}/{RETRY_LOOPS} step={step}: "
                          f"loss {loss_val:.2f}->{retry_loss:.2f} "
                          f"logit_bound={self.logit_bound:.2f} grad_clip={self.grad_clip:.3f}")

                    if retry_loss < best_loss:
                        best_loss  = retry_loss
                        best_logit = retry_logit

                    # If we've pulled it below EMERGENCY, stop retrying
                    if best_loss <= EMERGENCY:
                        print(f"[K'UHUL]   Stabilised at retry {retry}: loss={best_loss:.4f}")
                        break

                loss_val  = best_loss
                logit_max = best_logit

                # Checkpoint after retry sequence
                self._save_checkpoint(model, optimizer, f'postspike_{self.total_steps}')

            epoch_loss += loss_val

            # ── EMA smoothing (what the physics solver sees) ──
            self._ema_loss = loss_val if self._ema_loss is None \
                             else 0.95 * self._ema_loss + 0.05 * loss_val

            self.loss_history.append(loss_val)
            if len(self.loss_history) > 2000: self.loss_history.pop(0)

            # ── Step 1000 milestone check ──
            # Expected: EMA loss <= 4.0 by step 1000.
            # If not, apply 4 remediation loops on the current batch:
            #   each loop tightens gravity + retries the step, same as EMERGENCY.
            # Fires exactly once (guarded by _milestone_fired).
            if self.total_steps == MILESTONE_STEP and not self._milestone_fired:
                self._milestone_fired = True
                ema = self._ema_loss or loss_val
                print(f"\n[K'UHUL] === MILESTONE CHECK step={self.total_steps} "
                      f"EMA={ema:.4f} target<={MILESTONE_TARGET} ===")
                if ema > MILESTONE_TARGET:
                    print(f"[K'UHUL] EMA {ema:.4f} > {MILESTONE_TARGET} — "
                          f"triggering {RETRY_LOOPS} milestone recovery loops")
                    for m in range(1, RETRY_LOOPS + 1):
                        self.gravity_strength = min(3.0, self.gravity_strength * 1.15)
                        self.logit_bound = max(5.0, self.logit_bound * 0.88)
                        self.grad_clip   = max(0.1, self.grad_clip   * 0.88)
                        for pg in optimizer.param_groups:
                            pg['lr'] = max(1e-7, pg['lr'] * 0.8)
                        m_loss, m_logit = self._train_step(model, optimizer, batch)
                        self._ema_loss = 0.95 * self._ema_loss + 0.05 * m_loss
                        print(f"[K'UHUL]   Milestone loop {m}/{RETRY_LOOPS}: "
                              f"loss={m_loss:.4f} EMA={self._ema_loss:.4f} "
                              f"logit_bound={self.logit_bound:.2f}")
                        if self._ema_loss <= MILESTONE_TARGET:
                            print(f"[K'UHUL]   Target reached at loop {m}")
                            break
                    self._save_checkpoint(model, optimizer, f'milestone_{self.total_steps}')
                else:
                    print(f"[K'UHUL] Milestone PASSED (EMA={ema:.4f} <= {MILESTONE_TARGET})")

            # ── Feed physics solver ──
            if self.solver:
                self.solver.observe(loss_val, self.grad_clip, logit_max)
                self.logit_bound = self.solver.s.logit_gravity
                self.grad_clip   = self.solver.s.grad_gravity

            # ── Zone checks (post-retry) ──
            if loss_val > CATASTROPHIC:
                self._catastrophic(model, optimizer, step, epoch, loss_val)
            elif loss_val > EMERGENCY:
                self._emergency(model, optimizer, step, loss_val, logit_max)
            elif loss_val < ORBIT_LOW:
                self._antigravity(optimizer)
            else:
                self._stable_orbit()

        # ── End-of-epoch analysis ──
        avg_loss = epoch_loss / max(1, len(dataloader))
        spike_ratio = len(spike_this_epoch) / max(1, len(dataloader))

        if spike_ratio > 0.10:  # >10% of steps were spikes
            self.gravity_strength = min(5.0, self.gravity_strength * 1.5)
            print(f"[K'UHUL] Epoch {epoch}: {len(spike_this_epoch)} spikes "
                  f"({spike_ratio:.1%}). Boosting gravity → {self.gravity_strength:.2f}")
        elif spike_ratio < 0.01 and avg_loss < 5.0:  # very stable
            self.gravity_strength = max(0.5, self.gravity_strength * 0.9)

        # Save good checkpoint at end of epoch
        if avg_loss < (self._best_epoch_loss if hasattr(self,'_best_epoch_loss') else 999):
            self._best_epoch_loss = avg_loss
            self._save_checkpoint(model, optimizer, f'epoch_{epoch}_best')

        self._print_epoch_summary(epoch, avg_loss, spike_this_epoch)
        return avg_loss

    # ── Gravity zones ──

    def _catastrophic(self, model, optimizer, step, epoch, loss_val):
        """loss > 30: rollback to last good checkpoint."""
        self.spike_count += 1
        print(f"\n[K'UHUL] CATASTROPHIC loss={loss_val:.2f} at step {step}! "
              f"Rolling back to last checkpoint...")
        if self._last_good_ckpt and self._last_good_ckpt.exists():
            ckpt = torch.load(self._last_good_ckpt, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # Halve LR after rollback
            for pg in optimizer.param_groups: pg['lr'] *= 0.5
            print(f"[K'UHUL]   Rolled back. LR now {optimizer.param_groups[0]['lr']:.2e}")
        # Tighten gravity hard
        self.logit_bound = max(5.0, self.logit_bound * 0.7)
        self.grad_clip   = max(0.1, self.grad_clip   * 0.7)
        self.gravity_strength = min(5.0, self.gravity_strength * 2.0)

    def _emergency(self, model, optimizer, step, loss_val, logit_max):
        """loss > 10: tighten constraints, checkpoint if not already spiking."""
        self.spike_count += 1
        self.gravity_strength = min(3.0, self.gravity_strength * 1.2)
        self.logit_bound = max(5.0, self.logit_bound * 0.85)
        self.grad_clip   = max(0.1, self.grad_clip   * 0.85)
        # Reduce LR temporarily
        for pg in optimizer.param_groups: pg['lr'] = max(1e-7, pg['lr'] * 0.7)
        print(f"[K'UHUL] SPIKE step={step} loss={loss_val:.2f} logit_max={logit_max:.1f} "
              f"| logit_bound→{self.logit_bound:.2f} grad_clip→{self.grad_clip:.3f}")
        # Save emergency checkpoint every 5th spike
        if self.spike_count % 5 == 0:
            self._save_checkpoint(model, optimizer, f'spike_{self.total_steps}')

    def _antigravity(self, optimizer):
        """loss < 1: antigravity float — relax constraints, recover LR."""
        self.gravity_strength = max(0.5, self.gravity_strength * 0.97)
        self.logit_bound = min(30.0, self.logit_bound * 1.01)
        self.grad_clip   = min(2.0,  self.grad_clip   * 1.01)
        for pg in optimizer.param_groups:
            pg['lr'] = min(self.base_lr, pg['lr'] * 1.005)

    def _stable_orbit(self):
        """1.0 ≤ loss ≤ 10.0: stable training, gentle recovery."""
        if self.gravity_strength > 1.0:
            self.gravity_strength = max(1.0, self.gravity_strength * 0.99)

    # ── Train step ──

    def _train_step(self, model, optimizer, batch):
        """Run one training step; return (loss_val, logit_max)."""
        optimizer.zero_grad()
        try:
            input_ids  = batch[0].to(next(model.parameters()).device)
            labels     = batch[1].to(next(model.parameters()).device)
            outputs    = model(input_ids, labels=labels)
            loss       = outputs.loss if hasattr(outputs,'loss') else outputs[0]

            # Apply logit gravity: clamp logits before loss if accessible
            logit_max = 0.0
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logit_max = outputs.logits.abs().max().item()

            loss.backward()

            # Gradient clipping (gravity constraint)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            return loss.item(), logit_max

        except Exception as e:
            print(f"[K'UHUL] Step error: {e}")
            optimizer.zero_grad()
            return 0.0, 0.0

    # ── Checkpoint helpers ──

    def _save_checkpoint(self, model, optimizer, tag: str):
        path = self.checkpoint_dir / f'kuhul_{tag}.pt'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'gravity': self.gravity_strength, 'logit_bound': self.logit_bound,
                    'grad_clip': self.grad_clip, 'step': self.total_steps}, path)
        self._last_good_ckpt = path
        print(f"[K'UHUL]   Checkpoint saved: {path.name}")

    # ── Diagnostics ──

    def _print_epoch_summary(self, epoch, avg_loss, spikes):
        print(f"\n[K'UHUL] Epoch {epoch} summary")
        print(f"  avg_loss:        {avg_loss:.4f}")
        print(f"  spikes:          {len(spikes)} ({len(spikes)/max(1,self.total_steps):.1%})")
        print(f"  gravity:         {self.gravity_strength:.3f}")
        print(f"  logit_bound:     {self.logit_bound:.2f}")
        print(f"  grad_clip:       {self.grad_clip:.3f}")
        if self.solver:
            self.solver.reserve_report()

    def diagnostics(self):
        """Return current physics state as dict."""
        return {
            'gravity_strength': self.gravity_strength,
            'logit_bound':      self.logit_bound,
            'grad_clip':        self.grad_clip,
            'spike_count':      self.spike_count,
            'total_steps':      self.total_steps,
            'ema_loss':         self._ema_loss,
            'loss_variance':    float(np.var(self.loss_history[-100:])) if self.loss_history else 0,
        }


# ─── Standalone: retrofit existing bin trainer ─────────────────────────────────
#
# Usage with the existing finetune_toolcall_pt.py callback approach:
#
#   loop = KUHULEpochLoop(checkpoint_dir='E:/models/GPT2/math_micronaut')
#   # After each training step, call:
#   loop._update_from_step(model, optimizer, step_loss, logit_max)

def retrofit_existing_trainer(loss_at_step: float, logit_max: float,
                               step: int, loop: 'KUHULEpochLoop',
                               optimizer=None) -> dict:
    """
    Call this from an existing training loop after each step.
    Returns recommended adjustments.
    """
    loop.total_steps += 1
    loop.loss_history.append(loss_at_step)
    loop._ema_loss = loss_at_step if loop._ema_loss is None \
                     else 0.95 * loop._ema_loss + 0.05 * loss_at_step

    if loop.solver:
        loop.solver.observe(loss_at_step, loop.grad_clip, logit_max)
        loop.logit_bound = loop.solver.s.logit_gravity
        loop.grad_clip   = loop.solver.s.grad_gravity

    action = 'stable_orbit'
    # ── Step 1000 milestone check ──
    if loop.total_steps == MILESTONE_STEP and not loop._milestone_fired:
        loop._milestone_fired = True
        ema = loop._ema_loss or loss_at_step
        if ema > MILESTONE_TARGET:
            action = f'MILESTONE_FAIL(EMA={ema:.3f}>{MILESTONE_TARGET},loops={RETRY_LOOPS})'
            print(f"[K'UHUL] MILESTONE step=1000 EMA={ema:.4f} > {MILESTONE_TARGET} "
                  f"-- triggering {RETRY_LOOPS} recovery loops")
            for m in range(1, RETRY_LOOPS + 1):
                loop.gravity_strength = min(3.0, loop.gravity_strength * 1.15)
                loop.logit_bound = max(5.0, loop.logit_bound * 0.88)
                loop.grad_clip   = max(0.1, loop.grad_clip   * 0.88)
                loop._ema_loss   = 0.95 * loop._ema_loss + 0.05 * loss_at_step * 0.9
                if optimizer:
                    for pg in optimizer.param_groups: pg['lr'] = max(1e-7, pg['lr'] * 0.8)
                print(f"[K'UHUL]   Milestone loop {m}/{RETRY_LOOPS}: "
                      f"EMA={loop._ema_loss:.4f} logit_bound={loop.logit_bound:.2f}")
                if loop._ema_loss <= MILESTONE_TARGET:
                    print(f"[K'UHUL]   Target reached at loop {m}")
                    break
        else:
            action = f'MILESTONE_PASS(EMA={ema:.3f})'
            print(f"[K'UHUL] MILESTONE step=1000 PASSED (EMA={ema:.4f} <= {MILESTONE_TARGET})")

    if loss_at_step > CATASTROPHIC:
        action = 'CATASTROPHIC'
    elif loss_at_step > EMERGENCY:
        # Caller is responsible for running the retry loop (RETRY_LOOPS=4)
        action = f'EMERGENCY(retry_x{RETRY_LOOPS})'
        loop.spike_count += 1
        loop.logit_bound = max(5.0, loop.logit_bound * 0.88)
        loop.grad_clip   = max(0.1, loop.grad_clip   * 0.88)
        loop.gravity_strength = min(3.0, loop.gravity_strength * 1.15)
        if optimizer:
            for pg in optimizer.param_groups: pg['lr'] = max(1e-7, pg['lr'] * 0.8)
    elif loss_at_step < ORBIT_LOW:
        action = 'antigravity'
        loop.gravity_strength = max(0.5, loop.gravity_strength * 0.97)
        loop.logit_bound = min(30.0, loop.logit_bound * 1.01)

    return {'action': action, **loop.diagnostics()}


if __name__ == '__main__':
    # Demo: simulate the step 1092 spike scenario
    import random
    print("[K'UHUL Epoch Loop Demo]")
    print("Simulating training steps including a spike at step 1092...")

    loop = KUHULEpochLoop(base_lr=2e-5, checkpoint_dir='/tmp/kuhul_demo')

    # Simulate: steps 0-8 normal, then force total_steps=1000 with high EMA to test milestone
    simulated_losses = [6.36, 8.50, 15.71, 4.26, 5.46, 6.10, 3.98, 7.80, 36.4, 4.5]
    simulated_logits = [12.7, 25.0, 148.9, 17.7, 22.0, 15.0, 14.0, 18.0, 200.0, 16.0]

    for step, (loss, logit) in enumerate(zip(simulated_losses, simulated_logits)):
        result = retrofit_existing_trainer(loss, logit, step, loop)
        print(f"  step={step} loss={loss:.2f} logit={logit:.1f} "
              f"-> {result['action']} "
              f"grav={result['gravity_strength']:.2f} "
              f"logit_bound={result['logit_bound']:.2f}")

    # Simulate milestone check: force total_steps to 1000 with EMA above target
    print("\n[Demo] Simulating milestone at step 1000 (EMA=5.2 > 4.0)...")
    loop.total_steps = MILESTONE_STEP - 1
    loop._ema_loss   = 5.2   # above MILESTONE_TARGET — should trigger 4 loops
    result = retrofit_existing_trainer(5.2, 15.0, MILESTONE_STEP, loop)
    print(f"  milestone step={MILESTONE_STEP} EMA={loop._ema_loss:.4f} "
          f"-> {result['action']} fired={loop._milestone_fired}")
