"""
resume_math_cluster.py — Resume math µMODEL from s1500 using XVM CPU cluster method.

XVM cluster method (from v0.1.0-xvm-cpu-thread-cluster/train_loop.py):
  Split dataset into shards → stream each shard to the trainer → chain checkpoints.
  xvm_run_cpu_ticks_mt() = hardware_concurrency() threads across 32 fibers.
  Python port: torch.set_num_threads(FIBER_THREADS) + dataset split into FIBER_SHARDS.

Fiber mapping:
  XVMFiber.phase   → shard index (which chunk of data this fiber owns)
  XVMFiber.entropy → EMA loss for that shard (updated after each pass)
  XVMFiber.pressure → K'UHUL gravity_strength (tightens on spike)
  XVMState.shared  → cross-fiber shared state (best checkpoint path)

32-fiber split: dataset ÷ FIBER_SHARDS chunks, processed sequentially.
  Each shard = SHARD_STEPS training steps.
  Checkpoints chain: shard[k] resumes from shard[k-1] checkpoint.
  High-entropy shard (EMA > EMERGENCY) → re-run shard up to RETRY_LOOPS times.

Resume config:
  Start from:  E:/models/GPT2/math_micronaut/gpt2_medium_toolcall_s1500.safetensors
  Data:        C:/Users/canna/.gpu_trainer/bin/tokens_math_v2.bin
  Output:      E:/models/GPT2/math_micronaut/
  Remaining:   ~1500 steps to complete 3000-step target
  Fibers:      32 shards, 4 OS threads (50% of 8-core)
"""
from __future__ import annotations
import subprocess, sys, pathlib, struct, math, time
import numpy as np

# ─── XVM cluster constants (mirrors xvm_core.h) ─────────────────────────────

FIBER_SHARDS  = 32       # XVMState fiber count — data split into 32 chunks
FIBER_THREADS = 4        # os threads per run (50% of 8-core, matching xvm_run_cpu_ticks_mt)
SHARD_STEPS   = 50       # steps per shard call — fine-grained checkpoint chain
RETRY_LOOPS   = 4        # re-run shard if EMA > EMERGENCY (mirrors epoch loop)
EMERGENCY     = 8.0      # EMA loss threshold — triggers retry
TOTAL_REMAIN  = 1500     # remaining steps to close the 3000-step target

# ─── Paths ────────────────────────────────────────────────────────────────────

TRAINER   = pathlib.Path(r"C:\Users\canna\.gpu_trainer\finetune_toolcall_pt.py")
DATA_BIN  = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\tokens_math_v2.bin")
OUT_DIR   = pathlib.Path(r"E:\models\GPT2\math_micronaut")
RESUME_FROM = OUT_DIR / "gpt2_medium_toolcall_s1500.safetensors"
TMP_DIR   = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\cluster_tmp")

# ─── XVMFiber state (Python equivalent) ──────────────────────────────────────

class XVMFiber:
    """Python port of XVMFiber — tracks per-shard training state."""
    def __init__(self, fid: int, shard_path: pathlib.Path):
        self.fid      = fid
        self.pc       = 0          # current step within this fiber's shard
        self.phase    = fid        # shard index
        self.flags    = 1          # 1=active, 0=halted, 2=barrier_wait
        self.entropy  = 0.5        # EMA loss (starts at 0.5 = uncertain)
        self.pressure = 1.0        # gravity_strength
        self.shard    = shard_path # mini tokens.bin for this fiber
        self.ckpt     = None       # latest checkpoint from this fiber
        self.replays  = 0          # how many times this shard was re-run

    def update_entropy(self, loss: float):
        self.entropy = 0.9 * self.entropy + 0.1 * loss

    def tighten_pressure(self):
        self.pressure  = min(3.0, self.pressure * 1.15)

    def summary(self):
        return (f"F{self.fid:02d} phase={self.phase} ent={self.entropy:.3f} "
                f"pres={self.pressure:.2f} replays={self.replays}")

# ─── Shared state (XVMState.shared equivalent) ────────────────────────────────

class XVMShared:
    def __init__(self):
        self.best_ckpt   : pathlib.Path | None = RESUME_FROM
        self.total_steps : int   = 1500    # already completed
        self.global_ema  : float = 3.5     # approximate EMA from last training run
        self.spike_count : int   = 0

# ─── Dataset shard writer (mirrors split_tokens.py write_mini_tokens) ────────

def read_tokens_bin(path: pathlib.Path):
    with open(path, 'rb') as f:
        n_seq, seq_len = struct.unpack('<II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.int32).reshape(n_seq, seq_len).copy()
    return data, n_seq, seq_len

def write_shard_bin(tokens: np.ndarray, path: pathlib.Path):
    n_seq, seq_len = tokens.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', n_seq, seq_len))
        f.write(tokens.astype(np.int32).tobytes())

def split_into_fiber_shards(data: np.ndarray, n_shards: int, tmp_dir: pathlib.Path):
    """Split dataset into n_shards chunks, write each as mini tokens.bin."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n_seq = len(data)
    chunk = max(1, n_seq // n_shards)
    shards = []
    for i in range(n_shards):
        start = i * chunk
        end   = min(n_seq, start + chunk)
        if start >= n_seq: break
        p = tmp_dir / f"shard_{i:05d}.bin"
        write_shard_bin(data[start:end], p)
        shards.append(p)
    print(f"[cluster] Split {n_seq} seqs into {len(shards)} shards "
          f"(~{chunk} seqs/shard) in {tmp_dir}")
    return shards

# ─── Trainer runner (mirrors train_loop.py run_trainer) ──────────────────────

def run_trainer(fiber: XVMFiber, shared: XVMShared, steps: int, lr: float,
                logit_bound: float, grad_clip: float) -> float | None:
    """Run finetune_toolcall_pt.py on one fiber's shard. Returns last loss."""

    # Each fiber gets its own output dir so checkpoint names don't collide
    fiber_out = OUT_DIR / f"fiber_{fiber.fid:02d}"
    fiber_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAINER),
        "--model",      str(shared.best_ckpt),
        "--data",       str(fiber.shard),
        "--out_dir",    str(fiber_out),
        "--steps",      str(steps),
        "--batch",      "4",
        "--lr",         str(lr),
        "--log_every",  "10",
        "--ckpt_every", str(steps),   # save exactly once at end of shard
    ]

    print(f"  [F{fiber.fid:02d}] steps={steps} lr={lr:.1e} "
          f"bound={logit_bound:.1f} clip={grad_clip:.3f}", flush=True)

    env_extra = {"TORCH_NUM_THREADS": str(FIBER_THREADS)}
    import os
    env = {**os.environ, **env_extra}

    last_loss = None
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
        cwd=str(TRAINER.parent))

    for line in proc.stdout:
        line = line.rstrip('\n')
        print(f"    | {line}", flush=True)
        # parse loss from trainer output
        for tok in line.split():
            tok = tok.rstrip(',:')
            try:
                v = float(tok)
                if 0.0 < v < 50.0: last_loss = v
            except ValueError:
                pass
    proc.wait()
    elapsed = time.time() - t0

    # Find the checkpoint saved by this shard run (latest .safetensors in fiber dir)
    candidates = sorted(fiber_out.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    loss_str = f"{last_loss:.4f}" if last_loss else "?"
    if candidates:
        fiber.ckpt       = candidates[-1]
        shared.best_ckpt = candidates[-1]
        print(f"  [F{fiber.fid:02d}] done {elapsed:.0f}s  loss={loss_str}  ckpt={candidates[-1].name}")
    else:
        print(f"  [F{fiber.fid:02d}] done {elapsed:.0f}s  loss={loss_str}  WARNING: no checkpoint found")

    return last_loss

# ─── Main XVM cluster train loop ─────────────────────────────────────────────

def main():
    import torch
    torch.set_num_threads(FIBER_THREADS)
    print(f"[cluster] XVM CPU cluster — {FIBER_SHARDS} fibers × {FIBER_THREADS} threads")
    print(f"[cluster] Resuming from: {RESUME_FROM.name}")
    print(f"[cluster] Target: {TOTAL_REMAIN} more steps to close 3000-step run")
    print()

    if not RESUME_FROM.exists():
        print(f"ERROR: resume checkpoint not found: {RESUME_FROM}")
        sys.exit(1)

    # ── Load + shard dataset ──
    print("[cluster] Loading math dataset...")
    data, n_seq, seq_len = read_tokens_bin(DATA_BIN)
    print(f"[cluster] {n_seq} seqs × {seq_len} tokens")
    shard_paths = split_into_fiber_shards(data, FIBER_SHARDS, TMP_DIR)

    # ── Create fibers (XVMState) ──
    fibers = [XVMFiber(i, shard_paths[i]) for i in range(len(shard_paths))]
    shared = XVMShared()

    # Steps per fiber × total fibers to cover TOTAL_REMAIN
    steps_per_fiber = max(SHARD_STEPS, math.ceil(TOTAL_REMAIN / len(fibers)))
    lr = 1e-5          # continuation LR (lower than initial 2e-5)
    logit_bound = 20.0
    grad_clip   = 1.0

    print(f"[cluster] {len(fibers)} fibers × {steps_per_fiber} steps = "
          f"~{len(fibers) * steps_per_fiber} total steps")
    print()

    # ── XVM cluster execution — sequential fiber dispatch ──────────────────────
    # mirrors xvm_run_cpu_ticks: iterate fibers, each runs until pc >= steps
    # phase-sync: after each fiber, check entropy → tighten pressure if needed

    for fiber in fibers:
        shared.total_steps += steps_per_fiber
        print(f"\n{'='*60}")
        print(f"[cluster] Fiber {fiber.fid:02d}/{len(fibers)-1}  "
              f"global_step~{shared.total_steps}  {fiber.summary()}")
        print(f"{'='*60}")

        # XVM phase gate: if entropy (EMA) high, tighten pressure before dispatch
        if fiber.entropy > EMERGENCY:
            fiber.tighten_pressure()
            grad_clip   = max(0.1, grad_clip   * 0.88)
            logit_bound = max(5.0, logit_bound * 0.88)
            print(f"[cluster] F{fiber.fid:02d} high entropy={fiber.entropy:.3f} "
                  f"— tightening pressure={fiber.pressure:.2f}")

        # Run fiber
        loss = run_trainer(fiber, shared, steps_per_fiber, lr,
                           logit_bound, grad_clip)

        if loss is not None:
            fiber.update_entropy(loss)
            shared.global_ema = 0.95 * shared.global_ema + 0.05 * loss
            fiber.pc += steps_per_fiber

        # ── XVM barrier / retry: CLUSTER gravity ─────────────────────────────
        # When entropy is high, gravity fires on ALL fibers in a mini-cluster pass:
        # the same tightening that phase=2 (barrier wait) does in xvm_run_cpu_ticks_mt.
        # Each retry dispatches the CURRENT fiber AND its two neighbours (±1 phase),
        # simulating the cluster pressure_propagate opcode spreading to adjacent fibers.
        if fiber.entropy > EMERGENCY:
            cluster_radius = 2   # affect current fiber ± 2 neighbours
            affected = [fibers[max(0, fiber.fid - i)] for i in range(cluster_radius+1)
                        if fiber.fid - i >= 0]
            affected += [fibers[min(len(fibers)-1, fiber.fid + i)]
                         for i in range(1, cluster_radius+1)
                         if fiber.fid + i < len(fibers)]
            affected = list(dict.fromkeys(affected))   # deduplicate, preserve order

            for retry in range(RETRY_LOOPS):
                if fiber.entropy <= EMERGENCY:
                    break
                shared.spike_count += 1
                grad_clip   = max(0.1, grad_clip   * 0.88)
                logit_bound = max(5.0, logit_bound * 0.88)
                lr_retry    = max(1e-7, lr * (0.8 ** (retry + 1)))

                print(f"\n[cluster] GRAVITY retry {retry+1}/{RETRY_LOOPS}  "
                      f"cluster={[f.fid for f in affected]}  "
                      f"entropy={fiber.entropy:.3f}  lr={lr_retry:.1e}")

                # Run gravity on all affected cluster fibers
                for gf in affected:
                    gf.tighten_pressure()
                    loss_g = run_trainer(gf, shared, SHARD_STEPS, lr_retry,
                                         logit_bound, grad_clip)
                    if loss_g is not None:
                        gf.update_entropy(loss_g)
                        shared.global_ema = 0.95 * shared.global_ema + 0.05 * loss_g
                    gf.replays += 1

                print(f"[cluster]   after cluster gravity: "
                      f"EMA={shared.global_ema:.4f}  "
                      f"trigger_fiber_ent={fiber.entropy:.3f}")

        # ── Milestone check at global step ~3000 (cluster gravity) ────────────
        # Milestone also uses cluster gravity: run ALL fibers once more with
        # tightened constraints, same as the OP_PRESSURE_PROPAGATE broadcast.
        if shared.total_steps >= 3000 and shared.global_ema > 4.0:
            print(f"\n[cluster] MILESTONE step~3000 EMA={shared.global_ema:.4f} > 4.0")
            print(f"[cluster] Applying CLUSTER GRAVITY across all {len(fibers)} fibers")
            for m in range(1, RETRY_LOOPS + 1):
                grad_clip   = max(0.1, grad_clip   * 0.88)
                logit_bound = max(5.0, logit_bound * 0.88)
                lr_m        = max(1e-7, lr * (0.8 ** m))
                # Broadcast to all fibers — OP_PRESSURE_PROPAGATE
                for gf in fibers:
                    gf.tighten_pressure()
                    loss_m = run_trainer(gf, shared, SHARD_STEPS, lr_m,
                                         logit_bound, grad_clip)
                    if loss_m:
                        gf.update_entropy(loss_m)
                        shared.global_ema = 0.95 * shared.global_ema + 0.05 * loss_m
                print(f"[cluster]   milestone cluster loop {m}/{RETRY_LOOPS}: "
                      f"EMA={shared.global_ema:.4f}")
                if shared.global_ema <= 4.0:
                    print(f"[cluster]   milestone target reached at loop {m}")
                    break

        print(f"\n[cluster] global_step~{shared.total_steps}  "
              f"global_EMA={shared.global_ema:.4f}  spikes={shared.spike_count}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[cluster] XVM CPU cluster training COMPLETE")
    print(f"  Fibers run:   {len(fibers)}")
    print(f"  Total spikes: {shared.spike_count}")
    print(f"  Final EMA:    {shared.global_ema:.4f}")
    print(f"  Final ckpt:   {shared.best_ckpt}")
    print("="*60)

    # Clean up temp shards
    import shutil
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
        print(f"[cluster] Cleaned {TMP_DIR}")

if __name__ == '__main__':
    main()
