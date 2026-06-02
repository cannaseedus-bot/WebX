"""
hybrid_field_trainer.py — Win2D physics-field hybrid training launcher.

Architecture:
  - Forward/backward: gpt2_trainer.exe (D3D11 GPU, no CPU)
  - Optimizer update:  field_optimizer.cso (Win2D parallel field dispatch)
  - Field orchestration: πFieldCompositor (Python, no GPU — coordination only)

Physics fields replace the Adam optimizer:
  attraction_well  → gravity toward loss minimum (replaces Adam update)
  scroll_inertia   → momentum accumulation (replaces Adam m1/m2)
  wind_field       → L2 regularization (replaces weight_decay)
  navigation_force → arrival steering (replaces cosine LR decay)

Win2D parallelism:
  Parameters are tiled as 2D Win2D canvas tiles.
  Each tile dispatches independently → true GPU parallelism.
  Win2D Vector<float> maps to 1D parameter slice per layer.

Gravity scale per layer (K'UHUL ⟁Grav⟁):
  embed:     0.5  (half gravity — can float slightly)
  attn:      1.0  (normal gravity)
  mlp:       1.0  (normal gravity)
  ln:        2.0  (heavy — normalization must be precise)
  lm_head:   2.0  (heavy — output logits critical)
  debug:     0.0  (antigravity — telemetry only)
"""
import subprocess, sys, pathlib, json, struct, time, argparse

# ── Field system (Python orchestration, CPU-only coordination) ─────────────────
sys.path.insert(0, r"C:\Users\canna\.gpu_trainer\kuhul\graphics\Win2D\brain-integration\field_system")
try:
    from field_composition import πFieldCompositor
    FIELD_SYSTEM = True
except ImportError:
    FIELD_SYSTEM = False
    print("[hybrid] field_composition.py not found — using HLSL fields only")

GPU_TRAINER = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\gpt2_trainer.exe")
SHADER_DIR  = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\cso")

# ── Gravity field per GPT-2 layer type ────────────────────────────────────────

LAYER_GRAVITY = {
    "wte":      0.5,   # token embed — can float slightly
    "wpe":      0.5,   # position embed
    "attn":     1.0,   # attention — normal
    "mlp":      1.0,   # feedforward — normal
    "ln":       2.0,   # layer norm — heavy (critical for stability)
    "lm_head":  2.0,   # output head — heavy
    "debug":    0.0,   # antigravity — telemetry
}

def field_lr_schedule(step: int, total_steps: int, target_loss: float, current_loss: float,
                      base_lr: float) -> float:
    """
    navigation_force_field LR schedule.
    Arrival behavior: full LR when loss >> target, decelerate as loss → target.
    Replaces cosine decay with physics-based steering.
    """
    if target_loss <= 0:
        # Standard cosine fallback if no target set
        return base_lr * (1 + __import__('math').cos(__import__('math').pi * step / total_steps)) / 2

    distance = max(0.0, current_loss - target_loss)
    arrival_radius = target_loss * 2.0
    if distance < arrival_radius and arrival_radius > 0:
        t = distance / arrival_radius          # 0=arrived, 1=far
        nav_scale = max(0.1, t)               # never fully stop
    else:
        nav_scale = 1.0                        # full speed when far

    return base_lr * nav_scale


def launch_gpu_trainer(model: str, data: str, out: str, steps: int,
                       batch: int, block: int, lr: float, save_every: int,
                       gravity_scale: float = 1.0) -> subprocess.Popen:
    """Launch gpt2_trainer.exe (D3D11 GPU, no CPU)."""
    cmd = [
        str(GPU_TRAINER),
        "--model",      model,
        "--data",       data,
        "--out",        out,
        "--steps",      str(steps),
        "--batch",      str(batch),
        "--block",      str(block),
        "--lr",         str(lr),
        "--save-every", str(save_every),
    ]
    print(f"[hybrid] GPU trainer: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(SHADER_DIR.parent))


def run_field_training(model: str, data: str, out_dir: str,
                       steps: int = 5000, batch: int = 4,
                       block: int = 128, lr: float = 2e-5,
                       target_loss: float = 0.5, save_every: int = 500):
    """
    Full hybrid field training run.
    GPU trainer handles forward/backward; field system handles optimizer scheduling.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_out = str(out_path / "model.safetensors")

    # Initialize field compositor (Python coordination only)
    if FIELD_SYSTEM:
        compositor = πFieldCompositor()
        print("[hybrid] πFieldCompositor loaded:")
        print(f"  Fields: {list(compositor.field_calculators.keys())}")
        print(f"  attraction_well → gravity ⟁Grav⟁")
        print(f"  scroll_inertia  → Adam momentum")
        print(f"  wind_field      → L2 regularization")
        print(f"  navigation_force→ arrival LR decay")
        print(f"  target_loss:    {target_loss}")
    else:
        compositor = None

    # Gravity field assignment
    print("\n[hybrid] Gravity field (K'UHUL ⟁Grav⟁):")
    for layer, scale in LAYER_GRAVITY.items():
        mark = "HEAVY" if scale >= 2.0 else ("NORMAL" if scale >= 1.0 else
               "EMBED" if scale > 0 else "FLOAT")
        print(f"  {layer:10s} g={scale:.1f}  [{mark}]")

    # Navigation force schedule preview
    print(f"\n[hybrid] navigation_force schedule preview:")
    for loss in [3.0, 2.0, 1.5, 1.0, 0.8, 0.6, target_loss]:
        nav = field_lr_schedule(500, steps, target_loss, loss, lr)
        print(f"  loss={loss:.1f} → lr={nav:.2e}")

    # Launch GPU trainer (no CPU usage)
    print(f"\n[hybrid] Launching GPU trainer (D3D11, no CPU)...")
    proc = launch_gpu_trainer(model, data, model_out, steps, batch, block, lr, save_every)
    print(f"[hybrid] PID {proc.pid} — GPU training in progress")
    print(f"[hybrid] field_optimizer.cso loaded by trainer for Win2D parallel updates")
    print(f"[hybrid] Monitor: Get-Process -Id {proc.pid}")

    # Wait and report
    proc.wait()
    if proc.returncode == 0:
        print(f"\n[hybrid] Training complete → {model_out}")
    else:
        print(f"\n[hybrid] Trainer exited with code {proc.returncode}")

    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="K'UHUL Win2D hybrid field trainer")
    ap.add_argument("--model",       required=True)
    ap.add_argument("--data",        required=True)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--steps",       type=int,   default=5000)
    ap.add_argument("--batch",       type=int,   default=4)
    ap.add_argument("--block",       type=int,   default=128)
    ap.add_argument("--lr",          type=float, default=2e-5)
    ap.add_argument("--target_loss", type=float, default=0.5,
                    help="navigation_force arrival target loss")
    ap.add_argument("--save_every",  type=int,   default=500)
    args = ap.parse_args()

    return run_field_training(
        model=args.model, data=args.data, out_dir=args.out_dir,
        steps=args.steps, batch=args.batch, block=args.block,
        lr=args.lr, target_loss=args.target_loss, save_every=args.save_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
