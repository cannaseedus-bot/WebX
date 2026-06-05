"""
hybrid_train.py — K'UHUL Hybrid Training Orchestrator
Reads hybrid_trainer.kxml and dispatches each phase in order.

Phase map:
  Pop   — resolve base model + data paths
  Sek   — GPU pretraining  (D3D11 gpt2_trainer.exe, large corpus, high throughput)
  Ch'en — CPU finetuning   (PyTorch AdamW + cosine, toolcall corpus, precise)
  Xul   — export to GGUF + update model registry

The GPU/CPU split is the key insight:
  GPU learns the token manifold (WHAT patterns exist)
  CPU learns the toolcall surface (WHICH patterns map to which tool)

Usage:
  python hybrid_train.py                                    # defaults
  python hybrid_train.py --model small --gpu_steps 500 --cpu_steps 500
  python hybrid_train.py --model medium --gpu_steps 0      # CPU-only
  python hybrid_train.py --skip_gpu                        # skip GPU, CPU only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import struct
import subprocess
import sys
import time

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT     = pathlib.Path(__file__).parent
BIN_DIR  = ROOT / "bin"
TRAINER  = BIN_DIR / "gpt2_trainer.exe"

MODELS = {
    "small": {
        "base":     pathlib.Path(r"E:\models\GPT2\small-instruct\gpt2_small_ft_toolcall_fixed.safetensors"),
        "out_dir":  pathlib.Path(r"E:\models\GPT2\mini-GPT"),
        "gguf_name":"kuhul-tool-lite-q8.gguf",
        "registry_id": "kuhul-tool-lite",
        "params":   "124M",
    },
    "medium": {
        "base":     pathlib.Path(r"E:\models\GPT2\med-GPT\model.safetensors"),
        "out_dir":  pathlib.Path(r"E:\models\GPT2\med-GPT"),
        "gguf_name":"kuhul-tool-medium-q8.gguf",
        "registry_id": "kuhul-tool-medium",
        "params":   "345M",
    },
}

PRETRAIN_DATA = BIN_DIR / "tokens_toolcall.bin"   # toolcall corpus for both phases
WEBX_MODELS   = pathlib.Path(r"C:\Users\canna\.kuhul-v1\releases\v3.5.0-WebX\models")


# ─── Phase Pop — resolve config ───────────────────────────────────────────────

def phase_pop(model_key: str, gpu_steps: int) -> dict:
    cfg = MODELS[model_key]
    base = cfg["base"]
    if not base.exists():
        # Fall back to raw base model
        alt = base.parent / "model.safetensors"
        if alt.exists():
            base = alt
        else:
            raise FileNotFoundError(f"Base model not found: {base}")

    data = PRETRAIN_DATA
    if not data.exists():
        raise FileNotFoundError(f"Training data not found: {data}")

    print(f"[Pop] base     : {base.name}  ({base.stat().st_size/1e6:.0f} MB)")
    print(f"[Pop] data     : {data.name}  ({data.stat().st_size/1e6:.0f} MB)")
    print(f"[Pop] gpu steps: {gpu_steps}  cpu steps: determined per args")
    return {"base": base, "data": data, "cfg": cfg}


# ─── Phase Sek — GPU pretraining (D3D11) ─────────────────────────────────────

def phase_sek(ctx: dict, steps: int, batch: int = 4, lr: float = 3e-5,
              chunk_steps: int = 250) -> pathlib.Path:
    if steps <= 0:
        print("[Sek] GPU pretraining skipped.")
        return ctx["base"]

    base    = ctx["base"]
    out_dir = ctx["cfg"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ctx["data"], "rb") as f:
        n_seqs, block = struct.unpack("<II", f.read(8))
    print(f"[Sek] D3D11 pretrain: {steps} steps  batch={batch}  lr={lr:.2e}")
    print(f"      data: {n_seqs:,} seqs x {block} tokens")

    model_in = base
    step_done = 0
    n_chunks  = (steps + chunk_steps - 1) // chunk_steps

    for chunk in range(n_chunks):
        steps_this = min(chunk_steps, steps - step_done)
        step_done += steps_this
        ckpt = out_dir / f"hybrid_gpu_s{step_done:04d}.safetensors"

        cmd = [
            str(TRAINER),
            "--out",   str(ckpt),
            "--steps", str(steps_this),
            "--batch", str(batch),
            "--lr",    f"{lr:.2e}",
            "--block", str(block),
            "--data",  str(ctx["data"]),
            "--model", str(model_in),
        ]
        print(f"\n  [Sek chunk {chunk+1}/{n_chunks}] steps {step_done-steps_this+1}-{step_done}")
        t0 = time.monotonic()
        result = subprocess.run(cmd, cwd=str(BIN_DIR))
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            print(f"  [Sek] trainer exited {result.returncode} — using last checkpoint")
            return model_in

        print(f"  [Sek] done in {elapsed:.0f}s")
        model_in = ckpt

    print(f"[Sek] GPU pretrain complete -> {model_in.name}")
    return model_in


# ─── Phase Ch'en — CPU finetuning (PyTorch) ──────────────────────────────────

def phase_chen(ctx: dict, gpu_ckpt: pathlib.Path, steps: int,
               batch: int = 2, lr: float = 5e-5) -> pathlib.Path:
    # Load the PyTorch trainer from finetune_toolcall_pt.py
    ft_path = ROOT / "finetune_toolcall_pt.py"
    spec    = importlib.util.spec_from_file_location("_ft_", ft_path)
    ft      = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)


    out_dir = ctx["cfg"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Ch'en] CPU finetune: {steps} steps  batch={batch}  lr={lr:.2e}")
    print(f"        from: {gpu_ckpt.name}")

    data, _ = ft.load_bin(ctx["data"])
    model   = ft.load_gpt2(gpu_ckpt)

    ft.train(
        model      = model,
        data       = data,
        steps      = steps,
        batch      = batch,
        lr         = lr,
        log_every  = max(1, steps // 40),
        out_dir    = out_dir,
        ckpt_every = max(1, steps // 4),
    )

    final = out_dir / "hybrid_final.safetensors"
    last  = out_dir / f"gpt2_medium_toolcall_s{steps:04d}.safetensors"
    # finetune_toolcall_pt.py saves as gpt2_medium_toolcall_s*.safetensors
    candidates = sorted(out_dir.glob("gpt2_medium_toolcall_s*.safetensors"))
    if not candidates:
        candidates = sorted(out_dir.glob("hybrid_*.safetensors"))
    if candidates:
        shutil.copy2(str(candidates[-1]), str(final))
    print(f"[Ch'en] CPU finetune complete -> {final.name}")
    return final


# ─── Phase Xul — export to GGUF + registry update ────────────────────────────

def phase_xul(ctx: dict, weights: pathlib.Path) -> pathlib.Path:
    out_dir  = ctx["cfg"]["out_dir"]
    cfg      = ctx["cfg"]
    gguf_out = out_dir / cfg["gguf_name"]

    print(f"\n[Xul] converting to GGUF: {gguf_out.name}")

    # Remap bare keys to transformer.* prefix
    from safetensors.torch import load_file, save_file
    tmp = weights.with_suffix(".remap.safetensors")
    raw = load_file(str(weights))
    mapped = {}
    for k, v in raw.items():
        nk = k if k.startswith("transformer.") or k == "lm_head.weight" else "transformer." + k
        mapped[nk] = v.clone()
    if "lm_head.weight" not in mapped:
        mapped["lm_head.weight"] = mapped["transformer.wte.weight"].clone()
    save_file(mapped, str(tmp))

    # Run to_gguf.py
    gguf_script = out_dir / "to_gguf.py"
    if not gguf_script.exists():
        gguf_script = pathlib.Path(r"E:\models\GPT2\med-GPT\to_gguf.py")
    result = subprocess.run(
        [sys.executable, str(gguf_script), str(tmp),
         "--out", str(gguf_out), "--quant", "q8_0"],
        cwd=str(out_dir)
    )
    tmp.unlink(missing_ok=True)

    if result.returncode != 0:
        print("[Xul] GGUF conversion failed — keeping .safetensors only")
        return weights

    print(f"[Xul] GGUF written: {gguf_out.name}  ({gguf_out.stat().st_size/1e6:.0f} MB)")

    # Copy to WebX-3D models/ directory
    webx_dest = WEBX_MODELS / cfg["gguf_name"]
    try:
        shutil.copy2(str(gguf_out), str(webx_dest))
        print(f"[Xul] copied to WebX: {webx_dest.name}")
    except Exception as e:
        print(f"[Xul] WebX copy skipped: {e}")

    # Update model registry
    _update_registry(cfg, gguf_out)

    return gguf_out


def _update_registry(cfg: dict, gguf_path: pathlib.Path) -> None:
    reg_path = WEBX_MODELS / "model-registry.json"
    if not reg_path.exists():
        return
    try:
        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
        for m in reg["models"]:
            if m["id"] == cfg["registry_id"]:
                m["status"]   = "production"
                m["size_mb"]  = round(gguf_path.stat().st_size / 1e6, 1)
                m["training"]["trainer"] = "Hybrid (D3D11 GPU pretrain + PyTorch CPU finetune)"
                break
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)
        print(f"[Xul] model-registry.json updated: {cfg['registry_id']} -> production")
    except Exception as e:
        print(f"[Xul] registry update skipped: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      choices=["small", "medium"], default="small")
    ap.add_argument("--gpu_steps",  type=int,   default=500,
                    help="D3D11 pretraining steps (0 = skip GPU phase)")
    ap.add_argument("--cpu_steps",  type=int,   default=500,
                    help="PyTorch CPU finetuning steps")
    ap.add_argument("--gpu_lr",     type=float, default=3e-5)
    ap.add_argument("--cpu_lr",     type=float, default=5e-5)
    ap.add_argument("--gpu_batch",  type=int,   default=4)
    ap.add_argument("--cpu_batch",  type=int,   default=2)
    ap.add_argument("--skip_gpu",   action="store_true",
                    help="Skip GPU phase entirely (same as --gpu_steps 0)")
    ap.add_argument("--skip_export",action="store_true",
                    help="Skip GGUF export (useful for quick iteration)")
    ap.add_argument("--graph",      type=str,   default="hybrid_trainer.kxml",
                    help="KXML graph file (informational — read for metadata)")
    args = ap.parse_args()

    if args.skip_gpu:
        args.gpu_steps = 0

    print("=" * 62)
    print("K'UHUL HYBRID TRAINER")
    print(f"  graph  : {args.graph}")
    print(f"  model  : {args.model}")
    print("  Pop    : resolve config")
    print(f"  Sek    : GPU pretrain  {args.gpu_steps} steps  lr={args.gpu_lr:.2e}  batch={args.gpu_batch}")
    print(f"  Ch'en  : CPU finetune  {args.cpu_steps} steps  lr={args.cpu_lr:.2e}  batch={args.cpu_batch}")
    print(f"  Xul    : {'GGUF export + registry update' if not args.skip_export else 'skipped'}")
    print("=" * 62)

    t_start = time.monotonic()

    # ── Phase Pop ──────────────────────────────────────────────────────────────
    print("\n[Phase Pop] Resolving configuration...")
    ctx = phase_pop(args.model, args.gpu_steps)

    # ── Phase Sek ──────────────────────────────────────────────────────────────
    print("\n[Phase Sek] GPU Pretraining (D3D11)...")
    gpu_ckpt = phase_sek(ctx, args.gpu_steps, args.gpu_batch, args.gpu_lr)

    # ── Phase Ch'en ────────────────────────────────────────────────────────────
    print("\n[Phase Ch'en] CPU Finetuning (PyTorch)...")
    final_weights = phase_chen(ctx, gpu_ckpt, args.cpu_steps, args.cpu_batch, args.cpu_lr)

    # ── Phase Xul ──────────────────────────────────────────────────────────────
    if not args.skip_export:
        print("\n[Phase Xul] Exporting to GGUF...")
        gguf = phase_xul(ctx, final_weights)
    else:
        print("\n[Phase Xul] Export skipped.")
        gguf = final_weights

    elapsed = time.monotonic() - t_start
    print(f"\n{'=' * 62}")
    print(f"HYBRID TRAINING COMPLETE  ({elapsed/60:.1f} min)")
    print("  Pop   : config resolved")
    print(f"  Sek   : GPU pretrain {'done' if args.gpu_steps > 0 else 'skipped'}")
    print("  Ch'en : CPU finetune done")
    print(f"  Xul   : {gguf.name}")
    print("\nNext: load with llama.cpp or LM Studio")
    print(f"  {gguf}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
