"""
hybrid_math_micronaut.py — Hybrid iGPU→CPU training pipeline for math micronaut

Pipeline (4 K'UHUL phases):
  [Pop]    — resolve paths, tokenize KXML JSONL → .bin if needed
  [Sek]    — iGPU pretrain: gpt2_trainer.exe (D3D11 GPU Adam, high-throughput)
  [Ch'en]  — CPU finetune: finetune_toolcall_pt.py with geodesic+ARC attention
  [Xul]    — save final checkpoint, print summary

Key insight (why this works when pure GPU or pure CPU plateaus):
  GPU phase:  fast convergence on token distribution (WHAT token follows what)
              large batches, high LR, raw KXML tokens → learns structure cheaply
  CPU phase:  precision on tool-call surface (WHICH structure maps to which tool)
              small batches, cosine LR decay, geodesic attention, ARC bias
              the geodesic cache biases attention toward high-quality token arcs
              already accumulated from the GPU phase via shared sphere positions

The plateau at 3.5-4.0 on old runs was data (coder outputs, not KXML math).
New domain JSONL (build_domain_jsonl.py --domain math) feeds KXML-tagged
arithmetic graphs, sympy/numpy tool calls, and step-by-step reasoning chains.

Usage:
  python hybrid_math_micronaut.py
  python hybrid_math_micronaut.py --model E:/models/GPT2/math_micronaut/gpt2_medium_toolcall_s1500.safetensors
  python hybrid_math_micronaut.py --gpu-steps 500 --cpu-steps 500 --skip-gpu
  python hybrid_math_micronaut.py --jsonl E:/models/GPT2/domain_train.jsonl --domain math
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
import time
import shutil

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE         = pathlib.Path(__file__).resolve().parent
TRAINER_ROOT = HERE.parent.parent   # v3.5.0-WebX/
BIN_DIR      = TRAINER_ROOT / "native" / "bin"
TRAINER_EXE  = BIN_DIR / "gpt2_trainer.exe"

GEODESIC_CACHE = pathlib.Path(r"E:\models\GPT2\geodesic_cache")
MATH_OUT_DIR   = pathlib.Path(r"E:\models\GPT2\math_micronaut")
DEFAULT_BASE   = MATH_OUT_DIR / "gpt2_medium_toolcall_s1500.safetensors"
DEFAULT_JSONL  = pathlib.Path(r"E:\models\GPT2\domain_train.jsonl")
TMP_DIR        = MATH_OUT_DIR / "tmp_hybrid"

PYTHON = sys.executable


# ── Phase Pop — tokenize JSONL → .bin ────────────────────────────────────────

def phase_pop(jsonl: pathlib.Path, domain: str, block: int) -> pathlib.Path:
    """
    Tokenize KXML domain JSONL → uint32 tokens.bin the GPU trainer can read.
    Format: <n_seqs uint32><block uint32> then n_seqs*block int32 tokens.
    Filters to --domain records only. Pads/truncates to block length.
    Returns path to the .bin file.
    """
    try:
        import tiktoken
    except ImportError:
        print("[Pop] ERROR: tiktoken not installed. Run: pip install tiktoken")
        sys.exit(1)

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token   # 50256

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = TMP_DIR / f"tokens_{domain}_b{block}.bin"

    print(f"[Pop] tokenizing {jsonl.name} (domain={domain!r}, block={block})")
    seqs: list[list[int]] = []
    n_records = 0
    all_tokens: list[int] = []

    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if domain and r.get("domain", "") != domain:
                continue
            text = r.get("prompt", "") + r.get("response", "")
            toks = enc.encode(text, allowed_special={"<|endoftext|>"})
            all_tokens.extend(toks)
            all_tokens.append(EOT)
            n_records += 1

    if not all_tokens:
        print(f"[Pop] ERROR: no records found for domain={domain!r}")
        sys.exit(1)

    # Pack into fixed-length sequences (stride = block, no overlap)
    for i in range(0, len(all_tokens) - block, block):
        chunk = all_tokens[i:i + block + 1]
        seqs.append(chunk[:block + 1])

    if not seqs:
        print(f"[Pop] ERROR: not enough tokens to form even one block-{block} sequence")
        sys.exit(1)

    n_seqs = len(seqs)
    print(f"[Pop] {n_records} records -> {len(all_tokens):,} tokens -> {n_seqs:,} seqs x {block}")

    arr = np.zeros((n_seqs, block), dtype=np.int32)
    for i, seq in enumerate(seqs):
        arr[i] = seq[:block]

    with open(bin_path, "wb") as f:
        f.write(struct.pack("<II", n_seqs, block))
        f.write(arr.tobytes())

    print(f"[Pop] wrote {bin_path.name}  ({bin_path.stat().st_size // 1024} KB)")
    return bin_path


# ── Phase Sek — iGPU pretrain ─────────────────────────────────────────────────

def phase_sek(base_model: pathlib.Path, bin_path: pathlib.Path,
              steps: int, batch: int, block: int, lr: float,
              chunk_steps: int = 250) -> pathlib.Path:
    """
    Run gpt2_trainer.exe in chunks. Chains checkpoint → next chunk.
    Returns path to final GPU checkpoint.
    """
    if steps <= 0:
        print("[Sek] GPU phase skipped.")
        return base_model

    MATH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_in   = base_model
    step_done  = 0
    n_chunks   = (steps + chunk_steps - 1) // chunk_steps

    print(f"[Sek] iGPU pretrain: {steps} steps  batch={batch}  lr={lr:.2e}  chunks={n_chunks}")
    print(f"      base: {base_model.name}")
    print(f"      data: {bin_path.name}")

    for chunk in range(n_chunks):
        steps_this = min(chunk_steps, steps - step_done)
        step_done += steps_this
        ckpt = MATH_OUT_DIR / f"math_hybrid_gpu_s{step_done:04d}.safetensors"

        cmd = [
            str(TRAINER_EXE),
            "--model",      str(model_in),
            "--data",       str(bin_path),
            "--out",        str(ckpt),
            "--steps",      str(steps_this),
            "--batch",      str(batch),
            "--block",      str(block),
            "--lr",         f"{lr:.2e}",
            "--save-every", str(steps_this),
        ]

        print(f"\n  [Sek {chunk+1}/{n_chunks}] steps {step_done-steps_this+1}–{step_done}")
        t0 = time.monotonic()
        result = subprocess.run(cmd, cwd=str(BIN_DIR))
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            print(f"  [Sek] trainer exited {result.returncode} — using {model_in.name}")
            return model_in

        print(f"  [Sek] done in {elapsed:.0f}s  → {ckpt.name}")
        model_in = ckpt

    print(f"[Sek] GPU pretrain complete → {model_in.name}")
    return model_in


# ── Phase Ch'en — CPU finetune with geodesic attention ────────────────────────

def phase_chen(gpu_ckpt: pathlib.Path, bin_path: pathlib.Path,
               steps: int, batch: int, lr: float, log_every: int,
               ckpt_every: int, use_geodesic: bool) -> pathlib.Path:
    """
    Run finetune_toolcall_pt.py with geodesic+ARC attention from prebuilt cache.
    Cosine LR annealing already built into the trainer.
    """
    # Load finetune_toolcall_pt as a module
    ft_path = HERE / "finetune_toolcall_pt.py"
    spec    = importlib.util.spec_from_file_location("_ft_", ft_path)
    ft      = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)

    import torch

    print(f"\n[Ch'en] CPU finetune: {steps} steps  batch={batch}  lr={lr:.2e}")
    print(f"        from: {gpu_ckpt.name}")
    print(f"        data: {bin_path.name}")
    print(f"        geodesic: {use_geodesic}")

    data, block = ft.load_bin(bin_path)
    model       = ft.load_gpt2(gpu_ckpt)

    # ── Geodesic attention setup ──────────────────────────────────────────────
    arc_weights = None
    if use_geodesic and GEODESIC_CACHE.exists():
        try:
            geo_spec = importlib.util.spec_from_file_location(
                "_geo_", HERE / "geodesic_attention_bridge.py")
            geo = importlib.util.module_from_spec(geo_spec)
            geo_spec.loader.exec_module(geo)

            print(f"[Ch'en] loading geodesic cache from {GEODESIC_CACHE}")
            geo_map = geo.GeodesicSphereMap(cache_dir=GEODESIC_CACHE)
            geo_map.load_or_compile(model.wte.weight.detach())

            arc_weights = geo.ARCWeightMatrix(vocab_size=50257)
            # Load any pre-accumulated arc bias from previous runs
            rows_f = GEODESIC_CACHE / "arc_bias_rows.npy"
            cols_f = GEODESIC_CACHE / "arc_bias_cols.npy"
            vals_f = GEODESIC_CACHE / "arc_bias_vals.npy"
            if rows_f.exists() and cols_f.exists() and vals_f.exists():
                rows = np.load(str(rows_f)).astype(np.int32)
                cols = np.load(str(cols_f)).astype(np.int32)
                vals = np.load(str(vals_f)).astype(np.float32)
                for r, c, v in zip(rows, cols, vals):
                    arc_weights._pairs[(int(r), int(c))] = float(v)
                print(f"[Ch'en] loaded {len(rows):,} pre-accumulated ARC pairs")

            # Patch geodesic into every attention block
            for block_module in model.blocks:
                block_module.geo_map     = geo_map
                block_module.arc_weights = arc_weights
                block_module.geo_temperature = 1.0

            print(f"[Ch'en] geodesic+ARC attention active on all {len(model.blocks)} blocks")

        except Exception as e:
            print(f"[Ch'en] geodesic load failed ({e}) — using standard attention")
            arc_weights = None

    # ── Training ──────────────────────────────────────────────────────────────
    ft.train(
        model      = model,
        data       = data,
        steps      = steps,
        batch      = batch,
        lr         = lr,
        log_every  = log_every,
        ckpt_every = ckpt_every,
        out_dir    = MATH_OUT_DIR,
        cosine_total = steps,
        arc_weights  = arc_weights,
    )

    # ── Save updated ARC weights back to cache ────────────────────────────────
    if arc_weights is not None and arc_weights._pairs:
        try:
            arc_weights.save(GEODESIC_CACHE)
            print(f"[Ch'en] ARC weights updated → {GEODESIC_CACHE}")
        except Exception as e:
            print(f"[Ch'en] ARC save failed: {e}")

    # Locate final checkpoint
    candidates = sorted(MATH_OUT_DIR.glob("gpt2_medium_toolcall_s*.safetensors"))
    final = MATH_OUT_DIR / "math_micronaut_hybrid_final.safetensors"
    if candidates:
        shutil.copy2(str(candidates[-1]), str(final))
        print(f"[Ch'en] final → {final.name}")
    return final


# ── Phase Xul — summary ───────────────────────────────────────────────────────

def phase_xul(final: pathlib.Path, elapsed: float, args) -> None:
    print(f"\n{'='*62}")
    print(f"HYBRID MATH MICRONAUT COMPLETE  ({elapsed/60:.1f} min)")
    print(f"  base model : {pathlib.Path(args.model).name}")
    print(f"  GPU steps  : {args.gpu_steps}  lr={args.gpu_lr:.2e}")
    print(f"  CPU steps  : {args.cpu_steps}  lr={args.cpu_lr:.2e}")
    print(f"  geodesic   : {not args.no_geodesic}")
    print(f"  ARC cache  : {GEODESIC_CACHE}")
    print(f"  final ckpt : {final}")
    print(f"{'='*62}")
    print(f"\nNext steps:")
    print(f"  Load in LM Studio or llama.cpp with the KXML math system prompt")
    print(f"  Run more steps: --model {final} --gpu-steps 500 --cpu-steps 500")
    print(f"  Export GGUF:    python to_gguf.py {final} --out math_micronaut_q8.gguf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Hybrid iGPU→CPU math micronaut trainer")
    ap.add_argument("--model",      default=str(DEFAULT_BASE),
                    help="Base safetensors checkpoint to start from")
    ap.add_argument("--jsonl",      default=str(DEFAULT_JSONL),
                    help="Domain JSONL file (from build_domain_jsonl.py)")
    ap.add_argument("--domain",     default="math",
                    help="Domain filter applied to JSONL records (default: math)")
    ap.add_argument("--block",      type=int, default=128,
                    help="Token sequence length")
    ap.add_argument("--gpu-steps",  type=int, default=500, dest="gpu_steps",
                    help="iGPU D3D11 pretraining steps (0 = skip)")
    ap.add_argument("--gpu-lr",     type=float, default=3e-5, dest="gpu_lr")
    ap.add_argument("--gpu-batch",  type=int,   default=4,    dest="gpu_batch")
    ap.add_argument("--cpu-steps",  type=int, default=500, dest="cpu_steps",
                    help="CPU PyTorch finetune steps (0 = skip)")
    ap.add_argument("--cpu-lr",     type=float, default=5e-5, dest="cpu_lr")
    ap.add_argument("--cpu-batch",  type=int,   default=2,    dest="cpu_batch")
    ap.add_argument("--log-every",  type=int,   default=25,   dest="log_every")
    ap.add_argument("--ckpt-every", type=int,   default=100,  dest="ckpt_every")
    ap.add_argument("--skip-gpu",   action="store_true", dest="skip_gpu")
    ap.add_argument("--skip-cpu",   action="store_true", dest="skip_cpu")
    ap.add_argument("--no-geodesic",action="store_true", dest="no_geodesic",
                    help="Disable geodesic+ARC attention in CPU phase")
    ap.add_argument("--chunk-steps",type=int,   default=250,  dest="chunk_steps",
                    help="GPU trainer chunk size (prevents OOM on large runs)")
    args = ap.parse_args()

    if args.skip_gpu:
        args.gpu_steps = 0

    model_path = pathlib.Path(args.model)
    jsonl_path = pathlib.Path(args.jsonl)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)
    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}")
        sys.exit(1)

    print("=" * 62)
    print("K'UHUL HYBRID MATH MICRONAUT TRAINER")
    print(f"  [Pop]   tokenize {jsonl_path.name} domain={args.domain!r}")
    print(f"  [Sek]   iGPU pretrain  {args.gpu_steps} steps  lr={args.gpu_lr:.2e}  batch={args.gpu_batch}")
    print(f"  [Ch'en] CPU finetune   {args.cpu_steps} steps  lr={args.cpu_lr:.2e}  batch={args.cpu_batch}")
    print(f"  [Xul]   save + summary")
    print(f"  geodesic cache: {GEODESIC_CACHE}")
    print("=" * 62)

    t_start = time.monotonic()

    # [Pop] Tokenize
    print("\n[Phase Pop] Tokenizing KXML domain JSONL...")
    bin_path = phase_pop(jsonl_path, args.domain, args.block)

    # [Sek] iGPU pretrain
    print("\n[Phase Sek] iGPU Pretrain (D3D11)...")
    gpu_ckpt = phase_sek(
        model_path, bin_path,
        steps=args.gpu_steps, batch=args.gpu_batch,
        block=args.block, lr=args.gpu_lr,
        chunk_steps=args.chunk_steps,
    )

    # [Ch'en] CPU finetune
    if not args.skip_cpu and args.cpu_steps > 0:
        print("\n[Phase Ch'en] CPU Finetune (PyTorch + geodesic)...")
        final = phase_chen(
            gpu_ckpt, bin_path,
            steps=args.cpu_steps, batch=args.cpu_batch,
            lr=args.cpu_lr, log_every=args.log_every,
            ckpt_every=args.ckpt_every,
            use_geodesic=not args.no_geodesic,
        )
    else:
        print("[Ch'en] CPU phase skipped.")
        final = gpu_ckpt

    # [Xul] Summary
    phase_xul(final, time.monotonic() - t_start, args)


if __name__ == "__main__":
    main()
