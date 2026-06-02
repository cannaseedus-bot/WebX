"""
layered_train.py — Multi-layer training pipeline for GPT-2.

Stage order:
  instruct → code → math → toolcall

Each stage fine-tunes from the previous stage's checkpoint.
JSONL files must already exist (run xshard_to_jsonl.py first).

Usage:
  python layered_train.py                           # all stages, auto-detect base
  python layered_train.py --stages instruct,code    # only these stages
  python layered_train.py --skip_to code            # start from this stage
  python layered_train.py --base E:\\models\\GPT2\\med-GPT\\model.safetensors
  python layered_train.py --steps 500 --lr 1e-5
"""
from __future__ import annotations

import argparse
import pathlib
import json
import struct
import subprocess
import sys
import time

# ─── Paths ─────────────────────────────────────────────────────────────────────

TRAINER_DIR = pathlib.Path(r"C:\Users\canna\.gpu_trainer")
JSONL_DIR   = TRAINER_DIR / "bin" / "xshard_jsonl"
BIN_DIR     = TRAINER_DIR / "bin"
CKPT_DIR    = TRAINER_DIR / "bin" / "layered_ckpts"
TRAINER_PY  = TRAINER_DIR / "finetune_toolcall_pt.py"

# Ordered list of base model candidates (first existing one is used)
BASE_CANDIDATES = [
    pathlib.Path(r"E:\models\GPT2\mini-GPT\model.safetensors"),
    pathlib.Path(r"E:\models\GPT2\med-GPT\model.safetensors"),
    pathlib.Path(r"E:\models\GPT2\small-instruct\checkpoints\ckpt_epoch_01.safetensors"),
]

STAGE_JSONL = {
    "instruct": JSONL_DIR / "prompt_instruct_layer.jsonl",
    "code":     JSONL_DIR / "prompt_code_layer.jsonl",
    "math":     JSONL_DIR / "prompt_math_layer.jsonl",
    "toolcall": pathlib.Path(r"E:\models\GPT2\med-GPT\tokens_toolcall_v5.bin"),  # v5: 20,196 seqs, 76 contrastive + 3100 real harvested pairs
}

# Final safetensors name written by finetune_toolcall_pt.py into out_dir
TRAINER_FINAL_NAME = "gpt2_medium_ft_toolcall.safetensors"

STAGE_ORDER = ["instruct", "code", "math", "toolcall"]

# Per-stage step multipliers  (× --steps argument)
#
# Seq counts at batch=8:
#   instruct:  8,901 seqs → 1,112 steps/epoch  → 3× = 1,500 = 1.35 epochs
#   code:     48,536 seqs → 6,067 steps/epoch  → 2× = 1,000 = 0.16 epochs (exposure pass)
#   math:     42,228 seqs → 5,278 steps/epoch  → 2× = 1,000 = 0.19 epochs (exposure pass)
#   toolcall: 14,658 seqs → 1,832 steps/epoch  → 4× = 2,000 = 1.09 epochs (convergence pass)
#
# Loss targets:  instruct < 2.2   code < 2.3   math < 2.7   toolcall < 0.25
STAGE_STEPS = {
    "instruct": 6,    # 500 × 6 = 3000 steps  ~28% of epoch, loss target <1.8
    "code":     1,    # 500 × 1 =  500 steps   optional exposure pass
    "math":     1,    # 500 × 1 =  500 steps   optional exposure pass
    "toolcall": 8,    # 500 × 8 = 4000 steps  ~2.18 epochs on v5 data, loss target <0.2
}


# ─── Tokenisation ──────────────────────────────────────────────────────────────

def jsonl_to_bin(jsonl_path: pathlib.Path, bin_path: pathlib.Path,
                 block: int = 128, max_records: int = 0) -> int:
    """Tokenise a JSONL file to the GPU-trainer .bin format."""
    try:
        import tiktoken
    except ImportError:
        print("Installing tiktoken...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tiktoken", "-q"], check=True)
        import tiktoken

    enc  = tiktoken.get_encoding("gpt2")
    toks: list[int] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt   = r.get("prompt", "")
            response = r.get("response", "")
            if not prompt and not response:
                continue
            text = f"{prompt}\n{response}<|endoftext|>"
            toks.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))

    n    = len(toks) // block
    flat = toks[:n * block]
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<II", n, block))
        f.write(struct.pack(f"<{len(flat)}I", *flat))
    mb = bin_path.stat().st_size / 1e6
    print(f"  tokenised: {n:,} seqs x {block} ({mb:.1f} MB) -> {bin_path.name}")
    return n


# ─── Training ──────────────────────────────────────────────────────────────────

def run_stage(stage: str, model_path: pathlib.Path, base_out_dir: pathlib.Path,
              steps: int = 500, lr: float = 1e-5, batch: int = 8,
              block: int = 128, log_every: int = 10, ckpt_every: int = 100,
              max_records: int = 0) -> pathlib.Path:
    """Run one fine-tune stage. Returns path to the stage's final checkpoint."""

    # Apply per-stage step multiplier (toolcall needs 4x more steps)
    effective_steps = steps * STAGE_STEPS.get(stage, 1)

    print(f"\n{'='*62}")
    print(f"  STAGE: {stage.upper()}  ({effective_steps} steps = {steps} x {STAGE_STEPS.get(stage,1)}x)")
    print(f"  base:  {model_path.name}")
    print(f"  steps: {steps}  lr: {lr}  batch: {batch}")
    print(f"{'='*62}")

    stage_out = base_out_dir / stage
    stage_out.mkdir(parents=True, exist_ok=True)

    # Prepare .bin token file
    if stage == "toolcall":
        bin_path = STAGE_JSONL["toolcall"]
        if not bin_path.exists():
            raise FileNotFoundError(f"Toolcall bin not found: {bin_path}")
        n_seqs = struct.unpack_from("<I", bin_path.read_bytes(), 0)[0]
        print(f"  Using {bin_path.name} ({n_seqs:,} seqs)")
    else:
        jsonl = STAGE_JSONL[stage]
        if not jsonl.exists():
            raise FileNotFoundError(
                f"JSONL not found: {jsonl}\n"
                f"Run: python xshard_to_jsonl.py --domain {stage}"
            )
        bin_path = BIN_DIR / f"tokens_{stage}.bin"
        if not bin_path.exists():
            print(f"  Tokenising {jsonl.name}...")
            jsonl_to_bin(jsonl, bin_path, block=block, max_records=max_records)
        else:
            n_seqs = struct.unpack_from("<I", bin_path.read_bytes(), 0)[0]
            print(f"  Using cached {bin_path.name} ({n_seqs:,} seqs)")

    # finetune_toolcall_pt.py interface:
    #   --model, --data, --out_dir, --steps, --batch, --lr, --log_every, --ckpt_every
    cmd = [
        sys.executable, str(TRAINER_PY),
        "--model",      str(model_path),
        "--data",       str(bin_path),
        "--out_dir",    str(stage_out),
        "--steps",      str(effective_steps),
        "--batch",      str(batch),
        "--lr",         str(lr),
        "--log_every",  str(log_every),
        "--ckpt_every", str(ckpt_every),
    ]

    print(f"  cmd: python finetune_toolcall_pt.py [args]")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(TRAINER_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] trainer exited {result.returncode} after {elapsed:.0f}s")
        raise RuntimeError(f"Stage {stage} failed")

    # Trainer writes: out_dir/gpt2_medium_ft_toolcall.safetensors
    final = stage_out / TRAINER_FINAL_NAME
    if not final.exists():
        # Fall back to latest checkpoint
        ckpts = sorted(stage_out.glob("*.safetensors"))
        if not ckpts:
            raise RuntimeError(f"No safetensors found in {stage_out}")
        final = ckpts[-1]

    # Rename to stage-specific name for clarity
    renamed = base_out_dir / f"layered_{stage}.safetensors"
    import shutil
    shutil.copy2(str(final), str(renamed))

    print(f"  [OK] {elapsed:.0f}s  ->  {renamed.name}")
    return renamed


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",       default="", help="Starting safetensors model path")
    ap.add_argument("--stages",     default="instruct,toolcall",
                    help="Comma-separated list of stages to run (code/math optional)")
    ap.add_argument("--skip_to",    default="", help="Skip earlier stages, start from this one")
    ap.add_argument("--steps",      type=int,   default=500)
    ap.add_argument("--batch",      type=int,   default=8)
    ap.add_argument("--lr",         type=float, default=1e-5)
    ap.add_argument("--block",      type=int,   default=128)
    ap.add_argument("--log_every",  type=int,   default=10)
    ap.add_argument("--ckpt_every", type=int,   default=100)
    ap.add_argument("--max_records",type=int,   default=0,
                    help="Max records per stage JSONL (0=all)")
    ap.add_argument("--out_dir",    default=str(CKPT_DIR))
    args = ap.parse_args()

    stages  = [s.strip() for s in args.stages.split(",") if s.strip()]
    out_dir = pathlib.Path(args.out_dir)
    skip_to = args.skip_to.strip()

    # Find base model
    if args.base:
        base = pathlib.Path(args.base)
        if not base.exists():
            print(f"ERROR: model not found: {base}")
            sys.exit(1)
    else:
        # Auto-detect: prefer the most recent completed layered checkpoint
        # from ANY prior stage (not just stages in the current --stages list).
        # This prevents the "broken base" bug where --stages code,math,toolcall
        # misses layered_instruct.safetensors because instruct isn't in the list.
        out_dir_check = pathlib.Path(args.out_dir)
        prior_ckpts = []
        for stage in STAGE_ORDER:
            ckpt = out_dir_check / f"layered_{stage}.safetensors"
            if ckpt.exists():
                prior_ckpts.append(ckpt)
        if prior_ckpts:
            base = prior_ckpts[-1]  # most advanced completed stage
            print(f"Auto-detected prior checkpoint: {base.name}")
        else:
            base = next((p for p in BASE_CANDIDATES if p.exists()), None)
        if not base:
            print("ERROR: No base model found. Pass --base <path.safetensors>")
            sys.exit(1)

    print(f"Base model : {base.name}  ({base.stat().st_size/1e6:.0f} MB)")

    # Skip-to
    if skip_to and skip_to in stages:
        stages = stages[stages.index(skip_to):]

    # Auto-resume: skip stages whose checkpoint already exists
    remaining = []
    current_model = base
    for stage in stages:
        ckpt = out_dir / f"layered_{stage}.safetensors"
        if ckpt.exists():
            print(f"  [skip] {stage} checkpoint exists -> {ckpt.name}")
            current_model = ckpt
        else:
            remaining.append(stage)

    if not remaining:
        print("All stages already complete.")
        print(f"Final model: {current_model}")
        sys.exit(0)

    print(f"Stages to run : {remaining}")
    print(f"Steps/stage   : {args.steps}   LR: {args.lr}   Batch: {args.batch}")
    print()

    for stage in remaining:
        current_model = run_stage(
            stage, current_model, out_dir,
            steps=args.steps, lr=args.lr, batch=args.batch,
            block=args.block, log_every=args.log_every,
            ckpt_every=args.ckpt_every, max_records=args.max_records,
        )

    print(f"\nAll stages complete.")
    print(f"Final model: {current_model}")
    print(f"\nTo test:")
    print(f"  python test_toolcall_models.py")


if __name__ == "__main__":
    main()
