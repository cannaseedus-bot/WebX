"""
train_math_micronaut.py — Launch math_tool µMODEL training at 50% CPU.

µMODEL concept: Drivers/Kernels that consume schemas/TOML/YAML/XML/MD
as their behavior specification, backed by a fine-tuned specialist model.
The math_tool µMODEL:
  - Reads math YAML/TOML specs via SemanticReader
  - Routes arithmetic, calculus, linear algebra, statistics queries
  - Trained on math xshard data (3 shards, 14k records, 56k seqs)

XVM CPU cluster optimum (v0.1.0-xvm-cpu-thread-cluster):
  500 batches x 4 epochs = 2000 steps at batch=4
  Or: 3000 steps = ~1 epoch through math data

50% CPU: 4 threads (leaving 4 for routing model)
"""
import subprocess, sys, pathlib, torch

# Limit to 50% CPU before importing trainer
torch.set_num_threads(4)
print(f"[math-µMODEL] CPU threads: {torch.get_num_threads()} (50% of 8)")

TRAINER  = pathlib.Path(r"C:\Users\canna\.gpu_trainer\finetune_toolcall_pt.py")
MODEL    = pathlib.Path(r"E:\models\GPT2\mini-GPT\model.safetensors")
DATA     = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\tokens_math_v2.bin")
OUT_DIR  = pathlib.Path(r"E:\models\GPT2\math_micronaut")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, str(TRAINER),
    "--model",      str(MODEL),
    "--data",       str(DATA),
    "--out_dir",    str(OUT_DIR),
    "--steps",      "3000",     # XVM optimum: ~1 epoch, loss target < 1.5
    "--batch",      "4",        # 50% CPU (half of batch=8)
    "--lr",         "2e-5",     # Slightly higher LR — math needs faster convergence
    "--log_every",  "100",
    "--ckpt_every", "500",
]

print("[math-uMODEL]", ' '.join(cmd[2:]))
print("[math-uMODEL] 56,707 seqs x 256 | 3000 steps ~1 epoch")
print("[math-uMODEL] loss target <1.5 | driver: kxml_settings.xml + math_tool spec")

proc = subprocess.run(cmd, cwd=r"C:\Users\canna\.gpu_trainer")
sys.exit(proc.returncode)
