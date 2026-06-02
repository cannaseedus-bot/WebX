"""Build tokens_math_v2.bin at block=256 from math xshard JSONL.
µMODEL target: math_tool specialist on mathematical reasoning data.
"""
import json, struct, pathlib, tiktoken

JSONL   = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\xshard_jsonl\prompt_math_layer.jsonl")
OUT     = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\tokens_math_v2.bin")
BLOCK   = 256
MAX_REC = 30_000  # math data is smaller — use all 3 shards

enc  = tiktoken.get_encoding("gpt2")
toks = []
n    = 0

with open(JSONL, encoding="utf-8") as f:
    for line in f:
        if n >= MAX_REC: break
        try:
            r    = json.loads(line)
            p    = r.get("prompt","")
            resp = r.get("response","")
            if not p or not resp: continue
            text = f"### Instruction:\n{p}\n\n### Response:\n{resp}<|endoftext|>"
            toks.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))
            n += 1
        except Exception:
            pass

seqs = len(toks) // BLOCK
flat = toks[:seqs*BLOCK]
with open(OUT, "wb") as f:
    f.write(struct.pack("<II", seqs, BLOCK))
    f.write(struct.pack(f"<{len(flat)}I", *flat))

print(f"math v2 bin: {seqs:,} seqs x {BLOCK} ({OUT.stat().st_size/1e6:.1f} MB)  n={n:,}")
print(f"1 epoch at batch=4: {seqs//4:,} steps")
