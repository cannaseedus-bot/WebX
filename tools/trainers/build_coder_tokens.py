"""
build_coder_tokens.py — Convert E:\data\coder_outputs to a token bin
for fine-tuning the Coder Micronaut specialist model.

Format: {"id":N, "messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
→ ### Instruction:\n{user}\n\n### Response:\n{assistant}<|endoftext|>

7.1M total examples — we sample 80,000 for a focused coder specialist.
At block=256, batch=8: 1 epoch ≈ ~10,000 steps.
"""
import json
import pathlib
import random
import struct

import tiktoken

DATA_DIR = pathlib.Path(r"E:\data\coder_outputs")
OUT      = pathlib.Path(r"E:\models\GPT2\coder_micronaut\tokens_coder.bin")
MAX_REC  = 80_000
BLOCK    = 256
SEED     = 42

OUT.parent.mkdir(parents=True, exist_ok=True)

enc  = tiktoken.get_encoding("gpt2")
rng  = random.Random(SEED)

# Collect all chunk files, shuffle, sample from them
files = sorted(DATA_DIR.glob("*.jsonl"))
rng.shuffle(files)
print(f"Found {len(files)} chunk files  ({sum(f.stat().st_size for f in files)/1e9:.2f} GB total)")
print(f"Sampling {MAX_REC:,} records at block={BLOCK}...")

toks = []
n_seen = 0

for chunk_file in files:
    if n_seen >= MAX_REC:
        break
    try:
        content = chunk_file.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        continue

    lines = [l for l in content.splitlines() if l.strip()]
    rng.shuffle(lines)

    for line in lines:
        if n_seen >= MAX_REC:
            break
        try:
            e = json.loads(line)
        except Exception:
            continue

        # Handle both formats
        if "messages" in e:
            msgs = e["messages"]
            # Find user/assistant pairs
            user_text = asst_text = ""
            for m in msgs:
                if m.get("role") == "user":
                    user_text = str(m.get("content","")).strip()
                elif m.get("role") == "assistant":
                    asst_text = str(m.get("content","")).strip()
            if not user_text or not asst_text:
                continue
        elif "text" in e:
            # Raw text — use as response with a generic instruction
            raw = str(e["text"]).strip()
            if len(raw) < 20:
                continue
            user_text = "Complete the following code:"
            asst_text = raw
        else:
            continue

        # Skip very short or very long examples
        if len(user_text) < 10 or len(asst_text) < 20:
            continue
        if len(user_text) + len(asst_text) > 4000:
            asst_text = asst_text[:2000]  # truncate long responses

        text = (
            f"### Instruction:\n{user_text}\n\n"
            f"### Response:\n{asst_text}<|endoftext|>"
        )
        toks.extend(enc.encode(text, allowed_special={"<|endoftext|>"}))
        n_seen += 1

        if n_seen % 10_000 == 0:
            print(f"  {n_seen:,} / {MAX_REC:,} records  ({len(toks):,} tokens so far)")

print(f"\nTotal records: {n_seen:,}  Total tokens: {len(toks):,}")

n    = len(toks) // BLOCK
flat = toks[:n * BLOCK]
with open(OUT, "wb") as f:
    f.write(struct.pack("<II", n, BLOCK))
    f.write(struct.pack(f"<{len(flat)}I", *flat))

print(f"Coder bin: {n:,} seqs x {BLOCK} ({OUT.stat().st_size/1e6:.1f} MB)")
print(f"1 epoch at batch=8: {n//8:,} steps")
print(f"Output: {OUT}")
