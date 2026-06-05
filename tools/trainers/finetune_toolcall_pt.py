"""
finetune_toolcall_pt.py — PyTorch CPU fine-tune of GPT-2 medium on toolcall data.

No transformers required — pure torch + safetensors.
Weights use bare keys (no 'transformer.' prefix) and Conv1D layout [in, out].

Usage:
  python finetune_toolcall_pt.py
  python finetune_toolcall_pt.py --steps 500 --batch 2 --lr 1e-4 --log_every 25
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import struct
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from safetensors.torch import load_file, save_file


class FinetuneConfig(BaseModel):
    bin_dir:    pathlib.Path = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin")
    base_model: pathlib.Path = pathlib.Path(r"E:\models\GPT2\med-GPT\model.safetensors")
    out_dir:    pathlib.Path = pathlib.Path(r"E:\models\GPT2\med-GPT")

    @classmethod
    def from_env(cls) -> FinetuneConfig:
        kw: dict = {}
        if v := os.environ.get('KUHUL_BIN_DIR'):    kw['bin_dir']    = pathlib.Path(v)
        if v := os.environ.get('KUHUL_BASE_MODEL'): kw['base_model'] = pathlib.Path(v)
        if v := os.environ.get('KUHUL_OUT_DIR'):    kw['out_dir']    = pathlib.Path(v)
        return cls(**kw)

    def to_jsonl(self) -> str:
        return json.dumps({k: str(v) for k, v in self.model_dump().items()})

    def to_kxml(self) -> str:
        return (
            '<kxml:compute op="finetune_config" domain="trainer" phase="Pop">\n'
            f'  <step phase="Pop">bin_dir: {self.bin_dir}</step>\n'
            f'  <step phase="Wo">base_model: {self.base_model.name}</step>\n'
            f'  <result phase="Ch\'en">out_dir: {self.out_dir}</result>\n'
            '</kxml:compute>'
        )


_cfg      = FinetuneConfig.from_env()
BIN_DIR   = _cfg.bin_dir
BASE_MODEL = _cfg.base_model
OUT_DIR   = _cfg.out_dir
TRAIN_BIN  = BIN_DIR / "tokens_toolcall.bin"


# ─── Minimal GPT-2 implementation ─────────────────────────────────────────────
# Conv1D convention: weight shape [in_features, out_features]
# Forward: y = x @ W + b  (NOT F.linear which transposes W)

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)   # torch built-in — numerically stable


class GPT2Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_ctx: int):
        super().__init__()
        self.n_embd  = n_embd
        self.n_head  = n_head
        self.head_dim = n_embd // n_head
        self.n_ctx   = n_ctx

        # LayerNorms
        self.ln_1_w = nn.Parameter(torch.ones(n_embd))
        self.ln_1_b = nn.Parameter(torch.zeros(n_embd))
        self.ln_2_w = nn.Parameter(torch.ones(n_embd))
        self.ln_2_b = nn.Parameter(torch.zeros(n_embd))

        # Attention — Conv1D layout [in, out]
        self.c_attn_w = nn.Parameter(torch.empty(n_embd, 3 * n_embd))
        self.c_attn_b = nn.Parameter(torch.zeros(3 * n_embd))
        self.c_proj_w = nn.Parameter(torch.empty(n_embd, n_embd))
        self.c_proj_b = nn.Parameter(torch.zeros(n_embd))

        # MLP — Conv1D layout
        self.c_fc_w   = nn.Parameter(torch.empty(n_embd, 4 * n_embd))
        self.c_fc_b   = nn.Parameter(torch.zeros(4 * n_embd))
        self.m_proj_w = nn.Parameter(torch.empty(4 * n_embd, n_embd))
        self.m_proj_b = nn.Parameter(torch.zeros(n_embd))

    def _ln(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
            eps: float = 1e-5) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, unbiased=False, keepdim=True)
        return w * (x - mean) / (var + eps).sqrt() + b

    # Geodesic attention components — set by patch_model_geodesic()
    geo_map        = None
    arc_weights    = None
    geo_temperature: float = 1.0

    def _attn(self, x: torch.Tensor,
              token_ids: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = x @ self.c_attn_w + self.c_attn_b
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.geo_map is not None and self.geo_map.is_compiled():
            # ── Geodesic + ARC attention ──────────────────────────────────────
            from geodesic_attention_bridge import geodesic_arc_attention
            arc_bias = None
            if self.arc_weights is not None and token_ids is not None:
                raw = self.arc_weights.bias_for_tokens(token_ids[0, :T])
                if raw is not None:
                    arc_bias = raw.to(x.device).clamp(-5.0, 5.0)
            out = geodesic_arc_attention(q, k, v,
                                         arc_bias=arc_bias,
                                         temperature=self.geo_temperature,
                                         causal=True)
        else:
            # ── Standard dot-product attention ────────────────────────────────
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out @ self.c_proj_w + self.c_proj_b

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        h = gelu(x @ self.c_fc_w + self.c_fc_b)
        return h @ self.m_proj_w + self.m_proj_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self._ln(x, self.ln_1_w, self.ln_1_b))
        x = x + self._mlp(self._ln(x, self.ln_2_w, self.ln_2_b))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab: int, n_embd: int, n_layer: int, n_head: int, n_ctx: int):
        super().__init__()
        self.wte   = nn.Embedding(vocab, n_embd)
        self.wpe   = nn.Embedding(n_ctx,  n_embd)
        self.blocks = nn.ModuleList([GPT2Block(n_embd, n_head, n_ctx) for _ in range(n_layer)])
        self.ln_f_w = nn.Parameter(torch.ones(n_embd))
        self.ln_f_b = nn.Parameter(torch.zeros(n_embd))
        self.n_ctx  = n_ctx

    def _ln(self, x, w, b, eps=1e-5):
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, unbiased=False, keepdim=True)
        return w * (x - mean) / (var + eps).sqrt() + b

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            if hasattr(block, 'geo_map') and block.geo_map is not None:
                # Correct pre-norm residual: x = x + f(LN(x))
                x = x + block._attn(block._ln(x, block.ln_1_w, block.ln_1_b), token_ids=idx)
                x = x + block._mlp(block._ln(x, block.ln_2_w, block.ln_2_b))
            else:
                x = block(x)
        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        return x @ self.wte.weight.T


def load_gpt2(path: pathlib.Path) -> GPT2:
    raw = load_file(str(path))

    # Normalise keys: strip "transformer." prefix if present (HuggingFace format)
    # so both GPT-2 small (transformer.h.*) and medium (bare h.*) load the same way
    sd = {}
    for k, v in raw.items():
        sd[k[len("transformer."):] if k.startswith("transformer.") else k] = v

    # Detect config from normalised shapes
    wte_key = "wte.weight"
    n_embd  = sd[wte_key].shape[1]     # 768 small / 1024 medium
    vocab   = sd[wte_key].shape[0]     # 50257
    n_ctx   = sd["wpe.weight"].shape[0]  # 1024
    n_layer = max(int(k.split(".")[1]) for k in sd if k.startswith("h.")) + 1
    n_head  = n_embd // 64             # 12 small / 16 medium

    model = GPT2(vocab, n_embd, n_layer, n_head, n_ctx)

    # Load weights
    model.wte.weight.data.copy_(sd["wte.weight"])
    model.wpe.weight.data.copy_(sd["wpe.weight"])
    model.ln_f_w.data.copy_(sd["ln_f.weight"])
    model.ln_f_b.data.copy_(sd["ln_f.bias"])

    for i, block in enumerate(model.blocks):
        p = f"h.{i}."
        block.ln_1_w.data.copy_(sd[p + "ln_1.weight"])
        block.ln_1_b.data.copy_(sd[p + "ln_1.bias"])
        block.ln_2_w.data.copy_(sd[p + "ln_2.weight"])
        block.ln_2_b.data.copy_(sd[p + "ln_2.bias"])
        block.c_attn_w.data.copy_(sd[p + "attn.c_attn.weight"])
        block.c_attn_b.data.copy_(sd[p + "attn.c_attn.bias"])
        block.c_proj_w.data.copy_(sd[p + "attn.c_proj.weight"])
        block.c_proj_b.data.copy_(sd[p + "attn.c_proj.bias"])
        block.c_fc_w.data.copy_(sd[p + "mlp.c_fc.weight"])
        block.c_fc_b.data.copy_(sd[p + "mlp.c_fc.bias"])
        block.m_proj_w.data.copy_(sd[p + "mlp.c_proj.weight"])
        block.m_proj_b.data.copy_(sd[p + "mlp.c_proj.bias"])

    return model


def save_gpt2(model: GPT2, path: pathlib.Path) -> None:
    sd: dict[str, torch.Tensor] = {}
    sd["wte.weight"] = model.wte.weight.data.contiguous()
    sd["wpe.weight"] = model.wpe.weight.data.contiguous()
    sd["ln_f.weight"] = model.ln_f_w.data.contiguous()
    sd["ln_f.bias"]   = model.ln_f_b.data.contiguous()
    for i, block in enumerate(model.blocks):
        p = f"h.{i}."
        sd[p+"ln_1.weight"]         = block.ln_1_w.data.contiguous()
        sd[p+"ln_1.bias"]           = block.ln_1_b.data.contiguous()
        sd[p+"ln_2.weight"]         = block.ln_2_w.data.contiguous()
        sd[p+"ln_2.bias"]           = block.ln_2_b.data.contiguous()
        sd[p+"attn.c_attn.weight"]  = block.c_attn_w.data.contiguous()
        sd[p+"attn.c_attn.bias"]    = block.c_attn_b.data.contiguous()
        sd[p+"attn.c_proj.weight"]  = block.c_proj_w.data.contiguous()
        sd[p+"attn.c_proj.bias"]    = block.c_proj_b.data.contiguous()
        sd[p+"mlp.c_fc.weight"]     = block.c_fc_w.data.contiguous()
        sd[p+"mlp.c_fc.bias"]       = block.c_fc_b.data.contiguous()
        sd[p+"mlp.c_proj.weight"]   = block.m_proj_w.data.contiguous()
        sd[p+"mlp.c_proj.bias"]     = block.m_proj_b.data.contiguous()
    save_file(sd, str(path))


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_bin(path: pathlib.Path) -> tuple[torch.Tensor, int]:
    import numpy as np
    with open(path, "rb") as f:
        n_seqs, block = struct.unpack("<II", f.read(8))
        flat = np.frombuffer(f.read(n_seqs * block * 4), dtype=np.uint32)
    return torch.from_numpy(flat.astype(np.int64)).reshape(n_seqs, block), block


# ─── Training loop ─────────────────────────────────────────────────────────────

def train(model, data, steps, batch, lr, log_every, ckpt_every, out_dir,
          cosine_total: int = 0, arc_weights=None):
    n_seqs, block = data.shape
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    # cosine_total: full training length for the scheduler.
    # When called from the cluster runner, pass TOTAL_REMAIN (e.g. 1500) so the
    # LR only decays slightly within a single 50-step shard instead of collapsing
    # from 1e-5 → 1e-6 within the shard.
    t_max = cosine_total if cosine_total > steps else steps
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=lr*0.1)

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params/1e6:.1f}M params  {n_seqs:,} seqs x {block} tokens  batch={batch}")

    t0 = time.monotonic()
    acc = 0.0

    for step in range(1, steps + 1):
        rows = torch.randint(0, n_seqs, (batch,))
        x = data[rows, :-1]
        y = data[rows, 1:]

        opt.zero_grad(set_to_none=True)
        try:
            logits = model(x)
        except Exception:
            import traceback
            print(f"\n[CRASH] model(x) failed on step {step}:")
            traceback.print_exc()
            raise
        loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if not loss.isfinite():
            print(f"  [warn] step {step}: loss={loss.item()} — skipping update")
            continue

        loss.backward()
        # replace any stray NaN/inf grads before clipping
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sch.step()

        acc += loss.item()

        # ARC weight accumulation: record token co-occurrence quality
        # Pass full sequences (x = data[rows, :-1]) so record_batch sees token ids
        if arc_weights is not None:
            arc_weights.record_batch(x.detach(), loss.detach())

        if step % log_every == 0:
            elapsed = time.monotonic() - t0
            print(f"  step {step:4d}/{steps}  loss={acc/log_every:.4f}"
                  f"  lr={sch.get_last_lr()[0]:.2e}  {elapsed/step:.2f}s/step")
            acc = 0.0

        if step % ckpt_every == 0 or step == steps:
            ckpt = out_dir / f"gpt2_medium_toolcall_s{step:04d}.safetensors"
            save_gpt2(model, ckpt)
            print(f"  saved {ckpt.name}")

    final = out_dir / "gpt2_medium_ft_toolcall.safetensors"
    last  = out_dir / f"gpt2_medium_toolcall_s{steps:04d}.safetensors"
    if last.exists():
        shutil.copy2(str(last), str(final))
    print(f"  final -> {final.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",         type=int,   default=500)
    ap.add_argument("--batch",         type=int,   default=2)
    ap.add_argument("--lr",            type=float, default=5e-5)
    ap.add_argument("--log_every",     type=int,   default=25)
    ap.add_argument("--ckpt_every",    type=int,   default=100)
    ap.add_argument("--cosine_total",  type=int,   default=0,
                    help="Full LR schedule length. 0=use --steps.")
    ap.add_argument("--geodesic",      action="store_true", default=False,
                    help="Enable geodesic + ARC attention (requires Haswell iGPU for fast "
                         "map compilation; falls back to CPU NumPy if unavailable).")
    ap.add_argument("--data",          type=str,   default=str(TRAIN_BIN))
    ap.add_argument("--model",         type=str,   default=str(BASE_MODEL))
    ap.add_argument("--out_dir",       type=str,   default=str(OUT_DIR))
    args = ap.parse_args()

    data_path  = pathlib.Path(args.data)
    model_path = pathlib.Path(args.model)
    out_dir    = pathlib.Path(args.out_dir)

    for p, name in [(data_path, "data"), (model_path, "model")]:
        if not p.exists():
            print(f"[ERROR] {name} not found: {p}"); return
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GPT-2 MEDIUM TOOLCALL FINE-TUNE  (PyTorch CPU)")
    print(f"  data  : {data_path.name}")
    print(f"  model : {model_path.name}")
    print(f"  steps : {args.steps}  batch={args.batch}  lr={args.lr:.2e}")
    print("=" * 60)

    print("loading data ...")
    data, _ = load_bin(data_path)

    print("loading model ...")
    t0 = time.monotonic()
    model = load_gpt2(model_path)
    print(f"  loaded in {time.monotonic()-t0:.1f}s")

    arc_weights = None
    if args.geodesic:
        from geodesic_attention_bridge import build_geo_system, patch_model_geodesic
        print("[Geodesic] Compiling Haswell iGPU spherical map...")
        geo_map, arc_weights = build_geo_system(model)
        patch_model_geodesic(model, geo_map, arc_weights, temperature=1.0)
        print("[Geodesic] Geodesic + ARC attention active")

    train(model, data,
          steps        = args.steps,
          batch        = args.batch,
          lr           = args.lr,
          log_every    = args.log_every,
          ckpt_every   = args.ckpt_every,
          out_dir      = out_dir,
          cosine_total = args.cosine_total,
          arc_weights  = arc_weights)

    if arc_weights is not None:
        from geodesic_attention_bridge import CACHE_DIR
        arc_weights.save(CACHE_DIR)
        print(f"[Geodesic] ARC weights saved: {arc_weights.stats().to_jsonl()}")

    print("=" * 60)
    print("DONE")
    print(f"  {out_dir / 'gpt2_medium_ft_toolcall.safetensors'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
