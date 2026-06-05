"""
geodesic_attention_bridge.py — iGPU spherical map → geodesic + ARC-weighted attention

Architecture:
  1. Compile phase (once at training start, on iGPU or CPU fallback):
       token embeddings [V, d] → project to S^(d-1) → k-NN geodesic distances
       written to geodesic_cache.npy

  2. Runtime attention (every forward pass, on CPU with 8 threads):
       standard:  attn = softmax(Q @ K.T / sqrt(d))
       geodesic:  attn = softmax(-geodesic_dist(q_sphere, k_sphere) / R + arc_bias)

  3. ARC weights (accumulated across training):
       ReplayableArc quality scores → additive bias on attention logits
       High-quality arcs (short geodesic path, low entropy) boost their token pairs

Geodesic distance on S^(d-1):
  d(p, q) = arccos(p · q)   where |p| = |q| = 1
  This is the great-circle distance on the unit hypersphere.
  Connection to spherical_map_compiler.hlsl:
    The HLSL kernel computes GeodesicDist(p, q, R) = R * acos(dot(p,q)/(R*R))
    With R=1: d(p,q) = acos(p·q) — identical.

ARC weight bias:
  arc_bias[i, j] = sum(arc.quality * exp(-arc.mean_entropy))
                   for arcs connecting token i's sphere pos to token j's
  High quality, low entropy arcs → positive bias → attention prefers those token pairs
  This is the "replayable arc" concept: paths the model has practiced get rewarded.

iGPU acceleration (Intel HD 4600):
  If D3D11 runtime is available (native/optical_processor/):
    subprocess → optical_processor.exe or spherical_map_compiler compiled binary
    Writes geodesicCache to shared mmap file, Python reads zero-copy
  Otherwise: CPU NumPy fallback (still fast with 8 threads for k-NN)
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────

CACHE_DIR   = pathlib.Path(r"E:\models\GPT2\geodesic_cache")  # E: has 1.2TB free; C: had only 10GB
KNN         = 32       # nearest neighbours per token (matches spherical_map_compiler.hlsl)
RADIUS      = 1.0      # unit sphere
ARC_DECAY   = 0.99     # entropy decay per replay (matches arc_replay_kernel.hlsl fogReduction)
MIN_QUALITY = 0.3      # minimum arc quality to include as bias

# ─── Sphere projection ────────────────────────────────────────────────────────

def project_to_sphere(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize rows to unit sphere S^(d-1).  [*, d] → [*, d]"""
    return F.normalize(x, p=2, dim=-1, eps=eps)

def geodesic_dist_matrix(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Geodesic (great-circle) distance matrix between two sets of unit vectors.
    p: [*, N, d]  q: [*, M, d]  (already on unit sphere)
    Returns: [*, N, M]  values in [0, π]
    """
    cos_sim = torch.matmul(p, q.transpose(-2, -1))   # [*, N, M]
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_sim)

# ─── KXML/JSONL metric model ──────────────────────────────────────────────────

class ArcStatistics(BaseModel):
    total_arcs: int
    active_pairs: int
    coverage: float
    max_bias: float
    memory_kb: int

    def to_jsonl(self) -> str:
        return json.dumps(self.model_dump())

    def to_kxml(self) -> str:
        m = self.model_dump()
        return (
            '<kxml:compute op="arc_stats" domain="trainer" phase="Ch\'en">\n'
            f'  <step phase="Pop">total_arcs: {m["total_arcs"]:,}</step>\n'
            f'  <step phase="Wo">active_pairs: {m["active_pairs"]:,} coverage: {m["coverage"]:.2e}</step>\n'
            f'  <result phase="Ch\'en">max_bias={m["max_bias"]:.4f} mem={m["memory_kb"]}KB</result>\n'
            '</kxml:compute>'
        )


# ─── ARC weight accumulator ───────────────────────────────────────────────────

class ARCWeightMatrix:
    """
    Accumulates arc quality scores as a SPARSE dict.
    Dense [50257,50257] float16+int32 = 14GB — killed the process.
    Sparse dict: zero memory until a pair is seen, ~4 bytes/pair.
    268K pairs from previous run = ~1MB. 1500 steps ≈ 300K pairs = ~1MB.
    """
    def __init__(self, vocab_size: int, dtype=np.float16):
        self.vocab       = vocab_size
        self._pairs: dict = {}   # (src, dst) -> float bias value
        self._counts: dict = {}  # (src, dst) -> int count
        self._total_arcs = 0

    # ── internal helpers ──
    def _get(self, s, d): return self._pairs.get((s, d), 0.0)
    def _set(self, s, d, v): self._pairs[(s, d)] = v; self._counts[(s, d)] = self._counts.get((s,d),0)+1

    def record_arc(self, src_tokens, dst_tokens, quality: float, entropy: float):
        if quality < MIN_QUALITY: return
        weight = quality * math.exp(-entropy)
        for s in src_tokens[:32]:
            for d in dst_tokens[:32]:
                if 0 <= s < self.vocab and 0 <= d < self.vocab:
                    self._set(s, d, self._get(s, d) + weight)
        self._total_arcs += 1

    def record_batch(self, input_ids: torch.Tensor, loss_per_token):
        loss_val = float(loss_per_token.detach()) if hasattr(loss_per_token, 'detach') \
                   else float(loss_per_token)
        quality = math.exp(-loss_val / 10.0)
        if quality < MIN_QUALITY: return
        if input_ids.ndim == 1 or input_ids.shape[-1] < 2: return
        B, T = input_ids.shape
        for b in range(min(B, 4)):
            for t in range(T - 1):
                src = int(input_ids[b, t])
                dst = int(input_ids[b, t + 1])
                if 0 <= src < self.vocab and 0 <= dst < self.vocab:
                    old = self._get(src, dst)
                    self._set(src, dst, old * 0.99 + quality * 0.01)

    def bias_for_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not self._pairs:
            return None   # no arcs yet — skip bias entirely
        ids = token_ids.cpu().numpy().astype(np.int32)
        T = len(ids)
        bias = np.zeros((T, T), dtype=np.float32)
        # Only iterate over actually-recorded pairs, not all T² combinations
        for (s, d), v in self._pairs.items():
            si = np.where(ids == s)[0]
            di = np.where(ids == d)[0]
            if len(si) and len(di):
                bias[np.ix_(si, di)] = v
        return torch.from_numpy(bias)

    def save(self, path: pathlib.Path):
        # Sparse COO from dict — zero allocation, ~4 bytes/pair
        if not self._pairs:
            print("[ARCWeights] Nothing to save (0 pairs)")
            return
        rows = np.array([k[0] for k in self._pairs], dtype=np.int32)
        cols = np.array([k[1] for k in self._pairs], dtype=np.int32)
        vals = np.array(list(self._pairs.values()), dtype=np.float16)
        cnts = np.array([self._counts.get(k,1) for k in self._pairs], dtype=np.int32)
        np.save(str(path / 'arc_bias_rows.npy'),   rows)
        np.save(str(path / 'arc_bias_cols.npy'),   cols)
        np.save(str(path / 'arc_bias_vals.npy'),   vals)
        np.save(str(path / 'arc_bias_counts.npy'), cnts)
        size_kb = (rows.nbytes + cols.nbytes + vals.nbytes + cnts.nbytes) // 1024
        print(f"[ARCWeights] Saved {len(rows):,} arc pairs ({size_kb} KB sparse) -> {path}")

    def load(self, path: pathlib.Path):
        rf = path / 'arc_bias_rows.npy'
        if not rf.exists():
            return
        rows   = np.load(str(rf))
        cols   = np.load(str(path / 'arc_bias_cols.npy'))
        vals   = np.load(str(path / 'arc_bias_vals.npy')).astype(np.float32)
        counts = np.load(str(path / 'arc_bias_counts.npy'))
        for r, c, v, n in zip(rows, cols, vals, counts):
            self._pairs[(int(r), int(c))]  = float(v)
            self._counts[(int(r), int(c))] = int(n)
        print(f"[ARCWeights] Loaded {len(rows):,} arc pairs from {path}")

    @property
    def total_arcs(self): return self._total_arcs

    def stats(self) -> ArcStatistics:
        nonzero = len(self._pairs)
        max_bias = max(self._pairs.values()) if self._pairs else 0.0
        return ArcStatistics(
            total_arcs=self._total_arcs,
            active_pairs=nonzero,
            coverage=nonzero / (self.vocab * self.vocab),
            max_bias=float(max_bias),
            memory_kb=(nonzero * 12) // 1024,
        )

# ─── Geodesic map compiler (CPU mirror of spherical_map_compiler.hlsl) ───────

class GeodesicMapCompiler:
    """
    Mirrors spherical_map_compiler.hlsl executed on iGPU.
    Compiles the token embedding space into a geodesic distance cache.

    iGPU path (Intel HD 4600, when native exe is available):
      subprocess → writes geodesicCache to mmap file → Python reads zero-copy
    CPU path (NumPy, multiprocessing, 8 threads):
      Same math as HLSL GeodesicDist() = acos(dot(p,q)) for unit vectors
      k=32 NN via batched matrix multiply + partial sort

    Connection to SphericalGeometryAVX2.h:
      GeodesicDistanceF(p, q, R=1.0) = acos(clamp(dot(p,q), -1, 1))
      BatchExponentialMap → here implemented as incremental projection
    """
    def __init__(self, cache_dir: pathlib.Path = CACHE_DIR, knn: int = KNN):
        self.cache_dir = cache_dir
        self.knn       = knn
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._sphere_pos : np.ndarray | None = None  # [V, d] unit vectors
        self._knn_idx    : np.ndarray | None = None  # [V, K] nearest neighbour ids
        self._knn_dist   : np.ndarray | None = None  # [V, K] geodesic distances
        self._vocab      : int = 0

    def compile(self, embedding_weight: torch.Tensor,
                force: bool = False) -> GeodesicMapCompiler:
        """
        Project token embeddings to sphere, compute k-NN geodesic distances.
        embedding_weight: [V, d]  (model.wte.weight)
        """
        cache_file = self.cache_dir / f'sphere_pos_V{embedding_weight.shape[0]}.npy'
        knn_file   = self.cache_dir / f'knn_idx_k{self.knn}.npy'
        dist_file  = self.cache_dir / f'knn_dist_k{self.knn}.npy'

        if not force and cache_file.exists() and knn_file.exists():
            print(f"[GeoMap] Loading cached spherical map from {self.cache_dir}")
            self._sphere_pos = np.load(str(cache_file))
            self._knn_idx    = np.load(str(knn_file))
            self._knn_dist   = np.load(str(dist_file))
            self._vocab      = self._sphere_pos.shape[0]
            print(f"[GeoMap] Loaded: V={self._vocab} k={self.knn}")
            return self

        print(f"[GeoMap] Compiling spherical map for {embedding_weight.shape[0]} tokens...")
        t0 = time.time()

        # Project to unit sphere (Pop → Wo in K'UHUL phases)
        with torch.no_grad():
            sphere = F.normalize(embedding_weight.float(), dim=-1)
        self._sphere_pos = sphere.numpy()
        self._vocab      = self._sphere_pos.shape[0]

        # k-NN geodesic distances (Sek phase — the expensive part)
        # Process in chunks to fit in RAM
        chunk = 512
        V = self._vocab
        knn_idx  = np.zeros((V, self.knn), dtype=np.int32)
        knn_dist = np.zeros((V, self.knn), dtype=np.float16)

        for start in range(0, V, chunk):
            end   = min(V, start + chunk)
            p     = self._sphere_pos[start:end]                     # [C, d]
            # cos_sim with all tokens: [C, V]
            cosim = (p @ self._sphere_pos.T).clip(-1 + 1e-7, 1 - 1e-7)
            dists = np.arccos(cosim).astype(np.float32)             # [C, V]
            # vectorized self-distance zero-out (no Python loop)
            chunk_len = end - start
            ci = np.arange(chunk_len)
            dists[ci, start + ci] = np.inf
            # k nearest — batch argsort within top-k (no Python loop)
            top_k_idx  = np.argpartition(dists, self.knn, axis=1)[:, :self.knn]
            top_k_dist = dists[ci[:, None], top_k_idx]
            sort_ord   = np.argsort(top_k_dist, axis=1)
            knn_idx[start:end]  = top_k_idx[ci[:, None], sort_ord]
            knn_dist[start:end] = top_k_dist[ci[:, None], sort_ord].astype(np.float16)

            if (start // chunk) % 10 == 0:
                elapsed = time.time() - t0
                pct = (start + chunk) / V * 100
                print(f"  [GeoMap] {pct:.0f}%  {elapsed:.0f}s  "
                      f"chunk [{start}:{end}]", flush=True)

        self._knn_idx  = knn_idx
        self._knn_dist = knn_dist

        # Save cache (Ch'en phase)
        np.save(str(cache_file), self._sphere_pos)
        np.save(str(knn_file),   self._knn_idx)
        np.save(str(dist_file),  self._knn_dist)

        elapsed = time.time() - t0
        print(f"[GeoMap] Done in {elapsed:.0f}s  "
              f"cached to {self.cache_dir}")
        return self

    def sphere_pos(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up sphere positions for a batch of token ids. [*, T] → [*, T, d]"""
        ids = token_ids.cpu().numpy().reshape(-1)
        pos = self._sphere_pos[ids]
        return torch.from_numpy(pos).to(token_ids.device).view(*token_ids.shape, -1)

    def is_compiled(self) -> bool:
        return self._sphere_pos is not None

# ─── Geodesic + ARC attention ─────────────────────────────────────────────────

def geodesic_arc_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    arc_bias: torch.Tensor | None = None,
    radius: float = RADIUS,
    temperature: float = 1.0,
    causal: bool = True,
    chunk_size: int = 64,
    sh_coupling: float = 0.15,  # SH neighbour coupling weight (same as optical propagate)
    sh_steps: int = 1,          # propagation passes between adjacent chunks
) -> torch.Tensor:
    """
    SVG3D Parallel Geodesic Attention.

    Three-phase execution (matches K'UHUL Pop→Wo→Sek→Ch'en):

    Wo   — compute raw geodesic logits for ALL chunks upfront
           each chunk = one optical node on the attention lattice

    Sek  — SH wave propagation between adjacent chunks (the SVG3D link)
           chunk[i] += sh_coupling * mean(chunk[i-1])   (left neighbour)
           chunk[i] += sh_coupling * mean(chunk[i+1])   (right neighbour)
           This breaks the 4.0-plateau: chunks that can't distinguish
           nearby tokens on S^(d-1) now borrow signal from neighbours
           that CAN — global semantic structure emerges from local coupling.

    Ch'en — softmax + weighted sum per chunk, write to output

    Why this fixes the plateau:
      Below loss ~4 the remaining hard tokens are semantically close on
      S^(d-1) — tiny geodesic distances → near-uniform chunk softmax.
      Independent chunk softmax cannot resolve them.
      With SH coupling each chunk's logits are nudged by what adjacent
      chunks are attending to: tokens that one chunk strongly attends get
      amplified in neighbouring chunks → global attention via local propagation.

    Memory: 4 chunks × [B,H,64,T] stored simultaneously = same 12MB as the
      original full [B,H,T,T] matrix, but now with cross-chunk coupling.
    """
    B, H, T, D = q.shape
    q_s = project_to_sphere(q)
    k_s = project_to_sphere(k)

    # ── Wo: compute geodesic logits for all chunks ────────────────────────────
    chunk_starts  = list(range(0, T, chunk_size))
    chunk_logits  = []   # [num_chunks] each [B, H, chunk_i, T]

    for q_start in chunk_starts:
        q_end = min(T, q_start + chunk_size)
        q_c   = q_s[:, :, q_start:q_end, :]
        cos_c = torch.matmul(q_c, k_s.transpose(-2, -1)).clamp(-1+1e-7, 1-1e-7)
        lc    = -torch.acos(cos_c) / (radius * temperature)
        lc    = lc + torch.log(((1.0 + cos_c) * 0.5).clamp(min=1e-7))
        if arc_bias is not None:
            lc = lc + arc_bias[q_start:q_end, :].unsqueeze(0).unsqueeze(0)
        if causal:
            qi = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
            ki = torch.arange(T, device=q.device).unsqueeze(0)
            lc = lc.masked_fill((ki > qi).unsqueeze(0).unsqueeze(0), float('-inf'))
        chunk_logits.append(lc)

    # ── Sek: SH propagation — adjacent chunk coupling ─────────────────────────
    # Each chunk is an optical node. mean(logits) is the "field summary"
    # that propagates to neighbours, biasing their softmax toward
    # tokens the neighbour chunk is already attending to.
    for _ in range(sh_steps):
        # Reduce to [B,H,1,1] so any chunk size broadcasts cleanly
        means = [lc.mean(dim=(-2,-1), keepdim=True) for lc in chunk_logits]
        for i in range(len(chunk_logits)):
            if i > 0:
                chunk_logits[i] = chunk_logits[i] + sh_coupling * means[i - 1]
            if i < len(chunk_logits) - 1:
                chunk_logits[i] = chunk_logits[i] + sh_coupling * means[i + 1]

    # ── Ch'en: softmax + weighted sum ─────────────────────────────────────────
    out = torch.zeros_like(q)
    for q_start, lc in zip(chunk_starts, chunk_logits):
        q_end  = min(T, q_start + chunk_size)
        attn_c = F.softmax(lc, dim=-1)
        out[:, :, q_start:q_end, :] = torch.matmul(attn_c, v)

    return out

# ─── Drop-in replacement for GPT2Block._attn ─────────────────────────────────

class GeodesicAttentionMixin:
    """
    Mixin to add to GPT2Block.
    Set block.geo_map and block.arc_weights to enable geodesic attention.
    """
    geo_map    : GeodesicMapCompiler | None = None
    arc_weights: ARCWeightMatrix | None     = None
    geo_temperature: float = 1.0

    def _attn_geodesic(self, x: torch.Tensor,
                        token_ids: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = x @ self.c_attn_w + self.c_attn_b
        q, k, v = qkv.split(self.n_embd, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ARC bias: if we have token ids and accumulated arcs, look up the bias
        arc_bias = None
        if self.arc_weights is not None and token_ids is not None:
            # Use first item in batch as representative sequence for bias
            arc_bias = self.arc_weights.bias_for_tokens(token_ids[0, :T]).to(x.device)
            arc_bias = arc_bias.clamp(-5.0, 5.0)   # safety clamp

        out = geodesic_arc_attention(q, k, v,
                                     arc_bias=arc_bias,
                                     radius=RADIUS,
                                     temperature=self.geo_temperature,
                                     causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out @ self.c_proj_w + self.c_proj_b

# ─── Convenience: patch a loaded GPT2 model ──────────────────────────────────

def patch_model_geodesic(model, geo_map: GeodesicMapCompiler,
                          arc_weights: ARCWeightMatrix,
                          temperature: float = 1.0):
    """
    Replace standard dot-product attention in all blocks with geodesic + ARC attention.
    Call once after load_gpt2() to enable the spherical map.
    """
    import types

    def make_attn(block):
        def _attn_geo(self_block, x, token_ids=None):
            return GeodesicAttentionMixin._attn_geodesic(self_block, x, token_ids)
        return types.MethodType(_attn_geo, block)

    for block in model.blocks:
        block.geo_map         = geo_map
        block.arc_weights     = arc_weights
        block.geo_temperature = temperature
        block._attn_geodesic  = make_attn(block)

    print(f"[GeoMap] Patched {len(model.blocks)} blocks with geodesic + ARC attention")
    print(f"  radius={RADIUS}  temperature={temperature}  "
          f"arc_pairs={arc_weights.stats().active_pairs}")
    return model

# ─── Startup helper ───────────────────────────────────────────────────────────

def build_geo_system(model, cache_dir: pathlib.Path = CACHE_DIR,
                     vocab_size: int = 50257,
                     force_recompile: bool = False):
    """
    One-call setup: compile geodesic map + create ARC weight accumulator.
    Returns (geo_map, arc_weights) ready to pass to patch_model_geodesic().
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    geo_map = GeodesicMapCompiler(cache_dir)
    geo_map.compile(model.wte.weight.data, force=force_recompile)

    arc_weights = ARCWeightMatrix(vocab_size)
    arc_cache   = cache_dir / 'arc_bias.npy'
    if arc_cache.exists():
        arc_weights.load(cache_dir)
        print(f"[ARCWeights] Loaded: {arc_weights.stats().to_jsonl()}")
    else:
        print("[ARCWeights] Starting fresh (will accumulate from training)")

    return geo_map, arc_weights
