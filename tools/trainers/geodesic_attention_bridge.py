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
import math, pathlib, time, os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ───────────────────────────────────────────────────────────────────

CACHE_DIR   = pathlib.Path(r"C:\Users\canna\.gpu_trainer\bin\geodesic_cache")
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

# ─── ARC weight accumulator ───────────────────────────────────────────────────

class ARCWeightMatrix:
    """
    Accumulates arc quality scores into an attention bias matrix.
    Each (src_token, dst_token) pair records the quality of semantic arcs
    between those tokens observed during training.

    Connection to replayable-arcs.js / SphericalReplayBuffer:
      arc.quality  = 0.5 * length_score + 0.5 * entropy_score
      arc.entropy  = mean entropy along path (lower = more certain = better bias)
      Here we store: bias[i,j] += quality * exp(-entropy)
    """
    def __init__(self, vocab_size: int, dtype=np.float16):
        self.vocab  = vocab_size
        self._bias  = np.zeros((vocab_size, vocab_size), dtype=dtype)
        self._count = np.zeros((vocab_size, vocab_size), dtype=np.int32)
        self._total_arcs = 0

    def record_arc(self, src_tokens: np.ndarray, dst_tokens: np.ndarray,
                   quality: float, entropy: float):
        """
        Record a training arc from src_tokens → dst_tokens with given quality.
        quality ∈ [0,1], entropy ∈ [0,1]
        """
        if quality < MIN_QUALITY:
            return
        weight = quality * math.exp(-entropy)
        for s in src_tokens[:32]:  # cap per arc
            for d in dst_tokens[:32]:
                if 0 <= s < self.vocab and 0 <= d < self.vocab:
                    self._bias[s, d]  += weight
                    self._count[s, d] += 1
        self._total_arcs += 1

    def record_batch(self, input_ids: torch.Tensor, loss_per_token):
        """
        Auto-record arcs from a training batch.
        input_ids: [B, T] token ids  OR  [B] row indices into the dataset
        loss_per_token: scalar or [B, T] tensor
        Called after each training step.
        """
        loss_val = float(loss_per_token.detach()) if hasattr(loss_per_token, 'detach') \
                   else float(loss_per_token)
        quality = math.exp(-loss_val / 10.0)
        if quality < MIN_QUALITY:
            return
        if input_ids.ndim == 1 or input_ids.shape[-1] < 2:
            return   # row indices only — skip, need actual token ids
        B, T = input_ids.shape
        for b in range(min(B, 4)):   # cap at 4 sequences per step for speed
            for t in range(T - 1):
                src = int(input_ids[b, t])
                dst = int(input_ids[b, t + 1])
                if 0 <= src < self.vocab and 0 <= dst < self.vocab:
                    self._bias[src, dst] = 0.99 * self._bias[src, dst] + 0.01 * quality
                    self._count[src, dst] += 1

    def bias_for_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Return arc bias submatrix for a sequence of token ids.
        token_ids: [T]  →  bias: [T, T]  (additive attention logit bias)
        """
        ids = token_ids.cpu().numpy().astype(np.int32)
        bias = self._bias[np.ix_(ids, ids)]   # [T, T] submatrix
        return torch.from_numpy(bias.astype(np.float32))

    def save(self, path: pathlib.Path):
        np.save(str(path / 'arc_bias.npy'),  self._bias)
        np.save(str(path / 'arc_count.npy'), self._count)

    def load(self, path: pathlib.Path):
        bf = path / 'arc_bias.npy'
        cf = path / 'arc_count.npy'
        if bf.exists(): self._bias  = np.load(str(bf))
        if cf.exists(): self._count = np.load(str(cf))

    @property
    def total_arcs(self): return self._total_arcs

    def stats(self):
        nonzero = int((self._count > 0).sum())
        return {'total_arcs': self._total_arcs,
                'active_pairs': nonzero,
                'coverage': nonzero / (self.vocab * self.vocab),
                'max_bias': float(self._bias.max())}

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

        self._sphere_pos : Optional[np.ndarray] = None  # [V, d] unit vectors
        self._knn_idx    : Optional[np.ndarray] = None  # [V, K] nearest neighbour ids
        self._knn_dist   : Optional[np.ndarray] = None  # [V, K] geodesic distances
        self._vocab      : int = 0

    def compile(self, embedding_weight: torch.Tensor,
                force: bool = False) -> 'GeodesicMapCompiler':
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
            # zero out self-distance
            for i in range(end - start):
                dists[i, start + i] = float('inf')
            # k nearest
            top_k_idx = np.argpartition(dists, self.knn, axis=1)[:, :self.knn]
            for i in range(end - start):
                sorted_local = np.argsort(dists[i, top_k_idx[i]])
                knn_idx[start + i]  = top_k_idx[i][sorted_local]
                knn_dist[start + i] = dists[i, knn_idx[start + i]].astype(np.float16)

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
    q: torch.Tensor,           # [B, H, T, D]
    k: torch.Tensor,           # [B, H, T, D]
    v: torch.Tensor,           # [B, H, T, D]
    arc_bias: Optional[torch.Tensor] = None,   # [T, T] additive logit bias
    radius: float = RADIUS,
    temperature: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """
    Geodesic attention: replace Q·K^T with -arccos(q_norm·k_norm^T).

    Attention score = -geodesic_dist(q, k) / (radius × temperature)
                    + arc_bias (if provided)

    High quality arcs (short geodesic, low entropy) get positive bias
    and dominate the attention distribution — the model preferentially
    attends to token pairs it has "practiced" via replay.

    Connection to geo-weights.js GeodesicAttention:
      attend(queryPos, keyPositions) = softmax over exp(-geodesic_dist / R)
      This is the same but batched over heads.

    Connection to arc_replay_kernel.hlsl:
      Quality = 0.5 × length_score + 0.5 × entropy_score
      arc_bias encodes accumulated quality across training replays.
    """
    B, H, T, D = q.shape

    # Project to unit sphere (all ops on same manifold)
    q_s = project_to_sphere(q)   # [B, H, T, D]
    k_s = project_to_sphere(k)   # [B, H, T, D]

    # Geodesic distance matrix: arccos(q·k^T)
    cos_sim = torch.matmul(q_s, k_s.transpose(-2, -1))   # [B, H, T, T]
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    geo_dist = torch.acos(cos_sim)                        # [B, H, T, T]  ∈ [0, π]

    # Convert to logit: negative distance = closer tokens get higher weight
    logits = -geo_dist / (radius * temperature)            # [B, H, T, T]

    # Add ARC bias (accumulated quality from replayable arcs)
    if arc_bias is not None:
        logits = logits + arc_bias.unsqueeze(0).unsqueeze(0)  # broadcast over B, H

    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        logits = logits.masked_fill(mask, float('-inf'))

    # Parallel transport approximation: modulate logits by proximity factor
    # closer tokens (cos_sim→1) get a small bonus on top of the geodesic logit.
    # transport_scale ∈ [0,1]; apply in log-space to logits before softmax.
    transport_scale = (1.0 + cos_sim) * 0.5            # [B, H, T, T]  ∈ [0, 1]
    logits = logits + torch.log(transport_scale.clamp(min=1e-7))

    attn_w = F.softmax(logits, dim=-1)
    out    = torch.matmul(attn_w, v)                   # [B, H, T_q, D]
    return out

# ─── Drop-in replacement for GPT2Block._attn ─────────────────────────────────

class GeodesicAttentionMixin:
    """
    Mixin to add to GPT2Block.
    Set block.geo_map and block.arc_weights to enable geodesic attention.
    """
    geo_map    : Optional[GeodesicMapCompiler] = None
    arc_weights: Optional[ARCWeightMatrix]     = None
    geo_temperature: float = 1.0

    def _attn_geodesic(self, x: torch.Tensor,
                        token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
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
          f"arc_pairs={arc_weights.stats()['active_pairs']}")
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
        print(f"[ARCWeights] Loaded: {arc_weights.stats()}")
    else:
        print(f"[ARCWeights] Starting fresh (will accumulate from training)")

    return geo_map, arc_weights
