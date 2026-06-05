#!/usr/bin/env python3
"""
SMGM-16 checkpoint-compatible model class.

This module matches the structure observed in the saved checkpoints:
- token_embedding:   [1024, 768]
- position_embedding:[1024, 768]
- maya_embedding:    15 -> 192 -> 768
- phase_embedding:   1 -> 96 -> 768
- cards:             52 card modules with field_u/field_s/field_v and scalar
                      geometric tensors
- layers:            6 attention blocks
- norms:             6 pre-attention norms
- ffn_norms:         6 pre-FFN norms
- ffns:              6 MLP blocks
- ln_f:              final norm
- output_proj:       768 -> 1024 logits projection

The class is intentionally tolerant of the stale legacy constructor arguments
used by the older scripts in this tree. Those arguments are accepted, but the
actual checkpoint architecture is the source of truth.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_hidden(x: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Project or pad a tensor to the checkpoint hidden size without new params."""
    if x.size(-1) == hidden_size:
        return x
    if x.size(-1) > hidden_size:
        return x[..., :hidden_size]
    return F.pad(x, (0, hidden_size - x.size(-1)))


class CardSlot(nn.Module):
    """
    One learnable card slot from the checkpoint.

    Shapes are chosen to match the saved state dict exactly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.field_u = nn.Parameter(torch.zeros(16, 192))
        self.field_s = nn.Parameter(torch.zeros(192))
        self.field_v = nn.Parameter(torch.zeros(192, 768))
        self.amplitude = nn.Parameter(torch.zeros(4))
        self.gradient = nn.Parameter(torch.zeros(4))
        self.curvature = nn.Parameter(torch.zeros(1))
        self.pi_mod = nn.Parameter(torch.zeros(1))
        self.adjacency_strength = nn.Parameter(torch.zeros(8))
        self.register_buffer("maya_digit", torch.zeros(1, dtype=torch.long))
        self.register_buffer("adjacency", torch.zeros(8, dtype=torch.long))

    def summary_u(self) -> torch.Tensor:
        return self.field_u.mean(dim=0)

    def summary_v(self) -> torch.Tensor:
        return self.field_v.mean(dim=0)

    def scalar_bias(self) -> torch.Tensor:
        return (
            self.field_s.mean()
            + self.amplitude.mean()
            + self.gradient.mean()
            + self.curvature.mean()
            + self.pi_mod.mean()
            + self.adjacency_strength.mean()
        )


class AttentionBlock(nn.Module):
    """Attention block with the exact parameter surface used by the checkpoint."""

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.phase_scale = nn.Parameter(torch.ones(1))
        self.sigma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scale = self.phase_scale.clamp(min=1e-6) / math.sqrt(hidden)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            mask = attn_mask[:, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out) * self.sigma


class SMGM16(nn.Module):
    """
    Checkpoint-compatible SMGM-16 model.

    Legacy constructor arguments are accepted for compatibility with the older
    scripts in this repo, but the actual architecture is fixed by the checkpoint
    layout.
    """

    vocab_size: int = 1024
    hidden_size: int = 768
    max_positions: int = 1024
    num_layers: int = 6
    num_cards: int = 52

    def __init__(
        self,
        d_model: int = 768,
        layers: int = 6,
        stage_dims: tuple[int, ...] = (4, 4),
        k: int = 4,
        patch_dim: int = 64,
        *,
        vocab_size: int = 1024,
        hidden_size: int | None = None,
        max_positions: int = 1024,
        num_layers: int | None = None,
        num_cards: int = 52,
        maya_dim: int = 15,
        maya_hidden: int = 192,
        phase_hidden: int = 96,
    ) -> None:
        super().__init__()

        # The checkpoint is the source of truth.
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size or self.hidden_size)
        self.max_positions = int(max_positions)
        self.num_layers = int(num_layers or self.num_layers)
        self.num_cards = int(num_cards)
        self.maya_dim = int(maya_dim)
        self.maya_hidden = int(maya_hidden)
        self.phase_hidden = int(phase_hidden)

        # Legacy fields are retained for caller compatibility.
        self.legacy_d_model = d_model
        self.legacy_layers = layers
        self.legacy_stage_dims = tuple(stage_dims)
        self.legacy_k = k
        self.legacy_patch_dim = patch_dim

        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_positions, self.hidden_size)

        self.maya_embedding = nn.Sequential(
            nn.Linear(self.maya_dim, self.maya_hidden),
            nn.GELU(),
            nn.Linear(self.maya_hidden, self.hidden_size),
        )
        self.phase_embedding = nn.Sequential(
            nn.Linear(1, self.phase_hidden),
            nn.GELU(),
            nn.Linear(self.phase_hidden, self.hidden_size),
        )

        self.cards = nn.ModuleList(CardSlot() for _ in range(self.num_cards))
        self.norms = nn.ModuleList(nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers))
        self.layers = nn.ModuleList(AttentionBlock(self.hidden_size) for _ in range(self.num_layers))
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers))
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hidden_size, 3072),
                nn.GELU(),
                nn.Linear(3072, self.hidden_size),
            )
            for _ in range(self.num_layers)
        )

        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.pi_time = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Checkpoint compatibility helpers
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict: Any, strict: bool = True):  # type: ignore[override]
        """
        Accept either a raw state dict or the Lightning-style checkpoint dict
        with a nested "model" payload.
        """
        if isinstance(state_dict, dict):
            if isinstance(state_dict.get("model"), dict):
                state_dict = state_dict["model"]
            elif isinstance(state_dict.get("state_dict"), dict):
                state_dict = state_dict["state_dict"]
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        **kwargs: Any,
    ) -> SMGM16:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = cls(**kwargs)
        model.load_state_dict(checkpoint, strict=strict)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------
    def _token_features(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_f = token_ids.float()
        pos_f = position_ids.float()

        denom_tok = max(self.vocab_size - 1, 1)
        denom_pos = max(self.max_positions - 1, 1)

        frac = token_f / denom_tok
        pos_frac = pos_f / denom_pos

        features = [
            frac,
            frac.square(),
            torch.sqrt(frac.clamp_min(0.0) + 1e-6),
            torch.sin(math.pi * frac),
            torch.cos(math.pi * frac),
            torch.sin(2.0 * math.pi * frac),
            torch.cos(2.0 * math.pi * frac),
            pos_frac,
            pos_frac.square(),
            torch.sin(math.pi * pos_frac),
            torch.cos(math.pi * pos_frac),
            torch.remainder(token_f, 3.0) / 2.0,
            torch.remainder(token_f, 5.0) / 4.0,
            torch.remainder(token_f, 7.0) / 6.0,
            torch.remainder(token_f + pos_f + self.pi_time, 2.0),
        ]
        return torch.stack(features, dim=-1)

    def _continuous_features(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden_mean = x.mean(dim=-1)
        hidden_std = x.std(dim=-1, unbiased=False)
        hidden_abs = x.abs().mean(dim=-1)
        hidden_energy = x.pow(2).mean(dim=-1).sqrt()
        hidden_max = x.amax(dim=-1)
        hidden_min = x.amin(dim=-1)
        pos_f = position_ids.float() / max(self.max_positions - 1, 1)

        features = [
            hidden_mean,
            hidden_std,
            hidden_abs,
            hidden_energy,
            hidden_max,
            hidden_min,
            torch.sin(hidden_mean),
            torch.cos(hidden_mean),
            torch.sin(hidden_std),
            torch.cos(hidden_std),
            pos_f,
            pos_f.square(),
            torch.sin(math.pi * pos_f),
            torch.cos(math.pi * pos_f),
            torch.tanh(hidden_mean + self.pi_time),
        ]
        return torch.stack(features, dim=-1)

    def _phase_signal(self, token_ids: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            signal = token_ids.float() / max(self.vocab_size - 1, 1)
        else:
            if x is None:
                signal = token_ids.float()
            else:
                signal = x.mean(dim=-1)
        return signal.unsqueeze(-1)

    def _card_context(self, hidden: torch.Tensor, maya_latent: torch.Tensor, phase_latent: torch.Tensor):
        pooled_hidden = hidden.mean(dim=1)
        pooled_maya = maya_latent.mean(dim=1)[..., :192]
        pooled_phase = phase_latent.mean(dim=(1, 2))

        scores = []
        vectors = []
        for card in self.cards:
            card_u = card.summary_u()
            card_v = card.summary_v()
            score = (
                (pooled_hidden * card_v).sum(dim=-1)
                + (pooled_maya * card_u).sum(dim=-1)
                + card.scalar_bias().unsqueeze(0)
                + 0.05 * pooled_phase
                + 0.01 * card.maya_digit.float().mean()
                + 0.01 * card.adjacency.float().mean()
            )
            scores.append(score)
            vectors.append(card_v)

        card_logits = torch.stack(scores, dim=-1)
        card_gates = torch.softmax(card_logits, dim=-1)
        card_vectors = torch.stack(vectors, dim=0)  # [cards, hidden]
        context = torch.einsum("bc,ch->bh", card_gates, card_vectors)
        return card_gates, context

    def _stage_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        stats = torch.stack(
            [
                pooled.mean(dim=-1),
                pooled.std(dim=-1, unbiased=False),
                pooled.abs().mean(dim=-1),
                pooled.amax(dim=-1),
                pooled.amin(dim=-1),
                torch.sigmoid(self.pi_time).expand(pooled.size(0)),
            ],
            dim=-1,
        )
        return torch.softmax(stats, dim=-1)

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """
        Forward pass.

        Accepts either integer token IDs [B, S] or float inputs [B, S, D].
        Returns:
            logits [B, S, vocab]
            layer_gates: list of [B, num_cards]
            stage_probs: list of [B, 6]
        """
        if input_ids.dim() == 2 and input_ids.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
        ):
            token_ids = input_ids.clamp_(0, self.vocab_size - 1)
            batch, seq_len = token_ids.shape
            position_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, -1)
            hidden = self.token_embedding(token_ids) + self.position_embedding(position_ids % self.max_positions)
            maya_features = self._token_features(token_ids, position_ids)
            phase_signal = self._phase_signal(token_ids)
        elif input_ids.dim() == 3:
            hidden = _as_hidden(input_ids.float(), self.hidden_size)
            batch, seq_len, _ = hidden.shape
            position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(batch, -1)
            hidden = hidden + self.position_embedding(position_ids % self.max_positions)
            maya_features = self._continuous_features(hidden, position_ids)
            phase_signal = self._phase_signal(hidden, x=hidden)
        else:
            raise ValueError(
                "SMGM16 expects token IDs [batch, seq] or feature tensors [batch, seq, dim]."
            )

        maya_latent = self.maya_embedding(maya_features)
        phase_latent = self.phase_embedding(phase_signal)
        hidden = hidden + maya_latent + phase_latent

        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)

        layer_gates: list[torch.Tensor] = []
        stage_probs: list[torch.Tensor] = []

        for idx in range(self.num_layers):
            attn_input = self.norms[idx](hidden)
            attn_out = self.layers[idx](attn_input, attn_mask=attention_mask)
            hidden = hidden + attn_out

            card_gates, card_context = self._card_context(hidden, maya_latent, phase_latent)
            hidden = hidden + card_context.unsqueeze(1) * torch.sigmoid(self.pi_time)

            ffn_input = self.ffn_norms[idx](hidden)
            hidden = hidden + self.ffns[idx](ffn_input)

            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)

            layer_gates.append(card_gates)
            stage_probs.append(self._stage_probs(hidden))

        hidden = self.ln_f(hidden)
        logits = self.output_proj(hidden)

        if return_hidden:
            return logits, layer_gates, stage_probs, hidden
        return logits, layer_gates, stage_probs


def training_step(model: SMGM16, batch, optimizer=None, lambdas=None):
    """
    Compatibility training helper for the legacy scripts.

    The old scripts feed float tensors and compare them with MSE. The real model
    emits logits, so we compare a slice of the logits to the target width when
    needed.
    """
    x, y = batch
    out, layer_gates, stage_probs = model(x)

    if y.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        flat_targets = y.reshape(-1)
        valid_mask = flat_targets != -100
        if torch.any(valid_mask):
            loss_task = F.cross_entropy(
                out.reshape(-1, out.size(-1))[valid_mask],
                flat_targets[valid_mask],
            )
        else:
            loss_task = torch.zeros((), device=out.device, dtype=out.dtype)
    else:
        target = y
        if target.dim() == out.dim() and target.size(-1) != out.size(-1):
            target = _as_hidden(target, out.size(-1))
        elif target.dim() == out.dim() - 1:
            target = target.unsqueeze(-1).expand_as(out[..., :1])
        loss_task = F.mse_loss(out[..., : target.size(-1)], target[..., : out.size(-1)])

    if layer_gates:
        usage = torch.stack(layer_gates).mean(dim=(0, 1))
        loss_balance = usage.var()
    else:
        loss_balance = torch.tensor(0.0, device=out.device)

    if stage_probs:
        stage_balance_terms = []
        stage_entropy_terms = []
        for probs in stage_probs:
            probs = probs.float().clamp_min(1e-9)
            stage_balance_terms.append(probs.mean(dim=0).var())
            stage_entropy_terms.append(-(probs * probs.log()).sum(dim=-1).mean())
        stage_balance = torch.stack(stage_balance_terms).mean()
        stage_entropy = torch.stack(stage_entropy_terms).mean()
    else:
        stage_balance = torch.tensor(0.0, device=out.device)
        stage_entropy = torch.tensor(0.0, device=out.device)

    lambdas = lambdas or {}
    loss = (
        loss_task
        + float(lambdas.get("balance", 0.1)) * loss_balance
        + float(lambdas.get("stage_balance", 0.05)) * stage_balance
        + float(lambdas.get("stage_entropy", 0.02)) * stage_entropy
    )

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return float(loss.item())


def load_model(checkpoint_path: str | Path, device: str = "cpu", **model_kwargs) -> SMGM16:
    model = SMGM16(**model_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    checkpoints = [
        root / "checkpoints" / "checkpoint_epoch_10.pt",
        root / "checkpoints" / "model_fp32.pt",
    ]
    for checkpoint in checkpoints:
        if checkpoint.exists():
            model = load_model(checkpoint, device="cpu")
            print(f"{checkpoint.name}: loaded {model.__class__.__name__}")
            break
    else:
        model = SMGM16()
        print(model.__class__.__name__, "initialized")
