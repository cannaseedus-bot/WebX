"""
mayan_matrix_math.py — Mayan MatrixMath: tensors are living fields that evolve.

The model doesn't learn. It AGES.

Each forward pass = one mutation event = one "day" in the model's life.
Each 1000 steps = version boundary = New Year's Day.
SafeTensors = fossil records of evolution.

K'UHUL phase mapping:
    Pop   = load FluxTensor state (snapshot restore)
    Wo    = project with evolving Q/K/V weights
    Sek   = attention + carry field update
    Ch'en = mutate all tensors (the aging step)
    Xul   = save version snapshot

Attractor types (FLUXTensor):
    lorenz   sigma=10  rho=28   beta=8/3   — chaotic, sensitive to init
    rossler  a=0.2     b=0.2    c=5.7      — scroll-type chaos
    logistic r=3.9                          — 1D discrete chaos map
"""
from __future__ import annotations

import math
import pathlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── FluxTensor ───────────────────────────────────────────────────────────────

class FluxTensor(nn.Parameter):
    """A tensor that evolves with each forward pass. Not static — a living field."""

    mutation_rate: float
    half_life:     int
    creation_date: datetime
    mutation_count: int

    def __new__(cls, data: torch.Tensor, mutation_rate: float = 0.01, half_life: int = 365):
        instance = super().__new__(cls, data)
        instance.mutation_rate  = mutation_rate
        instance.half_life      = half_life
        instance.creation_date  = datetime.now()
        instance.mutation_count = 0
        return instance

    def mutate(self, gradient: torch.Tensor | None = None) -> FluxTensor:
        self.mutation_count += 1
        age_days     = (datetime.now() - self.creation_date).days
        decay_factor = 0.5 ** (age_days / max(1, self.half_life))
        with torch.no_grad():
            if gradient is not None:
                self.data.add_(gradient * self.mutation_rate * decay_factor)
            else:
                noise = torch.randn_like(self.data) * self.mutation_rate * decay_factor
                self.data.add_(noise)
        return self

# ─── FLUXTensor (chaotic attractor evolution) ─────────────────────────────────

class FLUXTensor(nn.Parameter):
    """FLUX = Forward Learning with Universal X-evolution.
    Evolves via chaotic attractor dynamics instead of gradients."""

    def __new__(cls, data: torch.Tensor, attractor: str = 'lorenz'):
        instance = super().__new__(cls, data)
        instance._attractor = attractor
        instance._state     = torch.zeros(3)
        return instance

    def evolve(self, steps: int = 1) -> FLUXTensor:
        s = self._state
        for _ in range(steps):
            if self._attractor == 'lorenz':
                sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
                dx = sigma * (s[1] - s[0])
                dy = s[0] * (rho - s[2]) - s[1]
                dz = s[0] * s[1] - beta * s[2]
                s  = s + torch.stack([dx, dy, dz]) * 0.01
            elif self._attractor == 'rossler':
                a, b, c = 0.2, 0.2, 5.7
                dx = -s[1] - s[2]
                dy =  s[0] + a * s[1]
                dz =  b    + s[2] * (s[0] - c)
                s  = s + torch.stack([dx, dy, dz]) * 0.01
            elif self._attractor == 'logistic':
                r  = 3.9
                s[0] = r * s[0] * (1.0 - s[0])
                s[1] = r * s[1] * (1.0 - s[1])
                s[2] = (s[0] + s[1]) / 2.0
        self._state = s
        with torch.no_grad():
            self.data.add_(s.view(1, 1, 3).expand_as(self.data[:1, :1, :3]).reshape(-1)[:self.data.numel()].reshape(self.data.shape) * 0.001)
        return self

# ─── MayanEvolutionLayer ──────────────────────────────────────────────────────

class MayanEvolutionLayer(nn.Module):
    """A transformer layer where attention IS the mutation mechanism.
    The carry_field is a living memory that evolves with every interaction."""

    def __init__(self, dim: int, evolution_rate: float = 0.01):
        super().__init__()
        self.evolution_rate = evolution_rate
        self.q_weight    = FluxTensor(torch.randn(dim, dim) / dim, mutation_rate=evolution_rate)
        self.k_weight    = FluxTensor(torch.randn(dim, dim) / dim, mutation_rate=evolution_rate)
        self.v_weight    = FluxTensor(torch.randn(dim, dim) / dim, mutation_rate=evolution_rate)
        self.carry_field = FluxTensor(torch.zeros(1, 1, dim),     mutation_rate=evolution_rate * 0.1)
        self.norm        = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        Q = x @ self.q_weight
        K = x @ self.k_weight
        V = x @ self.v_weight

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(x.shape[-1])
        scores = scores + self.carry_field.mean() * 0.1
        attn   = F.softmax(scores, dim=-1)
        out    = attn @ V

        self.carry_field.mutate(out.mean(dim=1, keepdim=True))

        x = self.norm(x + out + self.carry_field * 0.1)

        self.q_weight.mutate()
        self.k_weight.mutate()
        self.v_weight.mutate()

        return x, self.carry_field.data.mean().item()

# ─── MayanMatrixMath ──────────────────────────────────────────────────────────

class MayanMatrixMath(nn.Module):
    """A chat model where EVERY tensor is a FluxTensor that evolves.
    The model doesn't learn — it AGES."""

    def __init__(self, dim: int = 768, vocab_size: int = 50272, num_layers: int = 12):
        super().__init__()
        self.dim           = dim
        self.version       = '1.0.0'
        self.creation_date = datetime.now()

        self.token_embed = FluxTensor(torch.randn(vocab_size, dim) / dim)
        self.pos_embed   = FluxTensor(torch.randn(8192, dim)       / dim)

        self.layers = nn.ModuleList([
            MayanEvolutionLayer(dim, evolution_rate=0.01 * (1 + i / num_layers))
            for i in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(dim)
        self.lm_head     = nn.Linear(dim, vocab_size, bias=False)

        self.register_buffer('evolution_step', torch.tensor(0, dtype=torch.long))
        self._snapshot_dir = pathlib.Path('mayan_versions')
        self._snapshot_dir.mkdir(exist_ok=True)

    def forward(self, input_ids: torch.Tensor, return_evolution: bool = False):
        self.evolution_step += 1
        seq_len = input_ids.shape[1]

        x = (self.token_embed[input_ids]
             + self.pos_embed[torch.arange(seq_len, device=input_ids.device)])

        evolution_trace = []
        for layer in self.layers:
            x, carry = layer(x)
            evolution_trace.append(carry)

        logits = self.lm_head(self.output_norm(x))
        self._evolve_tensors()

        if return_evolution:
            return logits, evolution_trace
        return logits

    def _evolve_tensors(self):
        self.token_embed.mutate()
        self.pos_embed.mutate()
        if self.evolution_step % 1000 == 0:
            self._increment_version()

    def _increment_version(self):
        maj, mn, patch = (int(x) for x in self.version.split('.'))
        patch += 1
        if patch >= 10:
            patch = 0; mn += 1
        if mn >= 10:
            mn = 0; maj += 1
        self.version = f'{maj}.{mn}.{patch}'
        print(f'[mayan] evolved to version {self.version}  step={self.evolution_step.item()}')
        self.save_version_snapshot()

    def save_version_snapshot(self):
        path = self._snapshot_dir / f'mayan_v{self.version}_step{self.evolution_step.item()}.safetensors'
        torch.save({'version': self.version, 'step': self.evolution_step.item(),
                    'state': {k: v.cpu() for k, v in self.state_dict().items()},
                    'timestamp': datetime.now().isoformat()}, path)

    def total_mutations(self) -> int:
        return sum(
            getattr(p, 'mutation_count', 0)
            for p in self.parameters()
        )

# ─── MayanVersionCalendar ─────────────────────────────────────────────────────

class MayanVersionCalendar:
    """Tracks model evolution across time. Each 'year' = version boundary."""

    def __init__(self, model: MayanMatrixMath):
        self.model         = model
        self.version_history: list = []
        self.birth_date    = model.creation_date

    def simulate_years(self, years: int, steps_per_day: int = 10,
                       device: str = 'cpu', dim: int = 32):
        total_steps = years * 365 * steps_per_day
        model       = self.model.to(device)
        print(f'[calendar] born {self.birth_date}')
        print(f'[calendar] simulating {years} years ({total_steps} steps)')

        for step in range(total_steps):
            dummy = torch.randint(0, 1000, (1, dim), device=device)
            with torch.no_grad():
                model(dummy)

            if step > 0 and step % (365 * steps_per_day) == 0:
                year = step // (365 * steps_per_day)
                snap = {'year': year, 'version': model.version,
                        'step': model.evolution_step.item(), 'ts': datetime.now().isoformat()}
                self.version_history.append(snap)
                print(f'[calendar] year {year}  version {model.version}  '
                      f'mutations {model.total_mutations()}')

        return self.version_history

# ─── MayanVersionDay ─────────────────────────────────────────────────────────

class MayanVersionDay:
    """New Year's Day ritual for the model."""

    @staticmethod
    def celebrate(model: MayanMatrixMath, year: int):
        print(f'\n[version-day] Year {year} | {model.version} | '
              f'step {model.evolution_step.item()} | '
              f'mutations {model.total_mutations()}')
        model.save_version_snapshot()
        for p in model.parameters():
            if hasattr(p, 'mutation_count'):
                p.mutation_count = 0

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Mayan MatrixMath — tensors that age')
    ap.add_argument('--dim',        type=int, default=256,   help='hidden dim')
    ap.add_argument('--layers',     type=int, default=4,     help='num layers')
    ap.add_argument('--vocab',      type=int, default=50272, help='vocab size')
    ap.add_argument('--years',      type=int, default=10,    help='years to simulate')
    ap.add_argument('--steps_day',  type=int, default=10,    help='steps per day')
    ap.add_argument('--device',     default='cpu')
    args = ap.parse_args()

    model    = MayanMatrixMath(dim=args.dim, vocab_size=args.vocab, num_layers=args.layers)
    calendar = MayanVersionCalendar(model)
    history  = calendar.simulate_years(years=args.years, steps_per_day=args.steps_day, device=args.device)

    print(f'\n[mayan] final version:  {model.version}')
    print(f'[mayan] total mutations: {model.total_mutations()}')
    print(f'[mayan] version history: {len(history)} snapshots')
