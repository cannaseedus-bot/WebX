// mayan-orchestrator.js — Mayan Math Horizontal Semantic Orchestrator
//
// 8 horizontal folds with linear + geodesic entropy and π phase rotation.
//
// Key math:
//   Linear entropy  : S = -k Σ p_i log₁₃(p_i)         (Mayan base-13)
//   Geodesic entropy: S_geo = k(A·κ + P·L)              (hyperbolic surface)
//   Phase rotation  : ψ(θ) = e^(iθ) = cos θ + i·sin θ  (pentad: 72° steps)
//   Fold coherence  : C = |cos(Δφ/2)|                   (φ = n·π/4 per fold)
//   π-fold compress : R = π/(S+1)                       (compression ratio)
//
// Folds ↔ KXML phases:
//   fold_0(θ=0°)   ≡ Pop    fold_4(θ=180°) ≡ Ch'en
//   fold_1(θ=45°)  ≡ Wo     fold_5(θ=225°) ≡ Xul
//   fold_2(θ=90°)  ≡ Sek    fold_6(θ=270°) ≡ Pop(next)
//   fold_3(θ=135°) ≡ Attn   fold_7(θ=315°) ≡ Wo(next)

// ─── Mayan constants ─────────────────────────────────────────────────────────

export const MAYAN = Object.freeze({
  tzolkin:         260,       // sacred cycle
  haab:            365,       // solar cycle
  calendar_round:  18980,     // 52-year LCM
  baktun:          13,
  venus:           584,
  eclipse:         405,

  pi:   Math.PI,
  phi:  (1 + Math.sqrt(5)) / 2,
  e:    Math.E,
  sqrt2: Math.SQRT2,

  // Sacred proportions
  sacred_ratio:    13 / 8,    // 1.625
  pyramid_angle:   51.827,    // Chichen Itza
  equinox_ratio:   0.728,
  zenith_angle:    22.5,

  // Phase angles
  pentad:  [0, 72, 144, 216, 288],
  octad:   [0, 45, 90, 135, 180, 225, 270, 315],

  // Entropy
  entropy_base:    13,
  geodesic_kappa:  0.072,
  fold_count:      8,
});

// ─── Linear entropy fold (1D) ─────────────────────────────────────────────────

export class LinearEntropyFold {
  static entropy(probs, foldIndex = 0) {
    let s = 0;
    for (const p of probs) if (p > 0) s -= p * Math.log(p) / Math.log(MAYAN.entropy_base);
    return s * LinearEntropyFold.modulation(foldIndex);
  }

  static modulation(foldIndex) {
    const x = (foldIndex - 3.5) * 10;        // -35 to +35
    return 0.5 + (x + 40) / 80;
  }

  static position(foldIndex) {
    return (foldIndex - 3.5) * 10;
  }
}

// ─── Geodesic entropy fold (2D hyperbolic) ────────────────────────────────────

export class GeodesicEntropyFold {
  static area(r) {
    return Math.PI * (Math.exp(2 * r) - Math.exp(-2 * r)) / 4;
  }

  static perimeter(r) {
    return 2 * Math.PI * Math.sinh(r);
  }

  static distance(r1, r2, deltaTheta) {
    return Math.acosh(
      Math.cosh(r1) * Math.cosh(r2) -
      Math.sinh(r1) * Math.sinh(r2) * Math.cos(deltaTheta)
    );
  }

  static entropy(r, foldIndex = 0) {
    const area  = GeodesicEntropyFold.area(r);
    const perim = GeodesicEntropyFold.perimeter(r);
    const path  = GeodesicEntropyFold.distance(r, r, Math.PI / 4);
    const theta = (foldIndex * 45 * Math.PI) / 180;
    const mod   = 1 + MAYAN.geodesic_kappa * Math.sin(2 * theta);
    return (area * MAYAN.geodesic_kappa + perim * path) * mod;
  }
}

// ─── π Phase rotation ─────────────────────────────────────────────────────────

export class PiPhase {
  static rotate(amplitude, phase, rotation) {
    const p = phase + rotation;
    return { real: amplitude * Math.cos(p), imag: amplitude * Math.sin(p), phase: p, amplitude };
  }

  static pentad(amp, idx) {
    return PiPhase.rotate(amp, 0, (idx * 72 * Math.PI) / 180);
  }

  static octad(amp, idx) {
    return PiPhase.rotate(amp, 0, (idx * 45 * Math.PI) / 180);
  }

  static coherence(phaseA, phaseB) {
    return Math.abs(Math.cos((phaseA - phaseB) / 2));
  }

  static interfere(waves) {
    let sr = 0, si = 0;
    for (const w of waves) { sr += w.real; si += w.imag; }
    const amp   = Math.sqrt(sr * sr + si * si);
    const phase = Math.atan2(si, sr);
    return { real: sr, imag: si, amplitude: amp, phase,
             constructive: amp > waves.length / 2,
             destructive:  amp < waves.length / 4 };
  }

  static accumulatePiFold(foldIndex, steps) {
    let w = { real: 1, imag: 0, phase: 0, amplitude: 1 };
    for (let s = 0; s < steps; s++) {
      w = PiPhase.rotate(w.amplitude, w.phase, Math.PI + foldIndex * Math.PI / 4);
    }
    return w;
  }
}

// ─── Complete 8-fold orchestrator ─────────────────────────────────────────────

export class MayanOrchestrator {
  constructor() {
    this.tzolkin = 0;
    this.haab    = 0;
    this.baktun  = MAYAN.baktun;
  }

  async orchestrate(job) {
    this._advanceCalendar();

    const initEntropy  = this._baseEntropy(job.probabilities ?? [0.5, 0.3, 0.2]);
    const rotation     = initEntropy * MAYAN.pi;
    const phasedJob    = { ...job, phase: PiPhase.rotate(1, 0, rotation), entropy: initEntropy };

    const waves = [];
    const foldResults = {};

    for (let i = 0; i < MAYAN.fold_count; i++) {
      const linE  = LinearEntropyFold.entropy([job.uncertainty ?? 0.5], i);
      const geoE  = GeodesicEntropyFold.entropy(Math.log(1 + (job.complexity ?? 1)), i);
      const wave  = PiPhase.pentad(1, i % 5);
      const lw    = Math.abs(Math.cos(wave.phase));
      const gw    = Math.abs(Math.sin(wave.phase));

      const combined = { entropy: linE * lw + geoE * gw, wave };
      waves.push(combined);
      foldResults[`fold_${i}`] = combined;
    }

    const interference = PiPhase.interfere(waves.map(w => w.wave));
    const coherence    = this._totalCoherence(waves.map(w => w.wave.phase));

    const compressionRatio = MAYAN.pi / (initEntropy + 1);

    return {
      '@type': 'mayan_semantic_output',
      '@tzolkin': this.tzolkin,
      '@haab':    this.haab,
      '@baktun':  this.baktun,
      folds:      foldResults,
      interference,
      coherence,
      compression: {
        ratio:            compressionRatio,
        entropy_reduction: initEntropy * compressionRatio,
        fold_coherence:   interference.constructive ? 'COHERENT' : 'ENTANGLED',
        semantic_density: 1 / (compressionRatio + 0.001),
      },
      next_folds: this._suggestFolds(initEntropy),
    };
  }

  // ─── Private ───────────────────────────────────────────────────────────────

  _advanceCalendar() {
    this.tzolkin = (this.tzolkin + 1) % MAYAN.tzolkin;
    this.haab    = (this.haab    + 1) % MAYAN.haab;
    if (this.tzolkin === 0 && this.haab === 0) this.baktun++;
  }

  _baseEntropy(probs) {
    let s = 0;
    for (const p of probs) if (p > 0) s -= p * Math.log(p) / Math.log(MAYAN.entropy_base);
    return s;
  }

  _totalCoherence(phases) {
    let sum = 0, n = 0;
    for (let i = 0; i < phases.length; i++)
      for (let j = i + 1; j < phases.length; j++) {
        sum += PiPhase.coherence(phases[i], phases[j]);
        n++;
      }
    return n > 0 ? sum / n : 0;
  }

  _suggestFolds(entropy) {
    const angle = ((entropy * MAYAN.pi * 180) / MAYAN.pi) % 360;
    const idx   = Math.floor(angle / 45);
    return [`fold_${idx}`, `fold_${(idx + 1) % 8}`];
  }
}

// ─── Coherence matrix (8×8) ───────────────────────────────────────────────────

export function coherenceMatrix() {
  return Array.from({ length: 8 }, (_, i) =>
    Array.from({ length: 8 }, (__, j) =>
      Math.abs(Math.cos((Math.abs(i - j) * Math.PI / 4) / 2))
    )
  );
}
