// K'UHUL++ Phase Manager
// Controls the geometric execution phase — a value in [0, 2π] that represents
// the current position in the cyclic π-geometry execution substrate.
// The phase influences spherical coordinate transforms and wave operations.

// ------------------------------------------------------------------ //
// PhaseManager
// ------------------------------------------------------------------ //

const TWO_PI = 2 * Math.PI;

/**
 * Manages the execution phase of a K'UHUL++ program.
 * The phase is a continuous angle in [0, 2π) that advances as the program
 * executes glyph operations involving the manifold M.
 *
 * @example
 * const pm = new PhaseManager();
 * pm.advance(Math.PI / 4);  // advance by 45°
 * console.log(pm.getCurrentPhase());  // → 0.7854...
 */
export class PhaseManager {
    private phase:    number;
    private readonly initialPhase: number;
    private history:  number[] = [];

    /**
     * @param initialPhase - Starting phase in radians (default 0)
     */
    constructor(initialPhase = 0) {
        this.initialPhase = initialPhase % TWO_PI;
        this.phase        = this.initialPhase;
    }

    // ---- Getters ----

    /** Current phase angle in radians, always in [0, 2π) */
    getCurrentPhase(): number { return this.phase; }

    /** Current phase angle in degrees, always in [0°, 360°) */
    getCurrentPhaseDegrees(): number { return (this.phase * 180) / Math.PI; }

    /** Number of full 2π cycles completed since the last reset */
    getCycleCount(): number { return this.history.filter(d => d >= TWO_PI).length; }

    // ---- Mutation ----

    /**
     * Advance the phase by `delta` radians (may be negative for reverse).
     * The result is wrapped to [0, 2π).
     *
     * @param delta - Phase increment in radians
     * @returns New phase value
     */
    advance(delta: number): number {
        this.history.push(delta);
        this.phase = ((this.phase + delta) % TWO_PI + TWO_PI) % TWO_PI;
        return this.phase;
    }

    /**
     * Set the phase to an absolute value, wrapping to [0, 2π).
     *
     * @param value - Target phase in radians
     */
    setPhase(value: number): void {
        this.phase = ((value % TWO_PI) + TWO_PI) % TWO_PI;
    }

    /** Reset the phase to its initial value and clear history */
    reset(): void {
        this.phase   = this.initialPhase;
        this.history = [];
    }

    // ---- Interpolation ----

    /**
     * Linearly interpolate the phase toward a target value by `t` ∈ [0, 1].
     * Uses the shortest angular path (handles wrap-around).
     *
     * @param target - Target phase in radians
     * @param t      - Interpolation factor in [0, 1]
     */
    lerpTo(target: number, t: number): number {
        const from = this.phase;
        const to   = ((target % TWO_PI) + TWO_PI) % TWO_PI;
        let diff   = to - from;
        // Take shortest path
        if (diff >  Math.PI) diff -= TWO_PI;
        if (diff < -Math.PI) diff += TWO_PI;
        this.phase = ((from + diff * t) % TWO_PI + TWO_PI) % TWO_PI;
        return this.phase;
    }

    // ---- Serialisation ----

    /** Snapshot the phase state */
    snapshot(): { phase: number; history: number[] } {
        return { phase: this.phase, history: [...this.history] };
    }

    /** Restore from a snapshot */
    restore(snap: { phase: number; history: number[] }): void {
        this.phase   = snap.phase;
        this.history = [...snap.history];
    }
}
