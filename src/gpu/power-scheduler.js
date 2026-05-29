// Power Scheduler v1 — Intel iGPU power-state constants + adaptive scheduling rules
//
// Intel GPU P-states: P0 (max performance) → P3 (min power) → RC6 (idle park).
// Power state determines tile size, thread count, and Mayan base precision.
// Thermal override applies when temp >= 90°C (force P3).

// ─── Power state definitions ──────────────────────────────────────────────────

export const POWER_STATES = Object.freeze({
  P0: Object.freeze({
    id:          'P0',
    freqMHz:     1200,
    voltageV:    1.2,
    powerW:      45,
    performance: 1.0,
    tileSize:    16,
    threadCount: 140,
    mayanBase:   20,
    desc:        'Full performance — battery > 50% && temp < 80°C',
  }),
  P1: Object.freeze({
    id:          'P1',
    freqMHz:     900,
    voltageV:    1.0,
    powerW:      28,
    performance: 0.75,
    tileSize:    12,
    threadCount: 84,
    mayanBase:   16,
    desc:        'Balanced — battery > 20% && temp < 85°C',
  }),
  P2: Object.freeze({
    id:          'P2',
    freqMHz:     600,
    voltageV:    0.8,
    powerW:      15,
    performance: 0.5,
    tileSize:    8,
    threadCount: 56,
    mayanBase:   12,
    desc:        'Power saving — battery > 5% && temp < 90°C',
  }),
  P3: Object.freeze({
    id:          'P3',
    freqMHz:     350,
    voltageV:    0.7,
    powerW:      8,
    performance: 0.3,
    tileSize:    4,
    threadCount: 28,
    mayanBase:   8,
    desc:        'Minimum power — battery <= 5% || temp >= 90°C',
  }),
  RC6: Object.freeze({
    id:          'RC6',
    freqMHz:     0,
    voltageV:    0.0,
    powerW:      0.1,
    performance: 0,
    tileSize:    0,
    threadCount: 0,
    mayanBase:   0,
    desc:        'GPU parked — idle > 5 seconds',
  }),
});

// Ordered from highest to lowest performance (for selection logic)
export const POWER_STATE_ORDER = Object.freeze(['P0', 'P1', 'P2', 'P3', 'RC6']);

// Mayan precision config per power state
// When running in degraded power states, reduce Mayan base to lighten compute
export const MAYAN_PRECISION_BY_STATE = Object.freeze({
  P0:  { base: 20, digits: 5, label: 'full Long Count' },
  P1:  { base: 16, digits: 4, label: 'reduced Long Count' },
  P2:  { base: 12, digits: 3, label: 'tun-level precision' },
  P3:  { base: 8,  digits: 2, label: 'uinal-level precision' },
  RC6: { base: 0,  digits: 0, label: 'suspended' },
});

// ─── Policy selection ─────────────────────────────────────────────────────────

/**
 * Select a power state from telemetry readings.
 * batteryPct: 0–100, tempC: degrees Celsius, idleSecs: seconds since last GPU work.
 */
export function selectPowerState({ batteryPct = 100, tempC = 60, idleSecs = 0 } = {}) {
  if (idleSecs > 5) return POWER_STATES.RC6;
  if (tempC >= 90 || batteryPct <= 5) return POWER_STATES.P3;
  if (tempC >= 85 || batteryPct <= 20) return POWER_STATES.P2;
  if (tempC >= 80 || batteryPct <= 50) return POWER_STATES.P1;
  return POWER_STATES.P0;
}

// Adaptive batch size lookup — used by dispatch to size work items
export function batchSizeForState(state) {
  return {
    P0:  256,
    P1:  192,
    P2:  128,
    P3:  64,
    RC6: 0,
  }[state.id] ?? 64;
}

// ─── Thermal throttle rules ───────────────────────────────────────────────────

export const THERMAL_THRESHOLDS = Object.freeze({
  critical: 90,  // force P3
  high:     80,  // request voltage reduction
  nominal:  70,  // normal operation
  cool:     60,  // can sustain P0
});

export function thermalAction(tempC) {
  if (tempC >= THERMAL_THRESHOLDS.critical) return { action: 'throttle_p3',    undervolt: false };
  if (tempC >= THERMAL_THRESHOLDS.high)     return { action: 'undervolt_50mv', undervolt: true  };
  return { action: 'nominal', undervolt: false };
}

// ─── Work-steal heuristic ─────────────────────────────────────────────────────
// If GPU utilization < 30%, offload to CPU to avoid EU stalls.

export function shouldOffloadToCPU(gpuUtilization) {
  return gpuUtilization < 0.30;
}

// ─── Power-aware dispatch config ─────────────────────────────────────────────

export function dispatchConfig(state) {
  return Object.freeze({
    tileSize:      state.tileSize,
    threadCount:   state.threadCount,
    mayanBase:     state.mayanBase,
    batchSize:     batchSizeForState(state),
    compressionMode: state.id === 'P0' ? 'none'
                   : state.id === 'P1' ? 'mayan_delta'
                   : 'spectral_dct',
  });
}
