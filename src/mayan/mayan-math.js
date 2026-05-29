// Mayan Math v1 — base-20 positional system, Long Count calendar, AtomicMayan ops
// Compatible with ECMAScript BigInt (no division remainder sign issues).
//
// Long Count positional weights (modified base-20):
//   kin=1  uinal=20  tun=360  katun=7200  baktun=144000
// Digits 0..19 map to glyph strings (shell, dots, bars).

// ─── Glyph ↔ value maps ───────────────────────────────────────────────────────

export const MAYAN_GLYPHS = Object.freeze({
  '𝋠':          0n,   // shell (zero)
  '●':           1n,
  '●●':          2n,
  '●●●':         3n,
  '●●●●':        4n,
  '⎯':           5n,
  '⎯●':          6n,
  '⎯●●':         7n,
  '⎯●●●':        8n,
  '⎯●●●●':       9n,
  '⎯⎯':          10n,
  '⎯⎯●':         11n,
  '⎯⎯●●':        12n,
  '⎯⎯●●●':       13n,
  '⎯⎯●●●●':      14n,
  '⎯⎯⎯':         15n,
  '⎯⎯⎯●':        16n,
  '⎯⎯⎯●●':       17n,
  '⎯⎯⎯●●●':      18n,
  '⎯⎯⎯●●●●':     19n,
});

export const MAYAN_GLYPH_BY_VALUE = Object.freeze(
  Object.fromEntries(Object.entries(MAYAN_GLYPHS).map(([g, v]) => [String(v), g]))
);

// ─── Long Count positions ─────────────────────────────────────────────────────

export const LONG_COUNT_POSITIONS = Object.freeze([
  { index: 0, weight: 1n,       name: 'kines'   },
  { index: 1, weight: 20n,      name: 'uinals'  },
  { index: 2, weight: 360n,     name: 'tuns'    },
  { index: 3, weight: 7200n,    name: 'katuns'  },
  { index: 4, weight: 144000n,  name: 'baktuns' },
]);

// GMT correlation constant: Mayan epoch (0.0.0.0.0) → Julian Day 584283
export const MAYAN_JD_CORRELATION = 584283n;

// ─── Digit ↔ BigInt conversion ────────────────────────────────────────────────

// BigInt → Uint8Array(5): [kin, uinal, tun, katun, baktun]
export function bigIntToDigits(n) {
  const bn = BigInt(n);
  return new Uint8Array([
    Number(bn % 20n),
    Number((bn % 360n) / 20n),
    Number((bn % 7200n) / 360n),
    Number((bn % 144000n) / 7200n),
    Number(bn / 144000n),
  ]);
}

// Uint8Array(5) → BigInt
export function digitsToBase(digits) {
  return (
    BigInt(digits[0]) +
    BigInt(digits[1]) * 20n +
    BigInt(digits[2]) * 360n +
    BigInt(digits[3]) * 7200n +
    BigInt(digits[4]) * 144000n
  );
}

// Single-digit glyph string → BigInt value (0n–19n)
export function glyphToValue(glyph) {
  const v = MAYAN_GLYPHS[glyph.trim()];
  if (v === undefined) throw new Error(`Unknown Mayan glyph: "${glyph}"`);
  return v;
}

// BigInt value (0n–19n) → glyph string
export function valueToGlyph(n) {
  const g = MAYAN_GLYPH_BY_VALUE[String(n)];
  if (!g) throw new Error(`No Mayan glyph for value: ${n}`);
  return g;
}

// Space-separated glyph sequence (highest position first) → BigInt
// e.g. "𝋠 ⎯⎯●●" = 0 baktuns + 12 kines = 12n
export function mayanGlyphToBigInt(sequence) {
  const parts = sequence.trim().split(/\s+/);
  let result = 0n;
  for (let i = 0; i < parts.length; i++) {
    const posIdx  = parts.length - 1 - i;          // index into LONG_COUNT_POSITIONS
    const pos     = LONG_COUNT_POSITIONS[posIdx] ?? { weight: 1n };
    result += glyphToValue(parts[i]) * pos.weight;
  }
  return result;
}

// BigInt → space-separated glyph sequence (highest position first)
export function bigIntToMayanGlyph(n, positions = 5) {
  const digits = bigIntToDigits(BigInt(n));
  const glyphs = [];
  for (let i = positions - 1; i >= 0; i--) {
    glyphs.push(MAYAN_GLYPH_BY_VALUE[String(digits[i])] ?? '𝋠');
  }
  return glyphs.join(' ');
}

// ─── Long Count arithmetic ────────────────────────────────────────────────────

export function mayanAdd(a, b) {
  const av = a instanceof Uint8Array ? digitsToBase(a) : BigInt(a);
  const bv = b instanceof Uint8Array ? digitsToBase(b) : BigInt(b);
  return bigIntToDigits(av + bv);
}

export function mayanSub(a, b) {
  const av = a instanceof Uint8Array ? digitsToBase(a) : BigInt(a);
  const bv = b instanceof Uint8Array ? digitsToBase(b) : BigInt(b);
  if (av < bv) throw new Error('Mayan subtraction: result would be negative');
  return bigIntToDigits(av - bv);
}

export function mayanMul(a, scalar) {
  const av = a instanceof Uint8Array ? digitsToBase(a) : BigInt(a);
  return bigIntToDigits(av * BigInt(scalar));
}

// ─── Julian Day conversion ────────────────────────────────────────────────────

export function mayanToJD(digits) {
  return Number(digitsToBase(digits) + MAYAN_JD_CORRELATION);
}

export function jdToMayan(jd) {
  const n = BigInt(jd) - MAYAN_JD_CORRELATION;
  if (n < 0n) throw new Error('JD before Mayan epoch (JD 584283)');
  return bigIntToDigits(n);
}

// ─── AtomicMayan — ops on Uint8Array(5) ──────────────────────────────────────
// Works on both regular and SharedArrayBuffer-backed Uint8Array.
// For true Atomics, wrap compareExchange calls at the caller level.

export const AtomicMayan = Object.freeze({
  store(buffer, value) {
    const digits = bigIntToDigits(typeof value === 'string' ? mayanGlyphToBigInt(value) : BigInt(value));
    for (let i = 0; i < 5; i++) buffer[i] = digits[i];
    return buffer;
  },

  load(buffer) {
    return digitsToBase(buffer);
  },

  loadGlyph(buffer, positions = 5) {
    return bigIntToMayanGlyph(digitsToBase(buffer), positions);
  },

  add(buffer, addend) {
    const cur = digitsToBase(buffer);
    const add = typeof addend === 'string' ? mayanGlyphToBigInt(addend) : BigInt(addend);
    const next = cur + add;
    const digits = bigIntToDigits(next);
    for (let i = 0; i < 5; i++) buffer[i] = digits[i];
    return next;
  },

  sub(buffer, subtrahend) {
    const cur  = digitsToBase(buffer);
    const sub  = typeof subtrahend === 'string' ? mayanGlyphToBigInt(subtrahend) : BigInt(subtrahend);
    const next = cur - sub;
    if (next < 0n) throw new Error('AtomicMayan.sub: result would be negative');
    const digits = bigIntToDigits(next);
    for (let i = 0; i < 5; i++) buffer[i] = digits[i];
    return next;
  },

  fromBigInt: bigIntToDigits,
  toBigInt:   digitsToBase,
});

// ─── Formal grammar (EBNF reference, not parsed) ─────────────────────────────

export const ATOMIC_MATH_GRAMMAR = Object.freeze(`
AtomicMath ::= AtomicMayanMath | AtomicLinearAlgebra | AtomicMayanLinearHybrid

AtomicMayanMath ::=
    "@atomic.mayan." ("store" | "load" | "add" | "subtract" | "multiply" | "divide")
    "(" MayanNumber "," MayanValue ["," MayanValue] ")"

AtomicLinearAlgebra ::=
    "@atomic." ("vector" | "matrix" | "tensor") "."
    ("add" | "multiply" | "transpose" | "dot" | "inverse" | "eigenvalue")
    "(" OperandList ")"

MayanLinearHybrid ::=
    "@atomic.mayan.solve" "(" Matrix "," Vector "," Unknowns ")"

MayanValue ::= MayanGlyphSequence | BigInt | Integer

MayanGlyphSequence ::= (MayanDigit " ")* MayanDigit

MayanDigit ::=
    "𝋠" | "●" | "●●" | "●●●" | "●●●●" | "⎯" | "⎯●" | "⎯●●" |
    "⎯●●●" | "⎯●●●●" | "⎯⎯" | "⎯⎯●" | "⎯⎯●●" | "⎯⎯●●●" |
    "⎯⎯●●●●" | "⎯⎯⎯" | "⎯⎯⎯●" | "⎯⎯⎯●●" | "⎯⎯⎯●●●" | "⎯⎯⎯●●●●"
`);
